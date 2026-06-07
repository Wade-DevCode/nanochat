#!/usr/bin/env python3
"""
统一 Web 聊天服务器：由一个 FastAPI 实例同时提供 UI 和 API。

使用数据并行将请求分发到多张 GPU。每张 GPU 加载一份完整模型，
传入请求会分发给可用 worker。

启动示例：

- 单张可用 GPU（默认）
python -m scripts.chat_web

- 4 张 GPU
python -m scripts.chat_web --num-gpus 4

要聊天，请打开控制台打印的 URL。（如果在云机器上，请确认使用公网 IP）

端点：
  GET  /           - 聊天 UI
  POST /chat/completions - 聊天 API（仅流式）
  GET  /health     - 带 worker pool 状态的健康检查
  GET  /stats      - Worker pool 统计和 GPU 使用情况

滥用防护：
  - 每个请求最多 500 条消息
  - 每条消息最多 8000 字符
  - 整个对话最多 32000 字符
  - Temperature 限制在 0.0-2.0
  - Top-k 限制在 0-200（0 表示禁用 top-k filtering，使用完整词表）
  - Max tokens 限制在 1-4096
"""

import argparse
import json
import os
import torch
import asyncio
import logging
import random
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse
from pydantic import BaseModel
from typing import List, Optional, AsyncGenerator
from dataclasses import dataclass
from nanochat.common import compute_init, autodetect_device_type
from nanochat.checkpoint_manager import load_model
from nanochat.engine import Engine

# 滥用防护限制
MAX_MESSAGES_PER_REQUEST = 500
MAX_MESSAGE_LENGTH = 8000
MAX_TOTAL_CONVERSATION_LENGTH = 32000
MIN_TEMPERATURE = 0.0
MAX_TEMPERATURE = 2.0
MIN_TOP_K = 0 # 0 表示禁用 top-k filtering，使用完整词表
MAX_TOP_K = 200
MIN_MAX_TOKENS = 1
MAX_MAX_TOKENS = 4096

parser = argparse.ArgumentParser(description='NanoChat Web 服务器')
parser.add_argument('-n', '--num-gpus', type=int, default=1, help='要使用的 GPU 数量（默认：1）')
parser.add_argument('-i', '--source', type=str, default="sft", help="模型来源：sft|rl")
parser.add_argument('-t', '--temperature', type=float, default=0.8, help='默认生成 temperature')
parser.add_argument('-k', '--top-k', type=int, default=50, help='默认 top-k 采样参数')
parser.add_argument('-m', '--max-tokens', type=int, default=512, help='默认生成 max tokens')
parser.add_argument('-g', '--model-tag', type=str, default=None, help='要加载的模型 tag')
parser.add_argument('-s', '--step', type=int, default=None, help='要加载的 step')
parser.add_argument('-p', '--port', type=int, default=8000, help='服务器运行端口')
parser.add_argument('--device-type', type=str, default='', choices=['cuda', 'cpu', 'mps'], help='评估设备类型：cuda|cpu|mps。留空 => 自动检测')
parser.add_argument('--host', type=str, default='0.0.0.0', help='服务器绑定的 host')
args = parser.parse_args()

# 配置对话流量日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)

@dataclass
class Worker:
    """在指定 GPU 上加载了模型的 worker。"""
    gpu_id: int
    device: torch.device
    engine: Engine
    tokenizer: object

class WorkerPool:
    """Worker 池；每个 worker 在不同 GPU 上持有一个模型副本。"""

    def __init__(self, num_gpus: Optional[int] = None):
        if num_gpus is None:
            if device_type == "cuda":
                num_gpus = torch.cuda.device_count()
            else:
                num_gpus = 1 # e.g. cpu|mps
        self.num_gpus = num_gpus
        self.workers: List[Worker] = []
        self.available_workers: asyncio.Queue = asyncio.Queue()

    async def initialize(self, source: str, model_tag: Optional[str] = None, step: Optional[int] = None):
        """在每张 GPU 上加载模型。"""
        print(f"正在使用 {self.num_gpus} 张 GPU 初始化 worker pool...")
        if self.num_gpus > 1:
            assert device_type == "cuda", "只有 CUDA 支持多个 worker/GPU。cpu|mps 不支持。"

        for gpu_id in range(self.num_gpus):

            if device_type == "cuda":
                device = torch.device(f"cuda:{gpu_id}")
                print(f"正在 GPU {gpu_id} 上加载模型...")
            else:
                device = torch.device(device_type) # e.g. cpu|mps
                print(f"正在 {device_type} 上加载模型...")

            model, tokenizer, _ = load_model(source, device, phase="eval", model_tag=model_tag, step=step)
            engine = Engine(model, tokenizer)
            worker = Worker(
                gpu_id=gpu_id,
                device=device,
                engine=engine,
                tokenizer=tokenizer,
            )
            self.workers.append(worker)
            await self.available_workers.put(worker)

        print(f"全部 {self.num_gpus} 个 worker 已初始化！")

    async def acquire_worker(self) -> Worker:
        """从池中获取一个可用 worker。"""
        return await self.available_workers.get()

    async def release_worker(self, worker: Worker):
        """将 worker 归还到池中。"""
        await self.available_workers.put(worker)

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_k: Optional[int] = None

def validate_chat_request(request: ChatRequest):
    """验证聊天请求以防止滥用。"""
    # 检查消息数量
    if len(request.messages) == 0:
        raise HTTPException(status_code=400, detail="至少需要一条消息")
    if len(request.messages) > MAX_MESSAGES_PER_REQUEST:
        raise HTTPException(
            status_code=400,
            detail=f"消息过多。每个请求最多允许 {MAX_MESSAGES_PER_REQUEST} 条消息"
        )

    # 检查单条消息长度和整体对话长度
    total_length = 0
    for i, message in enumerate(request.messages):
        if not message.content:
            raise HTTPException(status_code=400, detail=f"消息 {i} 内容为空")

        msg_length = len(message.content)
        if msg_length > MAX_MESSAGE_LENGTH:
            raise HTTPException(
                status_code=400,
                detail=f"消息 {i} 过长。每条消息最多允许 {MAX_MESSAGE_LENGTH} 个字符"
            )
        total_length += msg_length

    if total_length > MAX_TOTAL_CONVERSATION_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"总对话过长。最多允许 {MAX_TOTAL_CONVERSATION_LENGTH} 个字符"
        )

    # 验证 role 值
    for i, message in enumerate(request.messages):
        if message.role not in ["user", "assistant"]:
            raise HTTPException(
                status_code=400,
                detail=f"消息 {i} 的 role 无效。必须是 'user'、'assistant' 或 'system'"
            )

    # 验证 temperature
    if request.temperature is not None:
        if not (MIN_TEMPERATURE <= request.temperature <= MAX_TEMPERATURE):
            raise HTTPException(
                status_code=400,
                detail=f"Temperature 必须在 {MIN_TEMPERATURE} 和 {MAX_TEMPERATURE} 之间"
            )

    # 验证 top_k
    if request.top_k is not None:
        if not (MIN_TOP_K <= request.top_k <= MAX_TOP_K):
            raise HTTPException(
                status_code=400,
                detail=f"top_k 必须在 {MIN_TOP_K} 和 {MAX_TOP_K} 之间"
            )

    # 验证 max_tokens
    if request.max_tokens is not None:
        if not (MIN_MAX_TOKENS <= request.max_tokens <= MAX_MAX_TOKENS):
            raise HTTPException(
                status_code=400,
                detail=f"max_tokens 必须在 {MIN_MAX_TOKENS} 和 {MAX_MAX_TOKENS} 之间"
            )

@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时在所有 GPU 上加载模型。"""
    print("正在跨 GPU 加载 nanochat 模型...")
    app.state.worker_pool = WorkerPool(num_gpus=args.num_gpus)
    await app.state.worker_pool.initialize(args.source, model_tag=args.model_tag, step=args.step)
    print(f"服务器已就绪：http://localhost:{args.port}")
    yield

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """提供聊天 UI。"""
    ui_html_path = os.path.join("nanochat", "ui.html")
    with open(ui_html_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    # 替换 API_URL 以使用同源地址
    html_content = html_content.replace(
        "const API_URL = `http://${window.location.hostname}:8000`;",
        "const API_URL = '';"
    )
    return HTMLResponse(content=html_content)


@app.get("/logo.svg")
async def logo():
    """提供 favicon 和 header 使用的 NanoChat logo。"""
    logo_path = os.path.join("nanochat", "logo.svg")
    return FileResponse(logo_path, media_type="image/svg+xml")

async def generate_stream(
    worker: Worker,
    tokens,
    temperature=None,
    max_new_tokens=None,
    top_k=None
) -> AsyncGenerator[str, None]:
    """以流式方式生成助手回复。"""
    temperature = temperature if temperature is not None else args.temperature
    max_new_tokens = max_new_tokens if max_new_tokens is not None else args.max_tokens
    top_k = top_k if top_k is not None else args.top_k

    assistant_end = worker.tokenizer.encode_special("<|assistant_end|>")
    bos = worker.tokenizer.get_bos_token_id()

    # 累积 token，以正确处理多字节 UTF-8 字符（如 emoji）
    accumulated_tokens = []
    # 跟踪上一个完整 UTF-8 字符串（不含替换字符）
    last_clean_text = ""

    for token_column, token_masks in worker.engine.generate(
        tokens,
        num_samples=1,
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        seed=random.randint(0, 2**31 - 1)
    ):
        token = token_column[0]

        # 停止条件
        if token == assistant_end or token == bos:
            break

        # 将 token 追加到序列
        accumulated_tokens.append(token)
        # 解码所有累积 token，以正确处理 UTF-8
        # decode 很高效，基本是查表和字符串拼接
        current_text = worker.tokenizer.decode(accumulated_tokens)
        # 只有文本不以替换字符结尾时才发出
        # 这样可确保不会发出不完整的 UTF-8 序列
        if not current_text.endswith('�'):
            # Extract only the new text since last clean decode
            new_text = current_text[len(last_clean_text):]
            if new_text:  # 只有存在新内容时才 yield
                yield f"data: {json.dumps({'token': new_text, 'gpu': worker.gpu_id}, ensure_ascii=False)}\n\n"
                last_clean_text = current_text

    yield f"data: {json.dumps({'done': True})}\n\n"

@app.post("/chat/completions")
async def chat_completions(request: ChatRequest):
    """聊天补全端点（仅流式）：使用 worker pool 支持多 GPU。"""

    # 基本验证，防止滥用
    validate_chat_request(request)

    # 将传入对话记录到控制台
    logger.info("="*20)
    for i, message in enumerate(request.messages):
        logger.info(f"[{message.role.upper()}]: {message.content}")
    logger.info("-"*20)

    # 从池中获取 worker（如果都忙则等待）
    worker_pool = app.state.worker_pool
    worker = await worker_pool.acquire_worker()

    try:
        # 构建对话 token
        bos = worker.tokenizer.get_bos_token_id()
        user_start = worker.tokenizer.encode_special("<|user_start|>")
        user_end = worker.tokenizer.encode_special("<|user_end|>")
        assistant_start = worker.tokenizer.encode_special("<|assistant_start|>")
        assistant_end = worker.tokenizer.encode_special("<|assistant_end|>")

        conversation_tokens = [bos]
        for message in request.messages:
            if message.role == "user":
                conversation_tokens.append(user_start)
                conversation_tokens.extend(worker.tokenizer.encode(message.content))
                conversation_tokens.append(user_end)
            elif message.role == "assistant":
                conversation_tokens.append(assistant_start)
                conversation_tokens.extend(worker.tokenizer.encode(message.content))
                conversation_tokens.append(assistant_end)

        conversation_tokens.append(assistant_start)

        # 流式响应；完成后释放 worker
        response_tokens = []
        async def stream_and_release():
            try:
                async for chunk in generate_stream(
                    worker,
                    conversation_tokens,
                    temperature=request.temperature,
                    max_new_tokens=request.max_tokens,
                    top_k=request.top_k
                ):
                    # 累积回复用于日志
                    chunk_data = json.loads(chunk.replace("data: ", "").strip())
                    if "token" in chunk_data:
                        response_tokens.append(chunk_data["token"])
                    yield chunk
            finally:
                # 将助手回复记录到控制台
                full_response = "".join(response_tokens)
                logger.info(f"[ASSISTANT] (GPU {worker.gpu_id}): {full_response}")
                logger.info("="*20)
                # 流式完成后将 worker 归还到池中
                await worker_pool.release_worker(worker)

        return StreamingResponse(
            stream_and_release(),
            media_type="text/event-stream"
        )
    except Exception as e:
        # 即使出错也确保释放 worker
        await worker_pool.release_worker(worker)
        raise e

@app.get("/health")
async def health():
    """健康检查端点。"""
    worker_pool = getattr(app.state, 'worker_pool', None)
    return {
        "status": "ok",
        "ready": worker_pool is not None and len(worker_pool.workers) > 0,
        "num_gpus": worker_pool.num_gpus if worker_pool else 0,
        "available_workers": worker_pool.available_workers.qsize() if worker_pool else 0
    }

@app.get("/stats")
async def stats():
    """获取 worker pool 统计信息。"""
    worker_pool = app.state.worker_pool
    return {
        "total_workers": len(worker_pool.workers),
        "available_workers": worker_pool.available_workers.qsize(),
        "busy_workers": len(worker_pool.workers) - worker_pool.available_workers.qsize(),
        "workers": [
            {
                "gpu_id": w.gpu_id,
                "device": str(w.device)
            } for w in worker_pool.workers
        ]
    }

if __name__ == "__main__":
    import uvicorn
    print("启动 NanoChat Web 服务器")
    print(f"Temperature: {args.temperature}, Top-k: {args.top_k}, Max tokens: {args.max_tokens}")
    uvicorn.run(app, host=args.host, port=args.port)
