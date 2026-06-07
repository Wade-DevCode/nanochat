"""
新的升级版聊天模式，因为上一个版本以来很多代码已经改变。

目前设计为仅单 GPU 运行：
python -m scripts.chat_cli
"""
import argparse
import torch
from nanochat.common import compute_init, autodetect_device_type
from nanochat.engine import Engine
from nanochat.checkpoint_manager import load_model

parser = argparse.ArgumentParser(description='和模型聊天')
parser.add_argument('-i', '--source', type=str, default="sft", help="模型来源：sft|rl")
parser.add_argument('-g', '--model-tag', type=str, default=None, help='要加载的模型 tag')
parser.add_argument('-s', '--step', type=int, default=None, help='要加载的 step')
parser.add_argument('-p', '--prompt', type=str, default='', help='向模型发送 prompt，并只获取一次回复')
parser.add_argument('-t', '--temperature', type=float, default=0.6, help='生成用 temperature')
parser.add_argument('-k', '--top-k', type=int, default=50, help='Top-k 采样参数')
parser.add_argument('--device-type', type=str, default='', choices=['cuda', 'cpu', 'mps'], help='评估设备类型：cuda|cpu|mps。留空 => 自动检测')
args = parser.parse_args()

# Init the model and tokenizer

device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
model, tokenizer, meta = load_model(args.source, device, phase="eval", model_tag=args.model_tag, step=args.step)

# Special tokens for the chat state machine
bos = tokenizer.get_bos_token_id()
user_start, user_end = tokenizer.encode_special("<|user_start|>"), tokenizer.encode_special("<|user_end|>")
assistant_start, assistant_end = tokenizer.encode_special("<|assistant_start|>"), tokenizer.encode_special("<|assistant_end|>")

# Create Engine for efficient generation
engine = Engine(model, tokenizer)

print("\nNanoChat 交互模式")
print("-" * 50)
print("输入 'quit' 或 'exit' 结束对话")
print("输入 'clear' 开始新对话")
print("-" * 50)

conversation_tokens = [bos]

while True:

    if args.prompt:
        # Get the prompt from the launch command
        user_input = args.prompt
    else:
        # Get the prompt interactively from the console
        try:
            user_input = input("\n用户：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

    # Handle special commands
    if user_input.lower() in ['quit', 'exit']:
        print("再见！")
        break

    if user_input.lower() == 'clear':
        conversation_tokens = [bos]
        print("对话已清空。")
        continue

    if not user_input:
        continue

    # Add User message to the conversation
    conversation_tokens.append(user_start)
    conversation_tokens.extend(tokenizer.encode(user_input))
    conversation_tokens.append(user_end)

    # Kick off the assistant
    conversation_tokens.append(assistant_start)
    generate_kwargs = {
        "num_samples": 1,
        "max_tokens": 256,
        "temperature": args.temperature,
        "top_k": args.top_k,
    }
    response_tokens = []
    print("\n助手：", end="", flush=True)
    for token_column, token_masks in engine.generate(conversation_tokens, **generate_kwargs):
        token = token_column[0] # pop the batch dimension (num_samples=1)
        response_tokens.append(token)
        token_text = tokenizer.decode([token])
        print(token_text, end="", flush=True)
    print()
    # we have to ensure that the assistant end token is the last token
    # so even if generation ends due to max tokens, we have to append it to the end
    if response_tokens[-1] != assistant_end:
        response_tokens.append(assistant_end)
    conversation_tokens.extend(response_tokens)

    # In the prompt mode, we only want a single response and exit
    if args.prompt:
        break
