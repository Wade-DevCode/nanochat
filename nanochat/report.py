"""
用于生成训练报告卡片的工具。代码比平时更乱，后续会修。
"""

import os
import re
import shutil
import subprocess
import socket
import datetime
import platform
import psutil
import torch

def run_command(cmd):
    """运行 shell 命令并返回输出；失败则返回 None。"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)
        # 如果有 stdout 就返回（即使 xargs 中某些文件失败）
        if result.stdout.strip():
            return result.stdout.strip()
        if result.returncode == 0:
            return ""
        return None
    except:
        return None

def get_git_info():
    """获取当前 git commit、branch 和 dirty 状态。"""
    info = {}
    info['commit'] = run_command("git rev-parse --short HEAD") or "unknown"
    info['branch'] = run_command("git rev-parse --abbrev-ref HEAD") or "unknown"

    # 检查仓库是否 dirty（有未提交改动）
    status = run_command("git status --porcelain")
    info['dirty'] = bool(status) if status is not None else False

    # 获取 commit message
    info['message'] = run_command("git log -1 --pretty=%B") or ""
    info['message'] = info['message'].split('\n')[0][:80]  # 第一行，截断

    return info

def get_gpu_info():
    """获取 GPU 信息。"""
    if not torch.cuda.is_available():
        return {"available": False}

    num_devices = torch.cuda.device_count()
    info = {
        "available": True,
        "count": num_devices,
        "names": [],
        "memory_gb": []
    }

    for i in range(num_devices):
        props = torch.cuda.get_device_properties(i)
        info["names"].append(props.name)
        info["memory_gb"].append(props.total_memory / (1024**3))

    # 获取 CUDA 版本
    info["cuda_version"] = torch.version.cuda or "unknown"

    return info

def get_system_info():
    """获取系统信息。"""
    info = {}

    # 基本系统信息
    info['hostname'] = socket.gethostname()
    info['platform'] = platform.system()
    info['python_version'] = platform.python_version()
    info['torch_version'] = torch.__version__

    # CPU 和内存
    info['cpu_count'] = psutil.cpu_count(logical=False)
    info['cpu_count_logical'] = psutil.cpu_count(logical=True)
    info['memory_gb'] = psutil.virtual_memory().total / (1024**3)

    # 用户和环境
    info['user'] = os.environ.get('USER', 'unknown')
    info['nanochat_base_dir'] = os.environ.get('NANOCHAT_BASE_DIR', 'out')
    info['working_dir'] = os.getcwd()

    return info

def estimate_cost(gpu_info, runtime_hours=None):
    """根据 GPU 类型和运行时间估算训练成本。"""

    # 粗略价格，来自 Lambda Cloud
    default_rate = 2.0
    gpu_hourly_rates = {
        "H100": 3.00,
        "A100": 1.79,
        "V100": 0.55,
    }

    if not gpu_info.get("available"):
        return None

    # 尝试从名称识别 GPU 类型
    hourly_rate = None
    gpu_name = gpu_info["names"][0] if gpu_info["names"] else "unknown"
    for gpu_type, rate in gpu_hourly_rates.items():
        if gpu_type in gpu_name:
            hourly_rate = rate * gpu_info["count"]
            break

    if hourly_rate is None:
        hourly_rate = default_rate * gpu_info["count"]  # 默认估算

    return {
        "hourly_rate": hourly_rate,
        "gpu_type": gpu_name,
        "estimated_total": hourly_rate * runtime_hours if runtime_hours else None
    }

def generate_header():
    """生成训练报告头部。"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    git_info = get_git_info()
    gpu_info = get_gpu_info()
    sys_info = get_system_info()
    cost_info = estimate_cost(gpu_info)

    header = f"""# nanochat 训练报告

生成时间：{timestamp}

## 环境

### Git 信息
- 分支：{git_info['branch']}
- Commit: {git_info['commit']} {"(dirty)" if git_info['dirty'] else "(clean)"}
- Message：{git_info['message']}

### 硬件
- 平台：{sys_info['platform']}
- CPU：{sys_info['cpu_count']} cores ({sys_info['cpu_count_logical']} logical)
- 内存：{sys_info['memory_gb']:.1f} GB
"""

    if gpu_info.get("available"):
        gpu_names = ", ".join(set(gpu_info["names"]))
        total_vram = sum(gpu_info["memory_gb"])
        header += f"""- GPU：{gpu_info['count']}x {gpu_names}
- GPU 显存：{total_vram:.1f} GB total
- CUDA 版本：{gpu_info['cuda_version']}
"""
    else:
        header += "- GPU：无可用 GPU\n"

    if cost_info and cost_info["hourly_rate"] > 0:
        header += f"""- 小时费率：${cost_info['hourly_rate']:.2f}/hour\n"""

    header += f"""
### 软件
- Python: {sys_info['python_version']}
- PyTorch: {sys_info['torch_version']}

"""

    # bloat 指标：只统计 git 跟踪的源文件行数/字符数
    extensions = ['py', 'md', 'rs', 'html', 'toml', 'sh']
    git_patterns = ' '.join(f"'*.{ext}'" for ext in extensions)
    files_output = run_command(f"git ls-files -- {git_patterns}")
    file_list = [f for f in (files_output or '').split('\n') if f]
    num_files = len(file_list)
    num_lines = 0
    num_chars = 0
    if num_files > 0:
        wc_output = run_command(f"git ls-files -- {git_patterns} | xargs wc -lc 2>/dev/null")
        if wc_output:
            total_line = wc_output.strip().split('\n')[-1]
            parts = total_line.split()
            if len(parts) >= 2:
                num_lines = int(parts[0])
                num_chars = int(parts[1])
    num_tokens = num_chars // 4  # 假设约 4 字符/token

    # 通过 uv.lock 统计依赖
    uv_lock_lines = 0
    if os.path.exists('uv.lock'):
        with open('uv.lock', 'r', encoding='utf-8') as f:
            uv_lock_lines = len(f.readlines())

    header += f"""
### Bloat
- 字符数：{num_chars:,}
- 行数：{num_lines:,}
- 文件数：{num_files:,}
- Tokens（约）：{num_tokens:,}
- 依赖数（uv.lock 行数）：{uv_lock_lines:,}

"""
    return header

# -----------------------------------------------------------------------------

def slugify(text):
    """将文本字符串转成 slug。"""
    return text.lower().replace(" ", "-")

# 期望的文件及其顺序
EXPECTED_FILES = [
    "tokenizer-training.md",
    "tokenizer-evaluation.md",
    "base-model-training.md",
    "base-model-loss.md",
    "base-model-evaluation.md",
    "chat-sft.md",
    "chat-evaluation-sft.md",
    "chat-rl.md",
    "chat-evaluation-rl.md",
]
# 当前关注的指标
chat_metrics = ["ARC-Easy", "ARC-Challenge", "MMLU", "GSM8K", "HumanEval", "ChatCORE"]

def extract(section, keys):
    """从 section 中提取单个 key 的简单函数。"""
    if not isinstance(keys, list):
        keys = [keys] # 方便调用
    out = {}
    for line in section.split("\n"):
        for key in keys:
            if key in line:
                out[key] = line.split(":")[1].strip()
    return out

def extract_timestamp(content, prefix):
    """从内容中提取带指定前缀的时间戳。"""
    for line in content.split('\n'):
        if line.startswith(prefix):
            time_str = line.split(":", 1)[1].strip()
            try:
                return datetime.datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
            except:
                pass
    return None

class Report:
    """维护一组日志，并生成最终 Markdown 报告。"""

    def __init__(self, report_dir):
        os.makedirs(report_dir, exist_ok=True)
        self.report_dir = report_dir

    def log(self, section, data):
        """将一节数据记录到报告。"""
        slug = slugify(section)
        file_name = f"{slug}.md"
        file_path = os.path.join(self.report_dir, file_name)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"## {section}\n")
            f.write(f"timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            for item in data:
                if not item:
                    # 跳过 None、空 dict 等 falsy 值
                    continue
                if isinstance(item, str):
                    # 直接写入字符串
                    f.write(item)
                else:
                    # 渲染 dict
                    for k, v in item.items():
                        if isinstance(v, float):
                            vstr = f"{v:.4f}"
                        elif isinstance(v, int) and v >= 10000:
                            vstr = f"{v:,.0f}"
                        else:
                            vstr = str(v)
                        f.write(f"- {k}: {vstr}\n")
            f.write("\n")
        return file_path

    def generate(self):
        """生成最终报告。"""
        report_dir = self.report_dir
        report_file = os.path.join(report_dir, "report.md")
        print(f"正在生成报告到 {report_file}")
        final_metrics = {} # 最后要作为表格添加的重要最终指标
        start_time = None
        end_time = None
        with open(report_file, "w", encoding="utf-8") as out_file:
            # 先写入 header
            header_file = os.path.join(report_dir, "header.md")
            if os.path.exists(header_file):
                with open(header_file, "r", encoding="utf-8") as f:
                    header_content = f.read()
                    out_file.write(header_content)
                    start_time = extract_timestamp(header_content, "Run started:")
                    # 捕获 bloat 数据，供后续 summary 使用（Bloat header 之后到 \n\n 之间的内容）
                    bloat_data = re.search(r"### Bloat\n(.*?)\n\n", header_content, re.DOTALL)
                    bloat_data = bloat_data.group(1) if bloat_data else ""
            else:
                start_time = None # 这样就不会写入总墙钟时间
                bloat_data = "[bloat data missing]"
                print(f"警告：{header_file} 不存在。是不是忘了运行 `nanochat reset`？")
            # 处理各个单独章节
            for file_name in EXPECTED_FILES:
                section_file = os.path.join(report_dir, file_name)
                if not os.path.exists(section_file):
                    print(f"警告：{section_file} 不存在，跳过")
                    continue
                with open(section_file, "r", encoding="utf-8") as in_file:
                    section = in_file.read()
                # 从此 section 中提取时间戳（最后一个 section 的时间戳会作为 end_time）
                if "rl" not in file_name:
                    # 跳过 RL section 的 end_time 计算，因为 RL 仍是实验性质
                    end_time = extract_timestamp(section, "timestamp:")
                # 从 section 中提取最重要的指标
                if file_name == "base-model-evaluation.md":
                    final_metrics["base"] = extract(section, "CORE")
                if file_name == "chat-evaluation-sft.md":
                    final_metrics["sft"] = extract(section, chat_metrics)
                if file_name == "chat-evaluation-rl.md":
                    final_metrics["rl"] = extract(section, "GSM8K") # RL 只评估 GSM8K
                # 追加此报告 section
                out_file.write(section)
                out_file.write("\n")
            # 添加最终指标表
            out_file.write("## 汇总\n\n")
            # 从 header 复制 bloat 指标
            out_file.write(bloat_data)
            out_file.write("\n\n")
            # 收集所有唯一指标名
            all_metrics = set()
            for stage_metrics in final_metrics.values():
                all_metrics.update(stage_metrics.keys())
            # 自定义排序：CORE 在最前，ChatCORE 在最后，其余在中间
            all_metrics = sorted(all_metrics, key=lambda x: (x != "CORE", x == "ChatCORE", x))
            # 固定列宽
            stages = ["base", "sft", "rl"]
            metric_width = 15
            value_width = 8
            # 写入表头
            header = f"| {'指标'.ljust(metric_width)} |"
            for stage in stages:
                header += f" {stage.upper().ljust(value_width)} |"
            out_file.write(header + "\n")
            # 写入分隔线
            separator = f"|{'-' * (metric_width + 2)}|"
            for stage in stages:
                separator += f"{'-' * (value_width + 2)}|"
            out_file.write(separator + "\n")
            # 写入表格行
            for metric in all_metrics:
                row = f"| {metric.ljust(metric_width)} |"
                for stage in stages:
                    value = final_metrics.get(stage, {}).get(metric, "-")
                    row += f" {str(value).ljust(value_width)} |"
                out_file.write(row + "\n")
            out_file.write("\n")
            # 计算并写入总墙钟时间
            if start_time and end_time:
                duration = end_time - start_time
                total_seconds = int(duration.total_seconds())
                hours = total_seconds // 3600
                minutes = (total_seconds % 3600) // 60
                out_file.write(f"总墙钟时间：{hours}h{minutes}m\n")
            else:
                out_file.write("总墙钟时间：unknown\n")
        # 同时把 report.md 复制到当前目录，方便查看
        print("正在将 report.md 复制到当前目录以便查看")
        shutil.copy(report_file, "report.md")
        return report_file

    def reset(self):
        """重置报告。"""
        # 移除 section 文件
        for file_name in EXPECTED_FILES:
            file_path = os.path.join(self.report_dir, file_name)
            if os.path.exists(file_path):
                os.remove(file_path)
        # 如果 report.md 存在则移除
        report_file = os.path.join(self.report_dir, "report.md")
        if os.path.exists(report_file):
            os.remove(report_file)
        # 生成并写入带开始时间戳的 header section
        header_file = os.path.join(self.report_dir, "header.md")
        header = generate_header()
        start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(header_file, "w", encoding="utf-8") as f:
            f.write(header)
            f.write(f"Run started: {start_time}\n\n---\n\n")
        print(f"已重置报告，并将 header 写入 {header_file}")

# -----------------------------------------------------------------------------
# nanochat 专用便捷函数

class DummyReport:
    def log(self, *args, **kwargs):
        pass
    def reset(self, *args, **kwargs):
        pass

def get_report():
    # 为方便起见，只有 rank 0 写入报告
    from nanochat.common import get_base_dir, get_dist_info
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    if ddp_rank == 0:
        report_dir = os.path.join(get_base_dir(), "report")
        return Report(report_dir)
    else:
        return DummyReport()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="生成或重置 nanochat 训练报告。")
    parser.add_argument("command", nargs="?", default="generate", choices=["generate", "reset"], help="要执行的操作（默认：generate）")
    args = parser.parse_args()
    if args.command == "generate":
        get_report().generate()
    elif args.command == "reset":
        get_report().reset()
