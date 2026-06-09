"""
ollama_import.py — Import một file GGUF vào Ollama và chạy thử.

Usage:
    python tools/ollama_import.py outputs/Qwen-Qwen3.5-4B/model-q4_k_m.gguf
    python tools/ollama_import.py outputs/Qwen-Qwen3.5-4B/model-q4_k_m.gguf --name my-model
    python tools/ollama_import.py outputs/Qwen-Qwen3.5-4B/model-q4_k_m.gguf --no-run
"""
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ChatML template — dùng cho Qwen và các model fine-tune với ChatML format
CHATML_TEMPLATE = """\
{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
{{ end }}{{ .Response }}<|im_end|>
"""


def check_ollama() -> None:
    result = subprocess.run(["ollama", "--version"], capture_output=True, text=True)
    if result.returncode != 0:
        print("Lỗi: Ollama chưa được cài đặt hoặc không có trong PATH.")
        print("Tải tại: https://ollama.com/download")
        sys.exit(1)
    print(f"Ollama: {result.stdout.strip()}")


def build_modelfile(gguf_path: Path, temperature: float, top_p: float) -> str:
    return f"""\
FROM {gguf_path.as_posix()}

PARAMETER temperature {temperature}
PARAMETER top_p {top_p}
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|im_start|>"

TEMPLATE \"\"\"{CHATML_TEMPLATE}\"\"\"
"""


def ollama_create(model_name: str, modelfile_content: str) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w", suffix="Modelfile", delete=False, encoding="utf-8"
    ) as f:
        f.write(modelfile_content)
        tmp_path = f.name

    print(f"\nModelfile tạm: {tmp_path}")
    print("─" * 60)
    print(modelfile_content)
    print("─" * 60)

    print(f"\nImporting → ollama create {model_name} ...")
    result = subprocess.run(
        ["ollama", "create", model_name, "-f", tmp_path],
        text=True,
    )
    Path(tmp_path).unlink(missing_ok=True)

    if result.returncode != 0:
        print("Lỗi: ollama create thất bại.")
        sys.exit(1)
    print(f"Import thành công: {model_name}")


def ollama_run(model_name: str) -> None:
    print(f"\nChạy: ollama run {model_name}")
    print("(Ctrl+C hoặc gõ /bye để thoát)\n")
    subprocess.run(["ollama", "run", model_name])


def main() -> None:
    parser = argparse.ArgumentParser(description="Import GGUF vào Ollama")
    parser.add_argument("gguf", type=Path, help="Đường dẫn tới file .gguf")
    parser.add_argument(
        "--name",
        default=None,
        help="Tên model trong Ollama (mặc định: tên thư mục cha)",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="Temperature (mặc định: 0.7)"
    )
    parser.add_argument(
        "--top-p", type=float, default=0.9, help="Top-p (mặc định: 0.9)"
    )
    parser.add_argument(
        "--no-run",
        action="store_true",
        help="Chỉ import, không chạy interactive chat",
    )
    args = parser.parse_args()

    gguf_path = args.gguf.resolve()
    if not gguf_path.exists():
        print(f"Lỗi: Không tìm thấy file: {gguf_path}")
        sys.exit(1)
    if gguf_path.suffix.lower() != ".gguf":
        print(f"Cảnh báo: File không có đuôi .gguf — tiếp tục anyway.")

    # Tên model mặc định = tên thư mục cha, chuyển về lowercase, thay khoảng trắng bằng dấu gạch ngang
    model_name = args.name or gguf_path.parent.name.lower().replace(" ", "-")

    check_ollama()

    modelfile = build_modelfile(gguf_path, args.temperature, args.top_p)
    ollama_create(model_name, modelfile)

    print(f"\nDùng model bất cứ lúc nào:")
    print(f"  ollama run {model_name}")
    print(f"  ollama rm  {model_name}   # xóa khi không cần")

    if not args.no_run:
        ollama_run(model_name)


if __name__ == "__main__":
    main()
