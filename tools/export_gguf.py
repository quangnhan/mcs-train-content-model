"""
export_gguf.py — Merge LoRA + convert sang GGUF, bypass Unsloth conversion bug.

Usage:
    python tools/export_gguf.py outputs/Qwen-Qwen3.5-4B
    python tools/export_gguf.py outputs/Qwen-Qwen3.5-4B --quant q8_0
    python tools/export_gguf.py outputs/Qwen-Qwen3.5-4B --skip-merge
"""
import argparse
import gc
import io
import json
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

# Ensure UTF-8 output on Windows terminals that default to cp1252
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

LLAMA_CPP_CACHE = Path.home() / ".cache" / "llama.cpp"
QUANT_DEFAULT = "q4_k_m"

# Types supported directly by convert_hf_to_gguf.py --outtype
DIRECT_QUANT_TYPES = {"f32", "f16", "bf16", "q8_0", "tq1_0", "tq2_0", "auto"}


def ensure_llama_cpp() -> Path:
    if not LLAMA_CPP_CACHE.exists():
        print(f"Cloning llama.cpp → {LLAMA_CPP_CACHE} ...")
        subprocess.run(
            ["git", "clone", "--depth=1",
             "https://github.com/ggerganov/llama.cpp",
             str(LLAMA_CPP_CACHE)],
            check=True,
        )
    req = LLAMA_CPP_CACHE / "requirements.txt"
    installed_flag = LLAMA_CPP_CACHE / ".requirements_installed"
    if req.exists() and not installed_flag.exists():
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "-r", str(req)],
            check=True,
        )
        # requirements pin numpy~=1.26.4 which crashes on Python 3.13 Windows;
        # force-upgrade to 2.x after the requirements install
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "--force-reinstall", "numpy>=2.0"],
            check=True,
        )
        installed_flag.touch()
    return LLAMA_CPP_CACHE


def ensure_llama_quantize(llama_cpp: Path) -> Path | None:
    """Find llama-quantize binary; download pre-built release if missing."""
    for name in ["llama-quantize.exe", "llama-quantize", "quantize.exe", "quantize"]:
        p = llama_cpp / name
        if p.exists():
            return p

    if sys.platform != "win32":
        print("llama-quantize not found. Build llama.cpp first: cmake -B build && cmake --build build -t llama-quantize")
        return None

    print("llama-quantize not found. Downloading pre-built Windows binary from GitHub releases...")
    try:
        api_url = "https://api.github.com/repos/ggerganov/llama.cpp/releases/latest"
        req = urllib.request.Request(api_url, headers={"Accept": "application/vnd.github+json",
                                                       "User-Agent": "export_gguf"})
        with urllib.request.urlopen(req, timeout=30) as r:
            release = json.loads(r.read())

        def _is_bin_zip_x64(a: dict) -> bool:
            n = a["name"]
            return (n.endswith(".zip")
                    and "win" in n
                    and "x64" in n
                    and "arm" not in n
                    and not n.startswith("cudart")   # cudart zips only have DLLs
                    and not n.startswith("llava"))

        # Prefer AVX2 x64 (pure CPU), then any x64 Windows zip
        asset = (
            next((a for a in release["assets"] if _is_bin_zip_x64(a) and "avx2" in a["name"]), None)
            or next((a for a in release["assets"] if _is_bin_zip_x64(a) and "avx" in a["name"]), None)
            or next((a for a in release["assets"] if _is_bin_zip_x64(a)), None)
        )
        if asset is None:
            print("No suitable Windows release asset found.")
            print("Available assets:", [a["name"] for a in release["assets"] if a["name"].endswith(".zip")])
            return None

        print(f"Downloading {asset['name']} ({asset['size'] // 1_000_000} MB)...")
        with urllib.request.urlopen(asset["browser_download_url"], timeout=300) as r:
            data = r.read()

        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            names = zf.namelist()
            quantize_member = next(
                (m for m in names if m.endswith("llama-quantize.exe") or m.endswith("quantize.exe")),
                None,
            )
            if not quantize_member:
                print("llama-quantize.exe not found inside the zip.")
                print("Zip contents (first 20):", names[:20])
                return None
            # Extract the binary AND all DLLs it depends on (in the same directory)
            bin_dir = Path(quantize_member).parent
            for member in names:
                if Path(member).parent == bin_dir or Path(member).parent == Path("."):
                    if member.endswith(".exe") or member.endswith(".dll"):
                        dest = llama_cpp / Path(member).name
                        dest.write_bytes(zf.read(member))
            quantize_dest = llama_cpp / Path(quantize_member).name
            print(f"Extracted binary + DLLs → {llama_cpp}")
            return quantize_dest
    except Exception as e:
        print(f"Failed to download llama-quantize: {e}")
        return None


def merge_lora(lora_dir: Path, merged_dir: Path) -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    config = json.loads((lora_dir / "adapter_config.json").read_text(encoding="utf-8"))
    base_id = config["base_model_name_or_path"]
    print(f"Base model: {base_id}")

    print("Loading base model (bfloat16)...")
    tokenizer = AutoTokenizer.from_pretrained(base_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    print("Applying LoRA & merging...")
    model = PeftModel.from_pretrained(model, str(lora_dir))
    model = model.merge_and_unload()

    merged_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving merged model → {merged_dir}")
    model.save_pretrained(str(merged_dir), safe_serialization=True, max_shard_size="4GB")
    tokenizer.save_pretrained(str(merged_dir))

    for extra in ["chat_template.jinja", "processor_config.json", "generation_config.json"]:
        src = lora_dir / extra
        if src.exists():
            import shutil
            shutil.copy(src, merged_dir / extra)

    del model, tokenizer
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass


def _run_convert(script: Path, merged_dir: Path, output_gguf: Path, outtype: str, llama_cpp: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(script),
         str(merged_dir),
         "--outfile", str(output_gguf),
         "--outtype", outtype],
        capture_output=True,
        text=True,
        cwd=str(llama_cpp),
    )
    if result.stdout:
        print(result.stdout[-4000:])
    if result.returncode != 0:
        print("LỖI convert_hf_to_gguf:\n", result.stderr[-3000:])
        sys.exit(1)


def patch_tokenizer_config(merged_dir: Path) -> None:
    """Fix tokenizer_class values that transformers doesn't recognize (e.g. TokenizersBackend)."""
    cfg_path = merged_dir / "tokenizer_config.json"
    if not cfg_path.exists():
        return
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    bad_classes = {"TokenizersBackend"}
    if cfg.get("tokenizer_class") in bad_classes:
        print(f"Patching tokenizer_class: {cfg['tokenizer_class']} → PreTrainedTokenizerFast")
        cfg["tokenizer_class"] = "PreTrainedTokenizerFast"
        cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")


def convert_to_gguf(merged_dir: Path, output_gguf: Path, quant: str, llama_cpp: Path) -> Path:
    """Returns the path of the final GGUF file (may differ from output_gguf on fallback)."""
    script = llama_cpp / "convert_hf_to_gguf.py"
    assert script.exists(), f"Không tìm thấy {script}"

    patch_tokenizer_config(merged_dir)

    if quant.lower() in DIRECT_QUANT_TYPES:
        # Single-step: convert directly with requested outtype
        print(f"Converting → {output_gguf}  (quant={quant}) ...")
        _run_convert(script, merged_dir, output_gguf, quant.lower(), llama_cpp)
        return output_gguf

    # Two-step: convert to f16 first, then quantize with llama-quantize
    f16_gguf = output_gguf.parent / "model-f16-temp.gguf"
    print(f"Step 1: Converting → f16 GGUF ({f16_gguf.name}) ...")
    _run_convert(script, merged_dir, f16_gguf, "f16", llama_cpp)

    quantize_bin = ensure_llama_quantize(llama_cpp)
    if quantize_bin is None:
        # Fall back: keep f16 as final output with correct name
        f16_out = output_gguf.parent / "model-f16.gguf"
        f16_gguf.rename(f16_out)
        print(f"\nKhông tìm thấy llama-quantize. Đã lưu f16 tại: {f16_out}")
        print(f"Để tạo {quant}, chạy: llama-quantize {f16_out.name} {output_gguf.name} {quant.upper()}")
        return f16_out

    print(f"Step 2: Quantizing f16 → {quant.upper()} ...")
    result = subprocess.run(
        [str(quantize_bin), str(f16_gguf), str(output_gguf), quant.upper()],
        capture_output=True, text=True,
    )
    if result.stdout:
        print(result.stdout[-2000:])
    if result.returncode != 0:
        print("LỖI llama-quantize:\n", result.stderr[-2000:])
        sys.exit(1)

    f16_gguf.unlink(missing_ok=True)
    print(f"Đã xóa file tạm {f16_gguf.name}")
    return output_gguf


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge LoRA → GGUF (bypass Unsloth bug)")
    parser.add_argument("model_dir", type=Path, help="e.g. outputs/Qwen-Qwen3.5-4B")
    parser.add_argument("--quant", default=QUANT_DEFAULT,
                        help="Quantization type: q4_k_m | q5_k_m | q8_0 | f16")
    parser.add_argument("--skip-merge", action="store_true",
                        help="Bỏ qua bước merge nếu safetensors đã tồn tại")
    args = parser.parse_args()

    model_dir = args.model_dir.resolve()
    lora_dir = model_dir / "lora"
    merged_dir = model_dir / "gguf"
    output_gguf = model_dir / f"model-{args.quant}.gguf"

    assert lora_dir.exists() and (lora_dir / "adapter_config.json").exists(), \
        f"Không tìm thấy LoRA adapter tại: {lora_dir}"

    # Merge step
    has_safetensors = bool(list(merged_dir.glob("model*.safetensors*")))
    if has_safetensors and args.skip_merge:
        print(f"Merged safetensors đã có ({merged_dir}), bỏ qua merge.")
    elif has_safetensors:
        ans = input("Đã tìm thấy merged safetensors. Bỏ qua bước merge? [Y/n]: ").strip().lower()
        if ans in ("", "y"):
            print("Bỏ qua merge.")
        else:
            merge_lora(lora_dir, merged_dir)
    else:
        merge_lora(lora_dir, merged_dir)

    llama_cpp = ensure_llama_cpp()
    final_gguf = convert_to_gguf(merged_dir, output_gguf, args.quant, llama_cpp)

    size_gb = final_gguf.stat().st_size / 1e9
    print(f"\nXong! {final_gguf}  ({size_gb:.2f} GB)")
    print("\n── Ollama ──────────────────────────────────────────────────────")
    print(f'  echo "FROM {final_gguf}" > Modelfile')
    print(f"  ollama create qwen-marketing -f Modelfile")
    print(f"  ollama run qwen-marketing")


if __name__ == "__main__":
    main()
