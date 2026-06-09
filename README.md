# Upload lên hugging face
```bash
pip install huggingface_hub
hf auth login
hf upload quangnhan145/Qwen3.5-4B-Marketing "outputs/Qwen-Qwen3.5-4B/model-q4_k_m.gguf" model-q4_k_m.gguf
```

# Copy file to TMA Server
scp -r d:\Github\mcs-train-content-model\dataset dc34rpa@192.168.92.26:/home/dc34rpa/nathan/
2026ai@BU3