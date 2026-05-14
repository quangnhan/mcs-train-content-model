# Copy file to TMA Server
scp -r d:\Github\mcs-train-content-model\dataset dc34rpa@192.168.92.26:/home/dc34rpa/nathan/
2026ai@BU3

# Convert gguf to ollama
1) Go to folder
cd ~/nathan/mcs_train_content_model_outputs/Qwen/Qwen3.5-2B_facebook_content_SFT/gguf_gguf

2) Create Modelfile (points at the trained Qwen3.5-2B GGUF, applies Qwen chat template + stop tokens)
cat > Modelfile << 'EOF'
FROM ./Qwen3.5-2B.F16.gguf

TEMPLATE """{{- if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}
{{- range .Messages }}<|im_start|>{{ .Role }}
{{ .Content }}<|im_end|>
{{ end }}<|im_start|>assistant
"""

SYSTEM """You are a helpful marketing and social media assistant fine-tuned on Facebook marketing content."""

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER num_ctx 4096
PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|endoftext|>"
EOF

3) Register model with Ollama
ollama create qwen3.5-2b-marketing -f Modelfile

4) Run model
ollama run qwen3.5-2b-marketing

# (Optional) serve via REST API
# ollama serve   # if not already running as a service
# curl http://localhost:11434/api/generate -d '{"model":"qwen3.5-2b-marketing","prompt":"Write a Facebook ad for a new coffee shop."}'