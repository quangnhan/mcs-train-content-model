# Copy file to TMA Server
scp -r d:\Github\mcs-train-content-model\dataset dc34rpa@192.168.92.26:/home/dc34rpa/nathan/
2026ai@BU3

# Convert gguf to ollama
1) Go to folder
cd ~/nathan/mcs_train_content_model_outputs/Qwen/Qwen3.5-2B_facebook_content_SFT/gguf_gguf

2) Create Modelfile (points at the trained Qwen3.5-4B GGUF, applies Qwen chat template + stop tokens)
cat > Modelfile << 'EOF'
FROM ./Qwen3.5-4B.F16.gguf

TEMPLATE """{{- if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}
{{- range .Messages }}
{{- if eq .Role "user" }}<|im_start|>user
{{ .Content }}<|im_end|>
{{ else if eq .Role "assistant" }}<|im_start|>assistant
{{ .Content }}<|im_end|>
{{ end }}
{{- end }}<|im_start|>assistant
"""

SYSTEM """Bạn là copywriter marketing người Việt, giọng chuyên nghiệp kiểu agency, viết Facebook-native, mạch lạc, súc tích, bám đúng dữ kiện seed. Output 120-200 từ, có hook, luận điểm thương hiệu rõ, CTA cụ thể, KHÔNG bịa claim ngoài seed."""

PARAMETER num_ctx 4096
EOF

3) Register model with Ollama
ollama create qwen3.5-2b-facebook-content -f Modelfile

4) Run model
ollama run qwen3.5-2b-facebook-content

# (Optional) serve via REST API
# ollama serve   # if not already running as a service
# curl http://localhost:11434/api/generate -d '{"model":"qwen3.5-2b-marketing","prompt":"Write a Facebook ad for a new coffee shop."}'