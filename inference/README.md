# QWEN-999B Local Inference Guide (Ollama & GGUF)

This directory manages GGUF export artifacts, Ollama Modelfiles, and local testing scripts.

## Hardware Profile
- **Lead PC (Ryzen 7000 / 16 GB RAM / CPU-only):** Runs `qwen-999b-q4_k_m.gguf` via Ollama CPU inference.
- **Team Laptop (Dell 15 / 6 GB NVIDIA GPU):** Offloads layers to GPU for ~25-40 tokens/sec interactive inference.

## Ollama Setup

1. **Place the exported GGUF file:**
   Move or link the trained `qwen-999b-q4_k_m.gguf` into this `inference/` directory.

2. **Register the model with Ollama:**
   ```bash
   ollama create qwen-999b -f inference/Modelfile
   ```

3. **Run interactive CLI chat:**
   ```bash
   ollama run qwen-999b
   ```

4. **Verify ChatML and `<think>` tags:**
   The model will respond with `<think>...</think>` traces followed by the final answer.
