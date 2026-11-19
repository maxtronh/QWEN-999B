---
name: gguf-ollama-packager
description: Merge LoRA adapters, export Q4_K_M GGUF, and generate Ollama Modelfile for QWEN-999B.
---
# Instructions
1. Export trained models using `model.save_pretrained_gguf(..., quantization_method="q4_k_m")` (~4.5–5.5 GB).
2. Generate `inference/Modelfile` with exact Qwen ChatML template and stop tokens (`<|im_end|>`, `<|endoftext|>`).
3. Include a local Ollama verification command (`ollama create qwen-999b -f inference/Modelfile`).
