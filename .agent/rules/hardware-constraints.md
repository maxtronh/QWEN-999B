# Hardware & Compute Guardrails (QWEN-999B)
- **Lead PC:** Ryzen 7000, 16 GB RAM, 500 MB AMD iGPU (CPU-only: data prep, scripting, Ollama Q4_K_M inference).
- **Team Laptop:** Dell 15, 6 GB NVIDIA GPU (ONLY `Qwen-1.5B` dry-run debugging and `7B Q4_K_M` local inference/RAG).
- **Training Compute:** Free Google Colab (T4 15 GB VRAM) and Kaggle (P100 16 GB / 2x T4).
- **Strict Prohibitions:** NEVER generate full-parameter BF16 training scripts or 14B+ training configs. Always use Unsloth 4-bit QLoRA (`fp16=True`, `bf16=False` on T4) with Hugging Face auto-checkpointing.
