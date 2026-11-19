---
name: unsloth-t4-trainer
description: Build Zero-OOM Unsloth 4-bit QLoRA notebooks for Colab T4/Kaggle and 6GB local debug scripts.
---
# Instructions
1. Before writing training code, verify latest Unsloth & TRL APIs via Context7 MCP.
2. Enforce T4-safe defaults: `load_in_4bit=True`, `max_seq_length=2048`, `per_device_train_batch_size=2`, `gradient_accumulation_steps=4`, `optim="adamw_8bit"`, `fp16=True`, `bf16=False`.
3. LoRA defaults: `r=16`, `lora_alpha=32`, target all linear projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`).
4. Always configure `save_strategy="steps"`, `save_steps=50`, `push_to_hub=True`, and `resume_from_checkpoint=True` to survive Colab disconnects.
5. For `01_debug_local_6gb.py`, use `Qwen2.5-1.5B-Instruct` for a 20-step pipeline smoke test on the Dell 15 (6 GB VRAM).
