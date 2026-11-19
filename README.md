# QWEN-999B: Frontier Reasoning Model

A specialized Qwen-based reasoning model fine-tuned for deep step-by-step logic (`<think>...</think>`), robust code generation, mathematical proofs, and natural technical Hinglish discourse.

## Project Structure
```text
QWEN-999B/
├── .agent/                  # Architecture rules, hardware constraints, and skills
│   ├── rules/
│   │   └── hardware-constraints.md
│   └── skills/
│       ├── repo-hygiene-guard/
│       ├── qwen-dataset-architect/
│       ├── unsloth-t4-trainer/
│       ├── gguf-ollama-packager/
│       ├── llm-eval-judge/
│       └── rag-agent-scaffolder/
├── data/
│   ├── README.md
│   └── scripts/
│       └── generate_reasoning_data.py   # Gemini API synthetic ChatML distillation
├── training/
│   ├── 01_debug_local_6gb.py            # 20-step smoke test for 6 GB NVIDIA GPU
│   └── 02_train_core_t4.ipynb           # Zero-OOM Unsloth QLoRA Colab T4 training
├── inference/
│   ├── Modelfile                        # Qwen ChatML Ollama Modelfile
│   └── README.md                        # Local deployment guide (Ryzen CPU / Dell GPU)
├── eval/
│   ├── test_prompts.json                # 20 benchmark hard prompts
│   └── README.md                        # Evaluation and judging protocol
└── .gitignore                           # Strict guards for weights, raw data & secrets
```

## Hardware Profile & Compute Guardrails
- **Lead PC:** Ryzen 7000, 16 GB RAM, 500 MB AMD iGPU (Data prep, scripting, Ollama Q4_K_M CPU inference).
- **Team Laptop:** Dell 15, 6 GB NVIDIA GPU (Quick `01_debug_local_6gb.py` smoke tests & local inference).
- **Training Compute:** Free Google Colab T4 (15 GB VRAM) & Kaggle (`fp16=True`, `bf16=False`, `save_steps=50`).
