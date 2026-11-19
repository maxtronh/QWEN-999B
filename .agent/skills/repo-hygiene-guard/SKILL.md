---
name: repo-hygiene-guard
description: Prevent large weights, datasets, and secrets from being committed to GitHub.
---
# Instructions
1. Ensure `.gitignore` blocks `*.gguf`, `*.safetensors`, `*.bin`, `*.pt`, `.env`, `data/raw/`, and `outputs/`.
2. Never hardcode Hugging Face, W&B, or GitHub tokens in notebooks or scripts—always read from environment variables or Colab Secrets.
