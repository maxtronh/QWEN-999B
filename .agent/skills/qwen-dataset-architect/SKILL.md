---
name: qwen-dataset-architect
description: Clean, deduplicate, and format raw datasets into Qwen ChatML JSONL for QWEN-999B training.
---
# Instructions
1. Convert all training data into strict ChatML JSONL (`messages` list with `system`, `user`, `assistant`).
2. Preserve `<think>...</think>` blocks for reasoning/distillation traces.
3. Run exact + fuzzy deduplication, drop empty/corrupt rows, and filter samples exceeding 2048 tokens.
4. Keep raw/large files out of Git; push verified splits to the private Hugging Face dataset repo.
