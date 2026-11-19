---
name: llm-eval-judge
description: Benchmark base Qwen vs trained QWEN-999B checkpoints on eval/test_prompts.json via Ollama.
---
# Instructions
1. Run all fixed prompts in `eval/test_prompts.json` against both base Qwen and the new checkpoint via local Ollama API.
2. Score reasoning accuracy, coding correctness, Hinglish/English clarity, and instruction adherence.
3. Save a clean Markdown comparison report in `eval/reports/`.
