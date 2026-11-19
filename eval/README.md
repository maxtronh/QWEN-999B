# QWEN-999B Evaluation & Benchmark Suite

This directory contains benchmark prompts and comparison reports.

## Files
- `test_prompts.json`: 20 standardized hard prompts across Reasoning, Coding, Math, and Hinglish.
- `reports/`: Markdown comparison reports between Base Qwen vs trained QWEN-999B checkpoints.

## Benchmarking Workflow
1. Run base Qwen (e.g. `qwen2.5:7b-instruct-q4_K_M` or `qwen2.5:1.5b`) on `test_prompts.json` via Ollama API (`http://localhost:11434/api/generate`).
2. Run newly trained QWEN-999B on the same prompts.
3. Compare reasoning trace quality (<think>...</think>), edge-case accuracy, code correctness, and natural Hinglish fluency.
4. Output Markdown report in `eval/reports/report_YYYYMMDD.md`.
