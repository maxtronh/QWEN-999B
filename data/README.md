# QWEN-999B Data Architecture

## Guidelines (from `qwen-dataset-architect` & `repo-hygiene-guard`)
1. **ChatML JSONL Standard:** All training files must be strictly formatted into `messages` (`system`, `user`, `assistant`).
2. **Reasoning Traces:** Assistant responses retain `<think>...</think>` tags for internal reasoning before presenting the final answer.
3. **Repo Hygiene:** Raw datasets (`data/raw/`), large archives, and generated JSONL files are excluded from Git via `.gitignore`.
4. **Token Limit:** Samples must not exceed 2048 tokens.

## Generating Synthetic Data
Run the generator with your `GEMINI_API_KEY`:
```bash
# Windows PowerShell
$env:GEMINI_API_KEY="your_api_key_here"
python data/scripts/generate_reasoning_data.py --samples 20 --topic all

# Linux / macOS
export GEMINI_API_KEY="your_api_key_here"
python data/scripts/generate_reasoning_data.py --samples 20 --topic all
```
