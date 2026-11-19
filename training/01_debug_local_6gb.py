"""
QWEN-999B: Local 6 GB VRAM Smoke Test (Debug Pipeline)
Target Hardware: Dell 15 (6 GB NVIDIA GPU)
Model: unsloth/Qwen2.5-1.5B-Instruct
Rules: .agent/rules/hardware-constraints.md & .agent/skills/unsloth-t4-trainer/SKILL.md
"""

import os
import sys
import torch
from datasets import Dataset

# Ensure reproducibility
torch.manual_seed(3407)

def check_gpu():
    if not torch.cuda.is_available():
        print("[WARNING] CUDA is not available! Running on CPU will be extremely slow.")
        return None
    device_name = torch.cuda.get_device_name(0)
    total_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"[INFO] Detected GPU: {device_name} with {total_mem:.2f} GB total VRAM.")
    if total_mem > 7.0:
        print("[INFO] VRAM > 6 GB detected; debug script will run well within bounds.")
    return total_mem

def get_debug_dataset(tokenizer):
    """
    Loads data/reasoning_data.jsonl if available; otherwise falls back to a curated
    in-memory ChatML dataset with <think>...</think> traces for immediate smoke testing.
    """
    jsonl_path = os.path.join("data", "reasoning_data.jsonl")
    if os.path.exists(jsonl_path):
        print(f"[INFO] Loading debug dataset from {jsonl_path}...")
        from datasets import load_dataset
        raw_ds = load_dataset("json", data_files=jsonl_path, split="train")
        # Pick first 40 samples for a 20-step smoke test
        num_samples = min(40, len(raw_ds))
        raw_ds = raw_ds.select(range(num_samples))
    else:
        print("[INFO] Local data/reasoning_data.jsonl not found. Generating in-memory ChatML mock samples...")
        mock_data = [
            {
                "messages": [
                    {"role": "system", "content": "You are QWEN-999B, a high-reasoning AI assistant specialized in logic, coding, and multilingual problem-solving."},
                    {"role": "user", "content": "Which is greater: 9.11 or 9.9?"},
                    {"role": "assistant", "content": "<think>\nCompare the numbers 9.11 and 9.9.\nBoth have the integer part 9.\nCompare the tenths place: in 9.11, the tenths digit is 1. In 9.9, the tenths digit is 9.\nSince 9 > 1, 9.9 is greater than 9.11 (9.90 > 9.11).\n</think>\n9.9 is greater than 9.11."}
                ]
            },
            {
                "messages": [
                    {"role": "system", "content": "You are QWEN-999B, a high-reasoning AI assistant specialized in logic, coding, and multilingual problem-solving."},
                    {"role": "user", "content": "Bhai ek Python function likho to check if a string is palindrome or not."},
                    {"role": "assistant", "content": "<think>\nUser is asking in Hinglish to check if a string is palindrome in Python.\nI should explain clearly and provide clean code.\nAlgorithm: clean non-alphanumeric or do simple slice reverse s == s[::-1].\n</think>\nHaan bhai, ye raha clean Python code palindrome check karne ke liye:\n\n```python\ndef is_palindrome(s: str) -> bool:\n    cleaned = ''.join(c.lower() for c in s if c.isalnum())\n    return cleaned == cleaned[::-1]\n\n# Example usage:\nprint(is_palindrome('Naman'))  # True\nprint(is_palindrome('Qwen'))   # False\n```"}
                ]
            },
            {
                "messages": [
                    {"role": "system", "content": "You are QWEN-999B, a high-reasoning AI assistant specialized in logic, coding, and multilingual problem-solving."},
                    {"role": "user", "content": "If 5 machines make 5 widgets in 5 minutes, how long do 100 machines take to make 100 widgets?"},
                    {"role": "assistant", "content": "<think>\nRate problem:\n5 machines produce 5 widgets in 5 minutes.\nRate of 1 machine: 5 widgets / (5 machines * 5 minutes) = 0.2 widgets per minute per machine.\nSo 1 machine takes 5 minutes to make 1 widget.\nFor 100 machines making 100 widgets:\nEach machine makes 1 widget simultaneously.\nTime required = 5 minutes.\n</think>\nIt takes **5 minutes**. Each machine makes 1 widget every 5 minutes, so 100 machines will produce 100 widgets in 5 minutes."}
                ]
            },
            {
                "messages": [
                    {"role": "system", "content": "You are QWEN-999B, a high-reasoning AI assistant specialized in logic, coding, and multilingual problem-solving."},
                    {"role": "user", "content": "Explain time complexity of binary search with edge cases."},
                    {"role": "assistant", "content": "<think>\nBinary search repeatedly divides sorted search space in half.\nRecurrence relation: T(n) = T(n/2) + O(1).\nBy Master Theorem, T(n) = O(log n).\nEdge cases: empty list, single element, target at boundary, duplicates.\n</think>\nBinary search operates in **O(log n)** time complexity.\nAt each step, half of the search interval is discarded. Space complexity is O(1) iterative or O(log n) recursive.\nKey edge cases to handle:\n1. Array is empty (return -1 immediately).\n2. Array has 1 element.\n3. Target is outside min/max range.\n4. Integer overflow when calculating midpoint: use `low + (high - low) // 2`."}
                ]
            }
        ] * 10  # 40 samples total
        raw_ds = Dataset.from_list(mock_data)

    def format_chatml(examples):
        convos = examples["messages"]
        texts = [
            tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False)
            for convo in convos
        ]
        return {"text": texts}

    formatted_ds = raw_ds.map(format_chatml, batched=True)
    return formatted_ds

def run_debug_training():
    print("=" * 60)
    print("Starting QWEN-999B 20-Step Local Debug Smoke Test (6 GB VRAM)")
    print("=" * 60)
    
    total_mem = check_gpu()

    try:
        from unsloth import FastLanguageModel
        from unsloth.chat_templates import get_chat_template
        from trl import SFTTrainer
        from transformers import TrainingArguments
    except ImportError as e:
        print(f"[ERROR] Required packages not found: {e}")
        print("Please install Unsloth and dependencies: pip install unsloth trl transformers accelerate bitsandbytes")
        sys.exit(1)

    model_name = "unsloth/Qwen2.5-1.5B-Instruct"
    max_seq_length = 2048

    print(f"\n[1/5] Loading model {model_name} in 4-bit...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,  # Auto-detection for float16
        load_in_4bit=True,
    )

    # Configure ChatML template for Qwen
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="qwen-2.5",
    )

    print("\n[2/5] Configuring LoRA adapters (Targeting all linear projections)...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=32,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    print("\n[3/5] Preparing dataset...")
    dataset = get_debug_dataset(tokenizer)

    if torch.cuda.is_available():
        vram_start = torch.cuda.memory_allocated(0) / (1024 ** 3)
        print(f"[VRAM] Allocated before training: {vram_start:.2f} GB")

    output_dir = os.path.join("outputs", "debug_local_6gb")
    os.makedirs(output_dir, exist_ok=True)

    print("\n[4/5] Setting up SFTTrainer (20 steps max)...")
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=2,
        max_steps=20,
        learning_rate=2e-4,
        fp16=True,
        bf16=False,  # Enforce bf16=False as per hardware-constraints
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        report_to="none",
        save_strategy="no",  # Debug only: no heavy checkpoints
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=1,
        packing=False,
        args=training_args,
    )

    print("\n[5/5] Executing 20-step smoke test...")
    train_result = trainer.train()

    if torch.cuda.is_available():
        vram_peak = torch.cuda.max_memory_allocated(0) / (1024 ** 3)
        vram_reserved = torch.cuda.max_memory_reserved(0) / (1024 ** 3)
        print("\n" + "=" * 60)
        print(f"[SUCCESS] 20 steps finished with loss: {train_result.training_loss:.4f}")
        print(f"[VRAM] Peak Allocated: {vram_peak:.2f} GB | Peak Reserved: {vram_reserved:.2f} GB")
        print(f"[STATUS] 6 GB GPU Limit Safe: {'PASS (Under 6GB)' if vram_reserved < 5.8 else 'WARN'}")
        print("=" * 60)

    # Save lightweight debug adapter
    adapter_path = os.path.join(output_dir, "adapter_test")
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print(f"[INFO] Smoke test adapter saved to {adapter_path}")

if __name__ == "__main__":
    run_debug_training()
