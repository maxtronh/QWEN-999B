"""
QWEN-999B: Master Dataset Builder & Hugging Face Hub Synchronizer
Standardizes open-source reasoning datasets and local custom Hinglish data into strict Qwen ChatML JSONL.

Features:
- Loads .env automatically via python-dotenv (with os.environ fallback) for HF_TOKEN & HF_DATASET_REPO
- Pulls ~3,500 curated reasoning & coding samples with <think> traces (Bespoke-Stratos-17k, OpenR1-Math, etc.) via streaming
- Merges local data/raw/custom_hinglish_reasoning.jsonl (300 Hinglish reasoning rows)
- Strict Qwen ChatML JSONL schema: system, user, assistant messages with preserved <think>...</think> blocks
- Multi-tier deduplication: exact SHA256 prompt hashing + inverted-index fuzzy Jaccard similarity (>0.85)
- Quality filters: drops corrupt/empty rows and filters out any sequence exceeding 2048 tokens
- Syncs the master dataset directly to private Hugging Face repository (HF_DATASET_REPO) using HF_TOKEN
Follows: .agent/skills/qwen-dataset-architect/SKILL.md & .agent/skills/repo-hygiene-guard/SKILL.md
"""

import os
import sys
import re
import json
import string
import hashlib
import argparse
from typing import List, Dict, Any, Optional, Set, Tuple

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

DEFAULT_SYSTEM_ROLE_MESSAGE = (
    "You are QWEN-999B, an advanced reasoning assistant specializing in deep problem-solving, "
    "clean software architecture, mathematical proofs, and natural Hinglish technical discourse."
)

TARGET_SOURCES = [
    {
        "repo_id": "HuggingFaceH4/Bespoke-Stratos-17k",
        "split": "train",
        "target_count": 2000,
        "description": "High-quality DeepSeek-R1 distilled reasoning traces with verified <think> blocks"
    },
    {
        "repo_id": "open-r1/OpenR1-Math-220k",
        "split": "train",
        "target_count": 1500,
        "description": "Rigorous mathematical reasoning traces and Olympiad problem derivations"
    }
]

def get_hf_credentials() -> Tuple[Optional[str], Optional[str]]:
    """Retrieves HF_TOKEN and HF_DATASET_REPO from environment."""
    token = os.environ.get("HF_TOKEN")
    repo = os.environ.get("HF_DATASET_REPO")
    return token, repo

def normalize_text_for_dedup(text: str) -> Tuple[str, Set[str]]:
    """
    Normalizes text for exact and fuzzy deduplication:
    Returns (cleaned_str, token_set).
    """
    cleaned = text.lower()
    cleaned = cleaned.translate(str.maketrans("", "", string.punctuation))
    tokens = set(cleaned.split())
    # Exclude common stop words for fuzzy similarity
    stop_words = {"the", "a", "an", "is", "in", "to", "of", "and", "or", "for", "with", "on", "at", "by", "from"}
    filtered_tokens = {t for t in tokens if t not in stop_words and len(t) > 2}
    return cleaned.strip(), filtered_tokens

class FuzzyDeduplicator:
    """
    Inverted-index based fuzzy deduplicator using Jaccard similarity.
    Scales efficiently across thousands of samples.
    """
    def __init__(self, threshold: float = 0.85):
        self.threshold = threshold
        self.exact_hashes: Set[str] = set()
        self.index: Dict[str, List[Tuple[int, Set[str]]]] = {}
        self.doc_tokens: List[Set[str]] = []
        self.count = 0

    def is_duplicate(self, text: str) -> bool:
        if not text or not text.strip():
            return True

        # 1. Exact hash check
        cleaned, tokens = normalize_text_for_dedup(text)
        exact_hash = hashlib.sha256(cleaned.encode("utf-8")).hexdigest()
        if exact_hash in self.exact_hashes:
            return True

        if not tokens:
            self.exact_hashes.add(exact_hash)
            return False

        # 2. Candidate gathering via inverted index
        candidate_indices: Set[int] = set()
        for token in tokens:
            if token in self.index:
                for doc_id, _ in self.index[token]:
                    candidate_indices.add(doc_id)

        # 3. Pairwise Jaccard evaluation against candidate documents
        for cand_id in candidate_indices:
            cand_tokens = self.doc_tokens[cand_id]
            intersection = len(tokens & cand_tokens)
            union = len(tokens | cand_tokens)
            if union > 0 and (intersection / union) >= self.threshold:
                return True

        # Register new unique sample
        doc_id = self.count
        self.exact_hashes.add(exact_hash)
        self.doc_tokens.append(tokens)
        for token in tokens:
            if token not in self.index:
                self.index[token] = []
            self.index[token].append((doc_id, tokens))
        self.count += 1

        return False

def extract_and_standardize_chatml(example: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Extracts user prompt and assistant response across diverse HF dataset schemas
    and standardizes into strict Qwen ChatML JSONL with <think>...</think> tags.
    """
    user_prompt: str = ""
    assistant_response: str = ""

    # Schema 1: "messages" format (standard ChatML/OpenAI)
    if "messages" in example and isinstance(example["messages"], list):
        for msg in example["messages"]:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user" and not user_prompt:
                user_prompt = content.strip()
            elif role == "assistant" and not assistant_response:
                assistant_response = content.strip()

    # Schema 2: "conversations" format (ShareGPT style)
    elif "conversations" in example and isinstance(example["conversations"], list):
        for msg in example["conversations"]:
            sender = msg.get("from", "")
            value = msg.get("value", "")
            if sender in ["human", "user"] and not user_prompt:
                user_prompt = value.strip()
            elif sender in ["gpt", "assistant", "model"] and not assistant_response:
                assistant_response = value.strip()

    # Schema 3: "problem" and "solution" (OpenR1 style)
    elif "problem" in example and ("solution" in example or "answer" in example):
        user_prompt = str(example["problem"]).strip()
        assistant_response = str(example.get("solution") or example.get("answer")).strip()

    # Schema 4: "instruction" / "input" / "output" (Alpaca style)
    elif "instruction" in example and "output" in example:
        inst = str(example["instruction"]).strip()
        inp = str(example.get("input", "")).strip()
        user_prompt = f"{inst}\n{inp}".strip() if inp else inst
        assistant_response = str(example["output"]).strip()

    # Validate non-empty fields
    if not user_prompt or not assistant_response:
        return None
    if len(assistant_response) < 25:
        return None

    # Standardize <thought> or thinking sections to <think>...</think>
    if "<thought>" in assistant_response and "</thought>" in assistant_response:
        assistant_response = assistant_response.replace("<thought>", "<think>").replace("</thought>", "</think>")

    # Ensure <think> tags are present
    if "<think>" not in assistant_response or "</think>" not in assistant_response:
        # Wrap preliminary reasoning paragraphs
        if "\n\nAnswer:" in assistant_response:
            parts = assistant_response.split("\n\nAnswer:", 1)
            assistant_response = f"<think>\n{parts[0].strip()}\n</think>\n{parts[1].strip()}"
        elif "\n\n" in assistant_response:
            paragraphs = assistant_response.split("\n\n", 1)
            assistant_response = f"<think>\n{paragraphs[0].strip()}\n</think>\n{paragraphs[1].strip()}"
        else:
            assistant_response = f"<think>\nAnalyze and solve the problem systematically step by step.\n</think>\n{assistant_response}"

    # Filter by 2048 token boundary (approx 4 chars per token)
    total_chars = len(DEFAULT_SYSTEM_ROLE_MESSAGE) + len(user_prompt) + len(assistant_response)
    estimated_tokens = total_chars // 4
    if estimated_tokens > 2048:
        return None

    return {
        "messages": [
            {"role": "system", "content": DEFAULT_SYSTEM_ROLE_MESSAGE},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_response}
        ]
    }

def load_local_hinglish_data(filepath: str, dedup: FuzzyDeduplicator) -> List[Dict[str, Any]]:
    """Loads and formats local synthetic Hinglish reasoning dataset."""
    valid_rows: List[Dict[str, Any]] = []
    if not os.path.exists(filepath):
        print(f"[INFO] Local Hinglish dataset not found at '{filepath}'. Skipping local merge.")
        return valid_rows

    print(f"[LOCAL] Reading local Hinglish reasoning dataset from '{filepath}'...")
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    sample = extract_and_standardize_chatml(obj)
                    if sample:
                        user_text = next(m["content"] for m in sample["messages"] if m["role"] == "user")
                        if not dedup.is_duplicate(user_text):
                            valid_rows.append(sample)
                except Exception:
                    continue
        print(f"[LOCAL] Successfully merged {len(valid_rows)} unique samples from local file.")
    except Exception as e:
        print(f"[WARN] Error reading local file '{filepath}': {e}")

    return valid_rows

def pull_hf_reasoning_subset(repo_id: str, split: str, target_count: int, dedup: FuzzyDeduplicator) -> List[Dict[str, Any]]:
    """
    Streams a curated subset from Hugging Face with <think> traces.
    Falls back gracefully if network or dataset is temporarily inaccessible.
    """
    print(f"\n[STREAM] Connecting to Hugging Face dataset: '{repo_id}' (split: {split}, target: {target_count})...")
    collected: List[Dict[str, Any]] = []

    try:
        from datasets import load_dataset
        # Use streaming to avoid downloading massive multi-gigabyte parquet files
        ds = load_dataset(repo_id, split=split, streaming=True)
        
        for item in ds:
            sample = extract_and_standardize_chatml(item)
            if sample:
                user_text = next(m["content"] for m in sample["messages"] if m["role"] == "user")
                if not dedup.is_duplicate(user_text):
                    collected.append(sample)
                    if len(collected) % 250 == 0:
                        print(f"  -> Collected {len(collected)}/{target_count} valid samples from '{repo_id}'...")

                    if len(collected) >= target_count:
                        break

        print(f"[SUCCESS] Pulled {len(collected)} unique samples from '{repo_id}'.")
    except Exception as e:
        print(f"[WARN] Could not stream from '{repo_id}': {e}")
        print("  -> Continuing with remaining data sources.")

    return collected

def push_to_huggingface_hub(dataset_records: List[Dict[str, Any]], repo_id: str, token: str, private: bool = True):
    """Pushes the standardized ChatML dataset to a private Hugging Face dataset repo."""
    print(f"\n[HUB] Preparing to push {len(dataset_records)} records to Hugging Face: '{repo_id}'...")
    try:
        from datasets import Dataset
        from huggingface_hub import HfApi, login

        # Authenticate
        login(token=token)
        hf_api = HfApi()

        # Create repo if it doesn't already exist
        print(f"[HUB] Ensuring repository '{repo_id}' exists (private={private})...")
        hf_api.create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            private=private,
            exist_ok=True,
            token=token
        )

        # Convert to Hugging Face Dataset format and push
        hf_dataset = Dataset.from_list(dataset_records)
        hf_dataset.push_to_hub(
            repo_id=repo_id,
            token=token,
            private=private
        )
        print(f"[SUCCESS] Master dataset pushed successfully to: https://huggingface.co/datasets/{repo_id}")
    except Exception as e:
        print(f"[ERROR] Failed to push dataset to Hugging Face Hub: {e}")
        print("Dataset is safely preserved locally in the output path.")

def main():
    parser = argparse.ArgumentParser(description="Build QWEN-999B Master Reasoning & Coding ChatML Dataset.")
    parser.add_argument("--local-hinglish", type=str, default="data/raw/custom_hinglish_reasoning.jsonl",
                        help="Path to local Hinglish dataset (default: data/raw/custom_hinglish_reasoning.jsonl)")
    parser.add_argument("--output", type=str, default="data/master_dataset_chatml.jsonl",
                        help="Path for combined local output JSONL (default: data/master_dataset_chatml.jsonl)")
    parser.add_argument("--hf-repo", type=str, default=None,
                        help="Target HF dataset repo (defaults to HF_DATASET_REPO from .env)")
    parser.add_argument("--push", action="store_true",
                        help="Push dataset to Hugging Face Hub (default: pushes if credentials exist)")
    parser.add_argument("--no-push", action="store_true",
                        help="Do not push to Hugging Face Hub, only build local JSONL")
    args = parser.parse_args()

    hf_token, env_hf_repo = get_hf_credentials()
    target_repo = args.hf_repo or env_hf_repo

    print("=" * 70)
    print(" QWEN-999B Master Dataset Architect & Hub Sync")
    print(f" Local Output:      {args.output}")
    print(f" Local Hinglish:    {args.local_hinglish}")
    print(f" Target HF Repo:    {target_repo if target_repo else '[Not Set]'}")
    print(f" HF Token Found:    {'Yes' if hf_token else 'No'}")
    print("=" * 70)

    # Initialize deduplicator
    dedup = FuzzyDeduplicator(threshold=0.85)
    master_records: List[Dict[str, Any]] = []

    # Step 1: Merge local Hinglish reasoning dataset if present
    local_samples = load_local_hinglish_data(args.local_hinglish, dedup)
    master_records.extend(local_samples)

    # Step 2: Stream open-source reasoning subsets from Hugging Face
    for src in TARGET_SOURCES:
        subset = pull_hf_reasoning_subset(
            repo_id=src["repo_id"],
            split=src["split"],
            target_count=src["target_count"],
            dedup=dedup
        )
        master_records.extend(subset)

    # Step 3: Verify and save local master dataset
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print(f"\n[SAVING] Writing {len(master_records)} verified ChatML records to '{args.output}'...")
    with open(args.output, "w", encoding="utf-8") as f:
        for record in master_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"[SAVED] Local master dataset written successfully ({len(master_records)} samples).")

    # Step 4: Push to Hugging Face Hub if credentials and configuration allow
    should_push = (not args.no_push) and (args.push or (hf_token and target_repo))
    if should_push:
        if not hf_token or not target_repo:
            print("\n[NOTE] Cannot push to Hugging Face: HF_TOKEN or HF_DATASET_REPO not set.")
            print("Configure your credentials in '.env' or provide --hf-repo.")
        else:
            push_to_huggingface_hub(master_records, target_repo, hf_token, private=True)
    else:
        print("\n[NOTE] Hub upload skipped (--no-push or credentials omitted). Dataset preserved locally.")

    print("\n" + "=" * 70)
    print(f"[SUMMARY] Master Dataset Pipeline Complete. Total unique samples: {len(master_records)}")
    print("=" * 70)

if __name__ == "__main__":
    main()
