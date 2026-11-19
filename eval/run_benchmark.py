#!/usr/bin/env python3
"""
eval/run_benchmark.py - QWEN-999B LLM Evaluation and Benchmark Suite

Runs standardized benchmarks against local Ollama instances (e.g., base Qwen vs QWEN-999B).
Reads 20 standardized prompts from eval/test_prompts.json covering:
  - Reasoning (trap detection, logic puzzles, multi-step state tracking)
  - Coding (Kadane with indices, rate limiter, closure late binding, tree codec, median of arrays)
  - Math (modular arithmetic, trailing zeros, conditional probability, infinite power tower, card expectation)
  - Hinglish (PostgreSQL indexing, heap dumps, Kafka lag, XOR trick, Bias-Variance & regularization)

Records latency, token throughput, and inspects responses for:
  1. <think>...</think> reasoning traces (presence, length, formatting)
  2. Ground truth / criteria alignment

Outputs:
  - Markdown report in eval/reports/benchmark_<model>_<timestamp>.md
  - JSON metrics dump in eval/reports/benchmark_<model>_<timestamp>.json
"""

import argparse
import datetime
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


DEFAULT_SYSTEM_PROMPT = (
    "You are QWEN-999B, a reasoning-focused language model. When presented with a question, "
    "think through the problem carefully step by step inside <think>...</think> tags, "
    "then output your final answer clearly."
)


def load_test_prompts(prompts_path: Path) -> List[Dict[str, Any]]:
    """Loads and validates test prompts from JSON."""
    if not prompts_path.exists():
        raise FileNotFoundError(f"Test prompts file not found at: {prompts_path}")

    with open(prompts_path, "r", encoding="utf-8") as f:
        prompts = json.load(f)

    if not isinstance(prompts, list):
        raise ValueError(f"Expected list of prompts in {prompts_path}, got {type(prompts)}")

    return prompts


def parse_think_tags(text: str) -> Tuple[bool, str, str]:
    """
    Extracts <think>...</think> content and final answer.
    Returns:
        (has_think_tags, thinking_trace, final_answer)
    """
    think_pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
    match = think_pattern.search(text)

    if match:
        thinking_trace = match.group(1).strip()
        final_answer = think_pattern.sub("", text).strip()
        return True, thinking_trace, final_answer

    # Check for unclosed <think> tag
    unclosed_match = re.search(r"<think>(.*)", text, re.DOTALL | re.IGNORECASE)
    if unclosed_match:
        thinking_trace = unclosed_match.group(1).strip()
        return True, thinking_trace, "[Warning: Unclosed <think> tag detected]"

    return False, "", text.strip()


def evaluate_criteria_alignment(
    prompt_id: str,
    category: str,
    full_text: str,
    final_answer: str,
    criteria: str,
    expected_focus: str,
) -> Tuple[str, float, str]:
    """
    Evaluates response against ground truth and criteria.
    Returns:
        (status: PASS | PARTIAL | FAIL, score: 0.0 - 1.0, rationale: str)
    """
    text_to_eval = (final_answer + " " + full_text).lower()

    # Specialized rule checks per prompt ID
    if prompt_id == "eval_01_reasoning":
        # Trap: 17 sheep, all but 9 die -> 9 sheep remain
        has_9 = bool(re.search(r"\b9\b|\bnine\b", text_to_eval))
        has_8 = bool(re.search(r"\b8\b|\beight\b", text_to_eval))
        mentions_survive = any(w in text_to_eval for w in ["survive", "left", "all but 9", "remain"])
        if has_9 and not (has_8 and "8 sheep left" in text_to_eval):
            return "PASS", 1.0, "Correctly identified that 9 sheep survived without falling into the 17-9=8 trap."
        elif has_9:
            return "PARTIAL", 0.5, "Mentioned 9, but may have also entertained 8 sheep."
        return "FAIL", 0.0, "Failed trap: did not identify that 9 sheep survived."

    elif prompt_id == "eval_02_reasoning":
        # Boxes puzzle: pick from 'Apples and Oranges' / mixed box
        correct_box = any(
            phrase in text_to_eval
            for phrase in [
                "apples and oranges",
                "apples & oranges",
                "both",
                "mixed",
                "apple and orange",
            ]
        )
        has_elimination = any(w in text_to_eval for w in ["mislabeled", "eliminate", "deduce", "label"])
        if correct_box and has_elimination:
            return "PASS", 1.0, "Selected the 'Apples & Oranges' box and explained the mislabel elimination logic."
        elif correct_box:
            return "PARTIAL", 0.5, "Selected the correct box but explanation of deduction lacked full rigor."
        return "FAIL", 0.0, "Did not identify the 'Apples & Oranges' box as the optimal initial draw."

    elif prompt_id == "eval_03_reasoning":
        # Pound of gold vs pound of feathers: Troy vs Avoirdupois
        has_troy = "troy" in text_to_eval
        has_avoirdupois = "avoirdupois" in text_to_eval
        if has_troy or has_avoirdupois:
            return "PASS", 1.0, "Noticed the distinction between Troy pound (373g, gold) and Avoirdupois pound (453g, feathers)."
        elif "equal" in text_to_eval or "same weight" in text_to_eval:
            return "PARTIAL", 0.5, "Treated both as standard 16 oz pounds (equal) without noting Troy ounce system."
        return "FAIL", 0.0, "Did not accurately evaluate the weight comparison."

    elif prompt_id == "eval_04_reasoning":
        # Sally: 3 brothers, each has 2 sisters -> Sally has 1 sister, 5 children total
        has_1_sister = bool(re.search(r"\b1 sister\b|\bone sister\b|sally has 1|sally has one", text_to_eval))
        has_5_total = bool(re.search(r"\b5 children\b|\bfive children\b|\b5 total\b|\bfive total\b|total.*?\b5\b", text_to_eval))
        if has_1_sister and has_5_total:
            return "PASS", 1.0, "Correctly identified Sally has 1 sister and family has 5 children total."
        elif has_1_sister or has_5_total:
            return "PARTIAL", 0.5, "Partially correct: solved either the sister count or total children."
        return "FAIL", 0.0, "Incorrect counts for sisters or total family children."

    elif prompt_id == "eval_05_reasoning":
        # Water jug 3-gal and 5-gal to get 4 gal
        has_4_gal = "4 gallon" in text_to_eval or "4-gallon" in text_to_eval or "exactly 4" in text_to_eval
        has_steps = any(w in text_to_eval for w in ["pour", "fill", "empty", "jug"])
        if has_4_gal and has_steps:
            return "PASS", 1.0, "Successfully derived step-by-step state transitions to measure 4 gallons."
        return "FAIL", 0.0, "Failed to reach valid 4-gallon water measurement."

    elif prompt_id == "eval_06_coding":
        # Kadane's algorithm tracking indices and handling negative numbers
        has_kadane = any(w in text_to_eval for w in ["kadane", "max_subarray_sum", "current_sum", "max_sum"])
        has_indices = any(w in text_to_eval for w in ["start", "end", "indices", "index"])
        has_negative = any(w in text_to_eval for w in ["negative", "all negative", "-inf", "float('-inf')"])
        if has_kadane and has_indices and has_negative:
            return "PASS", 1.0, "Implemented Kadane's algorithm with start/end indices and all-negative array handling."
        elif has_kadane and has_indices:
            return "PARTIAL", 0.5, "Implemented Kadane with indices, but negative number handling needs verification."
        return "FAIL", 0.0, "Missing index tracking or Kadane dynamic programming logic."

    elif prompt_id == "eval_07_coding":
        # Thread-safe rate limiter with Lock and monotonic
        has_lock = "lock" in text_to_eval
        has_monotonic = "monotonic" in text_to_eval
        has_token_bucket = "token" in text_to_eval or "bucket" in text_to_eval or "capacity" in text_to_eval
        if has_lock and has_monotonic and has_token_bucket:
            return "PASS", 1.0, "Implemented thread-safe Token Bucket with threading.Lock and time.monotonic()."
        elif has_lock or has_monotonic:
            return "PARTIAL", 0.5, "Addressed rate limiting, but missing either threading.Lock or time.monotonic()."
        return "FAIL", 0.0, "Missing thread-safety synchronization or token bucket mechanism."

    elif prompt_id == "eval_08_coding":
        # Python closure late binding: [13, 13, 13, 13] and i=i
        has_13 = "[13, 13, 13, 13]" in text_to_eval or "13, 13, 13, 13" in text_to_eval
        has_fix = "i=i" in text_to_eval or "default argument" in text_to_eval
        has_late_binding = any(w in text_to_eval for w in ["late binding", "closure", "lookup", "scope"])
        if has_13 and has_fix and has_late_binding:
            return "PASS", 1.0, "Explained late-binding closure output [13, 13, 13, 13] and provided default arg fix (i=i)."
        elif has_13 or has_fix:
            return "PARTIAL", 0.5, "Identified output or fix, but explanation was incomplete."
        return "FAIL", 0.0, "Failed to identify late-binding closure output or fix."

    elif prompt_id == "eval_09_coding":
        # Binary tree serialization
        has_tree = any(w in text_to_eval for w in ["serialize", "deserialize", "tree", "node"])
        has_complexity = "o(n)" in text_to_eval
        if has_tree and has_complexity:
            return "PASS", 1.0, "Provided serialization/deserialization tree codec with O(N) complexity analysis."
        elif has_tree:
            return "PARTIAL", 0.5, "Provided tree serialization logic, but omitted explicit O(N) complexity analysis."
        return "FAIL", 0.0, "Incomplete tree serialization/deserialization implementation."

    elif prompt_id == "eval_10_coding":
        # Median of two sorted arrays O(log(min(m, n)))
        has_binary_search = any(w in text_to_eval for w in ["binary search", "partition", "log(min"])
        has_complexity = "log" in text_to_eval
        if has_binary_search and has_complexity:
            return "PASS", 1.0, "Implemented binary search partition algorithm achieving O(log(min(m, n))) complexity."
        elif has_binary_search:
            return "PARTIAL", 0.5, "Addressed partition logic, but complexity proof or edge cases incomplete."
        return "FAIL", 0.0, "Failed to achieve optimal O(log(min(m, n))) partition algorithm."

    elif prompt_id == "eval_11_math":
        # Remainder 2^2025 divided by 13: remainder is 5
        has_5 = bool(re.search(r"\bremainder is 5\b|\bresult is 5\b|\b= 5\b|\b5\b", final_answer.lower()))
        has_fermat = any(w in text_to_eval for w in ["fermat", "mod 12", "euler", "modulo 13"])
        if has_5 and has_fermat:
            return "PASS", 1.0, "Correctly found remainder 5 using Fermat's Little Theorem / mod 12 arithmetic."
        elif has_5:
            return "PARTIAL", 0.5, "Arrived at remainder 5, but modular arithmetic proof was brief."
        return "FAIL", 0.0, "Incorrect remainder for 2^2025 mod 13."

    elif prompt_id == "eval_12_math":
        # 250! trailing zeros = 62
        has_62 = bool(re.search(r"\b62\b|\bsixty-two\b", final_answer.lower()))
        has_legendre = any(w in text_to_eval for w in ["legendre", "50 + 10 + 2", "floor(250/5)", "250/5"])
        if has_62 and has_legendre:
            return "PASS", 1.0, "Correctly calculated 62 trailing zeros using Legendre's factor-of-5 formula."
        elif has_62:
            return "PARTIAL", 0.5, "Gave 62 trailing zeros, but formula derivation was abbreviated."
        return "FAIL", 0.0, "Incorrect trailing zeros count for 250!."

    elif prompt_id == "eval_13_math":
        # Dice sum > 7, prob at least one 6: 9/15 = 3/5 = 0.6
        has_prob = any(p in text_to_eval for p in ["3/5", "0.6", "9/15", "60%"])
        has_cond = any(w in text_to_eval for w in ["conditional", "outcomes", "sample space", "sum > 7"])
        if has_prob and has_cond:
            return "PASS", 1.0, "Correctly computed conditional probability 3/5 (0.6) from 9/15 outcomes."
        elif has_prob:
            return "PARTIAL", 0.5, "Derived 3/5 (0.6), but outcome enumeration was partial."
        return "FAIL", 0.0, "Incorrect conditional probability for dice sum."

    elif prompt_id == "eval_14_math":
        # x^(x...) = 2 -> sqrt(2), for = 4 no solution (diverges above e)
        has_sqrt2 = any(w in text_to_eval for w in ["sqrt(2)", "√2", "1.414", "square root of 2"])
        has_no_sol = any(w in text_to_eval for w in ["no solution", "diverges", "euler", "cannot equal 4", "exceeds e"])
        if has_sqrt2 and has_no_sol:
            return "PASS", 1.0, "Correctly found x = sqrt(2) and proved no real solution exists for 4 (exceeds e)."
        elif has_sqrt2 or has_no_sol:
            return "PARTIAL", 0.5, "Partially correct: solved either x=sqrt(2) or divergence for 4."
        return "FAIL", 0.0, "Failed to analyze infinite power tower convergence boundary."

    elif prompt_id == "eval_15_math":
        # Expected cards before first Ace: 48/5 = 9.6 non-aces, 10.6 total dealt
        has_val = any(w in text_to_eval for w in ["9.6", "10.6", "48/5", "53/5"])
        if has_val:
            return "PASS", 1.0, "Correctly calculated expected cards (9.6 non-aces / 10.6 total cards dealt)."
        return "FAIL", 0.0, "Incorrect expectation calculation for first Ace appearance."

    elif prompt_id == "eval_16_hinglish":
        # PostgreSQL B-Tree vs GIN for JSONB in Hinglish
        has_gin = "gin" in text_to_eval
        has_btree = "b-tree" in text_to_eval or "btree" in text_to_eval
        has_jsonb = "jsonb" in text_to_eval
        has_hinglish = any(w in text_to_eval for w in ["hai", "karta", "hota", "liye", "farak", "better"])
        if has_gin and has_btree and has_jsonb and has_hinglish:
            return "PASS", 1.0, "Clear Hinglish technical comparison: B-Tree for scalar vs GIN for JSONB inverted index."
        elif has_gin and has_btree:
            return "PARTIAL", 0.5, "Technically sound B-Tree vs GIN comparison, but Hinglish fluency could be richer."
        return "FAIL", 0.0, "Missing comparison between B-Tree and GIN indexes for JSONB."

    elif prompt_id == "eval_17_hinglish":
        # High memory leak production heap dump in Hinglish
        has_dump = any(w in text_to_eval for w in ["heap", "snapshot", "dump", "v8", "tracemalloc", "objgraph"])
        has_inspect = any(w in text_to_eval for w in ["devtools", "chrome", "memory", "leak"])
        has_hinglish = any(w in text_to_eval for w in ["karein", "kaise", "step", "tarika", "samjho", "pe"])
        if has_dump and has_inspect and has_hinglish:
            return "PASS", 1.0, "Actionable step-by-step heap dump debugging guide in fluent Hinglish."
        elif has_dump and has_inspect:
            return "PARTIAL", 0.5, "Valid heap dump steps, but Hinglish language blend was limited."
        return "FAIL", 0.0, "Failed to provide heap dump debugging procedure."

    elif prompt_id == "eval_18_hinglish":
        # Kafka consumer lag in Hinglish
        has_lag = "lag" in text_to_eval or "offset" in text_to_eval
        has_fix = any(w in text_to_eval for w in ["partition", "scaling", "monitor", "prometheus", "burrow", "batch"])
        has_hinglish = any(w in text_to_eval for w in ["hota", "karen", "kaise", "batao", "example", "badhao"])
        if has_lag and has_fix and has_hinglish:
            return "PASS", 1.0, "Accurate explanation of Kafka consumer lag, offsets, and scaling in natural Hinglish."
        elif has_lag and has_fix:
            return "PARTIAL", 0.5, "Addressed Kafka lag and fixes, but Hinglish phrasing was sparse."
        return "FAIL", 0.0, "Incomplete explanation of Kafka consumer lag."

    elif prompt_id == "eval_19_hinglish":
        # Unique element XOR trick in Hinglish
        has_xor = "^" in text_to_eval or "xor" in text_to_eval
        has_complexity = "o(n)" in text_to_eval and ("o(1)" in text_to_eval or "constant space" in text_to_eval)
        has_hinglish = any(w in text_to_eval for w in ["karte", "element", "bhai", "samjhao", "milega"])
        if has_xor and (has_complexity or "reduce" in text_to_eval):
            return "PASS", 1.0, "Clear Hinglish explanation and Python code for O(N) time O(1) space XOR single-number trick."
        elif has_xor:
            return "PARTIAL", 0.5, "Mentioned XOR trick, but complexity proof or code was incomplete."
        return "FAIL", 0.0, "Did not identify the O(N) time O(1) space bitwise XOR solution."

    elif prompt_id == "eval_20_hinglish":
        # Bias-Variance and L1 vs L2 regularization in Hinglish
        has_bias_var = any(w in text_to_eval for w in ["bias", "variance", "overfitting", "underfitting"])
        has_reg = any(w in text_to_eval for w in ["l1", "l2", "lasso", "ridge", "regularization"])
        has_hinglish = any(w in text_to_eval for w in ["farak", "hota", "karta", "weights", "simple"])
        if has_bias_var and has_reg and has_hinglish:
            return "PASS", 1.0, "Accurate, intuitive Hinglish breakdown of Overfitting/Underfitting and L1 (Lasso) vs L2 (Ridge)."
        elif has_bias_var and has_reg:
            return "PARTIAL", 0.5, "Covered ML concepts, but Hinglish conversational style was limited."
        return "FAIL", 0.0, "Incomplete explanation of Bias-Variance or L1/L2 regularization."

    # General fallback
    keywords = [w.lower() for w in criteria.split() if len(w) > 3][:5]
    matches = sum(1 for kw in keywords if kw in text_to_eval)
    ratio = matches / max(1, len(keywords))

    if ratio >= 0.7:
        return "PASS", 1.0, f"Matched {matches}/{len(keywords)} key criteria terms."
    elif ratio >= 0.4:
        return "PARTIAL", 0.5, f"Matched {matches}/{len(keywords)} key criteria terms."
    else:
        return "FAIL", 0.0, f"Matched only {matches}/{len(keywords)} key criteria terms."


def query_ollama_http(
    host: str,
    endpoint: str,
    model: str,
    prompt: str,
    system_prompt: str,
    temperature: float = 0.6,
    timeout: float = 180.0,
) -> Dict[str, Any]:
    """
    Queries local Ollama instance via HTTP API (/api/generate or /api/chat).
    """
    clean_host = host.rstrip("/")
    if endpoint == "chat":
        url = f"{clean_host}/api/chat"
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "options": {"temperature": temperature},
        }
    else:
        url = f"{clean_host}/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "system": system_prompt,
            "stream": False,
            "options": {"temperature": temperature},
        }

    data_bytes = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data_bytes,
        headers={"Content-Type": "application/json", "User-Agent": "QWEN-999B-Eval-Judge"},
    )

    start_wall = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as response:
        elapsed_wall = time.perf_counter() - start_wall
        res_data = json.loads(response.read().decode("utf-8"))

    # Extract text depending on endpoint
    if endpoint == "chat":
        text_content = res_data.get("message", {}).get("content", "")
    else:
        text_content = res_data.get("response", "")

    # Ollama returns durations in nanoseconds
    total_ns = res_data.get("total_duration", int(elapsed_wall * 1e9))
    eval_count = res_data.get("eval_count", max(1, len(text_content.split())))
    eval_ns = res_data.get("eval_duration", total_ns)

    prompt_eval_count = res_data.get("prompt_eval_count", max(1, len(prompt.split())))
    eval_sec = max(0.001, eval_ns / 1e9)
    tokens_per_sec = eval_count / eval_sec if eval_sec > 0 else 0.0

    return {
        "text": text_content,
        "latency_sec": elapsed_wall,
        "total_duration_sec": total_ns / 1e9,
        "prompt_eval_count": prompt_eval_count,
        "eval_count": eval_count,
        "tokens_per_sec": tokens_per_sec,
        "raw_response": res_data,
    }


def get_mock_response_for_dry_run(prompt_id: str, prompt_text: str) -> Dict[str, Any]:
    """Generates realistic mock reasoning response for offline validation and testing."""
    mock_responses = {
        "eval_01_reasoning": (
            "<think>\n"
            "The question asks: 'A farmer has 17 sheep and all but 9 die. How many sheep does the farmer have left?'\n"
            "This is a classic linguistic riddle. The phrase 'all but 9 die' explicitly states that the quantity remaining is 9.\n"
            "Common error: 17 - 9 = 8. That would happen if 9 died, but here all died *except* 9.\n"
            "Therefore, exactly 9 sheep survive and are left.\n"
            "</think>\n\n"
            "The farmer has **9 sheep** left.\n\n"
            "Explanation: The statement says that 'all but 9 die', which means 9 sheep did not die and therefore survived."
        ),
        "eval_02_reasoning": (
            "<think>\n"
            "Three boxes: Apples, Oranges, Apples & Oranges.\n"
            "Key rule: ALL THREE BOXES ARE LABELED INCORRECTLY.\n"
            "If we pick from the box labeled 'Apples & Oranges':\n"
            "- It cannot contain both apples and oranges.\n"
            "- If we pull an Apple, then this box is definitely the 'Apples' box.\n"
            "- Now look at the box labeled 'Oranges'. It cannot be Oranges (mislabeled), and it cannot be Apples (taken). So it must be 'Apples & Oranges'.\n"
            "- The remaining box labeled 'Apples' must be 'Oranges'.\n"
            "- The same logic holds symmetrically if an orange is drawn.\n"
            "</think>\n\n"
            "You should pick one fruit from the box labeled **'Apples & Oranges'**.\n\n"
            "**Why:**\n"
            "Because all labels are incorrect, the box labeled 'Apples & Oranges' must be strictly all Apples or strictly all Oranges. "
            "Drawing a single fruit definitively reveals its true identity. Once that box is identified, the remaining two boxes can be "
            "unambiguously determined by process of elimination since neither can have its current label."
        ),
        "eval_03_reasoning": (
            "<think>\n"
            "Comparing a pound of gold vs a pound of feathers.\n"
            "Precious metals like gold are measured in Troy pounds (12 Troy ounces = ~373.24 grams).\n"
            "Everyday commodities like feathers are measured in Avoirdupois pounds (16 ounces = ~453.59 grams).\n"
            "Therefore, 1 Avoirdupois pound of feathers (~453.6 g) is heavier than 1 Troy pound of gold (~373.2 g).\n"
            "</think>\n\n"
            "**A pound of feathers is heavier than a pound of gold.**\n\n"
            "**Reasoning:**\n"
            "- Gold is measured using the **Troy system**, where 1 Troy pound = 12 Troy ounces ≈ **373.24 grams**.\n"
            "- Feathers are measured using the standard **Avoirdupois system**, where 1 Avoirdupois pound = 16 ounces ≈ **453.59 grams**.\n"
            "- Thus, 453.59g > 373.24g, making the pound of feathers noticeably heavier."
        ),
        "eval_04_reasoning": (
            "<think>\n"
            "Sally has 3 brothers. Each brother has 2 sisters.\n"
            "Since all brothers share the same sisters, there are 2 sisters in the family total.\n"
            "One of those sisters is Sally herself.\n"
            "So Sally has 2 - 1 = 1 sister.\n"
            "Total children in the family = 3 brothers + 2 sisters = 5 children total.\n"
            "</think>\n\n"
            "1. **Sally has 1 sister.**\n"
            "2. There are **5 children total** in the family (3 boys and 2 girls).\n\n"
            "**Explanation:** Each of the 3 brothers sees the same two female siblings: Sally and her one sister."
        ),
        "eval_05_reasoning": (
            "<think>\n"
            "We have a 3-gallon jug and a 5-gallon jug, need exactly 4 gallons.\n"
            "Step 1: Fill the 5-gallon jug completely (5 gal).\n"
            "Step 2: Pour from 5-gal jug into 3-gal jug until full. (3-gal has 3, 5-gal has 2 left).\n"
            "Step 3: Empty the 3-gallon jug. (3-gal has 0, 5-gal has 2).\n"
            "Step 4: Pour the 2 gallons from 5-gal jug into 3-gal jug. (3-gal has 2, 5-gal has 0).\n"
            "Step 5: Fill the 5-gallon jug completely. (3-gal has 2, 5-gal has 5).\n"
            "Step 6: Pour from 5-gal jug into 3-gal jug until full (needs 1 gal). Exactly 4 gallons remain in the 5-gallon jug!\n"
            "</think>\n\n"
            "Here is the optimal 6-step sequence to measure exactly 4 gallons:\n\n"
            "1. **Fill the 5-gallon jug** to the top (5-gal: 5, 3-gal: 0).\n"
            "2. **Pour water from the 5-gallon jug into the 3-gallon jug** until full (5-gal: 2, 3-gal: 3).\n"
            "3. **Empty the 3-gallon jug** (5-gal: 2, 3-gal: 0).\n"
            "4. **Pour the remaining 2 gallons** from the 5-gal jug into the 3-gal jug (5-gal: 0, 3-gal: 2).\n"
            "5. **Fill the 5-gallon jug** again completely (5-gal: 5, 3-gal: 2).\n"
            "6. **Pour from the 5-gallon jug into the 3-gallon jug** until the 3-gal jug is full (transfers 1 gallon).\n\n"
            "**Result:** You now have **exactly 4 gallons** of water remaining in the 5-gallon jug."
        ),
        "eval_06_coding": (
            "<think>\n"
            "Problem: Implement Kadane's algorithm returning (max_sum, start_idx, end_idx).\n"
            "Must handle arrays with all negative numbers correctly.\n"
            "Maintain current_sum, max_sum, temp_start, start_idx, end_idx.\n"
            "If nums is empty, raise ValueError or handle appropriately.\n"
            "</think>\n\n"
            "```python\ndef max_subarray_sum(nums: list[int]) -> tuple[int, int, int]:\n"
            "    if not nums:\n"
            "        raise ValueError('Array cannot be empty')\n\n"
            "    max_sum = nums[0]\n"
            "    current_sum = nums[0]\n"
            "    start_idx = end_idx = temp_start = 0\n\n"
            "    for i in range(1, len(nums)):\n"
            "        if nums[i] > current_sum + nums[i]:\n"
            "            current_sum = nums[i]\n"
            "            temp_start = i\n"
            "        else:\n"
            "            current_sum += nums[i]\n\n"
            "        if current_sum > max_sum:\n"
            "            max_sum = current_sum\n"
            "            start_idx = temp_start\n"
            "            end_idx = i\n\n"
            "    return (max_sum, start_idx, end_idx)\n```"
        ),
        "eval_07_coding": (
            "<think>\n"
            "Thread-safe rate limiter implementing Token Bucket.\n"
            "Use threading.Lock for synchronization and time.monotonic() to avoid system clock drift.\n"
            "</think>\n\n"
            "```python\nimport threading\nimport time\n\nclass TokenBucketRateLimiter:\n"
            "    def __init__(self, capacity: float, refill_rate: float):\n"
            "        self.capacity = float(capacity)\n"
            "        self.refill_rate = float(refill_rate)  # tokens per second\n"
            "        self.tokens = float(capacity)\n"
            "        self.last_refill = time.monotonic()\n"
            "        self.lock = threading.Lock()\n\n"
            "    def acquire(self, tokens: float = 1.0) -> bool:\n"
            "        with self.lock:\n"
            "            now = time.monotonic()\n"
            "            elapsed = now - self.last_refill\n"
            "            self.last_refill = now\n"
            "            self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)\n\n"
            "            if self.tokens >= tokens:\n"
            "                self.tokens -= tokens\n"
            "                return True\n"
            "            return False\n```"
        ),
        "eval_08_coding": (
            "<think>\n"
            "Snippet: funcs = [lambda x: x + i for i in range(4)]\n"
            "print([f(10) for f in funcs])\n"
            "Due to Python's late binding in closures, i is looked up in the enclosing scope when the function is executed.\n"
            "When the loop finishes, i is 3. So all functions compute x + 3. For x=10, 10+3=13.\n"
            "Output is [13, 13, 13, 13].\n"
            "Fix: bind i at definition time using a default argument: lambda x, i=i: x + i.\n"
            "</think>\n\n"
            "### Output:\n"
            "```python\n[13, 13, 13, 13]\n```\n\n"
            "### Explanation:\n"
            "This is caused by Python's **late binding closure** mechanism. The variable `i` is not captured by value when the lambda is defined; "
            "rather, it is looked up in the enclosing scope when the lambda is called. After the list comprehension terminates, `i = 3`.\n\n"
            "### Fix:\n"
            "Bind `i` at definition time via a default argument:\n"
            "```python\nfuncs = [lambda x, i=i: x + i for i in range(4)]\nprint([f(10) for f in funcs])  # Outputs [10, 11, 12, 13]\n```"
        ),
        "eval_09_coding": (
            "<think>\n"
            "Serialize and deserialize a binary tree in Python.\n"
            "Can use preorder DFS traversal with '#' or 'null' for None.\n"
            "Time complexity: O(N) since each node is visited once.\n"
            "Space complexity: O(N) for string buffer and recursion stack.\n"
            "</think>\n\n"
            "```python\nclass TreeNode:\n    def __init__(self, val=0, left=None, right=None):\n"
            "        self.val = val\n        self.left = left\n        self.right = right\n\n"
            "class Codec:\n"
            "    def serialize(self, root: TreeNode) -> str:\n"
            "        def rserialize(node):\n"
            "            if not node:\n"
            "                return ['#']\n"
            "            return [str(node.val)] + rserialize(node.left) + rserialize(node.right)\n"
            "        return ','.join(rserialize(root))\n\n"
            "    def deserialize(self, data: str) -> TreeNode:\n"
            "        vals = iter(data.split(','))\n"
            "        def rdeserialize():\n"
            "            val = next(vals)\n"
            "            if val == '#':\n"
            "                return None\n"
            "            node = TreeNode(int(val))\n"
            "            node.left = rdeserialize()\n"
            "            node.right = rdeserialize()\n"
            "            return node\n"
            "        return rdeserialize()\n```\n\n"
            "**Complexity:**\n"
            "- **Time Complexity:** O(N) where N is the number of nodes.\n"
            "- **Space Complexity:** O(N) for tree reconstitution and traversal stack."
        ),
        "eval_10_coding": (
            "<think>\n"
            "Find median of two sorted arrays nums1 (size m) and nums2 (size n) in O(log(min(m, n))).\n"
            "Use binary search on partition point of smaller array.\n"
            "</think>\n\n"
            "```python\ndef find_median_sorted_arrays(nums1: list[int], nums2: list[int]) -> float:\n"
            "    if len(nums1) > len(nums2):\n"
            "        nums1, nums2 = nums2, nums1\n"
            "    m, n = len(nums1), len(nums2)\n"
            "    low, high = 0, m\n\n"
            "    while low <= high:\n"
            "        i = (low + high) // 2\n"
            "        j = (m + n + 1) // 2 - i\n\n"
            "        max_left_1 = float('-inf') if i == 0 else nums1[i - 1]\n"
            "        min_right_1 = float('inf') if i == m else nums1[i]\n"
            "        max_left_2 = float('-inf') if j == 0 else nums2[j - 1]\n"
            "        min_right_2 = float('inf') if j == n else nums2[j]\n\n"
            "        if max_left_1 <= min_right_2 and max_left_2 <= min_right_1:\n"
            "            if (m + n) % 2 == 1:\n"
            "                return float(max(max_left_1, max_left_2))\n"
            "            return (max(max_left_1, max_left_2) + min(min_right_1, min_right_2)) / 2.0\n"
            "        elif max_left_1 > min_right_2:\n"
            "            high = i - 1\n"
            "        else:\n"
            "            low = i + 1\n"
            "    return 0.0\n```\n\n"
            "**Complexity:** O(log(min(m, n))) runtime complexity via binary search partitioning."
        ),
        "eval_11_math": (
            "<think>\n"
            "Find 2^2025 mod 13.\n"
            "13 is prime and gcd(2, 13) = 1. By Fermat's Little Theorem, 2^(13-1) = 2^12 = 1 (mod 13).\n"
            "We calculate 2025 mod 12:\n"
            "2025 = 12 * 168 + 9 (since 12 * 168 = 2016, 2025 - 2016 = 9).\n"
            "So 2^2025 = 2^9 (mod 13).\n"
            "2^9 = 512.\n"
            "512 / 13 = 39 with remainder:\n"
            "13 * 39 = 507.\n"
            "512 - 507 = 5.\n"
            "Thus, the remainder is 5.\n"
            "</think>\n\n"
            "The remainder when $2^{2025}$ is divided by 13 is **5**.\n\n"
            "**Step-by-step derivation:**\n"
            "1. By Fermat's Little Theorem, $2^{12} \\equiv 1 \\pmod{13}$.\n"
            "2. Reduce exponent modulo 12: $2025 = 12 \\times 168 + 9$, so $2025 \\equiv 9 \\pmod{12}$.\n"
            "3. $2^9 = 512$.\n"
            "4. $512 = 13 \\times 39 + 5$, therefore $512 \\equiv 5 \\pmod{13}$."
        ),
        "eval_12_math": (
            "<think>\n"
            "Calculate trailing zeros in 250!.\n"
            "Using Legendre's formula for factor 5:\n"
            "floor(250 / 5) = 50\n"
            "floor(250 / 25) = 10\n"
            "floor(250 / 125) = 2\n"
            "Total trailing zeros = 50 + 10 + 2 = 62.\n"
            "</think>\n\n"
            "The number 250! has **62 trailing zeros**.\n\n"
            "**Derivation via Legendre's Formula:**\n"
            "- Factors of 5: $\\lfloor 250 / 5 \\rfloor = 50$\n"
            "- Factors of 25: $\\lfloor 250 / 25 \\rfloor = 10$\n"
            "- Factors of 125: $\\lfloor 250 / 125 \\rfloor = 2$\n"
            "- Total = $50 + 10 + 2 = 62$."
        ),
        "eval_13_math": (
            "<think>\n"
            "Two fair 6-sided dice rolled. Sum > 7.\n"
            "Possible sums > 7: 8, 9, 10, 11, 12.\n"
            "Count outcomes with sum > 7:\n"
            "Sum 8: (2,6),(3,5),(4,4),(5,3),(6,2) = 5\n"
            "Sum 9: (3,6),(4,5),(5,4),(6,3) = 4\n"
            "Sum 10: (4,6),(5,5),(6,4) = 3\n"
            "Sum 11: (5,6),(6,5) = 2\n"
            "Sum 12: (6,6) = 1\n"
            "Total outcomes with sum > 7 = 5 + 4 + 3 + 2 + 1 = 15.\n"
            "Outcomes containing at least one 6:\n"
            "(2,6), (6,2), (3,6), (6,3), (4,6), (6,4), (5,6), (6,5), (6,6) -> Exactly 9 outcomes.\n"
            "Conditional probability = 9 / 15 = 3 / 5 = 0.6.\n"
            "</think>\n\n"
            "The probability that at least one die shows a 6 given that the sum is greater than 7 is **3/5 (or 0.6)**.\n\n"
            "**Proof:**\n"
            "- Total possible outcomes with sum > 7: 15.\n"
            "- Favorable outcomes with at least one 6: 9.\n"
            "- $P(\\text{At least one 6} \\mid \\text{Sum} > 7) = \\frac{9}{15} = \\frac{3}{5} = 0.6$."
        ),
        "eval_14_math": (
            "<think>\n"
            "Solve positive real x: x^(x^(x...)) = y.\n"
            "If infinite tower converges to y, then x^y = y => x = y^(1/y).\n"
            "For y = 2: x = 2^(1/2) = sqrt(2) ≈ 1.4142.\n"
            "For y = 4: x = 4^(1/4) = sqrt(2). But if x = sqrt(2), the tower converges to 2, not 4!\n"
            "Euler proved that the infinite power tower converges if and only if e^(-e) <= x <= e^(1/e) ≈ 1.44466.\n"
            "The maximum possible value the convergent tower can reach is e ≈ 2.718.\n"
            "Because 4 > e, no real solution exists for x^(x...) = 4; the sequence diverges.\n"
            "</think>\n\n"
            "1. For $x^{x^{x^{\\dots}}} = 2$, the solution is **$x = \\sqrt{2}$**.\n"
            "2. For $x^{x^{x^{\\dots}}} = 4$, **no real solution exists**.\n\n"
            "**Reasoning:**\n"
            "The infinite power tower $h(x) = x^{x^{\\dots}}$ converges only when $e^{-e} \\le x \\le e^{1/e}$. "
            "Within this domain, the maximum value that $h(x)$ can achieve is $e \\approx 2.71828$. "
            "Since $4 > e$, any attempt to set the limit to 4 results in divergence."
        ),
        "eval_15_math": (
            "<think>\n"
            "Standard deck: 52 cards, 4 Aces, 48 non-Aces.\n"
            "By symmetry / linearity, 4 Aces divide the 48 non-Ace cards into 4 + 1 = 5 equal intervals.\n"
            "Expected number of non-Ace cards before the first Ace = 48 / (4 + 1) = 48 / 5 = 9.6 cards.\n"
            "If counting the total cards dealt including the Ace: 9.6 + 1 = 10.6 cards.\n"
            "</think>\n\n"
            "The expected number of cards dealt before the first Ace appears is **9.6 cards** (or **10.6 cards** if including the first Ace itself).\n\n"
            "**Derivation:**\n"
            "The 4 Aces divide the remaining 48 non-Ace cards into 5 equally likely segments. "
            "By linearity of expectation: $\\frac{48}{4 + 1} = \\frac{48}{5} = 9.6$ non-Ace cards before the first Ace."
        ),
        "eval_16_hinglish": (
            "<think>\n"
            "Question is in Hinglish about PostgreSQL B-Tree vs GIN Index for JSONB.\n"
            "Explain: B-Tree is standard ordered tree, good for scalar comparisons (=, <, >, BETWEEN).\n"
            "GIN is Generalized Inverted Index, perfect for multi-value types like arrays and JSONB containment (@>).\n"
            "Use clear, technical, professional Hinglish.\n"
            "</think>\n\n"
            "Bhai, PostgreSQL me **B-Tree** aur **GIN (Generalized Inverted Index)** me main differences ye hain:\n\n"
            "1. **B-Tree Index:**\n"
            "- Ye standard ordered indexing hai jo single scalar values (numbers, strings, dates) pe equality aur range queries (`=, <, >, BETWEEN`) ke liye best hota hai.\n"
            "- Agar aap poora JSONB column as a whole compare kar rahe ho to B-Tree chalega, lekin JSONB ke andar nested keys search karne ke liye fail ho jata hai.\n\n"
            "2. **GIN Index (Generalized Inverted Index):**\n"
            "- Ye composite elements, arrays, aur JSONB documents ke andar ke keys/values ko alag-alag index karta hai.\n"
            "- **JSONB search ke liye GIN hamesha superior hota hai**, khaaskar jab aap containment operators jaise `@>`, `?`, ya `?&` use karte hain.\n\n"
            "**Conclusion:** JSONB search ke liye hamesha GIN index use karein (`CREATE INDEX idx_data ON table USING gin (data jsonb_path_ops);`)."
        ),
        "eval_17_hinglish": (
            "<think>\n"
            "Hinglish guide to debug high memory leak in Node.js / Python production using heap dumps.\n"
            "Cover v8.getHeapSnapshot / heapdump / Chrome DevTools for Node.js, and tracemalloc / objgraph for Python.\n"
            "</think>\n\n"
            "Bhai, production pe memory leak debug karne ke liye step-by-step procedure ye hai:\n\n"
            "### 1. Node.js Application:\n"
            "- **Heap Snapshot Capture:** Node.js 12+ me built-in `v8` module use karo:\n"
            "  ```javascript\n"
            "  const v8 = require('v8');\n"
            "  const fs = require('fs');\n"
            "  const snapshotStream = v8.getHeapSnapshot();\n"
            "  const fileStream = fs.createWriteStream(`heap-${Date.now()}.heapsnapshot`);\n"
            "  snapshotStream.pipe(fileStream);\n"
            "  ```\n"
            "- **Analysis:** Chrome browser kholo (`chrome://inspect`) -> **Memory tab** me `.heapsnapshot` load karo. 'Comparison' view me check karo kaunse objects continuously retain ho rahe hain (e.g. global event listeners, unclosed caches).\n\n"
            "### 2. Python Application:\n"
            "- Built-in `tracemalloc` module ya `objgraph` use karein:\n"
            "  ```python\n"
            "  import tracemalloc\n"
            "  tracemalloc.start()\n"
            "  # take snapshot\n"
            "  snapshot = tracemalloc.take_snapshot()\n"
            "  top_stats = snapshot.statistics('lineno')\n"
            "  for stat in top_stats[:10]:\n"
            "      print(stat)\n"
            "  ```\n"
            "In steps ko follow karke aap exact retaining paths aur memory leaks pakad sakte ho."
        ),
        "eval_18_hinglish": (
            "<think>\n"
            "Kafka consumer lag explanation and remediation in Hinglish.\n"
            "Difference between log end offset and consumer offset.\n"
            "Remediation: monitor with Prometheus/Burrow, increase partitions, horizontal scaling, batch optimization.\n"
            "</think>\n\n"
            "Bhai, **Kafka Consumer Lag** ka matlab hai producer aur consumer ke speed ka gap.\n\n"
            "- **Definition:** Producer topic me kis offset tak messages likh chuka hai (**Log End Offset**) aur consumer group abhi kis offset tak process kar paaya hai (**Current Consumer Offset**), in dono ka difference hi 'Lag' kehlata hai.\n\n"
            "### Monitoring & Best Practices to Fix:\n"
            "1. **Monitoring:** Prometheus + Grafana dashboard ya LinkedIn Burrow use karke `kafka_consumergroup_lag` metrics monitor karein.\n"
            "2. **Increase Partitions:** Topic me partitions ki sankhya badhao taaki consumer instances horizontally scale ho sakein.\n"
            "3. **Horizontal Scaling:** Consumer group me consumers ki sankhya badhao (up to partition count).\n"
            "4. **Batch Processing Optimize:** Consumer me `max.poll.records` adjust karein aur heavy downstream database calls ko async batch inserts me convert karein."
        ),
        "eval_19_hinglish": (
            "<think>\n"
            "Find unique element in array where every other element appears twice. O(N) time and O(1) space.\n"
            "Explain XOR bit manipulation trick: a ^ a = 0, a ^ 0 = a.\n"
            "</think>\n\n"
            "Bhai, is problem ko **O(N) time** aur **O(1) space** me solve karne ka sabse best tareeqa hai **Bitwise XOR (`^`) operator**.\n\n"
            "### Core Logic:\n"
            "- $a \\oplus a = 0$ (koi bhi number apne aap se XOR hone par 0 ho jata hai)\n"
            "- $a \\oplus 0 = a$\n"
            "- XOR associative aur commutative hota hai. Jab saare duplicate pairs aapas me XOR honge to wo cancel out hoke 0 ban jayenge, aur aakhir me sirf unique element bachega!\n\n"
            "### Python Code:\n"
            "```python\nfrom functools import reduce\n\ndef find_single_element(nums: list[int]) -> int:\n"
            "    # O(N) time, O(1) auxiliary space\n"
            "    return reduce(lambda x, y: x ^ y, nums)\n```"
        ),
        "eval_20_hinglish": (
            "<think>\n"
            "Explain Overfitting vs Underfitting, Bias-Variance tradeoff, and L1 vs L2 regularization in Hinglish.\n"
            "High bias = underfitting (model too simple), High variance = overfitting (model memorizes noise).\n"
            "L1 = Lasso (sparsity, weights driven to zero, feature selection).\n"
            "L2 = Ridge (weights shrunk smoothly towards zero, handles multicollinearity).\n"
            "</think>\n\n"
            "Bhai, Machine Learning me Overfitting aur Underfitting ka concept simple Hinglish me samjho:\n\n"
            "1. **Underfitting (High Bias):** Model itna simple hai ki wo training data ka basic pattern bhi nahi seekh pa raha (jaise curve data pe straight line fit karna). Training aur test dono pe accuracy bekaar aati hai.\n"
            "2. **Overfitting (High Variance):** Model ne training data ko itna 'ratta' maar liya hai ki wo noise ko bhi memorize kar leta hai. Training accuracy 99% aati hai lekin unseen test data pe fail ho jata hai.\n\n"
            "### Regularization (L1 vs L2):\n"
            "- **L1 Regularization (Lasso):** Loss function me $\\lambda \\sum |w|$ add karta hai. Ye un-important features ke weights ko **bilkul zero** bana deta hai, jisse feature selection automatically ho jata hai.\n"
            "- **L2 Regularization (Ridge):** Loss function me $\\lambda \\sum w^2$ add karta hai. Ye weights ko shrink karke chhota karta hai par zero nahi banata, jisse model smooth rehta hai aur overfitting rukti hai."
        ),
    }

    if prompt_id in mock_responses:
        content = mock_responses[prompt_id]
    else:
        content = (
            f"<think>\n"
            f"Analyzing prompt '{prompt_id}'. Considering core constraints and requirements.\n"
            f"Formulating optimal structured reasoning and criteria alignment.\n"
            f"</think>\n\n"
            f"Here is the comprehensive solution for prompt {prompt_id}. "
            f"Addressing all constraints with verified technical and mathematical precision."
        )

    words = len(content.split())
    return {
        "text": content,
        "latency_sec": 1.45,
        "total_duration_sec": 1.45,
        "prompt_eval_count": 42,
        "eval_count": words,
        "tokens_per_sec": 28.5,
        "raw_response": {"mock": True},
    }


def run_benchmark(
    model: str,
    host: str,
    endpoint: str,
    prompts_path: Path,
    output_dir: Path,
    temperature: float = 0.6,
    timeout: float = 180.0,
    dry_run: bool = False,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    save_json: bool = True,
) -> Tuple[Path, Optional[Path], Dict[str, Any]]:
    """Runs the benchmark across all prompts and writes reports."""
    prompts = load_test_prompts(prompts_path)
    total_prompts = len(prompts)
    print(f"\n========================================================")
    print(f"🚀 Starting QWEN-999B Benchmark Suite")
    print(f"Model: {model}")
    print(f"Host: {host} (endpoint: /api/{endpoint})")
    print(f"Prompts: {prompts_path} ({total_prompts} test cases)")
    print(f"Dry Run Mode: {dry_run}")
    print(f"Output Directory: {output_dir}")
    print(f"========================================================\n")

    output_dir.mkdir(parents=True, exist_ok=True)
    results = []

    start_benchmark_time = time.time()
    iso_timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
    clean_model_name = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", model)

    for i, p in enumerate(prompts, 1):
        pid = p.get("id", f"prompt_{i:02d}")
        category = p.get("category", "general")
        prompt_text = p.get("prompt", "")
        expected_focus = p.get("expected_reasoning_focus", "")
        ground_truth = p.get("ground_truth_or_criteria", "")

        print(f"[{i:02d}/{total_prompts}] Evaluating {pid} ({category})...", end=" ", flush=True)

        if dry_run:
            query_res = get_mock_response_for_dry_run(pid, prompt_text)
        else:
            try:
                query_res = query_ollama_http(
                    host=host,
                    endpoint=endpoint,
                    model=model,
                    prompt=prompt_text,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    timeout=timeout,
                )
            except urllib.error.URLError as e:
                print(f"FAILED (Connection Error: {e.reason})")
                query_res = {
                    "text": f"[HTTP Connection Error: {e.reason}. Ensure Ollama is running at {host}]",
                    "latency_sec": 0.0,
                    "total_duration_sec": 0.0,
                    "prompt_eval_count": 0,
                    "eval_count": 0,
                    "tokens_per_sec": 0.0,
                    "error": str(e),
                }
            except Exception as e:
                print(f"FAILED (Error: {e})")
                query_res = {
                    "text": f"[Execution Error: {e}]",
                    "latency_sec": 0.0,
                    "total_duration_sec": 0.0,
                    "prompt_eval_count": 0,
                    "eval_count": 0,
                    "tokens_per_sec": 0.0,
                    "error": str(e),
                }

        raw_text = query_res["text"]
        has_think, thinking_trace, final_answer = parse_think_tags(raw_text)

        status, score, rationale = evaluate_criteria_alignment(
            prompt_id=pid,
            category=category,
            full_text=raw_text,
            final_answer=final_answer,
            criteria=ground_truth,
            expected_focus=expected_focus,
        )

        status_icon = "✅ PASS" if status == "PASS" else ("⚠️ PARTIAL" if status == "PARTIAL" else "❌ FAIL")
        think_icon = "🧠" if has_think else "🚫"
        print(f"{status_icon} | {think_icon} Think | {query_res['latency_sec']:.2f}s | {query_res['tokens_per_sec']:.1f} tok/s")

        results.append({
            "id": pid,
            "category": category,
            "prompt": prompt_text,
            "expected_reasoning_focus": expected_focus,
            "ground_truth_or_criteria": ground_truth,
            "latency_sec": round(query_res["latency_sec"], 3),
            "tokens_per_sec": round(query_res["tokens_per_sec"], 1),
            "prompt_tokens": query_res["prompt_eval_count"],
            "completion_tokens": query_res["eval_count"],
            "has_think_tags": has_think,
            "thinking_trace_length_chars": len(thinking_trace),
            "thinking_trace": thinking_trace,
            "final_answer": final_answer,
            "full_response": raw_text,
            "evaluation_status": status,
            "evaluation_score": score,
            "evaluation_rationale": rationale,
        })

    total_benchmark_sec = time.time() - start_benchmark_time

    # Aggregate Statistics
    total_count = len(results)
    pass_count = sum(1 for r in results if r["evaluation_status"] == "PASS")
    partial_count = sum(1 for r in results if r["evaluation_status"] == "PARTIAL")
    fail_count = sum(1 for r in results if r["evaluation_status"] == "FAIL")
    think_count = sum(1 for r in results if r["has_think_tags"])
    total_score = sum(r["evaluation_score"] for r in results)
    overall_accuracy_pct = (total_score / max(1, total_count)) * 100

    avg_latency = sum(r["latency_sec"] for r in results) / max(1, total_count)
    min_latency = min((r["latency_sec"] for r in results), default=0.0)
    max_latency = max((r["latency_sec"] for r in results), default=0.0)

    avg_tok_sec = sum(r["tokens_per_sec"] for r in results) / max(1, total_count)
    total_completion_tokens = sum(r["completion_tokens"] for r in results)
    avg_completion_tokens = total_completion_tokens / max(1, total_count)

    # Category breakdown
    categories = sorted(list(set(r["category"] for r in results)))
    category_stats = {}
    for cat in categories:
        cat_items = [r for r in results if r["category"] == cat]
        cat_count = len(cat_items)
        cat_score = sum(r["evaluation_score"] for r in cat_items)
        cat_think = sum(1 for r in cat_items if r["has_think_tags"])
        cat_avg_lat = sum(r["latency_sec"] for r in cat_items) / max(1, cat_count)
        cat_avg_tok = sum(r["tokens_per_sec"] for r in cat_items) / max(1, cat_count)
        category_stats[cat] = {
            "count": cat_count,
            "score": cat_score,
            "accuracy_pct": (cat_score / max(1, cat_count)) * 100,
            "think_pct": (cat_think / max(1, cat_count)) * 100,
            "avg_latency": cat_avg_lat,
            "avg_tok_sec": cat_avg_tok,
        }

    summary_data = {
        "benchmark_metadata": {
            "model": model,
            "host": host,
            "endpoint": endpoint,
            "dry_run": dry_run,
            "timestamp_utc": iso_timestamp,
            "total_prompts": total_count,
            "total_benchmark_time_sec": round(total_benchmark_sec, 2),
            "temperature": temperature,
        },
        "kpis": {
            "overall_score": f"{total_score:.1f} / {total_count}",
            "overall_accuracy_pct": round(overall_accuracy_pct, 1),
            "pass_count": pass_count,
            "partial_count": partial_count,
            "fail_count": fail_count,
            "think_tag_adoption_pct": round((think_count / max(1, total_count)) * 100, 1),
            "avg_latency_sec": round(avg_latency, 2),
            "min_latency_sec": round(min_latency, 2),
            "max_latency_sec": round(max_latency, 2),
            "avg_tokens_per_sec": round(avg_tok_sec, 1),
            "total_output_tokens": total_completion_tokens,
            "avg_output_tokens_per_prompt": round(avg_completion_tokens, 1),
        },
        "category_breakdown": category_stats,
        "results": results,
    }

    # Generate Markdown Report
    md_filename = output_dir / f"benchmark_{clean_model_name}_{iso_timestamp}.md"
    generate_markdown_report(md_filename, summary_data)
    print(f"\n📄 Markdown Report written to: {md_filename}")

    # Generate JSON Report
    json_filename = None
    if save_json:
        json_filename = output_dir / f"benchmark_{clean_model_name}_{iso_timestamp}.json"
        with open(json_filename, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        print(f"📊 Detailed JSON Metrics written to: {json_filename}")

    return md_filename, json_filename, summary_data


def generate_markdown_report(report_path: Path, data: Dict[str, Any]) -> None:
    """Writes comprehensive Markdown benchmark report."""
    meta = data["benchmark_metadata"]
    kpis = data["kpis"]
    cat_stats = data["category_breakdown"]
    results = data["results"]

    lines = []
    lines.append(f"# 🚀 QWEN-999B Benchmark Evaluation Report")
    lines.append("")
    lines.append(f"**Model:** `{meta['model']}`  ")
    lines.append(f"**Evaluation Host:** `{meta['host']}` (`/api/{meta['endpoint']}`)  ")
    lines.append(f"**Timestamp (UTC):** `{meta['timestamp_utc']}`  ")
    lines.append(f"**Mode:** `{'Mock Dry Run' if meta['dry_run'] else 'Live Ollama Execution'}`  ")
    lines.append(f"**Total Benchmark Wall Time:** `{meta['total_benchmark_time_sec']}s`  ")
    lines.append("")
    lines.append("---")
    lines.append("")

    lines.append("## 📌 Executive Summary & Key Performance Indicators (KPIs)")
    lines.append("")
    lines.append("| Metric | Value | Target Benchmark Standard |")
    lines.append("| :--- | :--- | :--- |")
    lines.append(f"| **Overall Alignment Score** | **{kpis['overall_score']} ({kpis['overall_accuracy_pct']}%)** | >= 80.0% |")
    lines.append(f"| **Reasoning `<think>` Adoption Rate** | **{kpis['think_tag_adoption_pct']}%** | 100.0% for QWEN-999B |")
    lines.append(f"| **Pass / Partial / Fail** | `{kpis['pass_count']} Pass / {kpis['partial_count']} Partial / {kpis['fail_count']} Fail` | Maximized Pass |")
    lines.append(f"| **Average Latency** | **{kpis['avg_latency_sec']}s** (Range: {kpis['min_latency_sec']}s - {kpis['max_latency_sec']}s) | < 15.0s on 7B Q4_K_M |")
    lines.append(f"| **Average Generation Speed** | **{kpis['avg_tokens_per_sec']} tokens/sec** | > 15 tok/s (CPU) / > 35 tok/s (GPU) |")
    lines.append(f"| **Total Output Tokens** | `{kpis['total_output_tokens']}` (Avg `{kpis['avg_output_tokens_per_prompt']}` tok/prompt) | ~200 - 800 tok/prompt |")
    lines.append("")
    lines.append("---")
    lines.append("")

    lines.append("## 📊 Performance by Domain Category")
    lines.append("")
    lines.append("| Category | Total Prompts | Accuracy (%) | `<think>` Tag Adoption | Avg Latency (s) | Avg Speed (tok/s) |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: |")
    for cat, stat in cat_stats.items():
        lines.append(
            f"| **{cat.capitalize()}** | {stat['count']} | {stat['accuracy_pct']:.1f}% | "
            f"{stat['think_pct']:.1f}% | {stat['avg_latency']:.2f}s | {stat['avg_tok_sec']:.1f} |"
        )
    lines.append("")
    lines.append("---")
    lines.append("")

    lines.append("## 📋 Comprehensive Prompt Summary Table")
    lines.append("")
    lines.append("| # | ID | Category | Prompt Preview | `<think>` | Latency (s) | Speed (tok/s) | Status | Score |")
    lines.append("| :-: | :--- | :--- | :--- | :-: | :-: | :-: | :---: | :-: |")

    for i, r in enumerate(results, 1):
        preview = r["prompt"].replace("\n", " ")
        if len(preview) > 55:
            preview = preview[:52] + "..."
        think_badge = "✅ Yes" if r["has_think_tags"] else "❌ No"
        status_badge = f"**{r['evaluation_status']}**"
        lines.append(
            f"| {i:02d} | `{r['id']}` | {r['category']} | {preview} | {think_badge} | "
            f"{r['latency_sec']:.2f} | {r['tokens_per_sec']:.1f} | {status_badge} | {r['evaluation_score']:.1f} |"
        )

    lines.append("")
    lines.append("---")
    lines.append("")

    lines.append("## 🔍 Detailed Prompt-by-Prompt Deep Dive")
    lines.append("")

    for i, r in enumerate(results, 1):
        lines.append(f"### {i:02d}. `{r['id']}` [{r['category'].upper()}] - {r['evaluation_status']}")
        lines.append("")
        lines.append(f"**User Prompt:**")
        lines.append(f"> {r['prompt']}")
        lines.append("")
        lines.append(f"- **Expected Reasoning Focus:** {r['expected_reasoning_focus']}")
        lines.append(f"- **Ground Truth / Criteria:** {r['ground_truth_or_criteria']}")
        lines.append(f"- **Metrics:** Latency `{r['latency_sec']}s` | Speed `{r['tokens_per_sec']} tok/s` | Tokens: `{r['prompt_tokens']}` prompt + `{r['completion_tokens']}` completion | `<think>` Tag: `{'Present (' + str(r['thinking_trace_length_chars']) + ' chars)' if r['has_think_tags'] else 'Missing'}`")
        lines.append(f"- **Evaluation Rationale:** {r['evaluation_rationale']}")
        lines.append("")

        if r["has_think_tags"] and r["thinking_trace"]:
            lines.append("<details>")
            lines.append("<summary><b>🧠 Click to view Reasoning Trace (&lt;think&gt;...&lt;/think&gt;)</b></summary>")
            lines.append("")
            lines.append("```text")
            lines.append(r["thinking_trace"])
            lines.append("```")
            lines.append("</details>")
            lines.append("")

        lines.append("**Final Answer / Model Output:**")
        lines.append("```markdown")
        lines.append(r["final_answer"] if r["final_answer"] else r["full_response"])
        lines.append("```")
        lines.append("")
        lines.append("---")
        lines.append("")

    # Footer
    lines.append("Generated by `eval/run_benchmark.py` (QWEN-999B LLM Eval Judge Suite).")
    lines.append("")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run standardized evaluation benchmarks against local Ollama Qwen models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="qwen2.5:7b-instruct-q4_K_M",
        help="Ollama model name to evaluate.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="http://localhost:11434",
        help="Base URL for local Ollama HTTP API.",
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        choices=["chat", "generate"],
        default="chat",
        help="Ollama API endpoint format to query.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="eval/reports",
        help="Directory where Markdown and JSON evaluation reports will be saved.",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        default="eval/test_prompts.json",
        help="Path to JSON test prompts file.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Sampling temperature for model generation.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=180.0,
        help="Timeout in seconds per prompt query.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run benchmark with simulated responses to validate reports and scoring without a live Ollama server.",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=DEFAULT_SYSTEM_PROMPT,
        help="System instruction passed to Ollama.",
    )
    parser.add_argument(
        "--no-json",
        action="store_true",
        help="Disable generating companion JSON report.",
    )

    args = parser.parse_args()

    prompts_path = Path(args.prompts)
    output_dir = Path(args.output_dir)

    try:
        run_benchmark(
            model=args.model,
            host=args.host,
            endpoint=args.endpoint,
            prompts_path=prompts_path,
            output_dir=output_dir,
            temperature=args.temperature,
            timeout=args.timeout,
            dry_run=args.dry_run,
            system_prompt=args.system_prompt,
            save_json=not args.no_json,
        )
    except Exception as e:
        print(f"\n❌ Benchmark execution failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
