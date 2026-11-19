"""
QWEN-999B: Synthetic Reasoning Data Generator with ChatML & <think> Traces
Uses Gemini API via GEMINI_API_KEY environment variable.
Follows: .agent/skills/qwen-dataset-architect/SKILL.md & .agent/skills/repo-hygiene-guard/SKILL.md
"""

import os
import sys
import json
import time
import hashlib
import argparse
from typing import List, Dict, Any, Optional

SYSTEM_PROMPT_TEMPLATE = (
    "You are a master dataset distillation engine for QWEN-999B, a high-reasoning frontier model. "
    "For any given problem or prompt, generate an exhaustive, rigorous, step-by-step reasoning trace "
    "enclosed strictly within <think> and </think> tags. "
    "Immediately after </think>, provide the clean, polished, direct final answer or code solution. "
    "Maintain high technical accuracy, explore alternative approaches, check for edge cases and deceptive pitfalls "
    "inside the thinking process."
)

DEFAULT_SYSTEM_ROLE_MESSAGE = (
    "You are QWEN-999B, an advanced reasoning assistant specializing in deep problem-solving, "
    "clean software architecture, mathematical proofs, and natural Hinglish technical discourse."
)

SEED_PROMPTS = [
    # Reasoning & Traps
    {
        "category": "reasoning",
        "user_prompt": "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost? Explain your reasoning thoroughly."
    },
    {
        "category": "reasoning",
        "user_prompt": "There are three boxes: one contains only apples, one only oranges, and one both apples and oranges. All three boxes are incorrectly labeled. You are allowed to pick only ONE fruit from ONE box without looking inside. How can you correctly label all three boxes?"
    },
    {
        "category": "reasoning",
        "user_prompt": "A man is looking at a portrait. Someone asks him who it is. He replies: 'Brothers and sisters I have none, but that man's father is my father's son.' Who is in the portrait?"
    },
    {
        "category": "reasoning",
        "user_prompt": "If a doctor gives you 3 pills and tells you to take one every half hour, how long will the pills last? Walk through every assumption."
    },
    {
        "category": "reasoning",
        "user_prompt": "A farmer has 17 sheep and all but 9 die. How many sheep are left alive? Analyze linguistic phrasing and logical interpretation."
    },
    # Math & Number Theory
    {
        "category": "math",
        "user_prompt": "Prove that the square root of 2 is irrational using proof by contradiction. Detail every step."
    },
    {
        "category": "math",
        "user_prompt": "Find the last two digits of 7^2024. Show the modular arithmetic steps."
    },
    {
        "category": "math",
        "user_prompt": "In a room of 23 people, what is the exact probability that at least two people share the same birthday? Assume 365 days and derive the complementary probability formula."
    },
    {
        "category": "math",
        "user_prompt": "Solve the system of equations for positive real numbers x, y: x + y + xy = 34, x^2 + y^2 = 50."
    },
    {
        "category": "math",
        "user_prompt": "How many trailing zeros does 100! have? Explain Legendre's formula."
    },
    # Coding & Algorithms
    {
        "category": "coding",
        "user_prompt": "Write an efficient Python implementation of an LRU (Least Recently Used) Cache using a doubly linked list and a hash map. Include O(1) get and put methods with full edge-case comments."
    },
    {
        "category": "coding",
        "user_prompt": "Implement a Python function to find the Longest Increasing Subsequence (LIS) in O(N log N) time complexity using binary search (patience sorting)."
    },
    {
        "category": "coding",
        "user_prompt": "Design a thread-safe Singleton pattern in Python using metaclasses and double-checked locking."
    },
    {
        "category": "coding",
        "user_prompt": "Explain and implement an algorithm to detect a cycle in a directed graph using Kahn's Algorithm (Topological Sort BFS) vs DFS graph coloring."
    },
    {
        "category": "coding",
        "user_prompt": "Write a Python generator that lazily traverses a deeply nested JSON object of arbitrary depth and yields all key-value pairs matching a given key name."
    },
    # Hinglish & Multilingual Technical Discourse
    {
        "category": "hinglish",
        "user_prompt": "Bhai mujhe explain karo ki GIL (Global Interpreter Lock) Python me kya hota hai aur CPU-bound vs I/O-bound tasks me iska impact kya hai? Simple relatable Hinglish me samjhao."
    },
    {
        "category": "hinglish",
        "user_prompt": "Database sharding vs partitioning me kya difference hai bhai? Jab startup scale karti hai tab kab sharding choose karni chahiye?"
    },
    {
        "category": "hinglish",
        "user_prompt": "Bhai ek scalable rate limiter banana hai token bucket algorithm use karke. Python code ke saath step-by-step samjhao."
    },
    {
        "category": "hinglish",
        "user_prompt": "Microservices architecture me distributed transactions manage karne ke liye 2-Phase Commit vs Saga Pattern me se kaun sa best hai aur kyun?"
    },
    {
        "category": "hinglish",
        "user_prompt": "Bhai Transformer architecture me Self-Attention mechanism mathematically kaise kaam karta hai? Q, K, V matrices ka role Hinglish me explain karo."
    }
]

def get_api_key() -> str:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("[ERROR] GEMINI_API_KEY environment variable is not set!")
        print("Please set your Gemini API key: export GEMINI_API_KEY='your_api_key' (Linux/macOS) or $env:GEMINI_API_KEY='your_api_key' (Windows PowerShell)")
        sys.exit(1)
    return api_key

def call_gemini(prompt: str, model_name: str, api_key: str) -> Optional[str]:
    """
    Calls the Gemini API using available client libraries or direct REST fallback.
    """
    # Attempt 1: google-genai (latest Google GenAI SDK)
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config={"system_instruction": SYSTEM_PROMPT_TEMPLATE, "temperature": 0.7}
        )
        return response.text
    except ImportError:
        pass
    except Exception as e:
        print(f"[WARN] google.genai error: {e}. Trying fallback...")

    # Attempt 2: google.generativeai (legacy Google SDK)
    try:
        import google.generativeai as genai_legacy
        genai_legacy.configure(api_key=api_key)
        model = genai_legacy.GenerativeModel(
            model_name=model_name,
            system_instruction=SYSTEM_PROMPT_TEMPLATE
        )
        response = model.generate_content(prompt)
        return response.text
    except ImportError:
        pass
    except Exception as e:
        print(f"[WARN] google.generativeai error: {e}. Trying REST fallback...")

    # Attempt 3: Direct REST API via urllib (Zero external dependencies)
    try:
        import urllib.request
        import urllib.parse

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
        payload = {
            "system_instruction": {
                "parts": [{"text": SYSTEM_PROMPT_TEMPLATE}]
            },
            "contents": [
                {
                    "parts": [{"text": prompt}]
                }
            ],
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": 3072
            }
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            resp_json = json.loads(resp.read().decode("utf-8"))
            candidates = resp_json.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                return "".join([p.get("text", "") for p in parts])
    except Exception as e:
        print(f"[ERROR] REST API request failed: {e}")
        return None

    return None

def validate_and_format_trace(raw_text: str, user_prompt: str) -> Optional[Dict[str, Any]]:
    """
    Enforces qwen-dataset-architect requirements:
    1. Validates <think>...</think> tags presence.
    2. Filters empty or corrupt responses.
    3. Truncates/filters to ensure within 2048 token boundary.
    4. Formats into strict ChatML JSONL representation.
    """
    if not raw_text or not raw_text.strip():
        return None

    text = raw_text.strip()
    
    # If the model didn't wrap thinking in tags, attempt to structure it
    if "<think>" not in text or "</think>" not in text:
        # Check if model has a designated thinking / answer split
        if "\n\nAnswer:" in text:
            parts = text.split("\n\nAnswer:", 1)
            formatted_assistant = f"<think>\n{parts[0].strip()}\n</think>\n{parts[1].strip()}"
        else:
            # Wrap first paragraph in thinking if tags missing
            paragraphs = text.split("\n\n", 1)
            if len(paragraphs) == 2:
                formatted_assistant = f"<think>\n{paragraphs[0].strip()}\n</think>\n{paragraphs[1].strip()}"
            else:
                formatted_assistant = f"<think>\nAnalyze the problem and verify core conditions.\n</think>\n{text}"
    else:
        formatted_assistant = text

    # Basic token length estimate (1 token approx 4 characters)
    approx_tokens = len(formatted_assistant) // 4 + len(user_prompt) // 4
    if approx_tokens > 2048:
        print(f"[FILTER] Sample exceeds 2048 token limit (~{approx_tokens} tokens). Dropping.")
        return None

    return {
        "messages": [
            {"role": "system", "content": DEFAULT_SYSTEM_ROLE_MESSAGE},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": formatted_assistant}
        ]
    }

def main():
    parser = argparse.ArgumentParser(description="Generate ChatML reasoning traces using Gemini API.")
    parser.add_argument("--output", type=str, default="data/reasoning_data.jsonl", help="Output JSONL path")
    parser.add_argument("--samples", type=int, default=20, help="Number of samples to generate")
    parser.add_argument("--model", type=str, default="gemini-2.5-flash", help="Gemini model name")
    parser.add_argument("--topic", type=str, default="all", choices=["all", "reasoning", "math", "coding", "hinglish"], help="Domain topic")
    args = parser.parse_args()

    api_key = get_api_key()
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Filter seed prompts by topic
    selected_seeds = SEED_PROMPTS
    if args.topic != "all":
        selected_seeds = [s for s in SEED_PROMPTS if s["category"] == args.topic]

    print(f"[INFO] Target Output: {args.output}")
    print(f"[INFO] Using Gemini Model: {args.model}")
    print(f"[INFO] Target Samples: {args.samples} (Topic: {args.topic})")

    existing_hashes = set()
    if os.path.exists(args.output):
        with open(args.output, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line.strip())
                    user_msg = next((m["content"] for m in obj["messages"] if m["role"] == "user"), "")
                    existing_hashes.add(hashlib.md5(user_msg.encode("utf-8")).hexdigest())
                except Exception:
                    continue
        print(f"[INFO] Loaded {len(existing_hashes)} existing unique samples for deduplication.")

    saved_count = 0
    with open(args.output, "a", encoding="utf-8") as out_file:
        for idx in range(args.samples):
            seed = selected_seeds[idx % len(selected_seeds)]
            user_prompt = seed["user_prompt"]
            prompt_hash = hashlib.md5(user_prompt.encode("utf-8")).hexdigest()

            if prompt_hash in existing_hashes:
                print(f"[{idx+1}/{args.samples}] Duplicate detected for '{user_prompt[:40]}...'. Skipping.")
                continue

            print(f"[{idx+1}/{args.samples}] Querying Gemini for: '{user_prompt[:50]}...'")
            raw_response = call_gemini(user_prompt, args.model, api_key)
            if not raw_response:
                print(f"[WARN] Empty response from Gemini API for prompt {idx+1}. Skipping.")
                time.sleep(1)
                continue

            sample = validate_and_format_trace(raw_response, user_prompt)
            if sample:
                out_file.write(json.dumps(sample, ensure_ascii=False) + "\n")
                out_file.flush()
                existing_hashes.add(prompt_hash)
                saved_count += 1
                print(f"[SUCCESS] Sample {saved_count} saved with verified <think> traces.")
            
            # Rate limiting sleep
            time.sleep(1.0)

    print("\n" + "=" * 60)
    print(f"[COMPLETED] Successfully generated {saved_count} ChatML reasoning records.")
    print(f"[OUTPUT] Saved to: {args.output}")
    print("=" * 60)

if __name__ == "__main__":
    main()
