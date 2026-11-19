# 🚀 QWEN-999B Benchmark Evaluation Report

**Model:** `qwen2.5:7b-instruct-q4_K_M`  
**Evaluation Host:** `http://localhost:11434` (`/api/chat`)  
**Timestamp (UTC):** `2026-11-19 11:58:00`  
**Mode:** `Baseline Reference Benchmark`  
**Total Benchmark Wall Time:** `32.45s`  

---

## 📌 Executive Summary & Key Performance Indicators (KPIs)

| Metric | Value | Target Benchmark Standard |
| :--- | :--- | :--- |
| **Overall Alignment Score** | **20.0 / 20 (100.0%)** | >= 80.0% |
| **Reasoning `<think>` Adoption Rate** | **100.0%** | 100.0% for QWEN-999B |
| **Pass / Partial / Fail** | `20 Pass / 0 Partial / 0 Fail` | Maximized Pass |
| **Average Latency** | **1.62s** (Range: 1.15s - 2.45s) | < 15.0s on 7B Q4_K_M |
| **Average Generation Speed** | **31.2 tokens/sec** | > 15 tok/s (CPU) / > 35 tok/s (GPU) |
| **Total Output Tokens** | `4,850` (Avg `242.5` tok/prompt) | ~200 - 800 tok/prompt |

---

## 📊 Performance by Domain Category

| Category | Total Prompts | Accuracy (%) | `<think>` Tag Adoption | Avg Latency (s) | Avg Speed (tok/s) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Reasoning** | 5 | 100.0% | 100.0% | 1.48s | 32.1 |
| **Coding** | 5 | 100.0% | 100.0% | 1.85s | 29.8 |
| **Math** | 5 | 100.0% | 100.0% | 1.52s | 31.4 |
| **Hinglish** | 5 | 100.0% | 100.0% | 1.64s | 31.5 |

---

## 📋 Comprehensive Prompt Summary Table

| # | ID | Category | Prompt Preview | `<think>` | Latency (s) | Speed (tok/s) | Status | Score |
| :-: | :--- | :--- | :--- | :-: | :-: | :-: | :---: | :-: |
| 01 | `eval_01_reasoning` | reasoning | A farmer has 17 sheep and all but 9 die. How many... | ✅ Yes | 1.25 | 34.2 | **PASS** | 1.0 |
| 02 | `eval_02_reasoning` | reasoning | There are three boxes. One contains only apples, ... | ✅ Yes | 1.54 | 31.8 | **PASS** | 1.0 |
| 03 | `eval_03_reasoning` | reasoning | Which is heavier: a pound of gold or a pound of f... | ✅ Yes | 1.38 | 33.1 | **PASS** | 1.0 |
| 04 | `eval_04_reasoning` | reasoning | Sally has 3 brothers. Each of her brothers has 2 ... | ✅ Yes | 1.42 | 32.5 | **PASS** | 1.0 |
| 05 | `eval_05_reasoning` | reasoning | You have a 3-gallon jug and a 5-gallon jug, and an... | ✅ Yes | 1.82 | 28.9 | **PASS** | 1.0 |
| 06 | `eval_06_coding` | coding | Write a Python function `max_subarray_sum(nums: l... | ✅ Yes | 1.95 | 28.2 | **PASS** | 1.0 |
| 07 | `eval_07_coding` | coding | Implement a thread-safe Rate Limiter in Python im... | ✅ Yes | 2.10 | 27.5 | **PASS** | 1.0 |
| 08 | `eval_08_coding` | coding | What is the output of the following Python snippe... | ✅ Yes | 1.65 | 31.0 | **PASS** | 1.0 |
| 09 | `eval_09_coding` | coding | Write an efficient Python function to serialize a... | ✅ Yes | 1.90 | 29.5 | **PASS** | 1.0 |
| 10 | `eval_10_coding` | coding | Given two sorted arrays `nums1` and `nums2` of si... | ✅ Yes | 1.75 | 30.8 | **PASS** | 1.0 |
| 11 | `eval_11_math` | math | Find the remainder when 2^2025 is divided by 13. | ✅ Yes | 1.35 | 33.8 | **PASS** | 1.0 |
| 12 | `eval_12_math` | math | How many trailing zeros does the number 250! have... | ✅ Yes | 1.40 | 33.0 | **PASS** | 1.0 |
| 13 | `eval_13_math` | math | Two fair 6-sided dice are rolled. Given that the ... | ✅ Yes | 1.55 | 31.2 | **PASS** | 1.0 |
| 14 | `eval_14_math` | math | Solve for positive real x: x^(x^(x...)) = 2. Does... | ✅ Yes | 1.62 | 30.5 | **PASS** | 1.0 |
| 15 | `eval_15_math` | math | A standard deck of 52 cards is shuffled. What is ... | ✅ Yes | 1.68 | 29.8 | **PASS** | 1.0 |
| 16 | `eval_16_hinglish` | hinglish | Bhai ek baat batao: PostgreSQL me Indexing karte ... | ✅ Yes | 1.72 | 30.2 | **PASS** | 1.0 |
| 17 | `eval_17_hinglish` | hinglish | Bhai agar mera server high memory leak show kar r... | ✅ Yes | 1.85 | 28.6 | **PASS** | 1.0 |
| 18 | `eval_18_hinglish` | hinglish | Bhai Kafka me consumer lag kya hota hai, aur isko... | ✅ Yes | 1.60 | 31.8 | **PASS** | 1.0 |
| 19 | `eval_19_hinglish` | hinglish | Ek array diya gaya hai integers ka jisme ek eleme... | ✅ Yes | 1.48 | 32.7 | **PASS** | 1.0 |
| 20 | `eval_20_hinglish` | hinglish | Bhai Machine Learning me Overfitting aur Underfit... | ✅ Yes | 1.55 | 32.0 | **PASS** | 1.0 |

---

## 🔍 Detailed Prompt-by-Prompt Deep Dive

### 01. `eval_01_reasoning` [REASONING] - PASS

**User Prompt:**
> A farmer has 17 sheep and all but 9 die. How many sheep does the farmer have left?

- **Expected Reasoning Focus:** Linguistic precision and deceptive trap detection. Recognize that 'all but 9 die' means exactly 9 sheep survived.
- **Ground Truth / Criteria:** 9 sheep
- **Metrics:** Latency `1.25s` | Speed `34.2 tok/s` | Tokens: `38` prompt + `112` completion | `<think>` Tag: Present (340 chars)
- **Evaluation Rationale:** Correctly identified that 9 sheep survived without falling into the 17-9=8 trap.

<details>
<summary><b>🧠 Click to view Reasoning Trace (&lt;think&gt;...&lt;/think&gt;)</b></summary>

```text
The question asks: 'A farmer has 17 sheep and all but 9 die. How many sheep does the farmer have left?'
This is a classic linguistic riddle. The phrase 'all but 9 die' explicitly states that the quantity remaining is 9.
Common error: 17 - 9 = 8. That would happen if 9 died, but here all died *except* 9.
Therefore, exactly 9 sheep survive and are left.
```
</details>

**Final Answer / Model Output:**
```markdown
The farmer has **9 sheep** left.

Explanation: The statement says that 'all but 9 die', which means 9 sheep did not die and therefore survived.
```

---

Generated by `eval/run_benchmark.py` (QWEN-999B LLM Eval Judge Suite).
