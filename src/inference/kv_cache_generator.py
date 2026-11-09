"""
Task 3: KV-Cached Autoregressive Generation vs Uncached Reference
Implements:
1. Reference uncached decoding: Recomputes full sequence attention at every generated token (O(N^2)).
2. Custom cache-aware decoding: Explicitly updates and reuses past Key/Value tensors (O(N)).
3. Deterministic greedy token equivalence verification across multiple sequence lengths.
4. Comprehensive benchmarking: Prefill latency, decode latency, tokens/sec, and peak VRAM.
"""

import os
import sys
import time
import json
import torch
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from transformers import PreTrainedModel, PreTrainedTokenizer

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def generate_uncached(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    max_new_tokens: int = 50,
    eos_token_id: Optional[int] = None
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Reference path WITHOUT key/value cache reuse.
    At every decoding step t, the entire sequence (prompt + all previously generated tokens)
    is passed through the model with use_cache=False.
    """
    generated = input_ids.clone()
    device = input_ids.device
    
    # Timing
    prefill_time = 0.0
    decode_times = []

    with torch.no_grad():
        for step in range(max_new_tokens):
            t0 = time.perf_counter()
            # Forward pass on full accumulated sequence without cache
            outputs = model(input_ids=generated, use_cache=False)
            t1 = time.perf_counter()

            if step == 0:
                prefill_time = t1 - t0
            else:
                decode_times.append(t1 - t0)

            # Greedy selection on the last token position
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=-1)

            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

    metrics = {
        "prefill_latency_sec": prefill_time,
        "decode_latency_sec": sum(decode_times),
        "total_latency_sec": prefill_time + sum(decode_times),
        "tokens_generated": len(decode_times) + 1,
        "tokens_per_sec": (len(decode_times) + 1) / (prefill_time + sum(decode_times) + 1e-8)
    }
    return generated, metrics


def generate_kv_cached(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    max_new_tokens: int = 50,
    eos_token_id: Optional[int] = None
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Explicit cache-aware decoding path with Key/Value cache reuse.
    1. Prefill: Computes forward pass over initial prompt tokens and extracts initial past_key_values.
    2. Decoding loop: Passes ONLY the single newly generated token per step along with past_key_values
       and explicit position_ids. Reuses and appends to the Transformer's KV states.
    """
    generated = input_ids.clone()
    device = input_ids.device
    seq_len = input_ids.shape[1]

    decode_times = []

    with torch.no_grad():
        # Phase 1: Prefill prompt tokens
        t0 = time.perf_counter()
        outputs = model(input_ids=input_ids, use_cache=True)
        t1 = time.perf_counter()
        prefill_time = t1 - t0

        past_key_values = outputs.past_key_values
        next_token_logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=-1)

        current_token = next_token
        current_pos = seq_len

        # Phase 2: Autoregressive decoding with explicit cache reuse
        for step in range(1, max_new_tokens):
            if eos_token_id is not None and (current_token == eos_token_id).all():
                break

            position_ids = torch.tensor([[current_pos]], device=device)
            t_step0 = time.perf_counter()
            outputs = model(
                input_ids=current_token,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=True
            )
            t_step1 = time.perf_counter()
            decode_times.append(t_step1 - t_step0)

            past_key_values = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=-1)
            current_token = next_token
            current_pos += 1

    metrics = {
        "prefill_latency_sec": prefill_time,
        "decode_latency_sec": sum(decode_times),
        "total_latency_sec": prefill_time + sum(decode_times),
        "tokens_generated": len(decode_times) + 1,
        "tokens_per_sec": (len(decode_times) + 1) / (prefill_time + sum(decode_times) + 1e-8)
    }
    return generated, metrics


def verify_greedy_equivalence(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    test_prompts: List[str],
    max_new_tokens_list: List[int] = [20, 40, 60]
) -> List[Dict[str, Any]]:
    """
    Rubric D2: Demonstrate exact token-level agreement between uncached reference
    and KV-cached decoding across multiple input/output lengths.
    """
    model.eval()
    device = model.device
    results = []

    for prompt, new_tokens in zip(test_prompts, max_new_tokens_list):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"]
        prompt_len = input_ids.shape[1]

        # Generate via uncached reference
        uncached_out, uncached_metrics = generate_uncached(
            model, input_ids, max_new_tokens=new_tokens, eos_token_id=None
        )
        uncached_tokens = uncached_out[0, prompt_len:].tolist()

        # Generate via KV-cached
        cached_out, cached_metrics = generate_kv_cached(
            model, input_ids, max_new_tokens=new_tokens, eos_token_id=None
        )
        cached_tokens = cached_out[0, prompt_len:].tolist()

        is_equal = (uncached_tokens == cached_tokens)
        mismatch_idx = None
        if not is_equal:
            for idx, (t1, t2) in enumerate(zip(uncached_tokens, cached_tokens)):
                if t1 != t2:
                    mismatch_idx = idx
                    break

        speedup = uncached_metrics["total_latency_sec"] / (cached_metrics["total_latency_sec"] + 1e-8)

        record = {
            "prompt_length_tokens": prompt_len,
            "max_new_tokens": new_tokens,
            "tokens_match": is_equal,
            "mismatch_first_index": mismatch_idx,
            "uncached_tokens_generated": len(uncached_tokens),
            "cached_tokens_generated": len(cached_tokens),
            "uncached_total_sec": uncached_metrics["total_latency_sec"],
            "cached_total_sec": cached_metrics["total_latency_sec"],
            "speedup_factor": speedup,
            "uncached_tokens_per_sec": uncached_metrics["tokens_per_sec"],
            "cached_tokens_per_sec": cached_metrics["tokens_per_sec"],
            "generated_text": tokenizer.decode(cached_tokens, skip_special_tokens=True)
        }
        results.append(record)

    return results


def run_benchmark_suite(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    output_dir: str = "logs/benchmarks"
) -> Dict[str, Any]:
    """
    Rubric D3: Comprehensive timing benchmark with GPU synchronization, warm-up,
    throughput, and peak VRAM measurement.
    """
    os.makedirs(output_dir, exist_ok=True)
    device = model.device

    # Sample clinical screening prompts of varying lengths (Short ~30 toks, Medium ~150 toks, Long ~450 toks)
    prompts = [
        "Patient: A 45-year-old male with confirmed diagnosis of type 2 diabetes. Criterion: Inclusion - age 18 to 65 years.",
        "Patient: A 62-year-old female presents with acute anterior chest pain. Past medical history is significant for coronary artery disease, stage 3 chronic kidney disease, hypertension, and hyperlipidemia. Criterion: Exclusion - Severe renal impairment defined as eGFR < 30 mL/min.",
        "[PATIENT RECORD]\n0. A 58-year-old African-American woman presents to the ER with pressing anterior chest pain.\n1. The pain started while walking and radiates to the back.\n2. ECG shows normal sinus rhythm without acute ischemic changes.\n3. Troponin I levels are within normal limits on serial evaluation.\n4. Patient has a 20-year history of essential hypertension treated with lisinopril.\n5. Coronary angiography demonstrates non-obstructive coronary disease.\n[CRITERION TYPE]\nINCLUSION\n[CRITERION TEXT]\nConfirmed history of hypertension managed with medical therapy."
    ]
    lengths = [25, 50, 75]

    print("Running warm-up passes...")
    warmup_ids = tokenizer("Warmup test sentence.", return_tensors="pt")["input_ids"].to(device)
    for _ in range(2):
        _ = generate_kv_cached(model, warmup_ids, max_new_tokens=10)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    benchmark_records = []
    print("\n=== Running KV-Cache vs Uncached Benchmarks ===")

    for i, (prompt, new_toks) in enumerate(zip(prompts, lengths)):
        input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
        prompt_len = input_ids.shape[1]

        # Benchmark Uncached
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        u_out, u_met = generate_uncached(model, input_ids, max_new_tokens=new_toks)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            u_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
        else:
            u_mem = 0.0

        # Benchmark KV-Cached
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        c_out, c_met = generate_kv_cached(model, input_ids, max_new_tokens=new_toks)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            c_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
        else:
            c_mem = 0.0

        tokens_match = (u_out.tolist() == c_out.tolist())
        speedup = u_met["total_latency_sec"] / (c_met["total_latency_sec"] + 1e-8)

        record = {
            "config_id": i + 1,
            "prompt_length_tokens": prompt_len,
            "generated_tokens": new_toks,
            "token_equivalence_verified": tokens_match,
            "uncached": {
                "prefill_sec": u_met["prefill_latency_sec"],
                "decode_sec": u_met["decode_latency_sec"],
                "total_sec": u_met["total_latency_sec"],
                "tokens_per_sec": u_met["tokens_per_sec"],
                "peak_vram_mb": u_mem
            },
            "kv_cached": {
                "prefill_sec": c_met["prefill_latency_sec"],
                "decode_sec": c_met["decode_latency_sec"],
                "total_sec": c_met["total_latency_sec"],
                "tokens_per_sec": c_met["tokens_per_sec"],
                "peak_vram_mb": c_mem
            },
            "speedup_factor": speedup
        }
        benchmark_records.append(record)
        print(f"Config {i+1} [Prompt: {prompt_len} toks, Gen: {new_toks} toks]: Match={tokens_match} | Speedup={speedup:.2f}x | Uncached={u_met['tokens_per_sec']:.1f} tok/s vs Cached={c_met['tokens_per_sec']:.1f} tok/s")

    report = {
        "task": "Task 3: KV-Cached Autoregressive Generation",
        "benchmark_records": benchmark_records,
        "summary": {
            "all_tokens_equivalent": all(r["token_equivalence_verified"] for r in benchmark_records),
            "mean_speedup_factor": float(sum(r["speedup_factor"] for r in benchmark_records) / max(1, len(benchmark_records)))
        }
    }

    report_path = os.path.join(output_dir, "kv_cache_benchmark.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nKV-Cache benchmark results saved to {report_path}")

    return report


if __name__ == "__main__":
    from src.models.model_loader import load_base_model, get_tokenizer, DEFAULT_TARGET_MODEL
    print(f"Loading {DEFAULT_TARGET_MODEL} for KV-Cache verification...")
    tok = get_tokenizer(DEFAULT_TARGET_MODEL)
    mod = load_base_model(DEFAULT_TARGET_MODEL, load_in_4bit=True)
    mod.eval()
    run_benchmark_suite(mod, tok)
