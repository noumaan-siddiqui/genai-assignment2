# Research, Architecture Selection, and Compute Feasibility

This document addresses **Task 1** and **Rubric Criteria A1, A2, and A3** for *Generative AI Assignment #2*.

---

## 1. Literature Review & Critical Comparison (Rubric A1)

We investigated five foundational technical and scholarly sources spanning clinical eligibility reasoning, parameter-efficient adaptation, attention optimizations, and speculative inference:

1. **Jin et al. (2024)** - *TrialGPT: Matching Patients to Clinical Trials using Large Language Models*. EMNLP 2024 / arXiv:2404.09950.
   - **Relevance**: Establishes the clinical criterion annotation dataset and benchmark. Analyzes the challenge of long-context clinical reasoning, sentence-level grounding, and the risk of hallucinated eligibility.
   - **Critique / Connection**: While TrialGPT relies on proprietary GPT-4 zero-shot prompting, our implementation investigates supervised parameter-efficient adaptation on local open-weight Transformers, enforcing deterministic structured output and direct sentence grounding.

2. **Hu et al. (2022)** - *LoRA: Low-Rank Adaptation of Large Language Models*. ICLR 2022 / arXiv:2106.09685.
   - **Relevance**: Introduces rank decomposition matrices $W = W_0 + \Delta W$ where $\Delta W = B \cdot A$ with rank $r \ll d$.
   - **Critique / Connection**: LoRA reduces trainable parameters by $>99\%$ while freezing the pre-trained weights $W_0$. This eliminates catastrophic forgetting on clinical tokens and enables fine-tuning on a single 16 GB GPU (Kaggle T4) without full-precision optimizer overhead.

3. **Dao et al. (2022)** - *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*. NeurIPS 2022 / arXiv:2205.14135.
   - **Relevance**: Eliminates the $O(N^2)$ memory footprint of the attention matrix $QK^T$ by computing softmax and attention incrementally using SRAM tiling and online softmax normalizers (Milakov & Gimelshein, 2018).
   - **Critique / Connection**: Directly inspires **Task 4**, where we implement a reference PyTorch tiled online-softmax attention mechanism to verify numerical stability and memory savings against standard Scaled Dot-Product Attention (SDPA).

4. **Leviathan et al. (2023)** - *Fast Inference from Transformers via Speculative Decoding*. ICML 2023 / arXiv:2211.17192; & **Chen et al. (2023)** - *Accelerating Large Language Model Decoding with Speculative Sampling*.
   - **Relevance**: Proposes using a smaller draft model $M_q$ to generate $K$ candidate tokens cheaply, followed by a single parallel forward pass of target model $M_p$ to verify tokens, preserving exact target distribution.
   - **Critique / Connection**: Forms the algorithmic foundation for **Task 5**. In deterministic greedy decoding, speculative verification accepts candidate tokens where $\arg\max P_q = \arg\max P_p$, provably guaranteeing exact output equivalence while increasing tokens/second.

5. **Pope et al. (2023)** - *Efficiently Scaling Transformer Inference on TPU with Megatron and PagedAttention/KV-Caching*. arXiv:2211.05102; & **Kwon et al. (2023)** - *vLLM: Efficient Memory Management for Large Language Model Serving*. SOSP 2023.
   - **Relevance**: Explores the autoregressive memory-bandwidth bottleneck, showing that generative inference is memory-bound rather than compute-bound. Demonstrates that reusing cached Key/Value tensors avoids redundant $O(N^2)$ prompt recomputation.
   - **Critique / Connection**: Foundation for **Task 3**, where we build an explicit cache-aware autoregressive decoding loop and benchmark prefill vs. decode latency, memory footprint, and throughput.

---

## 2. Model Candidates & Feasibility Analysis (Rubric A2)

To satisfy **Task 1** and provide compatibility with **Task 5 (Speculative Decoding)**, we evaluated two primary open-weight model families:

| Dimension | Candidate A: Qwen2.5 Family (Selected) | Candidate B: Llama 3.2 Family (Alternative) |
| :--- | :--- | :--- |
| **Target Model** | `Qwen/Qwen2.5-1.5B-Instruct` (or `3B` / `7B`) | `meta-llama/Llama-3.2-3B-Instruct` |
| **Draft Model (Task 5)** | `Qwen/Qwen2.5-0.5B-Instruct` | `meta-llama/Llama-3.2-1B-Instruct` |
| **Vocabulary & Tokenizer** | 151,936 tokens (identical across 0.5B, 1.5B, 3B, 7B) | 128,256 tokens (shared across 1B & 3B) |
| **License** | **Apache 2.0** (Fully permissive, ungated) | Llama 3.2 Community License (Gated access) |
| **Architecture** | Causal Decoder, RoPE, GQA, RMSNorm, SwiGLU | Causal Decoder, RoPE, GQA, RMSNorm, SwiGLU |
| **Target Weight Footprint** | ~3.1 GB (FP16/BF16) or ~1.6 GB (4-bit NF4) | ~6.4 GB (FP16/BF16) or ~2.5 GB (4-bit NF4) |
| **Draft Weight Footprint** | ~1.0 GB (FP16) or ~0.4 GB (4-bit) | ~2.5 GB (FP16) or ~1.1 GB (4-bit) |
| **Target + Draft VRAM** | **~4.5 GB** (Fits effortlessly in 16 GB T4) | ~9.5 GB (Tighter fit with KV caches) |
| **Speculative Compatibility** | **100% Tokenizer Alignment** | 100% Tokenizer Alignment |
| **Kaggle Accessibility** | Instant download (No Hugging Face token hurdle) | Requires HF Token & Access Approval |

### Why `Qwen2.5-1.5B-Instruct` is Selected:
1. **Perfect Tokenizer Compatibility for Speculative Decoding**: Both `Qwen2.5-0.5B-Instruct` (draft) and `Qwen2.5-1.5B-Instruct` (target) share the exact same 152k BPE tokenizer, vocabulary embeddings, and chat template.
2. **Compute & Memory Budget on Kaggle T4**:
   - Free Kaggle environment provides an NVIDIA T4 GPU with **15,360 MB (15 GB)** VRAM.
   - Target (1.5B) + Draft (0.5B) combined in FP16 consume only **~4.5 GB VRAM**, leaving **>10 GB VRAM** for KV cache tensors, attention buffers, and training activations.
3. **Ungated Open Access**: Eliminates Kaggle execution failures caused by expired Hugging Face tokens or gated repository access checks.
4. **Context Length & Truncation**:
   - Maximum patient note length in the dataset is 235 words (~320 tokens); max criterion length is 155 words (~210 tokens).
   - Total prompt length is $< 650$ tokens.
   - Setting `max_seq_length = 1024` guarantees **0% truncation**, preserving all clinical evidence and patient sentences.

---

## 3. Supervised Adaptation Strategy (Rubric A3 & B3)

* **Adaptation Technique**: Low-Rank Adaptation (LoRA) on all linear attention projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`).
* **Hyperparameters**:
  * LoRA Rank ($r$): 16
  * LoRA Alpha ($\alpha$): 32
  * LoRA Dropout: 0.05
  * Learning Rate: $2 \times 10^{-4}$ (Cosine schedule with 10% warmup)
  * Batch Size: 4 (with Gradient Accumulation steps = 2, effective batch size = 8)
  * Epochs: 3 (sufficient convergence on 603 training examples)
  * Precision: BF16 / FP16 mixed precision
* **Training Objective**: Autoregressive causal language modeling loss $\mathcal{L}_{\text{CLM}}$ computed strictly over the structured target completion (assistant tokens).
