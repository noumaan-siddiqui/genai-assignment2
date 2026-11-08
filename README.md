# Generative AI - Assignment 2: Clinical Trial Matching & Advanced Transformer Inference

This repository contains the complete implementation for **Generative AI Assignment #2**, covering:
1. **Model Fine-Tuning**: Supervised adaptation (LoRA) for clinical eligibility matching and sentence-level evidence extraction on the **TrialGPT Criterion Annotations** dataset.
2. **Criterion-Level Verification**: Preserving inclusion/exclusion semantics, multi-class labels, and sentence evidence retrieval with failure analysis.
3. **KV-Cached Autoregressive Generation**: Cache-aware decoding flow with explicit key/value cache reuse, verified against uncached greedy decoding.
4. **Tiled Online-Softmax Attention**: Educational implementation of tiled/blocked attention with running maxima/normalizers (online softmax) compared against scaled dot-product attention (SDPA).
5. **Speculative Decoding**: Custom greedy speculative decoding utilizing a compatible draft model (e.g., Qwen2.5 / Llama family) with target verification and cache synchronization.
6. **Integrated Browser-Based Application**: Streamlit-based web interface for clinical eligibility screening, evidence inspection, report export, and interactive benchmark visualization.

---

## 📁 Repository Structure

```text
├── data/                    # Dataset split manifests (37/8/8) and metadata
├── notebooks/               # Kaggle/Colab runner notebooks for training & benchmarks
├── src/
│   ├── data/                # Data loading, 37/8/8 patient-grouped split, prompt formatting
│   ├── models/              # Model loaders, LoRA configs, tokenizer setup
│   ├── attention/           # SDPA reference and Tiled Online-Softmax attention
│   ├── inference/           # KV-cache generation and Speculative decoding implementations
│   ├── evaluation/          # Criterion classification & evidence sentence metrics (P/R/F1)
│   └── app/                 # Streamlit screening & inference benchmark web application
├── scripts/                 # Reproducible shell and python scripts
├── report/                  # IEEE-format research paper (LaTeX source)
├── requirements.txt         # Environment dependencies
└── README.md
```

---

## 🛠️ Installation & Setup

```bash
# Clone the repository
git clone https://github.com/noumaan-siddiqui/genai-assignment2.git
cd genai-assignment2

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## ⚡ Kaggle Integration Quick Guide

To connect and run this repository on Kaggle:

1. **Clone in Kaggle Notebook**:
   ```python
   !git clone https://github.com/noumaan-siddiqui/genai-assignment2.git
   %cd genai-assignment2
   !pip install -r requirements.txt
   ```
2. **Make `src` Importable**:
   ```python
   import sys
   sys.path.append("/kaggle/working/genai-assignment2")
   ```
3. **Model & Adapter Hosting**:
   Push trained LoRA adapters to Hugging Face Hub using `model.push_to_hub(...)` or download checkpoints to `/kaggle/working`.

---

## 📋 Evaluation Rubric Checklist

- [x] **Task 1**: Literature comparison & Decoder fine-tuning (TrialGPT dataset, 37/8/8 patient split with seed 42).
- [x] **Task 2**: Criterion-level eligibility & evidence verification (P/R/F1, confusion matrices, failure analysis).
- [x] **Task 3**: Custom KV-cached autoregressive generation vs uncached greedy decoding.
- [x] **Task 4**: Tiled attention with online softmax vs standard SDPA.
- [x] **Task 5**: Greedy speculative decoding with draft/target model pair.
- [x] **Task 6**: Browser-based interactive application (Streamlit).
- [x] **Report**: IEEE-format technical paper in LaTeX.
