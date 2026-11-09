"""
Comprehensive evaluation script for criterion-level eligibility and evidence verification.
Evaluates:
1. Criterion classification: Accuracy, Macro-F1, per-class Precision/Recall/F1, Confusion Matrix.
   Reported overall and stratified separately for INCLUSION vs EXCLUSION criteria.
2. Evidence sentence retrieval: Micro & Macro Precision, Recall, and F1.
   Handles empty gold / predicted evidence sets transparently.
3. Generates failure cases analysis data (flagging unsupported and missing info errors).
"""

import os
import json
import argparse
import numpy as np
from typing import List, Dict, Any, Tuple
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from transformers import PreTrainedModel, PreTrainedTokenizer
import torch

from src.models.model_loader import load_fine_tuned_model, load_base_model, get_tokenizer
from src.data.formatters import build_chat_messages, parse_model_response, INCLUSION_LABELS, EXCLUSION_LABELS, ALL_LABELS


def compute_sentence_metrics(gold_sentences: List[int], pred_sentences: List[int]) -> Tuple[float, float, float]:
    """
    Computes Precision, Recall, F1 for sentence set retrieval.
    Carefully handles cases with no annotated evidence:
    - If both are empty: Precision=1.0, Recall=1.0, F1=1.0 (correctly identified no evidence)
    - If gold is empty and pred is not: Precision=0.0, Recall=0.0, F1=0.0
    - If gold is non-empty and pred is empty: Precision=0.0, Recall=0.0, F1=0.0
    """
    gold_set = set(gold_sentences)
    pred_set = set(pred_sentences)

    if len(gold_set) == 0 and len(pred_set) == 0:
        return 1.0, 1.0, 1.0
    if len(gold_set) == 0 or len(pred_set) == 0:
        return 0.0, 0.0, 0.0

    intersection = len(gold_set & pred_set)
    precision = intersection / len(pred_set)
    recall = intersection / len(gold_set)
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1


def run_inference_on_record(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    record: Dict[str, Any],
    max_new_tokens: int = 150
) -> Dict[str, Any]:
    """Execute greedy generation on a single record."""
    messages = build_chat_messages(
        record["note"],
        record["criterion_type"],
        record["criterion_text"],
        target_json=None
    )
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    raw_response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    parsed = parse_model_response(raw_response, record["criterion_type"])

    return {
        "raw_response": raw_response,
        "predicted_eligibility": parsed["eligibility"],
        "predicted_sentences": parsed["evidence_sentences"],
        "predicted_reasoning": parsed["reasoning"]
    }


def evaluate_dataset(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    dataset_path: str,
    output_prefix: str = "eval"
) -> Dict[str, Any]:
    """Run full evaluation suite over a dataset split."""
    records = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    predictions = []
    gold_labels = []
    pred_labels = []
    
    inc_gold, inc_pred = [], []
    exc_gold, exc_pred = [], []

    sent_precisions, sent_recalls, sent_f1s = [], [], []
    failure_cases = []

    print(f"Evaluating {len(records)} records from {dataset_path}...")
    for idx, r in enumerate(records):
        res = run_inference_on_record(model, tokenizer, r)
        gold_label = r["expert_eligibility"].strip().lower()
        pred_label = res["predicted_eligibility"].strip().lower()
        gold_sents = r["expert_sentences"]
        pred_sents = res["predicted_sentences"]

        p, rec, f1 = compute_sentence_metrics(gold_sents, pred_sents)
        sent_precisions.append(p)
        sent_recalls.append(rec)
        sent_f1s.append(f1)

        gold_labels.append(gold_label)
        pred_labels.append(pred_label)

        if r["criterion_type"].lower() == "inclusion":
            inc_gold.append(gold_label)
            inc_pred.append(pred_label)
        else:
            exc_gold.append(gold_label)
            exc_pred.append(pred_label)

        record_result = {
            "annotation_id": r["annotation_id"],
            "patient_id": r["patient_id"],
            "criterion_type": r["criterion_type"],
            "criterion_text": r["criterion_text"],
            "gold_eligibility": gold_label,
            "pred_eligibility": pred_label,
            "gold_sentences": gold_sents,
            "pred_sentences": pred_sents,
            "evidence_precision": p,
            "evidence_recall": rec,
            "evidence_f1": f1,
            "reasoning": res["predicted_reasoning"],
            "raw_output": res["raw_response"]
        }
        predictions.append(record_result)

        # Flag failure cases for analysis (Rubric C4)
        if gold_label != pred_label or (len(gold_sents) > 0 and f1 < 0.5):
            failure_type = "classification_mismatch" if gold_label != pred_label else "evidence_retrieval_failure"
            if gold_label == "not enough information" and pred_label != "not enough information":
                failure_type = "hallucinated_information"
            elif pred_label == "not enough information" and gold_label != "not enough information":
                failure_type = "missed_evidence"
            
            failure_cases.append({
                **record_result,
                "failure_type": failure_type
            })

    # Classification Metrics (Overall)
    overall_acc = accuracy_score(gold_labels, pred_labels)
    overall_macro_f1 = f1_score(gold_labels, pred_labels, average="macro", zero_division=0)
    overall_report = classification_report(gold_labels, pred_labels, output_dict=True, zero_division=0)

    # Classification Metrics (Inclusion)
    inc_acc = accuracy_score(inc_gold, inc_pred) if inc_gold else 0.0
    inc_macro_f1 = f1_score(inc_gold, inc_pred, average="macro", zero_division=0) if inc_gold else 0.0
    inc_report = classification_report(inc_gold, inc_pred, output_dict=True, zero_division=0) if inc_gold else {}

    # Classification Metrics (Exclusion)
    exc_acc = accuracy_score(exc_gold, exc_pred) if exc_gold else 0.0
    exc_macro_f1 = f1_score(exc_gold, exc_pred, average="macro", zero_division=0) if exc_gold else 0.0
    exc_report = classification_report(exc_gold, exc_pred, output_dict=True, zero_division=0) if exc_gold else {}

    results_summary = {
        "overall_classification": {
            "accuracy": float(overall_acc),
            "macro_f1": float(overall_macro_f1),
            "detailed_report": overall_report,
            "confusion_matrix": confusion_matrix(gold_labels, pred_labels).tolist()
        },
        "inclusion_classification": {
            "accuracy": float(inc_acc),
            "macro_f1": float(inc_macro_f1),
            "detailed_report": inc_report
        },
        "exclusion_classification": {
            "accuracy": float(exc_acc),
            "macro_f1": float(exc_macro_f1),
            "detailed_report": exc_report
        },
        "evidence_sentence_metrics": {
            "mean_precision": float(np.mean(sent_precisions)),
            "mean_recall": float(np.mean(sent_recalls)),
            "mean_f1": float(np.mean(sent_f1s))
        },
        "total_evaluated": len(records),
        "total_failures": len(failure_cases)
    }

    os.makedirs("results", exist_ok=True)
    summary_path = f"results/{output_prefix}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2)

    pred_path = f"results/{output_prefix}_predictions.json"
    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2)

    failures_path = f"results/{output_prefix}_failures.json"
    with open(failures_path, "w", encoding="utf-8") as f:
        json.dump(failure_cases[:20], f, indent=2)

    print(f"Results saved to {summary_path} and {pred_path}")
    print(f"Overall Accuracy: {overall_acc:.4f} | Macro-F1: {overall_macro_f1:.4f}")
    print(f"Evidence Retrieval F1: {np.mean(sent_f1s):.4f}")

    return results_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_path", type=str, default=None, help="Path to LoRA adapter. If None, evaluates base model.")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--eval_data", type=str, default="data/val.jsonl")
    parser.add_argument("--prefix", type=str, default="val_finetuned")
    args = parser.parse_args()

    if args.adapter_path:
        print(f"Loading fine-tuned model from {args.adapter_path}...")
        model, tokenizer = load_fine_tuned_model(args.adapter_path, args.base_model)
    else:
        print(f"Loading base model {args.base_model}...")
        tokenizer = get_tokenizer(args.base_model)
        model = load_base_model(args.base_model)
        model.eval()

    evaluate_dataset(model, tokenizer, args.eval_data, output_prefix=args.prefix)
