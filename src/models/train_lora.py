"""
Fine-tuning script for clinical trial eligibility matching using QLoRA.
Optimized for Kaggle Dual NVIDIA T4 GPUs (30 GB combined VRAM).
- Uses 4-bit NormalFloat (NF4) quantization via bitsandbytes.
- Attaches 16-bit LoRA adapters.
- Uses gradient checkpointing and gradient accumulation to ensure 0% OOM risk.
- Supports multi-GPU distribution via device_map="auto".
"""

import os
import json
import argparse
import torch
from torch.utils.data import Dataset
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq
)
from src.models.model_loader import (
    DEFAULT_TARGET_MODEL,
    get_tokenizer,
    load_base_model,
    setup_lora_model
)
from src.data.formatters import build_chat_messages, format_training_target


class ClinicalTrialDataset(Dataset):
    def __init__(self, jsonl_path: str, tokenizer, max_length: int = 1024):
        self.samples = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.samples.append(json.loads(line))

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.features = [self._process_sample(s) for s in self.samples]

    def _process_sample(self, item):
        target_str = format_training_target(
            item["expert_eligibility"],
            item["expert_sentences"]
        )
        
        # Build prompt messages without assistant response
        prompt_messages = build_chat_messages(
            item["note"],
            item["criterion_type"],
            item["criterion_text"],
            target_json=None
        )
        # Build complete messages including assistant response
        full_messages = build_chat_messages(
            item["note"],
            item["criterion_type"],
            item["criterion_text"],
            target_json=target_str
        )

        prompt_text = self.tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True
        )
        full_text = self.tokenizer.apply_chat_template(
            full_messages,
            tokenize=False,
            add_generation_prompt=False
        )

        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        full_ids = self.tokenizer(full_text, add_special_tokens=False)["input_ids"]

        # Truncate if needed
        if len(full_ids) > self.max_length:
            full_ids = full_ids[:self.max_length]

        input_ids = full_ids
        attention_mask = [1] * len(input_ids)
        
        # Mask prompt tokens with -100 so loss is only computed on the target JSON output
        prompt_len = min(len(prompt_ids), len(full_ids))
        labels = [-100] * prompt_len + full_ids[prompt_len:]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx]


def train(args):
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    print(f"Loading tokenizer & base model: {args.model_name}...")
    tokenizer = get_tokenizer(args.model_name)
    base_model = load_base_model(
        args.model_name,
        load_in_4bit=args.use_qlora,
        device_map="auto"
    )

    print("Configuring LoRA adapter...")
    model = setup_lora_model(
        base_model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout
    )

    # Enable gradient checkpointing for memory safety on Kaggle T4s
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    print("Preparing datasets...")
    train_dataset = ClinicalTrialDataset(args.train_path, tokenizer, max_length=args.max_length)
    val_dataset = ClinicalTrialDataset(args.val_path, tokenizer, max_length=args.max_length)

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        gradient_checkpointing=args.gradient_checkpointing,
        report_to="none"
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        pad_to_multiple_of=8
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator
    )

    print("Starting training...")
    train_result = trainer.train()

    # Save final best adapter and tokenizer
    final_adapter_dir = os.path.join(args.output_dir, "final_adapter")
    model.save_pretrained(final_adapter_dir)
    tokenizer.save_pretrained(final_adapter_dir)
    print(f"Model saved to {final_adapter_dir}")

    # Save training log history for plotting curves
    log_history = trainer.state.log_history
    log_file = "logs/training_history.json"
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(log_history, f, indent=2)
    print(f"Saved training history to {log_file}")

    return train_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--train_path", type=str, default="data/train.jsonl")
    parser.add_argument("--val_path", type=str, default="data/val.jsonl")
    parser.add_argument("--output_dir", type=str, default="checkpoints/lora_qwen3b")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--use_qlora", action="store_true", default=True, help="Use 4-bit QLoRA")
    parser.add_argument("--no_qlora", action="store_false", dest="use_qlora", help="Disable QLoRA and use FP16/BF16")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    args = parser.parse_args()

    train(args)
