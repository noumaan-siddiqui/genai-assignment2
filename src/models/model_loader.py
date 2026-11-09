"""
Model loader, tokenizer setup, and PEFT LoRA configuration utilities.
"""

import os
import torch
from typing import Optional, Tuple
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, PeftModel, TaskType

DEFAULT_TARGET_MODEL = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_DRAFT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def get_tokenizer(model_name_or_path: str = DEFAULT_TARGET_MODEL) -> PreTrainedTokenizer:
    """Load and configure tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        padding_side="left"  # Recommended for causal decoder inference
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def load_base_model(
    model_name_or_path: str = DEFAULT_TARGET_MODEL,
    device_map: str = "auto",
    torch_dtype: Optional[torch.dtype] = None,
    load_in_4bit: bool = False
) -> PreTrainedModel:
    """Load pre-trained causal language model."""
    if torch_dtype is None:
        torch_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    quant_config = None
    if load_in_4bit:
        try:
            import bitsandbytes
        except ImportError:
            raise ImportError(
                "4-bit QLoRA requires bitsandbytes >= 0.46.1. Please run: pip install -U 'bitsandbytes>=0.46.1'"
            )
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True
        )

    model_kwargs = {
        "device_map": device_map,
        "quantization_config": quant_config,
        "trust_remote_code": True,
    }

    # Avoid torch_dtype deprecation warning in transformers >= 4.49
    try:
        import inspect
        if "dtype" in inspect.signature(AutoModelForCausalLM.from_pretrained).parameters:
            model_kwargs["dtype"] = torch_dtype
        else:
            model_kwargs["torch_dtype"] = torch_dtype
    except Exception:
        model_kwargs["dtype"] = torch_dtype

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        **model_kwargs
    )
    return model


def setup_lora_model(
    model: PreTrainedModel,
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: Optional[list] = None
) -> PeftModel:
    """Wrap base model with LoRA adapters for parameter-efficient fine-tuning."""
    if target_modules is None:
        # Standard projections for Qwen / modern Transformer architectures
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none"
    )

    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()
    return peft_model


def load_fine_tuned_model(
    adapter_path: str,
    base_model_name: str = DEFAULT_TARGET_MODEL,
    device_map: str = "auto",
    load_in_4bit: bool = False
) -> Tuple[PeftModel, PreTrainedTokenizer]:
    """Load fine-tuned model by loading base model and applying LoRA adapter."""
    tokenizer = get_tokenizer(base_model_name)
    base_model = load_base_model(base_model_name, device_map=device_map, load_in_4bit=load_in_4bit)
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    return model, tokenizer
