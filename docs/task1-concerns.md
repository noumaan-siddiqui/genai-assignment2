Using Kaggle's free GPUs (typically a single 16GB NVIDIA P100 or dual 15GB NVIDIA T4s) drastically changes your strategy for Phase 2. You have a strict VRAM budget, which means you cannot brute-force an 8B parameter model in full precision.

Here is exactly how to navigate Kaggle's hardware constraints for this assignment.

1. Model Selection (The Draft/Target Pair)
Because Task 5 requires speculative decoding with a smaller draft model and a larger target model, you must pick a model family that has both, shares a tokenizer, and fits within 16GB VRAM.   
GenAI_Assignment#2.pdf

    Llama 3.2 Family (Highly Recommended): Use Llama-3.2-3B as your target model and Llama-3.2-1B as your draft model. The 3B model is small enough to train comfortably on a Kaggle T4, while the 1B is perfectly sized for generating draft tokens.

    Qwen 2.5 Family: Use Qwen2.5-3B as the target and Qwen2.5-0.5B as the draft. Qwen models are incredibly memory-efficient and handle structured extraction tasks very well.

    Avoid 8B+ models: Even if you can barely squeeze an 8B model into Kaggle's memory for inference, you will likely hit an Out of Memory (OOM) error the moment you try to fine-tune it or run your custom KV cache loops.   
    GenAI_Assignment#2.pdf+ 1

2. Fine-Tuning Method: You Must Use QLoRA
Standard LoRA is not enough for Kaggle GPUs if you use anything larger than a 1B model. You need to use QLoRA (Quantized LoRA).

    How it works: You will use the bitsandbytes library to load your chosen base model (like Llama-3.2-3B) in 4-bit precision. This shrinks the model's memory footprint by about 75%. You then attach your LoRA adapters (which remain in 16-bit precision) on top.

    Memory Savers: In your Hugging Face TrainingArguments, you must enable gradient_checkpointing=True. Set your per_device_train_batch_size to 1 or 2, and use gradient_accumulation_steps (e.g., 4 or 8) to simulate a larger batch size without crashing the GPU.

3. Preprocessing: The Truncation Trap
The assignment explicitly states: "Truncation must be documented, especially when it could remove relevant patient evidence". This is a massive trap on Kaggle.   
GenAI_Assignment#2.pdf

    Because you are memory-constrained, you might be tempted to set your tokenizer's max_length to something small like 512 or 1024 tokens.

    Patient medical notes can be very long. If the expert evidence sentence (e.g., sentence ID 42) is located at the very end of the patient's chart, and your preprocessing function truncates the text before it reaches that sentence, your model has literally no way to learn the correct answer.

    Your Fix: In your preprocessing script, write a check to see if the target [Sentence ID] falls outside your maximum token length. If it does, you must document this in your IEEE report as a limitation, explaining exactly how many training examples were affected by your Kaggle hardware limits.   
    GenAI_Assignment#2.pdf

