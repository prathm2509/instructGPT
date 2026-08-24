"""Model loading and generation.

Torch and transformers are imported lazily inside functions so the rest of the
harness (extraction, grading, validation) runs on machines without them.

Quantization is opt-in (--quantize): bitsandbytes 4-bit needs CUDA. On a
CPU-only machine use the default fp32 load.
"""

import time

from .common import sample_seed


def load_model(model_id, quantize=False):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    kwargs = {}
    if quantize:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        kwargs["device_map"] = "auto"
    else:
        kwargs["torch_dtype"] = torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer


def format_prompt(tokenizer, user_text, mode, system_prompt):
    """Raw models receive the prompt unchanged; chat models get the template."""
    if mode == "chat":
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return user_text


def set_torch_seed(seed):
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_one(model, tokenizer, prompt_text, *, do_sample, temperature,
                 max_new_tokens, seed):
    """One generation. Returns (completion, n_input_tokens, n_output_tokens, seconds)."""
    import torch

    set_torch_seed(seed)
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    gen_args = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if do_sample:
        gen_args["temperature"] = temperature

    start = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(**inputs, **gen_args)
    elapsed = time.perf_counter() - start

    prompt_len = inputs["input_ids"].shape[1]
    completion_ids = output[0][prompt_len:]
    completion = tokenizer.decode(completion_ids, skip_special_tokens=True)
    return completion, prompt_len, len(completion_ids), elapsed


def generate_sample(model, tokenizer, user_text, *, mode, system_prompt,
                    do_sample, temperature, max_new_tokens,
                    base_seed, benchmark_version, model_id, condition,
                    problem_id, sample_index):
    """Generate one fully-described sample; returns a generation record."""
    seed = sample_seed(base_seed, benchmark_version, model_id, condition,
                       problem_id, sample_index)
    prompt_text = format_prompt(tokenizer, user_text, mode, system_prompt)
    completion, n_in, n_out, elapsed = generate_one(
        model, tokenizer, prompt_text,
        do_sample=do_sample, temperature=temperature,
        max_new_tokens=max_new_tokens, seed=seed,
    )
    return {
        "benchmark_version": benchmark_version,
        "problem_id": problem_id,
        "model_id": model_id,
        "model_mode": mode,
        "condition": condition,
        "sample_index": sample_index,
        "seed": seed,
        "do_sample": do_sample,
        "temperature": temperature if do_sample else None,
        "max_new_tokens": max_new_tokens,
        "prompt_text": prompt_text,
        "completion_text": completion,
        "n_input_tokens": n_in,
        "n_output_tokens": n_out,
        "elapsed_seconds": round(elapsed, 3),
        "error": None,
    }
