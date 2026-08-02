"""Run the four-condition few-shot boundary/instruction ablation.

Conditions:
    A: current few-shot prompt
    B: delimiters around examples and target
    C: delimiters plus explicit instructions
    D: explicit instructions without delimiters

The default decoding matches the original experiment: greedy decoding with
200 new tokens. For a stability audit, use --do-sample, --n-samples 5, and
--temperature 0.3.

Example, from Google Colab after cloning this repository:

    pip install transformers accelerate bitsandbytes
    python few_shot_ablation.py \
        --data-dir . \
        --task-ids 4,5,8,15,1,2,6,9,13,16 \
        --output few_shot_ablation.json

Sampling version:

    python few_shot_ablation.py \
        --data-dir . \
        --task-ids 4,5,8,15,1,2,6,9,13,16 \
        --n-samples 5 \
        --do-sample \
        --temperature 0.3 \
        --output few_shot_ablation_sampling.json
"""

import argparse
import json
import random
from collections import OrderedDict
from pathlib import Path


DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_TASK_IDS = "4,5,8,15,1,2,6,9,13,16"
SYSTEM_PROMPT = "You are a helpful assistant."

EXPLICIT_INSTRUCTION = (
    "The items above are examples of the task format.\n"
    "Solve the target question independently.\n"
    "Do not copy names, numbers, or answers from the examples.\n"
    "Return only the answer to the target question."
)


def parse_task_ids(value):
    try:
        task_ids = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("task IDs must be comma-separated integers") from exc
    if not task_ids:
        raise argparse.ArgumentTypeError("provide at least one task ID")
    if len(set(task_ids)) != len(task_ids):
        raise argparse.ArgumentTypeError("task IDs must be unique")
    return task_ids


def build_user_prompts(task):
    """Return the four user-message variants for one frozen task."""
    examples = task["few_shot_prefix"].rstrip()
    target = task["prompt"].strip()
    target_body, output_cue = split_output_cue(target)

    original = task["few_shot_prefix"] + task["prompt"]

    delimited = make_delimited_prompt(examples, target_body, output_cue)
    delimited_with_instruction = make_delimited_prompt(
        examples,
        target_body,
        output_cue,
        instruction=EXPLICIT_INSTRUCTION,
    )

    instruction_only = (
        f"{examples}\n\n"
        f"{EXPLICIT_INSTRUCTION}\n\n"
        f"{target}"
    )

    return OrderedDict(
        [
            ("A_original", original),
            ("B_delimiters", delimited),
            ("C_delimiters_and_instruction", delimited_with_instruction),
            ("D_instruction_only", instruction_only),
        ]
    )


def split_output_cue(target):
    """Keep Answer:/Code: after the target closing marker when present."""
    for cue in ("Answer:", "Code:"):
        if target.endswith(cue):
            return target[: -len(cue)].rstrip(), cue
    return target, ""


def make_delimited_prompt(examples, target, output_cue, instruction=None):
    parts = ["<EXAMPLES>", examples, "</EXAMPLES>"]
    if instruction:
        parts.extend(["", instruction])
    parts.extend(["", "<TARGET>", target, "</TARGET>"])
    if output_cue:
        parts.extend(["", output_cue])
    return "\n".join(parts)


def set_seed(seed):
    random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def load_model(model_name):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model.eval()

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer


def format_chat_prompt(tokenizer, user_text):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def generate(model, tokenizer, formatted_prompt, max_new_tokens, do_sample, temperature):
    import torch

    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    generation_args = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if do_sample:
        generation_args["temperature"] = temperature

    with torch.inference_mode():
        output = model.generate(**inputs, **generation_args)

    prompt_length = inputs["input_ids"].shape[1]
    return tokenizer.decode(output[0][prompt_length:], skip_special_tokens=True)


def load_tasks(data_dir, task_ids):
    prompts_path = data_dir / "prompts.json"
    with prompts_path.open(encoding="utf-8") as handle:
        prompts = json.load(handle)["prompts"]

    by_id = {task["id"]: task for task in prompts}
    missing = [task_id for task_id in task_ids if task_id not in by_id]
    if missing:
        raise ValueError(f"task IDs not found in {prompts_path}: {missing}")
    return [by_id[task_id] for task_id in task_ids]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("few_shot_ablation.json"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--task-ids", type=parse_task_ids, default=parse_task_ids(DEFAULT_TASK_IDS))
    parser.add_argument("--n-samples", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--do-sample",
        action="store_true",
        help="use temperature sampling instead of the original greedy decoding",
    )
    args = parser.parse_args()

    if args.n_samples < 1:
        parser.error("--n-samples must be at least 1")
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be at least 1")
    if args.do_sample and args.temperature <= 0:
        parser.error("--temperature must be greater than 0 when sampling")

    tasks = load_tasks(args.data_dir, args.task_ids)
    model, tokenizer = load_model(args.model)
    conditions = [
        "A_original",
        "B_delimiters",
        "C_delimiters_and_instruction",
        "D_instruction_only",
    ]

    results = {
        "metadata": {
            "model": args.model,
            "system_prompt": SYSTEM_PROMPT,
            "task_ids": args.task_ids,
            "n_samples": args.n_samples,
            "max_new_tokens": args.max_new_tokens,
            "do_sample": args.do_sample,
            "temperature": args.temperature if args.do_sample else None,
            "seed": args.seed,
            "conditions": {
                "A_original": "current few-shot prompt",
                "B_delimiters": "examples and target marked, no extra instruction",
                "C_delimiters_and_instruction": "delimiters plus explicit instruction",
                "D_instruction_only": "explicit instruction without delimiters",
            },
        },
        "tasks": [],
    }

    for task_index, task in enumerate(tasks):
        print(f"Task {task['id']} ({task['category']})")
        task_result = {
            "task_id": task["id"],
            "category": task["category"],
            "difficulty": task["difficulty"],
            "gold_answer": task["gold_answer"],
            "conditions": {},
        }

        user_prompts = build_user_prompts(task)
        for condition_index, condition in enumerate(conditions):
            user_text = user_prompts[condition]
            formatted_prompt = format_chat_prompt(tokenizer, user_text)
            samples = []

            for sample_index in range(args.n_samples):
                sample_seed = (
                    args.seed
                    + task_index * 10000
                    + condition_index * 1000
                    + sample_index
                )
                set_seed(sample_seed)
                text = generate(
                    model,
                    tokenizer,
                    formatted_prompt,
                    args.max_new_tokens,
                    args.do_sample,
                    args.temperature,
                )
                samples.append({"seed": sample_seed, "output": text})

            task_result["conditions"][condition] = {
                "user_prompt": user_text,
                "samples": samples,
            }
            print(f"  {condition}: {len(samples)} sample(s)")

        results["tasks"].append(task_result)

    output_path = args.output
    if not output_path.is_absolute():
        output_path = args.data_dir / output_path
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote {output_path}")


if __name__ == "__main__":
    main()
