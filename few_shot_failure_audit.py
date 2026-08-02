"""Instruct few-shot failure audit.

Extends base_vs_instruct.json with a sampling re-run of exactly the cases it
flagged as anomalous.

Problem found in Week 3's base-vs-instruct run: on a few tasks the *instruct*
model gets few-shot answers wrong more often than its own zero-shot answers,
sometimes producing values or names lifted from the few-shot examples rather
than the question being asked. The base model never does this — it just runs
on past the answer into repeated pattern text. So this looks like a failure
mode specific to instruction tuning, but the original run used greedy decoding
(do_sample=False), and greedy can lose a coin flip that sampling would win.

The experiment: for every task where instruct_few_shot is wrong but
instruct_zero_shot is right, rerun that task's few-shot prompt N times with
do_sample=True at low temperature (~0.3). Then decide whether the wrong answer
is a *stable* failure (same wrong answer every run — a real breakdown in how
the instruct model uses few-shot context) or an *unstable* one (different
wrong answers each run — greedy decoding got unlucky on this prompt).

No new models, no new harness, no new prompts — re-generation with a different
decoding setting on outputs already flagged. Fits in one Colab session and
feeds the report's failure-case section.

Usage:
    python few_shot_failure_audit.py --data-dir . --n-samples 5 --temperature 0.3

    # recompute verdicts from an existing few_shot_failure_audit.json
    # (no GPU, no model download — judging only):
    python few_shot_failure_audit.py --data-dir . --reclassify
"""

import argparse
import json
import re
from pathlib import Path

INSTRUCT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
SYSTEM_PROMPT = "You are a helpful assistant."


def normalize(text):
    return " ".join(text.lower().split())


def first_paragraph(text):
    for para in re.split(r"\n\s*\n", text):
        if para.strip():
            return para.strip()
    return ""


def judge(output, task, completion_style=False):
    """Return 'correct', 'wrong', or 'review' for one generation.

    Numeric gold answers (e.g. '10') are checked against the LAST number in
    scope, because these models write a reasoning block first and the final
    number is the answer. Using any number instead would misjudge outputs that
    mention the right value in passing and conclude something else (task 1's
    zero-shot answer did exactly this).

    completion_style=True scopes numeric judging to the FIRST paragraph, for
    few-shot completions that answer immediately and then drift into
    self-generated follow-up questions — without this, the last number comes
    from the drifted continuation, not the answer (task 4's audit run did
    exactly this). Falls back to the whole output if the first paragraph has
    no number (some few-shot answers are a reasoning block anyway).
    """
    category = task["category"]
    gold = task["gold_answer"]
    if category == "code":
        return "review"
    if re.fullmatch(r"-?\d+", gold.strip()):
        scope = first_paragraph(output) if completion_style else output
        numbers = re.findall(r"-?\d+", scope)
        if completion_style and not numbers:
            numbers = re.findall(r"-?\d+", output)
        return "correct" if numbers and numbers[-1] == gold.strip() else "wrong"
    pattern = r"\b" + re.escape(normalize(gold)) + r"\b"
    return "correct" if re.search(pattern, normalize(output)) else "wrong"


def answer_fingerprint(output, task, completion_style=False):
    """A stable string identifying what answer a generation produced."""
    gold = task["gold_answer"]
    if re.fullmatch(r"-?\d+", gold.strip()):
        scope = first_paragraph(output) if completion_style else output
        numbers = re.findall(r"-?\d+", scope)
        if completion_style and not numbers:
            numbers = re.findall(r"-?\d+", output)
        return numbers[-1] if numbers else "<no-number>"
    matches = list(re.finditer(r"answer\s*:", output, re.IGNORECASE))
    if matches:
        return normalize(output[matches[-1].end():])[:60]
    return normalize(output)[-60:]


def classify(samples, wrong_fingerprints):
    """'stable failure' vs 'unstable failure' vs greedy got unlucky."""
    correct = sum(1 for _, verdict in samples if verdict == "correct")
    if correct == len(samples):
        return "all correct (greedy was unlucky)"
    if correct > 0:
        return "mixed (greedy was unlucky; sampling recovers it)"
    if len(set(wrong_fingerprints)) <= 1:
        return "STABLE failure (same wrong answer every run)"
    return "UNSTABLE failure (different wrong answers each run)"


def summarize_entry(task, greedy_output, samples):
    """Judge a task's samples and build its audit record (no generation)."""
    judged = [(out, judge(out, task, completion_style=True)) for out in samples]
    unique_wrong = sorted(set(
        answer_fingerprint(out, task, completion_style=True)
        for out, verdict in judged if verdict == "wrong"
    ))
    return {
        "task_id": task["id"] if "id" in task else task["task_id"],
        "category": task["category"],
        "gold_answer": task["gold_answer"],
        "greedy_few_shot": greedy_output,
        "greedy_verdict": judge(greedy_output, task, completion_style=True),
        "n_correct": sum(1 for _, v in judged if v == "correct"),
        "unique_wrong_answers": unique_wrong,
        "verdict": classify(judged, unique_wrong),
        "samples": [{"output": out, "verdict": v} for out, v in judged],
    }


def print_summary(audit):
    for entry in audit:
        print(f"task {entry['task_id']:>2} ({entry['category']:>12}) gold={entry['gold_answer']!r:>12} "
              f"correct {entry['n_correct']}/{entry['n_samples']}  wrong answers seen: {entry['unique_wrong_answers']}  -> {entry['verdict']}")


def load_models():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(INSTRUCT_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        INSTRUCT_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
    )
    return model, tokenizer


def generate(model, tokenizer, formatted_prompt, temperature, max_new_tokens):
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
    )
    return tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def instruct_generate(model, tokenizer, text, temperature, max_new_tokens):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return generate(model, tokenizer, formatted, temperature, max_new_tokens)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=".", help="dir holding prompts.json and base_vs_instruct.json")
    parser.add_argument("--n-samples", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reclassify", action="store_true",
                        help="re-judge an existing few_shot_failure_audit.json in place (no GPU)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_path = data_dir / "few_shot_failure_audit.json"

    if args.reclassify:
        with open(out_path) as f:
            old = json.load(f)
        audit = []
        for e in old:
            entry = summarize_entry(e, e["greedy_few_shot"],
                                    [s["output"] for s in e["samples"]])
            entry["n_samples"] = len(e["samples"])
            entry["temperature"] = e.get("temperature")
            audit.append(entry)
        with open(out_path, "w") as f:
            json.dump(audit, f, indent=2)
        print(f"Re-judged {len(audit)} tasks in {out_path} (samples unchanged)\n")
        print_summary(audit)
        return

    with open(data_dir / "prompts.json") as f:
        harness = json.load(f)
    with open(data_dir / "base_vs_instruct.json") as f:
        results = json.load(f)

    prompts_by_id = {p["id"]: p for p in harness["prompts"]}

    candidates = []
    manual_review = []
    for r in results:
        few_verdict = judge(r["instruct_few_shot"], prompts_by_id[r["task_id"]], completion_style=True)
        zero_verdict = judge(r["instruct_zero_shot"], prompts_by_id[r["task_id"]])
        if few_verdict == "review":
            manual_review.append(r["task_id"])
        elif few_verdict == "wrong" and zero_verdict == "correct":
            candidates.append(r)

    print(f"Tasks where instruct_few_shot is wrong but instruct_zero_shot is right: {[c['task_id'] for c in candidates]}")
    if manual_review:
        print(f"Code tasks auto-skipped (answer not machine-checkable, judge by eye): {manual_review}")

    if not candidates:
        print("No candidates. Nothing to re-run.")
        return

    import torch
    torch.manual_seed(args.seed)
    model, tokenizer = load_models()

    audit = []
    for r in candidates:
        task = prompts_by_id[r["task_id"]]
        few_shot_prompt = task["few_shot_prefix"] + task["prompt"]

        samples = [
            instruct_generate(model, tokenizer, few_shot_prompt, args.temperature, args.max_new_tokens)
            for _ in range(args.n_samples)
        ]
        entry = summarize_entry(task, r["instruct_few_shot"], samples)
        entry["n_samples"] = args.n_samples
        entry["temperature"] = args.temperature
        audit.append(entry)

    with open(out_path, "w") as f:
        json.dump(audit, f, indent=2)

    print(f"\nWrote {out_path}\n")
    print_summary(audit)


if __name__ == "__main__":
    main()
