"""Prompt-construction registry.

Each condition maps a benchmark task to the exact user text sent to the model.
`direct` is the frozen baseline; cot / few_shot_cot / self_consistency are
registered here in the next session without touching run.py.
"""


def direct_prompt(task):
    return task["prompt"]


CONDITIONS = {
    "direct": direct_prompt,
}

# CoT zero-shot instruction, frozen now so the wording cannot drift between sessions.
COT_INSTRUCTION = "Think step by step, then give the final answer."


def cot_prompt(task):
    """Zero-shot chain-of-thought (not yet scheduled; wording frozen here)."""
    body = task["prompt"].rstrip()
    for cue in ("Answer:", "Code:"):
        if body.endswith(cue):
            body = body[: -len(cue)].rstrip()
            break
    return f"{body}\n{COT_INSTRUCTION}"


def few_shot_prompt(task):
    return task["few_shot_prefix"] + task["prompt"]


# Registered but not part of the frozen baseline. self_consistency is a grading
# mode over N cot samples, not a separate prompt.
CONDITIONS["cot"] = cot_prompt
CONDITIONS["few_shot"] = few_shot_prompt
