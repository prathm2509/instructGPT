"""Prompt-construction registry.

Each condition maps a benchmark task to the exact user text sent to the model.
`direct` is the frozen baseline. E1 conditions transform only the in-memory
few-shot prefix; the frozen benchmark file is never modified.
"""

import random
import re


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


# This seed defines the E1 corruption, independently of generation seeds.
# It is recorded in harness.run's config.json for reproducibility.
E1_LABEL_PERMUTATION_SEED = 20260827

# The benchmark uses several field names, but each demonstration has one
# answer-bearing line (or a multiline Code: field).
_ANSWER_MARKER = re.compile(
    r"^(?P<indent>\s*)(?P<marker>Answer:|Sentiment:|Classify:|Subject:|"
    r"Extract[^:]*:|Code:)(?P<value>.*)$", re.IGNORECASE
)


def _demonstration_blocks(prefix):
    return prefix.rstrip().split("\n\n")


def _answer_parts(block):
    lines = block.splitlines()
    for index, line in enumerate(lines):
        match = _ANSWER_MARKER.match(line)
        if match:
            return lines, index, match
    raise ValueError(f"few-shot demonstration has no recognized answer marker: {block!r}")


def _replace_answer(block, value):
    lines, index, match = _answer_parts(block)
    lines[index] = (f"{match.group('indent')}{match.group('marker')} {value}")
    # Code answers can span multiple lines. A replacement is deliberately a
    # single placeholder so it cannot accidentally retain the gold program.
    del lines[index + 1:]
    return "\n".join(lines)


def few_shot_format_only_prompt(task):
    blocks = _demonstration_blocks(task["few_shot_prefix"])
    return "\n\n".join(_replace_answer(block, "?") for block in blocks) + task["prompt"]


def few_shot_random_label_prompt(task):
    blocks = _demonstration_blocks(task["few_shot_prefix"])
    answers = []
    for block in blocks:
        lines, index, match = _answer_parts(block)
        first_line = match.group("value").lstrip()
        remainder = "\n".join(lines[index + 1:])
        answers.append("\n".join(part for part in (first_line, remainder) if part))
    shuffled = list(answers)
    random.Random(E1_LABEL_PERMUTATION_SEED + int(task["id"])).shuffle(shuffled)
    transformed = [_replace_answer(block, value) for block, value in zip(blocks, shuffled)]
    return "\n\n".join(transformed) + task["prompt"]


# Registered but not part of the frozen baseline. self_consistency is a grading
# mode over N cot samples, not a separate prompt.
CONDITIONS["cot"] = cot_prompt
CONDITIONS["few_shot"] = few_shot_prompt
CONDITIONS["few_shot_random_label"] = few_shot_random_label_prompt
CONDITIONS["few_shot_format_only"] = few_shot_format_only_prompt

CONDITION_METADATA = {
    "few_shot_random_label": {
        "e1_label_permutation_seed": E1_LABEL_PERMUTATION_SEED,
    },
}
