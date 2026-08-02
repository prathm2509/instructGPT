"""Grade few_shot_ablation.json outputs per condition.

Mirrors the extraction rules in grade.py so the ablation results can be
compared with the original base-vs-instruct grades:

  - numeric answers: last number in the first paragraph for few-shot-style
    completions (fall back to whole output)
  - yes/no: first standalone yes/no in scope
  - label: first label mentioned; longer labels match first
  - name: gold name contained in the conclusion (first paragraph few-shot)
  - string: normalized containment of the gold string
  - code: exec the generated code and run the frozen test cases

The ablation stores the user prompt per cell; the output cue (`Answer:` /
`Code:`) and whether the prompt used a delimiter condition are used to scope
extraction the same way grade.py scopes base_few_shot cells.

Writes few_shot_ablation_metrics.json next to the input.
"""

import argparse
import contextlib
import copy
import io
import json
import re
from pathlib import Path

CONDITIONS = [
    "A_original",
    "B_delimiters",
    "C_delimiters_and_instruction",
    "D_instruction_only",
]

TASK_SPECS = {
    4: {"type": "number"},
    5: {"type": "name"},
    8: {"type": "name"},
    15: {"type": "string"},
    1: {"type": "number"},
    2: {"type": "number"},
    6: {"type": "yesno"},
    9: {"type": "label", "labels": ["negative", "positive", "neutral"]},
    13: {"type": "string"},
    16: {"type": "code", "fn": "add",
         "tests": [((2, 3), 5), ((-1, 4), 3), ((0, 0), 0)]},
}

NUM_RE = r"-?\$?\d[\d,]*\.?\d*"


def first_paragraph(text):
    for para in re.split(r"\n\s*\n", text):
        if para.strip():
            return para.strip()
    return ""


def nonempty_lines(text):
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


def clean_number(tok):
    tok = tok.replace("$", "").replace(",", "").rstrip(".")
    try:
        return float(tok)
    except ValueError:
        return None


def extract_number_completion(text):
    stripped = re.sub(r"^\s*answer\s*:\s*", "", text, flags=re.IGNORECASE)
    m = re.search(NUM_RE, stripped)
    return clean_number(m.group()) if m else None


def extract_label(scope_text, labels):
    low = scope_text.lower()
    hits = []
    masked = low
    for label in sorted(labels, key=len, reverse=True):
        idx = masked.find(label)
        if idx != -1:
            hits.append((idx, label))
            masked = masked.replace(label, "#" * len(label))
    return min(hits)[1] if hits else None


def extract_yesno(scope_text):
    m = re.search(r"\b(yes|no)\b", scope_text, re.IGNORECASE)
    return m.group(1).lower() if m else None


def normalize(s):
    return re.sub(r"\s+", " ", s.strip().strip('"').strip().rstrip(".")).lower()


def grade_code(output, spec):
    detail = {"extracted": None, "flags": []}
    if not output.strip():
        detail["flags"].append("empty_output")
        return False, detail
    code = re.sub(r"```(?:python)?|```", "", output)
    namespace = {}
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            exec(code, namespace)
    except Exception as e:
        detail["flags"].append(f"exec_error: {type(e).__name__}")
        return False, detail
    fn = namespace.get(spec["fn"])
    if not callable(fn):
        detail["flags"].append("function_not_defined")
        detail["flags"].append("needs_review")
        return False, detail
    passed = 0
    for args, expected in spec["tests"]:
        try:
            if fn(*copy.deepcopy(args)) == expected:
                passed += 1
        except Exception:
            pass
    detail["extracted"] = f"{passed}/{len(spec['tests'])} tests passed"
    return passed == len(spec["tests"]), detail


def is_completion_style(condition):
    return condition.startswith("A_")


def grade_text(output, gold, spec, condition):
    detail = {"extracted": None, "flags": []}
    if not output.strip():
        detail["flags"].append("empty_output")
        return False, detail

    completion_style = is_completion_style(condition)
    para = first_paragraph(output)
    lines = nonempty_lines(output)
    kind = spec["type"]

    if kind == "number":
        gold_num = clean_number(gold)
        scope = para if completion_style else output
        val = extract_number_completion(scope)
        if completion_style and val is None:
            val = extract_number_completion(output)
            detail["flags"].append("needs_review")
        detail["extracted"] = val
        return val is not None and val == gold_num, detail

    if kind == "yesno":
        scope = para if completion_style else output
        val = extract_yesno(scope)
        detail["extracted"] = val
        if val is None:
            detail["flags"].append("needs_review")
        return val == gold.lower(), detail

    if kind == "label":
        scope = para if completion_style else " ".join(lines[:2])
        val = extract_label(scope, spec["labels"])
        detail["extracted"] = val
        if val is None:
            detail["flags"].append("needs_review")
        return val == gold.lower(), detail

    if kind == "name":
        scope = para if completion_style else " ".join(lines[-3:])
        detail["extracted"] = scope[:80]
        return normalize(gold) in normalize(scope), detail

    if kind == "string":
        scope = para if completion_style else output
        detail["extracted"] = scope[:80]
        return normalize(gold) in normalize(scope), detail

    raise ValueError(f"unknown answer type {kind}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path, default=Path("few_shot_ablation.json"))
    args = parser.parse_args()

    with args.input.open(encoding="utf-8") as handle:
        data = json.load(handle)

    rows = []
    for task in data["tasks"]:
        tid = task["task_id"]
        spec = TASK_SPECS[tid]
        cell = {"task_id": tid, "category": task["category"], "gold": task["gold_answer"]}
        for condition in CONDITIONS:
            cond = task["conditions"][condition]
            output = cond["samples"][0]["output"]
            if spec["type"] == "code":
                correct, detail = grade_code(output, spec)
            else:
                correct, detail = grade_text(output, task["gold_answer"], spec, condition)
            cell[condition] = {"correct": correct, "extracted": detail["extracted"],
                               "flags": detail["flags"]}
        rows.append(cell)

    summary = {}
    for condition in CONDITIONS:
        correct = sum(1 for r in rows if r[condition]["correct"])
        summary[condition] = {
            "correct": correct,
            "n": len(rows),
            "accuracy": round(correct / len(rows), 3),
        }

    out = {
        "summary": summary,
        "cells": rows,
    }
    out_path = args.input.parent / "few_shot_ablation_metrics.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("summary:")
    for condition in CONDITIONS:
        s = summary[condition]
        print(f"  {condition:32s} {s['correct']}/{s['n']} acc {s['accuracy']:.3f}")

    print("\nper task:")
    for r in rows:
        marks = "".join("C" if r[c]["correct"] else "." for c in CONDITIONS)
        print(f"  task {r['task_id']:>2} ({r['category']:>12})  {marks}")


if __name__ == "__main__":
    main()
