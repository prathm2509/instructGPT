"""Stage 1: auto-grade base_vs_instruct.json.

Reads ../base_vs_instruct.json (+ optional overrides.json), writes grades.json.

Extraction is condition-aware because the two prompt formats put the answer in
different places:
  - few-shot completions start with the answer, then drift into self-generated
    Q/A pairs -> grade the FIRST paragraph only
  - zero-shot instruct answers are chain-of-thought prose that states the
    answer at the END -> grade with final-answer rules (answer is X, **X**,
    last "= X", last number)
Code tasks are graded by executing the generated function against test cases,
not by string match. Cells where no answer could be extracted are flagged
needs_review so a human can settle them in overrides.json.
"""

import contextlib
import copy
import io
import json
import re
from pathlib import Path

HERE = Path(__file__).parent
RESULTS_PATH = HERE.parent / "base_vs_instruct.json"
OVERRIDES_PATH = HERE / "overrides.json"
OUT_PATH = HERE / "grades.json"

CONDITIONS = ["base_zero_shot", "base_few_shot", "instruct_zero_shot", "instruct_few_shot"]

# Frozen 20-task set. answer_type drives which extraction/grading rule runs.
TASK_SPECS = {
    1: {"type": "number"},
    2: {"type": "number"},
    3: {"type": "number"},
    4: {"type": "number"},
    5: {"type": "name"},
    6: {"type": "yesno"},
    7: {"type": "yesno"},
    8: {"type": "name"},
    9: {"type": "label", "labels": ["negative", "positive", "neutral"]},
    10: {"type": "label", "labels": ["negative", "positive", "neutral"]},
    11: {"type": "label", "labels": ["not spam", "spam"]},
    12: {"type": "label", "labels": ["biology", "physics", "history"]},
    13: {"type": "string"},
    14: {"type": "string"},
    15: {"type": "string"},
    16: {"type": "code", "fn": "add",
         "tests": [((2, 3), 5), ((-1, 4), 3), ((0, 0), 0)]},
    17: {"type": "code", "fn": "is_even",
         "tests": [((4,), True), ((7,), False), ((0,), True)]},
    18: {"type": "code", "fn": "max_of_two",
         "tests": [((3, 9), 9), ((10, 5), 10), ((-2, -3), -2)]},
    19: {"type": "code", "fn": "reverse_list",
         "tests": [(([1, 2, 3],), [3, 2, 1]), (([],), []), ((["a"],), ["a"])]},
    20: {"type": "number"},
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


def extract_number_final(text):
    """Final-answer number from chain-of-thought prose, in precedence order."""
    m = re.findall(r"answer is\s*(" + NUM_RE + ")", text, re.IGNORECASE)
    if m:
        return clean_number(m[-1])
    m = re.findall(r"\*\*[^*\d]*(" + NUM_RE + r")[^*]*\*\*", text)
    if m:
        return clean_number(m[-1])
    m = re.findall(r"=\s*(" + NUM_RE + ")", text)
    if m:
        return clean_number(m[-1])
    m = re.findall(NUM_RE, text)
    if m:
        return clean_number(m[-1])
    return None


def extract_number_completion(text):
    """First number in a completion-style answer (few-shot first paragraph)."""
    stripped = re.sub(r"^\s*answer\s*:\s*", "", text, flags=re.IGNORECASE)
    m = re.search(NUM_RE, stripped)
    return clean_number(m.group()) if m else None


def extract_label(scope_text, labels):
    """First label mentioned; longer labels matched first so 'not spam' wins over 'spam'."""
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
    """Execute generated code and check the target function against tests."""
    detail = {"extracted": None, "flags": []}
    if not output.strip():
        detail["flags"].append("empty_output")
        return False, detail
    code = re.sub(r"```(?:python)?|```", "", output)
    namespace = {}
    try:
        # model-generated toy functions from our own run; swallow example prints
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


def grade_text(output, gold, spec, condition):
    """Extraction + comparison for all non-code answer types."""
    detail = {"extracted": None, "flags": []}
    if not output.strip():
        detail["flags"].append("empty_output")
        return False, detail

    few_shot = condition.endswith("few_shot")
    para = first_paragraph(output)
    lines = nonempty_lines(output)
    kind = spec["type"]

    if kind == "number":
        gold_num = clean_number(gold)
        if few_shot:
            val = extract_number_completion(para)
            if val is None:  # model ignored the completion format; fall back
                val = extract_number_final(output)
                detail["flags"].append("needs_review")
        else:
            val = extract_number_final(output)
        detail["extracted"] = val
        return val is not None and val == gold_num, detail

    if kind == "yesno":
        scope = para if few_shot else output
        val = extract_yesno(scope)
        detail["extracted"] = val
        if val is None:
            detail["flags"].append("needs_review")
        return val == gold.lower(), detail

    if kind == "label":
        # zero-shot scope is the first 2 lines: labels echoed deep in drifted
        # text (e.g. a restated "(biology, physics, or history)") must not count
        scope = para if few_shot else " ".join(lines[:2])
        val = extract_label(scope, spec["labels"])
        detail["extracted"] = val
        if val is None:
            detail["flags"].append("needs_review")
        return val == gold.lower(), detail

    if kind == "name":
        # conclusion scope: mentions of the gold name mid-reasoning must not count
        scope = para if few_shot else " ".join(lines[-3:])
        detail["extracted"] = scope[:80]
        return normalize(gold) in normalize(scope), detail

    if kind == "string":
        scope = para if few_shot else output
        detail["extracted"] = scope[:80]
        return normalize(gold) in normalize(scope), detail

    raise ValueError(f"unknown answer type {kind}")


def main():
    tasks = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    overrides = {}
    if OVERRIDES_PATH.exists():
        overrides = json.loads(OVERRIDES_PATH.read_text(encoding="utf-8"))

    grades = []
    for task in tasks:
        tid = task["task_id"]
        spec = TASK_SPECS[tid]
        row = {"task_id": tid, "category": task["category"],
               "difficulty": task["difficulty"], "answer_type": spec["type"],
               "gold_answer": task["gold_answer"], "conditions": {}}
        for cond in CONDITIONS:
            output = task[cond]
            if spec["type"] == "code":
                auto, detail = grade_code(output, spec)
            else:
                auto, detail = grade_text(output, task["gold_answer"], spec, cond)
            cell = {"auto_correct": auto, "final_correct": auto,
                    "extracted": detail["extracted"], "flags": detail["flags"],
                    "overridden": False}
            ov = overrides.get(str(tid), {}).get(cond)
            if ov is not None:
                cell["final_correct"] = bool(ov["correct"])
                cell["overridden"] = True
                cell["override_reason"] = ov.get("reason", "")
            row["conditions"][cond] = cell
        grades.append(row)

    OUT_PATH.write_text(json.dumps(grades, indent=2), encoding="utf-8")

    n_review = sum(1 for r in grades for c in r["conditions"].values()
                   if "needs_review" in c["flags"])
    n_override = sum(1 for r in grades for c in r["conditions"].values()
                     if c["overridden"])
    print(f"graded {len(grades)} tasks x {len(CONDITIONS)} conditions -> {OUT_PATH.name}")
    print(f"needs_review: {n_review} cells, overrides applied: {n_override}")


if __name__ == "__main__":
    main()
