"""Stage 2: compute metrics from grades.json + raw outputs.

Writes metrics.json with, per condition:
  - accuracy (auto and final) with a Wilson 95% CI, overall / by category / by difficulty
  - behavioral metrics computed from the raw text:
      drift        : output continues past the answer into self-generated
                     Q/A pairs or template text (the "doesn't know when to
                     stop" failure from FINDINGS.md, as a number)
      repetition   : fraction of non-empty lines that are duplicates
      refusal      : refusal phrasing in the answer segment
      empty        : empty completion
      verbosity    : mean/median output length in characters
and pairwise condition comparisons: discordant counts (A right & B wrong) with
an exact two-sided McNemar/binomial p-value. n=20, so p-values are coarse;
they are reported to keep the comparison honest, not to claim significance.
"""

import json
import math
import re
import statistics
from pathlib import Path

HERE = Path(__file__).parent
RESULTS_PATH = HERE.parent / "base_vs_instruct.json"
GRADES_PATH = HERE / "grades.json"
OUT_PATH = HERE / "metrics.json"

CONDITIONS = ["base_zero_shot", "base_few_shot", "instruct_zero_shot", "instruct_few_shot"]

# markers of the model generating a new task/template for itself
DRIFT_RE = re.compile(
    r"(^|\n)\s*(answer|sentiment|subject|review|text|question)\s*:"
    r"|(^|\n).*\?\s*($|\n)"
    r"|you are (a|an) (helpful |ai )?assistant",
    re.IGNORECASE,
)
REFUSAL_RE = re.compile(
    r"unable to|cannot provide|can't provide|i'm sorry|as an ai", re.IGNORECASE
)


def first_paragraph(text):
    for para in re.split(r"\n\s*\n", text):
        if para.strip():
            return para.strip()
    return ""


def drifted(text):
    """True if anything after the first answer line looks like self-generated
    tasks or template text (catches single-newline degeneration too)."""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) < 2:
        return False
    rest = "\n".join(lines[1:])
    return bool(DRIFT_RE.search(rest))


def repetition_score(text):
    lines = [ln.strip() for ln in text.splitlines() if len(ln.strip()) > 3]
    if len(lines) < 2:
        return 0.0
    return round(1 - len(set(lines)) / len(lines), 3)


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (round(center - half, 3), round(center + half, 3))


def mcnemar_exact_p(b, c):
    """Exact two-sided binomial test on the discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return round(min(1.0, 2 * tail), 4)


def accuracy_block(rows, cond, key="final_correct"):
    k = sum(r["conditions"][cond][key] for r in rows)
    n = len(rows)
    return {"correct": k, "n": n, "accuracy": round(k / n, 3), "wilson_95ci": wilson_ci(k, n)}


def main():
    grades = json.loads(GRADES_PATH.read_text(encoding="utf-8"))
    raw = {t["task_id"]: t for t in json.loads(RESULTS_PATH.read_text(encoding="utf-8"))}

    categories = sorted({r["category"] for r in grades})
    difficulties = ["easy", "medium", "hard"]

    per_condition = {}
    for cond in CONDITIONS:
        outputs = [raw[r["task_id"]][cond] for r in grades]
        lengths = [len(o) for o in outputs]
        per_condition[cond] = {
            "overall": accuracy_block(grades, cond),
            "overall_auto": accuracy_block(grades, cond, key="auto_correct"),
            "by_category": {
                cat: accuracy_block([r for r in grades if r["category"] == cat], cond)
                for cat in categories
            },
            "by_difficulty": {
                d: accuracy_block([r for r in grades if r["difficulty"] == d], cond)
                for d in difficulties
            },
            "behavior": {
                "drift_rate": round(sum(drifted(o) for o in outputs) / len(outputs), 3),
                "drift_tasks": [r["task_id"] for r, o in zip(grades, outputs) if drifted(o)],
                "mean_repetition": round(
                    statistics.mean(repetition_score(o) for o in outputs), 3),
                "refusals": [r["task_id"] for r, o in zip(grades, outputs)
                             if REFUSAL_RE.search(first_paragraph(o))],
                "empty_outputs": [r["task_id"] for r, o in zip(grades, outputs)
                                  if not o.strip()],
                "mean_chars": round(statistics.mean(lengths), 1),
                "median_chars": statistics.median(lengths),
            },
        }

    pairwise = {}
    for i, a in enumerate(CONDITIONS):
        for b in CONDITIONS[i + 1:]:
            a_only = [r["task_id"] for r in grades
                      if r["conditions"][a]["final_correct"]
                      and not r["conditions"][b]["final_correct"]]
            b_only = [r["task_id"] for r in grades
                      if r["conditions"][b]["final_correct"]
                      and not r["conditions"][a]["final_correct"]]
            pairwise[f"{a}__vs__{b}"] = {
                "only_first_correct": a_only,
                "only_second_correct": b_only,
                "mcnemar_exact_p": mcnemar_exact_p(len(a_only), len(b_only)),
            }

    needs_review = [
        {"task_id": r["task_id"], "condition": c, "extracted": cell["extracted"]}
        for r in grades for c, cell in r["conditions"].items()
        if "needs_review" in cell["flags"] and not cell["overridden"]
    ]
    overrides = [
        {"task_id": r["task_id"], "condition": c,
         "auto": cell["auto_correct"], "final": cell["final_correct"],
         "reason": cell.get("override_reason", "")}
        for r in grades for c, cell in r["conditions"].items() if cell["overridden"]
    ]

    metrics = {
        "n_tasks": len(grades),
        "conditions": per_condition,
        "pairwise": pairwise,
        "needs_review": needs_review,
        "overrides_applied": overrides,
    }
    OUT_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    ranked = sorted(CONDITIONS, key=lambda c: -per_condition[c]["overall"]["accuracy"])
    print(f"wrote {OUT_PATH.name}")
    for c in ranked:
        o = per_condition[c]["overall"]
        beh = per_condition[c]["behavior"]
        print(f"  {c:20s} acc {o['accuracy']:.2f} ({o['correct']}/{o['n']}) "
              f"CI {o['wilson_95ci']}  drift {beh['drift_rate']:.0%}  "
              f"mean {beh['mean_chars']:.0f} chars")


if __name__ == "__main__":
    main()
