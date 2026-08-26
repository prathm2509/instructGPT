"""Compare two graded runs: the direct baseline vs the CoT treatment.

Produces everything the CoT writeup needs and mints the comparison JSON:

    python compare_runs.py --baseline runs/<direct-run-id> --treatment runs/<cot-run-id>
                           [--out research/2026-08-26-cot-vs-direct-comparison.json]

Both run dirs must already be graded (predictions.json + metrics.json +
parse_failures.json are written by `python -m harness.grade --run-dir <dir>`).

Reports, per model:
  - overall accuracy + Wilson 95% CI for both runs and the delta
  - accuracy by category (both runs)
  - mean output tokens (the cost axis of CoT)
  - parse-failure counts from each run
  - flip table: tasks where direct is right and CoT is wrong (regressions)
    and the reverse (fixes), with task ids + categories

Markdown tables are printed for copy-paste into the report; the same numbers
are written to --out as JSON.
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from harness.grade import wilson_ci  # noqa: E402


def load_predictions(run_dir):
    path = Path(run_dir) / "predictions.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing - grade the run first: "
            f"python -m harness.grade --run-dir {run_dir}")
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("predictions") or []


def load_parse_failures(run_dir):
    path = Path(run_dir) / "parse_failures.json"
    if not path.exists():
        return 0
    return json.loads(path.read_text(encoding="utf-8")).get("n", 0)


def cell_table(preds, key_builder):
    table = {}
    for p in preds:
        table[key_builder(p)] = p
    return table


def compare_predictions(baseline, treatment):
    """Pure comparison of two prediction lists -> report dict.

    Keys are (model_id, problem_id); both runs must use the same graders.
    """
    base = cell_table(baseline, lambda p: (p["model_id"], p["problem_id"]))
    treat = cell_table(treatment, lambda p: (p["model_id"], p["problem_id"]))

    models = sorted({k[0] for k in base} | {k[0] for k in treat})
    report = {"models": {}}
    for model in models:
        b_rows = {pid: p for (m, pid), p in base.items() if m == model}
        t_rows = {pid: p for (m, pid), p in treat.items() if m == model}
        pids = sorted(set(b_rows) | set(t_rows))
        missing = [pid for pid in pids if pid not in b_rows or pid not in t_rows]
        if missing:
            raise ValueError(
                f"{model}: predictions missing for problem ids {missing} in one run")

        def summarize(rows):
            n = len(rows)
            k = sum(1 for p in rows.values() if p["correct"])
            parsed = sum(1 for p in rows.values() if p["status"] == "parsed")
            tokens = [p.get("n_output_tokens") or 0 for p in rows.values()]
            return {
                "n": n, "n_correct": k, "accuracy": round(k / n, 3),
                "wilson_95ci": wilson_ci(k, n),
                "n_parsed": parsed,
                "n_unparsed_or_failed": n - parsed,
                "mean_output_tokens": round(sum(tokens) / n, 1) if n else 0.0,
            }

        by_cat = {}
        fixes, regressions = [], []
        for pid in pids:
            b, t = b_rows[pid], t_rows[pid]
            cat = t.get("category") or b.get("category")
            by_cat.setdefault(cat, []).append((b, t))
            if b["correct"] and not t["correct"]:
                regressions.append({"problem_id": pid, "category": cat})
            elif not b["correct"] and t["correct"]:
                fixes.append({"problem_id": pid, "category": cat})

        b_sum, t_sum = summarize(b_rows), summarize(t_rows)
        report["models"][model] = {
            "baseline": b_sum,
            "treatment": t_sum,
            "delta_accuracy": round(t_sum["accuracy"] - b_sum["accuracy"], 3),
            "delta_percentage_points": round(
                (t_sum["accuracy"] - b_sum["accuracy"]) * 100, 1),
            "by_category": {
                cat: {
                    "n": len(rows),
                    "baseline_correct": sum(1 for b, _ in rows if b["correct"]),
                    "treatment_correct": sum(1 for _, t in rows if t["correct"]),
                }
                for cat, rows in sorted(by_cat.items())
            },
            "flips": {
                "regressions": regressions,   # direct right, CoT wrong
                "fixes": fixes,               # direct wrong, CoT right
                "n_regressions": len(regressions),
                "n_fixes": len(fixes),
            },
        }
    return report


def print_markdown(report, baseline_id, treatment_id):
    print(f"## CoT vs direct comparison")
    print(f"direct: `{baseline_id}` | cot: `{treatment_id}`")
    print(f"parse failures: direct {report['baseline_parse_failures']} | "
          f"cot {report['treatment_parse_failures']}")
    for model, m in report["models"].items():
        b, t = m["baseline"], m["treatment"]
        print(f"\n### {model}")
        print("| | direct | cot |")
        print("|---|---:|---:|")
        print(f"| accuracy | {b['accuracy']*100:.1f}% | {t['accuracy']*100:.1f}% "
              f"(delta {m['delta_accuracy']*100:+.1f} pp) |")
        print(f"| Wilson 95% CI | {b['wilson_95ci']} | {t['wilson_95ci']} |")
        print(f"| correct | {b['n_correct']}/{b['n']} | {t['n_correct']}/{t['n']} |")
        print(f"| mean output tokens | {b['mean_output_tokens']} | "
              f"{t['mean_output_tokens']} |")
        print("\nby category:")
        print("| category | n | direct ok | cot ok |")
        print("|---|---:|---:|---:|")
        for cat, row in m["by_category"].items():
            print(f"| {cat} | {row['n']} | {row['baseline_correct']} | "
                  f"{row['treatment_correct']} |")
        print("\nflips (direct right / CoT wrong): "
              f"{m['flips']['n_regressions']}")
        for f in m["flips"]["regressions"]:
            print(f"  - task {f['problem_id']} ({f['category']})")
        print("flips (direct wrong / CoT right): "
              f"{m['flips']['n_fixes']}")
        for f in m["flips"]["fixes"]:
            print(f"  - task {f['problem_id']} ({f['category']})")
    print("\nHand-read the chains for arithmetic, logic, and the 4 code tasks "
          "(direct was 0/4) - flag fluent-but-illogical chains.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True,
                        help="graded run dir of the direct baseline")
    parser.add_argument("--treatment", type=Path, required=True,
                        help="graded run dir of the CoT run")
    parser.add_argument("--out", type=Path, default=HERE / "research"
                        / "2026-08-26-cot-vs-direct-comparison.json")
    args = parser.parse_args(argv)

    baseline = load_predictions(args.baseline)
    treatment = load_predictions(args.treatment)
    if not baseline or not treatment:
        parser.error("one of the runs has no predictions")

    report = compare_predictions(baseline, treatment)
    report["baseline_run"] = args.baseline.name
    report["treatment_run"] = args.treatment.name
    report["baseline_parse_failures"] = load_parse_failures(args.baseline)
    report["treatment_parse_failures"] = load_parse_failures(args.treatment)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    print(f"wrote {args.out}")

    print_markdown(report, args.baseline.name, args.treatment.name)


if __name__ == "__main__":
    main()