"""Grade a run directory: generations.jsonl -> predictions.jsonl + metrics.json.

Usage:
    python -m harness.grade --run-dir runs/<run-id>

Reads config.json for the benchmark and condition. Every generation record gets
a prediction record — nothing is silently dropped. Records whose extraction
status is unparsed / ambiguous / malformed are additionally copied to
parse_failures.jsonl with the full completion text.

Extraction scope follows the old grade.py convention: few-shot (completion-
style) conditions grade the first paragraph; everything else grades the whole
completion with final-answer precedence.

Code tasks are graded by executing the generated function against the task's
frozen test cases (ported from eval/grade.py), not by string extraction.
"""

import argparse
import contextlib
import copy
import io
import json
import math
import re
from pathlib import Path

from .common import read_jsonl, utc_now, write_json
from .extract import EXTRACTION_VERSION, answers_match, extract_answer

FEWSHOT_PREFIX = "few_shot"


def scope_for_condition(condition):
    return "first_paragraph" if condition.startswith(FEWSHOT_PREFIX) else "full"


def grade_code(output, code_tests):
    detail = {"flags": [], "extracted": None}
    if not output.strip():
        detail["flags"].append("empty_output")
        return False, detail
    code = re.sub(r"```(?:python)?|```", "", output)
    namespace = {}
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            exec(code, namespace)
    except Exception as exc:
        detail["flags"].append(f"exec_error: {type(exc).__name__}")
        return False, detail
    fn = namespace.get(code_tests["fn"])
    if not callable(fn):
        detail["flags"].append("function_not_defined")
        return False, detail
    passed = 0
    for fn_args, expected in code_tests["tests"]:
        try:
            if fn(*copy.deepcopy(fn_args)) == expected:
                passed += 1
        except Exception:
            pass
    detail["extracted"] = f"{passed}/{len(code_tests['tests'])} tests passed"
    return passed == len(code_tests["tests"]), detail


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    denom = 1 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return [round(center - half, 3), round(center + half, 3)]


def grade_record(record, task, condition):
    answer_type = task["answer_type"]
    completion = record["completion_text"]
    base = {
        "problem_id": record["problem_id"],
        "model_id": record["model_id"],
        "condition": condition,
        "sample_index": record["sample_index"],
        "seed": record["seed"],
        "gold_answer": task["gold_answer"],
        "answer_type": answer_type,
        "extraction_version": EXTRACTION_VERSION,
        "n_output_tokens": record.get("n_output_tokens"),
    }
    if record.get("error"):
        base.update({"status": "unparsed", "prediction": None, "strategy": None,
                     "candidates": [], "unit": None,
                     "flags": ["generation_error"], "correct": False})
        return base
    if answer_type == "code":
        correct, detail = grade_code(completion, task["code_tests"])
        base.update({"status": "parsed" if detail["extracted"] else "unparsed",
                     "prediction": detail["extracted"], "strategy": "exec",
                     "candidates": [], "unit": None,
                     "flags": detail["flags"], "correct": correct})
        return base
    extraction = extract_answer(
        completion, answer_type,
        scope=scope_for_condition(condition),
        labels=task.get("labels"),
        gold=task["gold_answer"],
    )
    base.update(extraction)
    base["correct"] = answers_match(extraction, task["gold_answer"], answer_type)
    return base


def summarize(predictions):
    by_model = {}
    for pred in predictions:
        by_model.setdefault(pred["model_id"], []).append(pred)
    summary = {}
    for model_id, preds in by_model.items():
        n = len(preds)
        n_correct = sum(1 for p in preds if p["correct"])
        n_parsed = sum(1 for p in preds if p["status"] == "parsed")
        by_category = {}
        for p in preds:
            by_category.setdefault(p["category"], []).append(p)
        summary[model_id] = {
            "n": n,
            "n_correct": n_correct,
            "accuracy": round(n_correct / n, 3) if n else 0.0,
            "wilson_95ci": wilson_ci(n_correct, n),
            "n_parsed": n_parsed,
            "n_unparsed_or_failed": n - n_parsed,
            "mean_output_tokens": round(
                sum(p.get("n_output_tokens") or 0 for p in preds) / n, 1) if n else 0.0,
            "by_category": {
                cat: {
                    "n": len(rows),
                    "n_correct": sum(1 for r in rows if r["correct"]),
                }
                for cat, rows in sorted(by_category.items())
            },
        }
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    config = json.loads((args.run_dir / "config.json").read_text(encoding="utf-8"))
    with open(config["benchmark"], encoding="utf-8") as handle:
        benchmark = json.load(handle)
    tasks_by_id = {t["id"]: t for t in benchmark["prompts"]}
    condition = config["condition"]

    generations = read_jsonl(args.run_dir / "generations.jsonl")
    categories = {t["id"]: t["category"] for t in benchmark["prompts"]}

    predictions, failures = [], []
    for record in generations:
        pred = grade_record(record, tasks_by_id[record["problem_id"]], condition)
        pred["category"] = categories[record["problem_id"]]
        predictions.append(pred)
        if pred["status"] != "parsed":
            failures.append({**pred, "completion_text": record["completion_text"]})

    write_json(args.run_dir / "predictions.json", {
        "graded_utc": utc_now(),
        "extraction_version": EXTRACTION_VERSION,
        "n": len(predictions),
        "predictions": predictions,
    })
    write_json(args.run_dir / "parse_failures.json", {
        "n": len(failures),
        "failures": failures,
    })
    metrics = {
        "run_id": config["run_id"],
        "condition": condition,
        "extraction_version": EXTRACTION_VERSION,
        "models": summarize(predictions),
    }
    write_json(args.run_dir / "metrics.json", metrics)

    print(f"graded {len(predictions)} generations "
          f"({len(failures)} unparsed/ambiguous/malformed -> parse_failures.json)")
    for model_id, s in metrics["models"].items():
        print(f"  {model_id}: {s['n_correct']}/{s['n']} "
              f"acc {s['accuracy']:.3f} CI {s['wilson_95ci']} "
              f"({s['n_unparsed_or_failed']} not parsed)")


if __name__ == "__main__":
    main()
