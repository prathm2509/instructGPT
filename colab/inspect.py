"""Human-readable checks and summaries for a run directory.

Purely offline: reads config.json + generations.jsonl (+ predictions.json when
graded). Used by the Colab runner for the by-hand smoke check and the end-of-run
summary, and locally any time.

Usage:
    python colab/inspect.py --run-dir runs/<id> --checklist
    python colab/inspect.py --run-dir runs/<id> --summary
    python colab/inspect.py --run-dir runs/<id> --show-completions [--problem-ids 1,5,9]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from harness.common import read_jsonl  # noqa: E402


def load_config(run_dir):
    return json.loads((Path(run_dir) / "config.json").read_text(encoding="utf-8"))


def load_predictions(run_dir):
    path = Path(run_dir) / "predictions.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))["predictions"]


def by_key(records):
    table = {}
    for rec in records:
        table[(rec["model_id"], rec["problem_id"], rec["sample_index"])] = rec
    return table


def checklist(run_dir, config, records, predictions):
    preds = by_key(predictions) if predictions else {}
    print(f"{'model':<28} {'pid':>3} {'tokens':>5} {'trunc?':<7} {'status':<10} prediction")
    for rec in records:
        key = (rec["model_id"], rec["problem_id"], rec["sample_index"])
        pred = preds.get(key)
        truncated = rec["n_output_tokens"] is not None and rec["n_output_tokens"] >= config["max_new_tokens"]
        status = pred["status"] if pred else "-"
        pred_str = str(pred["prediction"]) if pred and pred["prediction"] is not None else ""
        if isinstance(pred_str, float):
            pred_str = f"{pred_str:g}"
        print(f"{rec['model_id']:<28} {rec['problem_id']:>3} "
              f"{rec['n_output_tokens'] or 0:>5} {'YES' if truncated else 'no':<7} "
              f"{status:<10} {pred_str[:40]}")
    print()
    for rec in records:
        tail = rec["prompt_text"][-90:].replace("\n", " / ")
        print(f"[{rec['model_id']} pid={rec['problem_id']}] prompt tail: ...{tail}")
    print("\ncheck: prompt tail ends with 'Think step by step...' (not 'Answer:'),")
    print("completions are not all truncated at max_new_tokens, and parse statuses")
    print("still find a final answer after the chain.")


def summary(run_dir, config, records):
    n_trunc = sum(1 for r in records
                  if r["n_output_tokens"] is not None
                  and r["n_output_tokens"] >= config["max_new_tokens"])
    models = sorted({r["model_id"] for r in records})
    print(f"records: {len(records)} | truncated at {config['max_new_tokens']}: {n_trunc}")
    for model_id in models:
        rows = [r for r in records if r["model_id"] == model_id]
        tok = [r["n_output_tokens"] or 0 for r in rows]
        print(f"  {model_id}: n={len(rows)} mean_output_tokens={sum(tok)/len(tok):.1f} max={max(tok)}")
    print(f"\nmean output tokens here vs direct baseline "
          f"(92.1 base / 102.0 instruct) is the cost axis of the writeup.")


def show_completions(run_dir, records, problem_ids):
    for rec in records:
        if problem_ids and rec["problem_id"] not in problem_ids:
            continue
        print("=" * 72)
        print(f"{rec['model_id']} | problem {rec['problem_id']} | "
              f"sample {rec['sample_index']} | {rec['n_output_tokens']} tokens")
        print("-" * 72)
        print(rec["completion_text"])
        print()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--checklist", action="store_true")
    mode.add_argument("--summary", action="store_true")
    mode.add_argument("--show-completions", action="store_true")
    parser.add_argument("--problem-ids", default=None,
                        help="comma-separated ids for --show-completions")
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir)
    config = load_config(run_dir)
    generations = read_jsonl(run_dir / "generations.jsonl")
    predictions = load_predictions(run_dir)
    records = [r for r in generations if not r.get("error")]

    if args.checklist:
        checklist(run_dir, config, records, predictions)
    elif args.summary:
        summary(run_dir, config, records)
    elif args.show_completions:
        pids = {int(p) for p in args.problem_ids.split(",")} if args.problem_ids else None
        show_completions(run_dir, records, pids)


if __name__ == "__main__":
    main()