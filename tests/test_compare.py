"""Tests for the offline tooling added for the CoT experiment:
compare_runs, colab/dedup_generations, colab/inspect.

All fixtures are synthetic run dirs with the same JSONL/JSON shapes that
harness.run and harness.grade produce; no torch / GPU needed.
"""

import json
import shutil
import uuid
import unittest
from pathlib import Path

from compare_runs import compare_predictions, load_predictions
from colab.dedup_generations import main as dedup_main
from colab.inspect import main as inspect_main
from harness.common import append_jsonl, write_json

BENCH_VERSION = "benchmark_v1"
MODEL_A = "Qwen/Qwen2.5-1.5B"
MODEL_B = "Qwen/Qwen2.5-1.5B-Instruct"

TMP_BASE = Path(__file__).resolve().parent.parent / ".test-tmp"


def make_run_dir(prefix="run-"):
    """Temp run dir under the repo (plain mkdir; mkdtemp's 0o700 ACLs clash
    with the Windows sandbox)."""
    TMP_BASE.mkdir(exist_ok=True)
    p = TMP_BASE / f"{prefix}{uuid.uuid4().hex[:10]}"
    p.mkdir()
    return p


def cleanup(*paths):
    for p in paths:
        shutil.rmtree(p, ignore_errors=True)


def make_config(run_dir, condition="cot"):
    write_json(run_dir / "config.json", {
        "run_id": run_dir.name,
        "benchmark_version": BENCH_VERSION,
        "models": [{"model_id": MODEL_A, "mode": "raw"},
                   {"model_id": MODEL_B, "mode": "chat"}],
        "condition": condition,
        "problem_ids": "all",
        "n_samples": 1,
        "do_sample": False,
        "max_new_tokens": 400,
    })


def make_generation(run_dir, model_id, pid, correct, tokens=20, status="parsed",
                    condition="cot", prediction="ok"):
    rec = {
        "benchmark_version": BENCH_VERSION,
        "problem_id": pid,
        "model_id": model_id,
        "model_mode": "raw" if model_id == MODEL_A else "chat",
        "condition": condition,
        "sample_index": 0,
        "seed": 1234,
        "do_sample": False,
        "max_new_tokens": 400,
        "prompt_text": f"problem {pid}",
        "completion_text": f"completion {pid}",
        "n_input_tokens": 20,
        "n_output_tokens": tokens,
        "elapsed_seconds": 1.0,
        "error": None,
    }
    append_jsonl(run_dir / "generations.jsonl", rec)
    pred = {
        "problem_id": pid,
        "model_id": model_id,
        "condition": condition,
        "sample_index": 0,
        "seed": 1234,
        "gold_answer": "gold",
        "answer_type": "number",
        "extraction_version": "extract_v1",
        "n_output_tokens": tokens,
        "status": status,
        "prediction": prediction,
        "strategy": "last_number",
        "candidates": [],
        "unit": None,
        "flags": [],
        "correct": correct,
        "category": "logic" if pid in (2, 3) else "math",
    }
    append_jsonl(run_dir / "predictions.jsonl", pred)
    return rec, pred


def finalize_predictions(run_dir):
    preds = []
    with (run_dir / "predictions.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                preds.append(json.loads(line))
    write_json(run_dir / "predictions.json", {
        "graded_utc": "2026-08-26T00:00:00Z",
        "extraction_version": "extract_v1",
        "n": len(preds),
        "predictions": preds,
    })


def make_graded_run(condition, flips):
    """flips: dict pid -> (direct_correct, cot_correct) for pids 1..3."""
    tmp = make_run_dir()
    make_config(tmp, condition)
    for pid in (1, 2, 3):
        correct_b, correct_t = flips[pid]
        if condition == "direct":
            correct = correct_b
        else:
            correct = correct_t
        make_generation(tmp, MODEL_A, pid, correct)
        make_generation(tmp, MODEL_B, pid, correct)
    finalize_predictions(tmp)
    return tmp


class CompareRunsTest(unittest.TestCase):
    def test_flip_table_and_runs(self):
        # pid 1: direct right, cot wrong -> regression
        # pid 2: direct wrong, cot right -> fix
        # pid 3: both right -> no flip
        direct = make_graded_run("direct", {1: (True, False), 2: (False, True),
                                            3: (True, True)})
        cot = make_graded_run("cot", {1: (True, False), 2: (False, True),
                                      3: (True, True)})
        try:
            report = compare_predictions(load_predictions(direct),
                                         load_predictions(cot))
            for model in (MODEL_A, MODEL_B):
                m = report["models"][model]
                self.assertEqual(m["flips"]["n_regressions"], 1)
                self.assertEqual(m["flips"]["n_fixes"], 1)
                self.assertEqual([f["problem_id"] for f in m["flips"]["regressions"]], [1])
                self.assertEqual([f["problem_id"] for f in m["flips"]["fixes"]], [2])
                self.assertEqual(m["baseline"]["n_correct"], 2)
                self.assertEqual(m["treatment"]["n_correct"], 2)
                self.assertEqual(m["by_category"]["math"]["n"], 1)
                self.assertEqual(m["by_category"]["logic"]["n"], 2)
        finally:
            cleanup(direct, cot)

    def test_missing_predictions_raises(self):
        d = make_run_dir()
        try:
            with self.assertRaises(FileNotFoundError):
                load_predictions(d)
        finally:
            cleanup(d)


class DedupTest(unittest.TestCase):
    def test_removes_exact_duplicates(self):
        run_dir = make_run_dir()
        try:
            make_config(run_dir)
            make_generation(run_dir, MODEL_A, 1, True)
            make_generation(run_dir, MODEL_A, 1, True)  # duplicate
            make_generation(run_dir, MODEL_A, 2, False)
            dedup_main(["--run-dir", str(run_dir)])
            from harness.common import read_jsonl

            records = read_jsonl(run_dir / "generations.jsonl")
            self.assertEqual(len(records), 2)
        finally:
            cleanup(run_dir)


class InspectTest(unittest.TestCase):
    def test_checklist_and_summary_run_without_torch(self):
        run_dir = make_run_dir()
        try:
            make_config(run_dir)
            make_generation(run_dir, MODEL_A, 1, True, tokens=400)  # truncated
            make_generation(run_dir, MODEL_A, 2, False, tokens=30)
            finalize_predictions(run_dir)
            # should not raise; capture output would be nice but smoke is enough
            inspect_main(["--run-dir", str(run_dir), "--checklist"])
            inspect_main(["--run-dir", str(run_dir), "--summary"])
            inspect_main(["--run-dir", str(run_dir), "--show-completions",
                          "--problem-ids", "1"])
        finally:
            cleanup(run_dir)


if __name__ == "__main__":
    unittest.main()