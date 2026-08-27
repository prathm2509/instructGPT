"""Generation runner.

Usage (smoke test, CPU):
    python -m harness.run --benchmark benchmark_v1.json \
        --model Qwen/Qwen2.5-0.5B:raw --model Qwen/Qwen2.5-0.5B-Instruct:chat \
        --condition direct --problem-ids 1,2,3 --n-samples 3 --temperature 0.3

Usage (greedy baseline, Colab with GPU):
    python -m harness.run --benchmark benchmark_v1.json \
        --model Qwen/Qwen2.5-1.5B:raw --model Qwen/Qwen2.5-1.5B-Instruct:chat \
        --condition direct --problem-ids all --n-samples 1 --quantize

Model specs are "model_id:mode" where mode is raw (plain text) or chat
(apply_chat_template with the system prompt). Default mode: raw.

Every generation is appended to <output-dir>/generations.jsonl immediately, so
a crashed run loses nothing already written. Failures are written as records
with an error field, never silently dropped.

Resume: when generations.jsonl already exists in the output directory, cells
whose (model, condition, problem, sample) record is already present are skipped
instead of regenerated, so rerunning the same command (e.g., after a Colab
disconnect) only produces the missing cells. Identical seeds make this exact
for greedy decoding.
"""

import argparse
import json
import traceback
from datetime import datetime
from pathlib import Path

from .common import append_jsonl, read_jsonl, utc_now, write_json
from .conditions import CONDITIONS, CONDITION_METADATA
from .models import generate_sample, load_model

DEFAULT_MAX_NEW_TOKENS = 200
DEFAULT_SEED = 20260823
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."


def parse_model_spec(value):
    if ":" in value:
        model_id, mode = value.rsplit(":", 1)
    else:
        model_id, mode = value, "raw"
    if mode not in ("raw", "chat"):
        raise argparse.ArgumentTypeError(
            f"model mode must be raw or chat, got {mode!r} in {value!r}"
        )
    if not model_id:
        raise argparse.ArgumentTypeError("model id may not be empty")
    return {"model_id": model_id, "mode": mode}


def parse_problem_ids(value):
    if value.strip().lower() == "all":
        return "all"
    try:
        ids = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "problem ids must be 'all' or comma-separated integers"
        ) from exc
    if not ids:
        raise argparse.ArgumentTypeError("provide at least one problem id")
    if len(set(ids)) != len(ids):
        raise argparse.ArgumentTypeError("problem ids must be unique")
    return ids


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--model", dest="models", type=parse_model_spec,
                        action="append", required=True,
                        help="model_id:raw|chat (repeatable)")
    parser.add_argument("--condition", choices=sorted(CONDITIONS), required=True)
    parser.add_argument("--problem-ids", type=parse_problem_ids, default="all")
    parser.add_argument("--n-samples", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    sampling = parser.add_mutually_exclusive_group()
    sampling.add_argument("--greedy", action="store_true",
                          help="greedy decoding (default when --temperature is absent)")
    sampling.add_argument("--temperature", type=float, default=None,
                          help="sampling temperature; presence implies do_sample=True")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="base seed")
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="default: <benchmark_dir>/runs/<timestamp>-<condition>")
    parser.add_argument("--quantize", action="store_true",
                        help="4-bit bitsandbytes load (requires CUDA)")
    return parser


def validate_args(parser, args):
    if args.n_samples < 1:
        parser.error("--n-samples must be at least 1")
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be at least 1")
    if args.temperature is not None and args.temperature <= 0:
        parser.error("--temperature must be greater than 0")


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(parser, args)

    do_sample = args.temperature is not None

    with args.benchmark.open(encoding="utf-8") as handle:
        benchmark = json.load(handle)
    benchmark_version = benchmark["version"]
    tasks = benchmark["prompts"]
    by_id = {task["id"]: task for task in tasks}

    if args.problem_ids == "all":
        problem_ids = [task["id"] for task in tasks]
    else:
        missing = [pid for pid in args.problem_ids if pid not in by_id]
        if missing:
            parser.error(f"problem ids not in benchmark: {missing}")
        problem_ids = args.problem_ids

    output_dir = args.output_dir
    if output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = args.benchmark.parent / "runs" / f"{stamp}-{args.condition}"
    output_dir.mkdir(parents=True, exist_ok=True)
    generations_path = output_dir / "generations.jsonl"

    config = {
        "run_id": output_dir.name,
        "created_utc": utc_now(),
        "benchmark": str(args.benchmark.resolve()),
        "benchmark_version": benchmark_version,
        "models": args.models,
        "condition": args.condition,
        "problem_ids": problem_ids,
        "n_samples": args.n_samples,
        "do_sample": do_sample,
        "temperature": args.temperature if do_sample else None,
        "max_new_tokens": args.max_new_tokens,
        "base_seed": args.seed,
        "seed_policy": "sha256(base_seed|benchmark_version|model_id|condition|problem_id|sample_index)[:8hex]",
        "system_prompt": args.system_prompt,
        "quantize": args.quantize,
    }
    if args.condition in CONDITION_METADATA:
        config["condition_metadata"] = CONDITION_METADATA[args.condition]
    write_json(output_dir / "config.json", config)
    print(f"run dir: {output_dir}")

    build_prompt = CONDITIONS[args.condition]
    n_written = n_errors = n_skipped = 0
    started = utc_now()

    # Resume: skip cells whose successful records already exist in this dir.
    # Error records are kept, but those cells are retried, not skipped.
    existing = set()
    if generations_path.exists():
        for rec in read_jsonl(generations_path):
            if rec.get("error"):
                continue
            existing.add((rec.get("benchmark_version"), rec.get("model_id"),
                          rec.get("condition"), rec.get("problem_id"),
                          rec.get("sample_index")))
    if existing:
        print(f"resuming: {len(existing)} cell(s) already present, will be skipped")

    for spec in args.models:
        print(f"loading {spec['model_id']} ({spec['mode']})")
        model, tokenizer = load_model(spec["model_id"], quantize=args.quantize)
        for pid in problem_ids:
            task = by_id[pid]
            user_text = build_prompt(task)
            for sample_index in range(args.n_samples):
                cell = (benchmark_version, spec["model_id"], args.condition,
                        pid, sample_index)
                if cell in existing:
                    n_skipped += 1
                    continue
                record = None
                try:
                    record = generate_sample(
                        model, tokenizer, user_text,
                        mode=spec["mode"], system_prompt=args.system_prompt,
                        do_sample=do_sample, temperature=args.temperature,
                        max_new_tokens=args.max_new_tokens,
                        base_seed=args.seed, benchmark_version=benchmark_version,
                        model_id=spec["model_id"], condition=args.condition,
                        problem_id=pid, sample_index=sample_index,
                    )
                except Exception as exc:  # record, never silently drop
                    record = {
                        "benchmark_version": benchmark_version,
                        "problem_id": pid,
                        "model_id": spec["model_id"],
                        "model_mode": spec["mode"],
                        "condition": args.condition,
                        "sample_index": sample_index,
                        "seed": None,
                        "do_sample": do_sample,
                        "temperature": args.temperature if do_sample else None,
                        "max_new_tokens": args.max_new_tokens,
                        "prompt_text": user_text,
                        "completion_text": "",
                        "n_input_tokens": None,
                        "n_output_tokens": None,
                        "elapsed_seconds": None,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    n_errors += 1
                    traceback.print_exc()
                append_jsonl(generations_path, record)
                n_written += 1
                print(f"  problem {pid} sample {sample_index}: "
                      f"{record['n_output_tokens']} tokens "
                      f"in {record['elapsed_seconds']}s"
                      + (f"  ERROR {record['error']}" if record["error"] else ""))
        del model
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    summary = {
        "run_id": config["run_id"],
        "started_utc": started,
        "finished_utc": utc_now(),
        "n_records_written": n_written,
        "n_records_skipped": n_skipped,
        "n_errors": n_errors,
        "expected_records": len(args.models) * len(problem_ids) * args.n_samples,
    }
    write_json(output_dir / "run_summary.json", summary)
    print(f"wrote {n_written} new records ({n_skipped} already present, "
          f"{n_errors} errors) -> {generations_path}")


if __name__ == "__main__":
    main()
