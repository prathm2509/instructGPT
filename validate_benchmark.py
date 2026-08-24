"""Validate and freeze benchmark_v1.json.

    python validate_benchmark.py           # checks only
    python validate_benchmark.py --freeze  # checks, then writes benchmark_manifest.json

Freeze semantics: once benchmark_manifest.json exists, rerunning --freeze with
a changed benchmark file is an error (bump the version field instead of
silently editing a frozen benchmark). Rerunning with an unchanged file just
reprints the summary.

Checks:
  - exactly 50 tasks, unique ids, required fields per answer_type
  - gold answers parse under extract_v1 (self-extraction: "Answer: <gold>"
    must extract and match)
  - no duplicate prompt strings
  - gold answer does not appear in the few-shot prefix (whole-word for
    numbers, substring for names/strings; yes/no and labels exempt since
    exemplars necessarily use the same answer vocabulary)
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

from harness.common import sha256_file, utc_now, write_json
from harness.extract import answers_match, extract_answer, parse_number_token

HERE = Path(__file__).parent
BENCHMARK_PATH = HERE / "benchmark_v1.json"
MANIFEST_PATH = HERE / "benchmark_manifest.json"

ANSWER_TYPES = {"number", "yesno", "label", "name", "string", "code"}
DIFFICULTIES = {"easy", "medium", "hard"}
REQUIRED = {"id", "category", "difficulty", "answer_type", "prompt",
            "few_shot_prefix", "gold_answer"}


def check_tasks(data):
    errors = []
    tasks = data.get("prompts", [])

    if len(tasks) != 50:
        errors.append(f"expected 50 tasks, found {len(tasks)}")

    ids = [t.get("id") for t in tasks]
    if len(set(ids)) != len(ids):
        errors.append("task ids are not unique")

    seen_prompts = {}
    for task in tasks:
        tid = task.get("id", "?")
        missing = REQUIRED - set(task)
        if missing:
            errors.append(f"task {tid}: missing fields {sorted(missing)}")
            continue

        atype = task["answer_type"]
        if atype not in ANSWER_TYPES:
            errors.append(f"task {tid}: unknown answer_type {atype!r}")
        if task["difficulty"] not in DIFFICULTIES:
            errors.append(f"task {tid}: unknown difficulty {task['difficulty']!r}")
        if atype == "label" and not task.get("labels"):
            errors.append(f"task {tid}: label task needs labels")
        if atype == "label" and task["gold_answer"].lower() not in [
                label.lower() for label in task.get("labels", [])]:
            errors.append(f"task {tid}: gold {task['gold_answer']!r} not in labels")
        if atype == "code":
            tests = task.get("code_tests")
            if not tests or "fn" not in tests or not tests.get("tests"):
                errors.append(f"task {tid}: code task needs code_tests.fn and tests")

        prompt = task["prompt"]
        if prompt in seen_prompts:
            errors.append(f"task {tid}: duplicate prompt of task {seen_prompts[prompt]}")
        seen_prompts[prompt] = tid

        # gold must parse under extract_v1
        if atype == "number" and parse_number_token(task["gold_answer"]) is None:
            errors.append(f"task {tid}: numeric gold {task['gold_answer']!r} does not parse")
        if atype == "yesno" and task["gold_answer"].lower() not in ("yes", "no"):
            errors.append(f"task {tid}: yesno gold must be yes or no")
        if atype in ("number", "yesno", "label"):
            ext = extract_answer(f"Answer: {task['gold_answer']}", atype,
                                 labels=task.get("labels"))
            if ext["status"] != "parsed" or not answers_match(
                    ext, task["gold_answer"], atype):
                errors.append(f"task {tid}: gold self-extraction failed: {ext}")

        # gold must not leak into the few-shot prefix. Leakage channel that
        # matters: an exemplar ANSWER equal to the gold teaches "just say
        # <gold>". A gold number appearing as a quantity inside an exemplar
        # question (or vocabulary like "knight" in a knights/knaves setup) is
        # priming, not leakage, and is allowed.
        prefix = task["few_shot_prefix"]
        gold = task["gold_answer"]
        exemplar_answers = re.findall(r"answer\s*:\s*(.+)", prefix, re.IGNORECASE)
        if atype == "number":
            gold_num = parse_number_token(gold)
            for ans in exemplar_answers:
                if parse_number_token(ans.strip()) == gold_num:
                    errors.append(
                        f"task {tid}: gold {gold!r} is an exemplar answer in few_shot_prefix")
        elif atype in ("name", "string"):
            from harness.extract import normalize_text
            gold_norm = normalize_text(gold)
            for ans in exemplar_answers:
                if gold_norm == normalize_text(ans):
                    errors.append(
                        f"task {tid}: gold {gold!r} is an exemplar answer in few_shot_prefix")

    return errors, tasks


def build_manifest(data, tasks, file_hash):
    categories = Counter(t["category"] for t in tasks)
    difficulties = Counter(t["difficulty"] for t in tasks)
    return {
        "frozen_utc": utc_now(),
        "benchmark_file": BENCHMARK_PATH.name,
        "benchmark_version": data["version"],
        "benchmark_sha256": file_hash,
        "n_tasks": len(tasks),
        "tasks_by_category": dict(sorted(categories.items())),
        "tasks_by_difficulty": dict(sorted(difficulties.items())),
        "models": {
            "primary": {
                "base": "Qwen/Qwen2.5-1.5B",
                "instruct": "Qwen/Qwen2.5-1.5B-Instruct",
            },
            "fallback": {
                "base": "Qwen/Qwen2.5-0.5B",
                "instruct": "Qwen/Qwen2.5-0.5B-Instruct",
            },
            "note": "Model IDs are CLI arguments to harness.run; the freeze binds "
                    "the primary pair per the NEXT.md milestone, with the 0.5B pair "
                    "as the proven fallback (used for the 2026-08-23 CPU smoke test).",
        },
        "decoding": {
            "direct": {"do_sample": False, "temperature": None,
                       "n_samples": 1, "max_new_tokens": 200},
            "smoke": {"do_sample": True, "temperature": 0.3,
                      "n_samples": 3, "max_new_tokens": 200},
            "self_consistency_planned": {"do_sample": True, "temperature": 0.3,
                                         "n_samples": [3, 5]},
        },
        "extraction_version": "extract_v1",
        "seed_policy": "sha256(base_seed|benchmark_version|model_id|condition|problem_id|sample_index)[:8 hex]",
        "base_seed": 20260823,
        "system_prompt": "You are a helpful assistant.",
        "history": [{
            "version": data["version"],
            "sha256": file_hash,
            "frozen_utc": utc_now(),
            "note": "initial freeze: 20 tasks copied verbatim from prompts.json "
                    "v1.0 + 30 new tasks (ids 21-50)",
        }],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", action="store_true",
                        help="write benchmark_manifest.json after checks pass")
    args = parser.parse_args(argv)

    with BENCHMARK_PATH.open(encoding="utf-8") as handle:
        data = json.load(handle)
    errors, tasks = check_tasks(data)

    categories = Counter(t["category"] for t in tasks)
    difficulties = Counter(t["difficulty"] for t in tasks)
    print(f"{data['version']}: {len(tasks)} tasks")
    print(f"  categories:  {dict(sorted(categories.items()))}")
    print(f"  difficulties: {dict(sorted(difficulties.items()))}")

    if errors:
        print(f"\nFAILED with {len(errors)} error(s):")
        for error in errors:
            print(f"  - {error}")
        return 1

    file_hash = sha256_file(BENCHMARK_PATH)
    print(f"  sha256: {file_hash}")
    print("all checks passed")

    if not args.freeze:
        return 0

    if MANIFEST_PATH.exists():
        old = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        if old["benchmark_sha256"] != file_hash:
            print("\nREFUSED: benchmark file changed after freeze "
                  f"(manifest has {old['benchmark_sha256'][:12]}..., "
                  f"file is {file_hash[:12]}...). "
                  "Bump the version field instead of editing a frozen benchmark.")
            return 1
        print("manifest already exists and matches; nothing to do")
        return 0

    manifest = build_manifest(data, tasks, file_hash)
    write_json(MANIFEST_PATH, manifest)
    print(f"wrote {MANIFEST_PATH.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
