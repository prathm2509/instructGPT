"""Remove exact duplicate generation records in place (resume safety).

If the same cell (model, condition, problem, sample_index) was generated more
than once — e.g. a resumed run appended it again — keep the FIRST record and
drop later ones. Greedy decoding with a deterministic seed reproduces the same
output, so this never changes which record is graded, only removes repeats.

Usage:
    python colab/dedup_generations.py --run-dir runs/<id>
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from harness.common import read_jsonl  # noqa: E402


def key(rec):
    return (rec["benchmark_version"], rec["model_id"], rec["condition"],
            rec["problem_id"], rec["sample_index"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args(argv)

    path = Path(args.run_dir) / "generations.jsonl"
    records = read_jsonl(path)
    seen, kept, removed = set(), [], 0
    for rec in records:
        k = key(rec)
        if k in seen:
            removed += 1
            continue
        seen.add(k)
        kept.append(rec)

    if removed == 0:
        print(f"dedup: {len(records)} records, no duplicates")
        return 0
    with path.open("w", encoding="utf-8") as handle:
        for rec in kept:
            handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"dedup: {len(records)} -> {len(kept)} ({removed} duplicates removed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())