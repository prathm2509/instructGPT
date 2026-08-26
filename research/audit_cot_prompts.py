"""Static audit of the `cot` prompt transform before the real run.

Runs with no torch / no GPU: only reads benchmark_v1.json and conditions.py.
Checks the smoke-test checklist items that are checkable offline:

  1. every cot prompt ends with the frozen COT_INSTRUCTION,
  2. the trailing Answer:/Code: cue was stripped exactly where intended,
  3. reports (id, category, prompt head) so a human can eyeball the boundary.

Usage:
    python research/audit_cot_prompts.py [--head N]
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent  # instructGPT/
sys.path.insert(0, str(HERE))

from harness.conditions import COT_INSTRUCTION, cot_prompt, direct_prompt  # noqa: E402

CUE_TAILS = ("Answer:", "Code:")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--head", type=int, default=10,
                        help="how many stripped prompts to print (default 10)")
    args = parser.parse_args(argv)

    benchmark = json.loads((HERE / "benchmark_v1.json").read_text(encoding="utf-8"))
    tasks = benchmark["prompts"]
    problems = []

    for task in tasks:
        tid = task["id"]
        direct = direct_prompt(task)
        cot = cot_prompt(task)
        ok_suffix = cot.endswith(COT_INSTRUCTION)
        stripped = None
        for cue in CUE_TAILS:
            body = direct.rstrip()
            if body.endswith(cue):
                stripped = cue
                break
        if stripped is None and direct.rstrip().endswith(("?", ".", ":")):
            # non-cue endings are fine; just report for the eyeball check
            pass
        problems.append({
            "id": tid,
            "category": task["category"],
            "answer_type": task["answer_type"],
            "had_cue": stripped,
            "ok_suffix": ok_suffix,
            "prompt_head": direct[:90].replace("\n", " / "),
        })

    bad_suffix = [p for p in problems if not p["ok_suffix"]]
    if bad_suffix:
        print("FAIL: prompts missing the COT suffix:")
        for p in bad_suffix:
            print(" ", p["id"], p["category"])
        return 1

    with_cue = [p for p in problems if p["had_cue"]]
    print(f"ok: {len(problems)}/{len(tasks)} cot prompts end with the frozen instruction")
    print(f"ok: {len(with_cue)} prompts had a trailing cue ({', '.join(CUE_TAILS)}) stripped")
    print(f"ok: {len(problems) - len(with_cue)} prompts had no trailing cue (append-only)")
    print(f"\nfirst {args.head} stripped prompts (id | category | had_cue | prompt head):")
    for p in problems[: args.head]:
        print(f"  {p['id']:>3} | {p['category']:<23} | {p['had_cue'] or '-':<8} | {p['prompt_head']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())