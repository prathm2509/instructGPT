# Base vs Instruct: a small InstructGPT experiment

This is a small experiment I am running, in parallel with reading the InstructGPT paper.
The InstructGPT paper argues that post-training on human preferences makes language models more helpful and better at following instructions than base language models.
Rather than reproducing RLHF—which is outside the scope of this project and my compute budget—I compare a base model and its instruction-tuned counterpart on the same set of prompts.
This experiment accompanies my reading of InstructGPT as part of a broader ten-week study of modern LLMs.

Models: `Qwen/Qwen2.5-0.5B` (base) vs `Qwen/Qwen2.5-0.5B-Instruct`, on 20 frozen tasks (math, logic, classification, extraction, code), each under zero-shot and few-shot prompting. Everything runs on a free Colab T4.

## What's here

| File | What it is |
|---|---|
| `prompts.json` | the frozen 20-task prompt set (prompt, few-shot prefix, gold answer) |
| `base_vs_instruct.ipynb` | generates the completions → `base_vs_instruct.json` |
| `eval/` | deterministic grading pipeline (no LLM judges) → [RESULTS.md](RESULTS.md) |
| [RESULTS.md](RESULTS.md) | generated metrics: accuracy with CIs, drift/repetition/refusals, head-to-head task flips |
| `few_shot_failure_audit.py` | follow-up: re-samples the tasks where instruct few-shot underperforms instruct zero-shot, to separate stable failures from greedy-decoding luck → `few_shot_failure_audit.json` |
| [FINDINGS.md](FINDINGS.md) | my interpretation of the results |
| `next_experiment.md` | the audit's original design note (now executed) |

## Reproducing

1. **Generate completions** — run `base_vs_instruct.ipynb` on a GPU runtime (Colab T4 is enough); it writes `base_vs_instruct.json`.
2. **Grade and report** — no GPU needed:
   ```
   cd eval
   python grade.py && python metrics.py && python report.py
   ```
   Grading rules, metric definitions, and the human-override workflow are documented in [eval/README.md](eval/README.md).
3. **Failure audit** — on a GPU runtime:
   ```
   python few_shot_failure_audit.py --data-dir . --n-samples 5 --temperature 0.3
   ```
   To re-judge an existing audit JSON without a GPU: add `--reclassify`.


## Benchmark harness (`benchmark_v1`, frozen 2026-08-23)

A general harness replaces the notebook as the source of truth for generation: model IDs,
raw-vs-chat mode, n-sampling, temperature, and seed policy are CLI arguments, and every raw
generation is saved to `runs/<run-id>/generations.jsonl` with full metadata. Freeze record:
`benchmark_manifest.json` (sha256 `a52e11f09632f7f34ab3dae1683ee8389fc917b63f29211b1bd3111245e2b2ac`).

| File | What it is |
|---|---|
| `harness/run.py` | generation runner (n-sampling, temperature, seeds, raw/chat) → `generations.jsonl` |
| `harness/models.py` | lazy model load + one fully-described record per sample |
| `harness/extract.py` | `extract_v1` versioned answer extraction (number/yesno/label/name/string) |
| `harness/grade.py` | `generations.jsonl` → `predictions.json` + `metrics.json` + `parse_failures.json` |
| `harness/common.py` / `conditions.py` | seed policy, JSONL IO; prompt constructors (`direct` frozen, `cot`/`few_shot` stubbed) |
| `benchmark_v1.json` | frozen 50-task benchmark (20 inherited from `prompts.json` + 30 new) |
| `validate_benchmark.py` | structural checks + `--freeze` manifest hash |
| `tests/test_extraction.py` | 30 unit tests (extraction + seed policy) |

Commands (need a writable HuggingFace cache; on a CPU-only sandbox set `HF_HOME` to a
writable dir first / on Colab it just works):

```
# tests + freeze check
python -m unittest discover -s tests -t . -v
python validate_benchmark.py

# smoke test: 3 problems (1 number, 5 name, 9 label) x 2 models x N=3, T=0.3
python -m harness.run --benchmark benchmark_v1.json \
  --model Qwen/Qwen2.5-0.5B:raw --model Qwen/Qwen2.5-0.5B-Instruct:chat \
  --condition direct --problem-ids 1,5,9 --n-samples 3 --temperature 0.3

# baseline: greedy direct, all 50 x 2 models (use 1.5B IDs + --quantize on a GPU)
python -m harness.run --benchmark benchmark_v1.json \
  --model Qwen/Qwen2.5-0.5B:raw --model Qwen/Qwen2.5-0.5B-Instruct:chat \
  --condition direct --problem-ids all --n-samples 1 --greedy

# grade a run
python -m harness.grade --run-dir runs/<run-id>
```

Session handoff + full status: `research/2026-08-24-benchmark-harness-implementation.md`.


   just a note: i do use ai to format these properly but most of the findings are through my experimentation. (just putting it out there to be completely candid to myself on this learning journey)
