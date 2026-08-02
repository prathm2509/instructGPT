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


   just a note: i do use ai to format these properly but most of the findings are through my experimentation. (just putting it out there to be completely candid to myself on this learning journey)
