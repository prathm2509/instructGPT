# Direct Baseline Report: Qwen2.5 Base vs Instruct

**Date:** 2026-08-25  
**Benchmark:** `benchmark_v1`  
**Benchmark size:** 50 frozen tasks  
**Run ID:** `20260825-081614-direct`  
**Condition:** direct answering, greedy decoding, one sample per task  
**Extraction:** `extract_v1`

## Research Question

How does the instruction-tuned `Qwen/Qwen2.5-1.5B-Instruct` model compare with the base `Qwen/Qwen2.5-1.5B` model on the same frozen 50-task benchmark under direct answering?

This is a baseline comparison. It is not a reproduction of the InstructGPT paper and does not isolate every possible effect of instruction tuning.

## Method

Both models received the same 50 benchmark tasks. The base model used raw prompt formatting; the instruction-tuned model used the chat format with the harness system prompt. Decoding was greedy (`do_sample=false`, `N=1`, `max_new_tokens=200`).

The run was executed in the remote GPU environment because the local environment could not write the HuggingFace cache. The generation command was:

```bash
python -m harness.run --benchmark benchmark_v1.json \
  --model Qwen/Qwen2.5-1.5B:raw \
  --model Qwen/Qwen2.5-1.5B-Instruct:chat \
  --condition direct --problem-ids all --n-samples 1 --greedy
```

Grading was deterministic and used the frozen answer key. Code tasks were graded by executing the generated function against the frozen test cases.

## Results

### Overall

| Model | Correct | Accuracy | Wilson 95% CI | Parsed | Unparsed/ambiguous/malformed | Mean output tokens |
|---|---:|---:|---:|---:|---:|---:|
| `Qwen/Qwen2.5-1.5B` | 31/50 | 62.0% | 48.2%-74.1% | 42/50 | 8 | 92.1 |
| `Qwen/Qwen2.5-1.5B-Instruct` | 37/50 | 74.0% | 60.4%-84.1% | 45/50 | 5 | 102.0 |

The instruct model was **6 tasks** and **12 percentage points** more accurate than the base model. There were 100 generations and zero generation errors. The run completed from `2026-08-25T08:16:14Z` to `2026-08-25T09:48:51Z`, approximately 92 minutes.

### Accuracy by Category

| Category | Base | Instruct |
|---|---:|---:|
| Arithmetic word problem | 3/8 (37.5%) | 5/8 (62.5%) |
| Classification | 6/7 (85.7%) | 6/7 (85.7%) |
| Code | 0/4 (0.0%) | 0/4 (0.0%) |
| Extraction | 6/6 (100.0%) | 6/6 (100.0%) |
| Logic | 7/14 (50.0%) | 11/14 (78.6%) |
| Math | 9/11 (81.8%) | 9/11 (81.8%) |

## Interpretation

The instruction-tuned model performed better overall in this direct-answering condition. The largest category difference was in logic, where it scored 11/14 compared with the base model's 7/14. The two models were identical on classification, extraction, and math in this sample, while the instruct model was better on arithmetic word problems.

Both models scored 0/4 on code tasks. This is an important baseline result, but the four code outputs and the code grader should be inspected before interpreting it as a substantive model limitation. The category contains only four tasks.

The confidence intervals are wide and overlap. These results should therefore be treated as descriptive evidence for this benchmark, not as a statistically conclusive estimate of a general instruction-tuning effect.

## Extraction Audit

Thirteen generations were written to `parse_failures.json` because they were unparsed, ambiguous, or malformed: 8 from the base model and 5 from the instruct model. These records must remain part of the evaluation accounting; they were not silently discarded. A manual audit of those records is still needed to distinguish model formatting failures from extraction limitations.

## Reproducibility

The benchmark is frozen in `instructGPT/benchmark_v1.json` with manifest hash:

```text
a52e11f09632f7f34ab3dae1683ee8389fc917b63f29211b1bd3111245e2b2ac
```

The run outputs are identified by `20260825-081614-direct` and include the generation metadata, predictions, metrics, and parse-failure records. The repository harness records the model, mode, condition, sample index, seed, decoding settings, prompts, completions, token counts, and errors for every generation.

## Next Experiment

Before self-consistency, inspect the code-task outputs and parse failures, then add majority-vote aggregation to `harness/grade.py`. The next planned comparisons are zero-shot CoT and self-consistency with CoT at `N=3` and `N=5`, compared against this direct greedy baseline.

## Confidence and Limitations

- **High confidence:** the run completed with the expected 100 records and no generation errors.
- **High confidence:** the reported counts match the supplied `metrics.json` and `run_summary.json`.
- **Moderate confidence:** category-level differences, because several categories contain few tasks.
- **Open question:** whether the 0/4 code result reflects model capability, prompt design, or a grading mismatch.
- **Open question:** whether the 13 parse failures are genuine malformed answers or extraction cases that should be improved without changing the frozen benchmark.
