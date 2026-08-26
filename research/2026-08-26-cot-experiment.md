# Experiment: Zero-Shot Chain-of-Thought vs Direct Baseline

**Date planned:** 2026-08-26
**Status:** ready to run
**Benchmark:** `benchmark_v1` (frozen; sha256 `a52e11f0…`)
**Compare against:** `20260825-081614-direct`
**Extraction:** `extract_v1` (do not change mid-run)

## Research question

Does zero-shot chain-of-thought change accuracy on the frozen 50-task set relative to greedy direct answering, for `Qwen2.5-1.5B` base and instruct?

This is Kojima-style zero-shot CoT (`Think step by step…`), not Wei et al. few-shot CoT. The existing `few_shot_prefix` fields are question→answer exemplars, not rationales. Say that in the writeup.

The CoT paper found gains mainly around 100B. At 1.5B the interesting outcome is that chains can be fluent and still illogical, or that CoT can hurt. Record that if it happens.

## Frozen settings

| Axis | Value |
|---|---|
| Benchmark | `instructGPT/benchmark_v1.json` — do not edit |
| Models | `Qwen/Qwen2.5-1.5B:raw` and `Qwen/Qwen2.5-1.5B-Instruct:chat` |
| Condition | `cot` in `harness/conditions.py` |
| Prompt transform | strip a trailing `Answer:` / `Code:`, then append `Think step by step, then give the final answer.` |
| Decoding | greedy, `N=1` |
| `max_new_tokens` | **400** (direct used 200; CoT needs the extra room) |
| Seed / extract / grade | same as the direct baseline |
| System prompt | `You are a helpful assistant.` (chat model only) |

If Colab time is short, run instruct only. Both models is better: the other axis of the report is post-training.

## What this session does not do

- Do not start self-consistency. That needs the Self-Consistency paper and majority-vote in `grade.py`.
- Do not rewrite `few_shot_prefix` on `benchmark_v1.json`.
- Do not change `extract_v1`. Log bad parses; a later `extract_v2` can fix them.
- Do not read Let's Verify or add a verifier this session.

## Session

Needs a writable HuggingFace cache and a GPU (Colab T4 with `--quantize` is enough). Local sandbox cannot write `C:\Users\DELL\.cache\huggingface`.

### 1. Smoke test

Problems `1` (number), `5` (name), `9` (label) × both models.

```bash
cd instructGPT
python -m harness.run --benchmark benchmark_v1.json \
  --model Qwen/Qwen2.5-1.5B:raw \
  --model Qwen/Qwen2.5-1.5B-Instruct:chat \
  --condition cot --problem-ids 1,5,9 --n-samples 1 --greedy \
  --max-new-tokens 400 --quantize
python -m harness.grade --run-dir runs/<smoke-run-id>
```

Check by hand:

- The stored prompt ends with the CoT instruction, not `Answer:`.
- Completions are not obviously truncated at 400 tokens.
- `extract_v1` can still find a final answer after the chain.
- Code/name/label extraction is not obviously broken by the extra prose.

Fix only clear harness bugs. Then continue.

### 2. Full run

100 generations. Direct at 200 tokens took ~92 minutes; budget ~2 hours.

```bash
python -m harness.run --benchmark benchmark_v1.json \
  --model Qwen/Qwen2.5-1.5B:raw \
  --model Qwen/Qwen2.5-1.5B-Instruct:chat \
  --condition cot --problem-ids all --n-samples 1 --greedy \
  --max-new-tokens 400 --quantize
python -m harness.grade --run-dir runs/<run-id>
```

### 3. Writeup

Write `instructGPT/research/2026-08-26-cot-baseline-report.md` after the run, same shape as the direct report:

- overall accuracy + Wilson CI vs `20260825-081614-direct`
- accuracy by category
- mean output tokens (this is the cost axis)
- parse-failure count; do not drop them
- flip table: direct✓ / CoT✗ and the reverse, by model
- 8–10 hand-read chains, especially arithmetic, logic, and the four code tasks (direct was 0/4)

## How to read the result

Treat category splits as descriptive. Several categories are small; the CIs will overlap.

A chain that reaches the right answer with a wrong step is still counted correct by `extract_v1`. Flag those in the hand-read. That is faithfulness evidence, not a grading bug.

Truncation at 400 is a harness limit, not a model failure. Note it if it happens.

## After this

1. Read Self-Consistency (L3).
2. Add majority-vote aggregation in `harness/grade.py`.
3. Few-shot CoT: a small shared rationale bank, not 50 new prefixes.
4. Self-consistency at T=0.3, N=3 then N=5, graded as majority over CoT samples.
