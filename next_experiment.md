# Next experiment: instruct few-shot failure audit

> **Status: executed 2026-08-02** — implemented as `few_shot_failure_audit.py`, results in `few_shot_failure_audit.json`. Kept as the original design note.

Extends `base_vs_instruct.json` — same notebook, same two models already loaded, no new downloads.

## What was found

Across the 20-prompt set, `instruct_few_shot` answers are wrong more often than `instruct_zero_shot`, sometimes producing values or names that don't even appear in the prompt (e.g. task 5 answers "Frank," task 8 answers "Fay" — neither name is in the problem). Base few-shot doesn't show this; it just runs on past the answer into repeated pattern text.

## The experiment

For every task where `instruct_few_shot` is wrong but `instruct_zero_shot` is right, rerun that task's few-shot prompt 3–5 times with `do_sample=True` at low temperature (~0.3–0.5) instead of greedy decoding.

Check: is the wrong answer a stable failure (same wrong answer each time — real breakdown in how the instruct model uses few-shot context) or an unstable one (different wrong answers each run — greedy decoding got unlucky on this specific prompt)?

## Why this and not something bigger

No new models, no new harness, no new prompts — just re-generation with a different decoding setting on outputs already flagged as anomalous. Fits in the same session.

## Output

A short table: task_id, wrong answers seen across N samples, stable or unstable. Feeds directly into the report's failure-case section — this is exactly the kind of faithfulness evidence Week 6 needs, gathered a week early.
