# Zero-Shot Chain-of-Thought vs Direct Baseline

**Experiment:** does telling a 1.5B model to "think step by step" change accuracy?
**Benchmark:** `benchmark_v1` — 50 frozen tasks (sha256 `a52e11f0…`)
**Models:** `Qwen/Qwen2.5-1.5B` (raw) and `Qwen/Qwen2.5-1.5B-Instruct` (chat)
**Compared runs:** `20260826-direct-q4` (baseline) vs `20260826-111745-cot` (treatment)
**Date:** 2026-08-26 · **Extraction:** `extract_v1` · **Decoding:** greedy, N=1, 4-bit quantized

---

## TL;DR

Zero-shot chain-of-thought did **not** improve accuracy at 1.5B. The instruction-tuned
model actually got *one task worse* (38→37), and both models paid a token premium for the
privilege — instruct's mean output rose **84%** (93.4 → 171.7 tokens) for a net *loss*.

But the flat headline hides a genuinely interesting result: CoT **reallocated** errors rather
than reducing them. Math and arithmetic got better (base math 8/11 → **11/11**); classification
and extraction got worse (base classification 5/7 → **1/7**). The failure modes are legible in
the chains themselves — empty outputs, prose-wrapped code that breaks the grader, and reasoning
loops that never conclude. This is the small-scale behavior the Kojima zero-shot CoT paper would
predict: fluent chains, no accuracy gain, real cost.

**Key numbers (matched 4-bit precision):**

| Model | Direct | CoT | Δ |
|---|---:|---:|---:|
| Base 1.5B | 31/50 (62.0%) | 31/50 (62.0%) | 0.0 pp |
| Instruct 1.5B | 38/50 (76.0%) | 37/50 (74.0%) | **−2.0 pp** |

---

## 1. Research question

Does appending *"Think step by step, then give the final answer."* change accuracy on the frozen
50-task set, relative to greedy direct answering, for base and instruction-tuned `Qwen2.5-1.5B`?

This is **Kojima-style zero-shot CoT** (an instruction), not **Wei-style few-shot CoT** (worked
examples with rationales). The motivating expectation comes from Kojima et al.: "Let's think step
by step" boosts accuracy mainly at the ~100B scale. At 1.5B the interesting outcomes are not "it
helps," but rather *how the chains fail* — and whether any improvement shows up at all.

## 2. Method

**One variable changed.** Both runs use the same 50 tasks, the same two models, greedy decoding
(`do_sample=false`, `N=1`), the same seed policy, and the same `extract_v1` grader with the frozen
answer key. The only prompt difference is the `cot` transform in `harness/conditions.py`: strip a
trailing `Answer:`/`Code:`, then append the CoT instruction. `direct` sends the prompt unchanged.

**Two deliberate differences from the *original* fp32 baseline.** The first direct baseline
(`20260825-081614-direct`, reported 2026-08-25) ran fp32 and scored base 31/50 / instruct 37/50.
For this comparison I re-ran the direct condition in the **same 4-bit precision** as the CoT run,
because quantization is not output-identical to fp32:

| Run | Base accuracy | Instruct accuracy | Base code |
|---|---|---:|---:|
| direct, fp32 (original) | 31/50 | 37/50 | 0/4 |
| direct, 4-bit (this baseline) | 31/50 | **38/50** | **1/4** |

4-bit quantization moved instruct by one task and flipped a base code task from wrong to right.
That is direct evidence that precision noise is on the order of **1–2 task flips** at this size —
small, but exactly the size of the effect under study. So the CoT treatment is compared against the
matched-precision `20260826-direct-q4`, which makes any residual CoT delta a *prompt* effect rather
than a precision artifact. The fp32 run is kept as a historical anchor.

**Token cap.** Direct used `max_new_tokens=200` (the frozen baseline setting); CoT used 400 to give
chains room. This is a real asymmetry: the direct run truncated **18 outputs** at its 200-token cap
(base 8, instruct 10), while the CoT run truncated exactly **1** (base task 37, at 400). The cap
difference slightly favors CoT and is noted as a limitation rather than corrected post hoc.

Both runs executed on a Colab T4 (4-bit `bitsandbytes` nf4), writing 100 records each with **0
generation errors**. Grading is deterministic; code tasks are graded by executing the generated
function against the frozen test cases.

## 3. Results

### 3.1 Overall accuracy

| Model | Direct (q4) | CoT (q4) | Δ accuracy |
|---|---:|---:|---:|
| `Qwen/Qwen2.5-1.5B` | 31/50 (62.0%) | 31/50 (62.0%) | **0.0 pp** |
| `Qwen/Qwen2.5-1.5B-Instruct` | 38/50 (76.0%) | 37/50 (74.0%) | **−2.0 pp** |

Wilson 95% CIs overlap in every case (base 48.2–74.1% both; instruct 62.6–85.7% → 60.4–84.1%),
so the headline is best read as "no effect or a slight negative," not as a precise −2.

### 3.2 Accuracy by category

| Category | n | Base direct | Base CoT | Instruct direct | Instruct CoT |
|---|---:|---:|---:|---:|---:|
| Arithmetic word problem | 8 | 4/8 | **6/8** | 5/8 | 6/8 |
| Classification | 7 | 5/7 | **1/7** | **7/7** | 5/7 |
| Code | 4 | **1/4** | 0/4 | 0/4 | 0/4 |
| Extraction | 6 | **6/6** | 5/6 | **6/6** | 4/6 |
| Logic | 14 | 7/14 | 8/14 | 10/14 | 11/14 |
| Math | 11 | 8/11 | **11/11** | 10/11 | **11/11** |

The redistribution is the story. Math improved for **both** models (base 8→11, instruct 10→11),
and base arithmetic rose 4→6. The price was paid in base classification (5→1) and instruct
extraction (6→4) plus a small classification dip (7→5).

### 3.3 Cost axis

| Model | Direct mean tokens | CoT mean tokens | Δ |
|---|---:|---:|---:|
| Base | 88.4 | 98.6 | +10.2 (+12%) |
| Instruct | 93.4 | **171.7** | **+78.3 (+84%)** |

Instruct emits ~84% more tokens per answer for a net loss of one task. The base model's small +12%
understates the story only because so many base CoT outputs are *empty* (see §4.1).

### 3.4 Parse failures and truncation

| | Direct | CoT |
|---|---:|---:|
| Parse failures (base) | 7 | 15 |
| Parse failures (instruct) | 6 | 7 |
| **Total** | **13** | **22** |
| Truncated at cap | 18 (at 200) | 1 (at 400) |

Parse failures nearly doubled for the base model (7→15), driven by empty outputs and
yes/no-candidate misses. None were discarded: every unparsed/ambiguous/malformed record is retained
in each run's `parse_failures.json`.

### 3.5 Flip table (per-task movement)

*First-pass counts from the graded predictions; see §5 for the corrections that hand-reading forces.*

**Base model**
- Direct right → CoT wrong (**8 regressions**): tasks 6, 7, 9, 10, 12, 16, 45, 49
- Direct wrong → CoT right (**8 fixes**): tasks 4, 20, 28, 31, 32, 38, 39, 40

**Instruct model**
- Direct right → CoT wrong (**6 regressions**): tasks 6, 14, 15, 20, 46, 47
- Direct wrong → CoT right (**5 fixes**): tasks 28, 32, 33, 43, 44

Raw net movement: base 0 (8↔8), instruct −1 (5 vs 6). Hand-reading changes that conclusion — see §5.

## 4. What the numbers hide — the chains, read by hand

The flip table is a first pass; the completions themselves carry the real result. Seven mechanisms
account for nearly all movement.

### 4.1 The base model's classification collapse is empty outputs

Tasks **9, 10, 12, 45, 47** (classification) and **44** (logic) produced **zero characters** from the
raw base model under the CoT instruction — 1 token, then end-of-sequence. Direct answering got 5/7
classification right; CoT got 1/7 because the model, asked to "think step by step" on a label-style
prompt, simply **emitted nothing**. This is a failure to *engage*, not a wrong answer: the added
instruction converts a correct short answer into silence. It accounts for most of base's
classification collapse and five of its six new parse failures.

### 4.2 Code: chain prose breaks the executor (base task 16)

Base's direct answer to "write a function `add`" was a clean code fence that executes:

```python
def add(x, y):
    return x + y
```

Under CoT the **same correct function** is wrapped in prose — *"Sure, here's the Python function
`add` that takes two numbers…"* — and the `exec`-based code grader fails on the wrapper with an
`IndentationError`/`SyntaxError`. Base's code "regression" (1/4 → 0/4) is therefore **not a reasoning
failure**: the code is right, but `extract_v1`'s code path executes the raw completion and cannot
separate the function from the chain. Instruct shows the same artifact (its four code tasks fail
`exec_error: SyntaxError` in *both* conditions, hence no flip). This is a known `extract_v1`
limitation to log — a future `extract_v2` that pulls the *last* code block would fix it without
changing the benchmark.

### 4.3 Gold-containment produces a false-correct (instruct task 43)

`extract_v1` grades `name`/`string` answers by checking whether the **gold string appears anywhere in
the completion** (documented in `harness/extract.py`). Task 43 asks who finished last; the gold is
**Noah**. Instruct's chain *mis-transcribes* the second premise — it writes "Noah finished ahead of
Olivia" where the prompt says "Olivia finished ahead of Liam" — and concludes **"Olivia finished
last," the wrong answer**. But because the chain restates "Noah" while reasoning, gold-containment
scores it correct.

This is faithfulness evidence, not a grading bug: the model got the task wrong and a keyword rescued
it. It means the flip table **over-counts instruct fixes by one**, so instruct's true CoT movement is
**−2 tasks, not −1** (and the recorded 37 may still be optimistic; a full audit of gold-containment
cells could move it further).

### 4.4 Yes/no chains never say "yes" or "no" (tasks 6, 7; base 37)

Tasks 6 and 7 are yes/no logic. CoT chains reason about the premises but never emit the literal
`yes`/`no` that `extract_v1` matches (`no_yesno_candidate`), so they score wrong even where the
reasoning is sound. This is the shared cause of base's regressions on 6 and 7 and instruct's on 6.

### 4.5 Base task 37: a fluent loop, cut off at 400 tokens

Task 37 ("If the alarm rings, the dog barks. The dog did not bark. Did the alarm ring?") makes the
base model repeat *"Step N: Determine if the alarm ringing is the cause of the dog barking…"*
essentially verbatim until it hits the 400-token cap — the run's single truncation. It is the
cleanest instance of the plan's predicted failure: a fluent, well-formatted chain that never
concludes. (It is also, again, a yes/no task whose chain never states a yes/no.)

### 4.6 Task 20 flips both directions at once — "all but 9"

"A farmer has 17 sheep. All but 9 die. How many are left?" (gold: **9**). Base CoT answers **9**
(a fix); instruct CoT answers **17 − 9 = 8** (a regression) because it reads "all but 9" as "9 die."
The same task, same condition, helps one model and hurts the other — a reminder that the flip
table's net counts obscure task-level disagreement between the two models.

### 4.7 Where CoT genuinely helps, the chains are correct and well-formed

The gains are real, not artifacts. Base task 31 sets up `x + (x+1) + (x+2) = 51` and lands on 18;
instruct task 44 derives **63** from the two digit constraints; both models solve task 32's
proportion correctly. The wins concentrate in math and arithmetic — the one place a short chain is a
genuine scaffold at this scale — and their chains are visibly sound (correct algebra, correct
answer), not just lucky final tokens.

## 5. Interpretation

The cleanest summary: **at 1.5B, zero-shot CoT reallocates correctness instead of adding it, and it
charges a token tax for the rearrangement.**

- **No accuracy gain.** Base is exactly unchanged (31/50); instruct is one-to-two tasks worse once
  the task-43 false-correct is corrected. The CIs overlap throughout.
- **Real redistribution.** Math and arithmetic improve (base math → 11/11; base arithmetic 4→6);
  classification and extraction degrade (base classification 5→1; instruct extraction 6→4).
- **The failures are legible.** Empty base outputs on label prompts, prose-wrapped code, yes/no
  chains that never answer, and a non-terminating loop are all *visible in the text*, not hidden in
  aggregate scores. This is the small-scale analog of the Kojima finding: the chains are fluent but
  don't buy accuracy, and at this scale the extra instruction can actively break tasks the model
  would otherwise answer correctly.
- **The "who last" false-correct is the most important single data point.** It shows the grader
  trusting a keyword over a conclusion, which is precisely the faithfulness gap the plan flagged as
  worth recording rather than "fixing" mid-run.

## 6. Grading and faithfulness caveats

These are properties of the *pipeline*, not the models, and belong in the report rather than in a
silent patch:

1. **Name/string gold-containment** credits any mention of the gold string, so chains that restate
   the gold while reasoning can be scored correct even with a wrong conclusion (task 43).
2. **Code grading executes the raw completion**, so chain prose around otherwise-correct code causes
   spurious `exec` failures (task 16 base; instruct's four code tasks in both conditions).
3. **Yes/no extraction requires the literal token**, so chains that express the answer in other
   words score as unparsed (tasks 6, 7, 37).
4. **The token caps differ** (direct 200, CoT 400); direct truncated 18 outputs at its cap, CoT 1.

None of these were changed during the run (the plan freezes `extract_v1`); each is a candidate for a
future `extract_v2`, and each is recorded here so the numbers can be read honestly.

## 7. Confidence and limitations

- **High confidence:** both runs wrote 100 records with 0 generation errors; every count above is
  read from `predictions.json` / `metrics.json` / `generations.jsonl`.
- **High confidence:** the headline (no CoT gain; token cost). The null/slightly-negative effect is
  robust to the CIs and to the known artifacts.
- **Moderate confidence:** per-category and per-task movement. Categories are small (4–14 tasks) and
  several flips trace to pipeline artifacts (§6) rather than reasoning.
- **Open questions:** (a) whether base's empty CoT outputs are a raw-vs-chat formatting interaction
  rather than a model property — a chat-format base + CoT run would separate them; (b) how many more
  gold-containment cells are false-corrects on *either* side — a full name/string audit is pending;
  (c) whether the direct run's 18 truncations at 200 tokens cost it any correct answers it would
  have reached with more room.

## 8. Reproducibility

- Benchmark: `benchmark_v1.json`, sha256 `a52e11f09632f7f34ab3dae1683ee8389fc917b63f29211b1bd3111245e2b2ac`
- Direct run: `runs/20260826-direct-q4/` — `python -m harness.run --condition direct --max-new-tokens 200 --quantize` + `python -m harness.grade`
- CoT run: `runs/20260826-111745-cot/` — `python -m harness.run --condition cot --max-new-tokens 400 --quantize` + `python -m harness.grade`
- Comparison: `research/2026-08-26-cot-vs-direct-comparison.json` (generated by `compare_runs.py`)
- Prompt-transform audit: `research/audit_cot_prompts.py` (50/50 prompts verified before the run)

Each run dir contains `config.json`, `generations.jsonl`, `predictions.json`, `metrics.json`,
`parse_failures.json`, and `run_summary.json`.

## 9. Next steps

1. **Faithfulness audit:** hand-check every name/string cell in both runs (gold-containment) and the
   four code cells, then write down the `extract_v2` rules they motivate (pull the last code block;
   require the gold in the *conclusion*, not anywhere in the chain).
2. **Self-consistency:** majority vote over `N=3,5` CoT samples (T=0.3) is the natural next
   comparison — it tests whether the null headline is stable under sampling and whether base's empty
   outputs disappear once decoding is non-greedy.
3. **Few-shot CoT:** a small shared rationale bank (not 50 new prefixes) to contrast Kojima
   zero-shot against Wei-style exemplars at this scale.
