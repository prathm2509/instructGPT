# Eval pipeline

Turns the raw completions in `../base_vs_instruct.json` into the metrics in
`../RESULTS.md`. Three stages, each writing a JSON artifact the next one reads:

```
python grade.py     # -> grades.json    (per task x condition: extracted answer, correct?)
python metrics.py   # -> metrics.json   (accuracy + CIs, behavior metrics, head-to-head)
python report.py    # -> ../RESULTS.md, metrics_chart.png
```

No LLM is involved anywhere in grading — every number is reproducible from the
raw JSON with these scripts.

## Grading rules

Extraction is **condition-aware**, because the two prompt formats put the
answer in different places:

- **Few-shot** completions start with the answer and then drift into
  self-generated Q/A pairs — so only the **first paragraph** is graded. If it
  contains no parseable answer, grading falls back to the zero-shot rules and
  the cell is flagged `needs_review`.
- **Zero-shot** instruct answers are chain-of-thought prose that states the
  answer at the **end** — numbers are extracted in precedence order:
  `answer is X` → `**X**` → last `= X` → last number in the output.

Per answer type:

| Type | Tasks | Rule |
|---|---|---|
| number | 1–4, 20 | numeric match after stripping `$`, commas, trailing `.` |
| yes/no | 6, 7 | first standalone yes/no in scope |
| label | 9–12 | first label mentioned; longer labels matched first (`not spam` beats `spam`); zero-shot scope is the first 2 lines so labels echoed deep in drifted text don't count |
| name | 5, 8 | gold name in the conclusion (last 3 lines zero-shot, first paragraph few-shot) so mid-reasoning mentions don't count |
| string | 13–15 | normalized containment of the gold string |
| code | 16–19 | **executed**: the generated function is called on 3 test cases; all must pass |

## Human overrides

Auto-extraction fails on some outputs (e.g. task 6 base answers "Whiskers is
an animal." — semantically a yes, but no yes/no token to extract). Those cells
are flagged `needs_review` and listed in RESULTS.md. To rule on one, add it to
`overrides.json` and re-run the pipeline:

```json
{
  "6": {
    "base_zero_shot": { "correct": true, "reason": "affirms the conclusion without saying yes" }
  }
}
```

Overrides are applied on top of auto-grades, counted, and listed in RESULTS.md
with their reasons — corrections are visible, never silent.

## Metric definitions

- **Accuracy** — final (post-override) correct / 20, with a Wilson 95% interval.
- **Drift** — anything after the first answer line matches self-generated task
  markers (`Answer:`, `Review:`, a question line, chat-template text). This is
  the "doesn't know when to stop" failure as a number.
- **Repetition** — mean fraction of duplicate non-empty lines per output.
- **Refusal** — refusal phrasing in the answer segment only, so refusals
  *quoted inside* fabricated examples don't false-positive.
- **Head-to-head** — for each condition pair, which tasks only one got right,
  with an exact two-sided McNemar/binomial p-value on the discordant set.

## Limitations

n = 20 tasks, one greedy sample per cell, so the confidence intervals are wide
and no pairwise comparison reaches significance — the numbers support the
*direction* of the claims in FINDINGS.md, not effect sizes. The task specs in
`grade.py` hardcode the frozen 20-task set; new tasks need a spec entry.
