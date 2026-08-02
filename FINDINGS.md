# Findings: base versus instruction-tuned prompting

## Scope

This experiment compares `Qwen/Qwen2.5-0.5B` with
`Qwen/Qwen2.5-0.5B-Instruct` on 20 frozen tasks under zero-shot and few-shot
prompting. It is an experiment about observable differences between a base
model and an instruction-tuned model; it is not a reproduction of InstructGPT
or a way to isolate SFT, reward modelling, and PPO separately.

## Findings from the first run

1. **Instruction tuning substantially reduced output drift in this sample.**
   Drift was measured as continuation into self-generated questions, examples,
   or template text:

   ```text
   Base zero-shot:       10%
   Base few-shot:        80%
   Instruct zero-shot:    0%
   Instruct few-shot:     5%
   ```

   Therefore, the accurate claim is not that every base completion drifted. The
   base few-shot condition drifted frequently, while the instruct conditions
   almost never did under this evaluation's drift heuristic.

2. **Few-shot prompting had different effects on the two models.**

   ```text
   Base accuracy:       55% zero-shot -> 65% few-shot
   Instruct accuracy:   70% zero-shot -> 60% few-shot
   ```

   Few-shot prompting improved the base model's accuracy in this sample but
   also increased its drift from 10% to 80%. It reduced the instruct model's
   accuracy on these 20 tasks. These are descriptive results, not reliable
   estimates of general performance: the confidence intervals are wide and the
   exact paired comparisons were not statistically significant (`p = 0.625` for
   both zero-shot versus few-shot comparisons).

3. **Task 9 shows an over-refusal in the instruction-tuned model.** The prompt
   explicitly contains the review text, but the instruct model says that the
   statement was not provided. This is a useful example of an instruction-tuned
   model being less helpful than the base model on a simple classification task.
   Because the model is Qwen2.5-Instruct rather than InstructGPT, this should not
   be described as proof that InstructGPT or RLHF caused the refusal.

## Few-shot failure audit

The follow-up audit sampled the four tasks where instruct few-shot was wrong but
instruct zero-shot was correct. It used five samples per task at temperature
0.3.

1. **Task 4:** 0/5 samples were correct. The model repeatedly answered with an
   incorrect number and often continued by creating additional word problems.
   This is evidence of a stable failure under this prompt and decoding setup.

2. **Task 5:** 0/5 samples were correct. The model repeatedly answered
   `Frank`, a name appearing in a few-shot example, instead of `Carol`. This is
   better described as example interference or structural failure than as a
   completely unsupported hallucination.

3. **Task 8:** 2/5 samples contained the correct final answer, `Amy`. This is
   consistent with the original greedy answer being sensitive to decoding, but
   the sampled explanations were sometimes internally confused. Final-answer
   accuracy alone does not establish that the model understood the logic.

4. **Task 15:** 0/5 samples were correct. The wrong email addresses combined
   pieces from the target and the demonstrations, such as `contactus@example.com`
   and `contact@site.net`. This is a clear example of few-shot example
   interference in this prompt.

These four tasks were selected after looking at the first-run results. They are
useful case studies, but they cannot establish how common these failures are.
Five samples also provide only weak evidence for a general claim about
stability.

## Limitations

- The evaluation has only 20 hand-designed tasks and one original generation
  per condition.
- The confidence intervals are wide; no headline accuracy comparison reaches
  statistical significance.
- The drift, repetition, and extraction measures are heuristics. Code tasks are
  executed, but several natural-language outputs require fixed extraction rules.
- The four-task audit is post-selected and therefore should be reported as a
  failure-case analysis, not as an unbiased estimate of failure frequency.
- Comparing a base model with an instruction-tuned model cannot identify which
  post-training component caused an observed behavior.

## Next experiment: few-shot boundary and instruction ablation

The next question is whether the few-shot failures result mainly from unclear
demonstration boundaries, from insufficient instruction to solve the target,
or from deeper reasoning interference.

Use the same model, target tasks, demonstrations, decoding settings, and output
budget in three conditions:

```text
A. Current few-shot prompt.

B. The same examples and target, but with explicit sections:
   Examples:
   ...
   End examples.

   Target:
   ...
   Answer:

C. Condition B plus:
   Solve only the target question.
   Do not copy names, numbers, or answers from the examples.
   Return only the target answer.
```

Run at least five, preferably ten, samples per task at the same temperature.
Use the four audited tasks for a targeted pilot, then add a new held-out set if
the effect is large enough to justify a broader claim.

Measure accuracy, drift, repetition, and whether the output copies an entity or
value from the demonstrations.

```text
B improves over A:       demonstration boundaries are likely part of the problem.
C improves over B:       explicit instruction-following guidance is important.
Drift falls but accuracy does not:
                          the model learned to stop but not to solve the task.
Neither condition helps: deeper semantic interference or reasoning failure is
                          more likely.
```

## Ablation results (executed 2026-08-03)

The four-condition ablation ran on the Instruct model with greedy decoding on 10
tasks (the four audited failures plus six held-out tasks). Outputs are in
`few_shot_ablation.json`; automatic grades are in
`few_shot_ablation_metrics.json`.

### Accuracy

```text
A (original few-shot)                     4/10
B (delimiters)                            6/10
C (delimiters + instruction)              6/10
D (instruction only)                      6/10
```

A is a grader undercount: task 2 answered the equation correctly but the
first-number extraction rule hit a `1` inside the first paragraph, so treat the
real A score as 5/10 (tasks 1, 2, 6, 13, 16). The honest picture is then:

```text
A (original)                              5/10
B / C / D                                 6/10
```

No condition reached a ceiling; all four conditions failed on the same two
tasks (task 4 arithmetic, and task 2 under the scoped rules), which signals a
remaining difficulty floor.

### What changed, task by task

- Task 5 (relational logic, "shortest"): A wrong (`Frank`), B/C/D all correct
  (`Carol`). Clean win for separating examples from the target.
- Task 9 (sentiment): A wrong (`neutral`), B/C/D all correct (`negative`).
- Task 15 (email extraction): A and B wrong (copied/blended), C and D correct.
- Task 8 (line-order logic): B correct (`Amy`), A/C/D wrong. A alone does not
  fully explain this; B helps, but the improvement is not robust to adding an
  instruction.
- Task 16 (code): A is the only pass. B/C/D drift into prose or keep generating
  after the function, so code generation does not benefit from any of the new
  formats.
- Task 4 (arithmetic): stable failure across A/B/C/D; no format fixed it.

### Conclusion

The delimiters and instructions help selectively, not universally. The clearest
win is on tasks where the target and examples share the same words: separating
the sections fixed task 5 and task 9, and the explicit instruction fixed the
email-extraction task 15. The added instruction was not necessary for those
cases; the delimiters alone (B) already recovered task 5 and task 9.

The new formats did not help everywhere. Task 8 regressed when the instruction
was added on top of delimiters (C), and tasks 2, 4, and 16 still failed under
B/C/D. The instruction's "do not copy" warning helped the email task but did not
fix the deeper arithmetic and code-generation problems.

So the ablation supports a narrow conclusion: the original few-shot failures
are partly a prompt-structure problem, and the best fixes are either delimiters
alone or delimiters plus an instruction, depending on the task. It does not
support a general claim that explicit instructions fix few-shot prompting. The
remaining failures on tasks 2, 4, and 16 are not fixed by any prompt variant
tested here.

A useful next step is to sample each condition (10 draws at temperature 0.3)
before drawing conclusions, because the current run is one greedy draw per
cell.
