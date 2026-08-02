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
