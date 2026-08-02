after the first run these are the things i observed
Instruction tuning fixes the "doesn't know when to stop" problem — every single base zero-shot and few-shot completion runs past the answer into noise or repetition; instruct zero-shot almost never does.
Few-shot prompting is not uniformly good — it helps the base model terminate and often improves its accuracy, but it actively hurts the instruct model on this prompt set, which is a genuine, checkable claim you can make with these 20 examples as evidence.
Task 9's instruct zero-shot refusal ("I am unable to provide...") despite the review text being right there in the prompt is worth a specific callout — that's the alignment/refusal behavior InstructGPT introduces, showing up as an over-cautious failure mode rather than a helpful one.

on running the audit for instruct model in few-shot setting I observed this:
a) on task 4 and 5 the model genuinely fails even when we have 5 separate runs for it. it fails to answer correctly 5 times
b) on task 8 it does answer the question correctly 2/5 times, which shows that the greedy decoding just got unlucky there.
in task 5 and 8 i initially believed that the model confabulated the wrong names 'Frank' (in task 5) and 'Fay' (in task 8) and stated this somewhere too. however in both prompts few-shot examples, both these names appear. so it is more of a failure to understand the structure rather than pure hallucination.
c) on task 15 the model fails to answer correctly on all 5 attempts, and it is a very interesting one as it just combines the suffixes of the example emails in the examples with 'contact' or 'contactus'
so the instruct model works terribly with the few shot examples on the 4 tasks that I picked to test it on (based on few shot failing and zero shot succeeding on them)
it either answers wrong and keeps going on creating more sample problems itself (task 4), either gets all wrong (task 5), gets unlucky with greedy decoding (task 8) or creates answers from blending words from the prompt and few shot examples (task 15)