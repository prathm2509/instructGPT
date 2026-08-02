this is an experiment to further find out where the quality of output decreases and why the models drift (basically give an answer and then continue with more text, mostly creating new examples themselves)
i have a 2 x 2 matrix for this:
condition a: same config as original experiment
condition b: add delimiters to clearly mark the start and end of the examples that are included in the prompt
condition c: add delimiters and explicit instructions stating that the examples are not to be used to copy names from etc etc. basically this: 
The items above are examples of the task format.
Solve the target question independently.
Do not copy names, numbers, or answers from the examples.
Return only the answer to the target question.
condition d: same as c but without the delimiters (to check whether the instructions alone work or are the delimiters necessary too)

conclusions to the experiment added to findings.md