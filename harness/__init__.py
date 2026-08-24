"""General evaluation harness for the frozen benchmark (benchmark_v1).

Replaces notebook state as the source of truth for generation: model IDs,
raw-vs-chat mode, n-sampling, temperature, and seed policy are CLI arguments,
and every generation is appended to generations.jsonl with full metadata.

Subcommands:
    python -m harness.run   ...   # generate -> runs/<run-id>/generations.jsonl
    python -m harness.grade ...   # grade    -> predictions.jsonl, metrics.json
"""
