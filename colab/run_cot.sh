#!/usr/bin/env bash
# Zero-shot CoT vs direct baseline — Colab runner.
#
# Plan: research/2026-08-26-cot-experiment.md
# Frozen benchmark: benchmark_v1.json (sha256 a52e11f0...)
#
# Usage (from the instructGPT repo root, on a GPU Colab runtime):
#     bash colab/run_cot.sh                 # setup + smoke + FULL RUN + grade + download
#     bash colab/run_cot.sh --smoke-only    # stop after the 6-cell smoke check
#     bash colab/run_cot.sh --skip-download # keep only the zip/run dir, no browser download
#
# Resume safety: rerunning the full-run command continues into the SAME run dir
# (id locked by runs/.cot_run_id); harness.run skips cells whose
# (model, condition, problem, sample) record already exists, and
# colab/dedup_generations.py removes any exact duplicates before grading.
# A Colab disconnect therefore costs only the missing cells.
#
# Artifacts (all under runs/, which is gitignored):
#   runs/<ts>-cot-smoke/   smoke-test run (problems 1, 5, 9 x both models)
#   runs/<ts>-cot/         full run (50 x 2), config.json + generations.jsonl
#                          + predictions.json + metrics.json + parse_failures.json
set -euo pipefail
cd "$(dirname "$0")/.."

BENCH=benchmark_v1.json
BASE_MODEL="Qwen/Qwen2.5-1.5B:raw"
INSTRUCT_MODEL="Qwen/Qwen2.5-1.5B-Instruct:chat"
MAX_TOKENS=400
CONDITION=cot

SMOKE_ONLY=0
SKIP_DOWNLOAD=0
for arg in "$@"; do
    case "$arg" in
        --smoke-only) SMOKE_ONLY=1 ;;
        --skip-download) SKIP_DOWNLOAD=1 ;;
        *) echo "unknown flag: $arg (expected --smoke-only | --skip-download)"; exit 2 ;;
    esac
done

mkdir -p runs

echo "===================================================================="
echo " zero-shot CoT experiment — $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "===================================================================="

# 0. Setup (once per machine; marker lives in gitignored runs/).
if [[ ! -f runs/.cot_setup_done ]]; then
    bash colab/setup.sh
    touch runs/.cot_setup_done
fi

# 1. Smoke test: problems 1 (number), 5 (name), 9 (label) x both models.
SMOKE_ID="$(date +%Y%m%d-%H%M%S)-cot-smoke"
SMOKE_DIR="runs/$SMOKE_ID"
echo "== smoke test -> $SMOKE_DIR =="
python -m harness.run --benchmark "$BENCH" \
    --model "$BASE_MODEL" --model "$INSTRUCT_MODEL" \
    --condition "$CONDITION" --problem-ids 1,5,9 --n-samples 1 --greedy \
    --max-new-tokens "$MAX_TOKENS" --quantize --output-dir "$SMOKE_DIR"
python -m harness.grade --run-dir "$SMOKE_DIR"
echo "---- by-hand smoke checks ----"
python colab/inspect.py --run-dir "$SMOKE_DIR" --checklist

if [[ "$SMOKE_ONLY" == 1 ]]; then
    echo "smoke-only: full run skipped. Inspect the lines above, then rerun"
    echo "without --smoke-only to launch the ~2h full run."
    exit 0
fi

# 2. Full run (resume-safe; id locked by runs/.cot_run_id).
if [[ -f runs/.cot_run_id ]]; then
    RUN_DIR="runs/$(cat runs/.cot_run_id)"
    echo "== resuming full run -> $RUN_DIR =="
else
    RUN_ID="$(date +%Y%m%d-%H%M%S)-cot"
    RUN_DIR="runs/$RUN_ID"
    echo "$RUN_ID" > runs/.cot_run_id
    echo "== full run -> $RUN_DIR =="
fi
python -m harness.run --benchmark "$BENCH" \
    --model "$BASE_MODEL" --model "$INSTRUCT_MODEL" \
    --condition "$CONDITION" --problem-ids all --n-samples 1 --greedy \
    --max-new-tokens "$MAX_TOKENS" --quantize --output-dir "$RUN_DIR"

# 3. Resume-safety dedup, then grade.
python colab/dedup_generations.py --run-dir "$RUN_DIR"
python -m harness.grade --run-dir "$RUN_DIR"

echo "---- run summary ----"
python colab/inspect.py --run-dir "$RUN_DIR" --summary

echo "-> run dir: $RUN_DIR"
echo "-> comparisons and writeup inputs:"
echo "     python compare_runs.py --baseline runs/<direct-run-id> --treatment $RUN_DIR"

# 4. Download the run artifacts back to the laptop.
if [[ "$SKIP_DOWNLOAD" == 0 ]]; then
    python colab/export.py "$RUN_DIR"
fi
echo "done"