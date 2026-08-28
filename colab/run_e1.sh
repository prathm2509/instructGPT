#!/usr/bin/env bash
# E1 label-corruption ablation (icl-why) — Colab runner.
#
# Plan: icl-why/PLAN.md §E1 · protocol: research/experiments/2026-08-28-icl-e1-label-corruption.md
# Frozen benchmark: benchmark_v1.json (sha256 a52e11f0...)
#
# Tests H1 vs H2 (Min et al. 2022): does corrupting the labels in the few-shot
# prefix hurt? Three conditions x 50 problems x Qwen2.5-1.5B base, greedy.
# The `direct` arm reuses the existing graded run (deterministic seed policy
# makes it bit-identical), so only these three conditions need GPU time.
#
# Persistence: everything lands under E1_ROOT (default: Google Drive), so a
# Colab disconnect loses nothing — rerun the same command and harness.run
# skips cells already in generations.jsonl (per-condition run ids are locked
# by marker files on Drive).
#
# Usage (from the instructGPT repo root, on a GPU Colab runtime, Drive mounted):
#     bash colab/run_e1.sh                 # setup + smoke + 3 condition runs + grade + zip
#     bash colab/run_e1.sh --smoke-only    # stop after the 9-cell smoke check
#     bash colab/run_e1.sh --skip-zip      # skip the per-condition zip step
#
# Budget on a free T4 with 4-bit quantization: ~5.5 s/generation, so each
# condition is ~7-9 min including model load and grading; the full script is
# ~30 min. Quota fragmented? Run one session per condition — resume stitches.
set -euo pipefail
cd "$(dirname "$0")/.."

BENCH=benchmark_v1.json
BASE_MODEL="Qwen/Qwen2.5-1.5B:raw"
MAX_TOKENS=200
CONDITIONS=(few_shot few_shot_random_label few_shot_format_only)

E1_ROOT="${E1_ROOT:-/content/drive/MyDrive/icl-why/e1}"
# Presence test: for the Drive default, test the MOUNT POINT itself — anything
# deeper (icl-why/e1) may not exist until mkdir -p below creates it.
case "$E1_ROOT" in
    /content/drive/MyDrive | /content/drive/MyDrive/*) E1_TEST="/content/drive/MyDrive" ;;
    *) E1_TEST="$(dirname "$E1_ROOT")" ;;
esac

SMOKE_ONLY=0
SKIP_ZIP=0
for arg in "$@"; do
    case "$arg" in
        --smoke-only) SMOKE_ONLY=1 ;;
        --skip-zip) SKIP_ZIP=1 ;;
        *) echo "unknown flag: $arg (expected --smoke-only | --skip-zip)"; exit 2 ;;
    esac
done

mkdir -p runs

# Persistent root: Drive when mounted, local fallback otherwise (still
# resume-safe within the session, but wiped on runtime recycle).
if [[ -d "$E1_TEST" ]]; then
    mkdir -p "$E1_ROOT/runs" "$E1_ROOT/zips"
    export HF_HOME="${HF_HOME:-$E1_ROOT/hf-cache}"
    echo "persistent root: $E1_ROOT (hf cache: $HF_HOME)"
else
    E1_ROOT="runs/e1"
    mkdir -p "$E1_ROOT/runs" "$E1_ROOT/zips"
    echo "WARNING: $E1_TEST not found - is Drive mounted?"
    echo "         Falling back to LOCAL $E1_ROOT (not disconnect-safe)."
fi

echo "===================================================================="
echo " E1 label-corruption ablation — $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "===================================================================="

# 0. Setup: once per runtime (packages do not survive a recycled Colab VM).
if [[ ! -f runs/.e1_setup_done ]]; then
    bash colab/setup.sh
    touch runs/.e1_setup_done
fi

# 1. Smoke test: problems 1 (number), 5 (name), 9 (label) x all three
#    conditions, so every prompt builder is exercised end-to-end. One smoke
#    dir per condition (config.json records a single condition, and grading
#    scope keys off it). Smoke dirs are ephemeral by design — only the full
#    runs need to persist.
SMOKE_STAMP="$(date +%Y%m%d-%H%M%S)"
for condition in "${CONDITIONS[@]}"; do
    SMOKE_DIR="runs/${SMOKE_STAMP}-e1-smoke-$condition"
    echo "== smoke test ($condition) -> $SMOKE_DIR =="
    python -m harness.run --benchmark "$BENCH" \
        --model "$BASE_MODEL" \
        --condition "$condition" --problem-ids 1,5,9 --n-samples 1 --greedy \
        --max-new-tokens "$MAX_TOKENS" --quantize --output-dir "$SMOKE_DIR"
    python colab/dedup_generations.py --run-dir "$SMOKE_DIR"
    python -m harness.grade --run-dir "$SMOKE_DIR"
    echo "---- by-hand smoke checks ($condition) ----"
    python colab/inspect.py --run-dir "$SMOKE_DIR" --checklist
done

if [[ "$SMOKE_ONLY" == 1 ]]; then
    echo "smoke-only: condition runs skipped. Inspect the lines above, then"
    echo "rerun without --smoke-only (~7-9 min per condition)."
    exit 0
fi

# 2. Full runs: one run dir per condition, id locked by a marker on the
#    persistent root so a disconnect + rerun resumes into the same dir.
for condition in "${CONDITIONS[@]}"; do
    marker="$E1_ROOT/.e1_run_id_$condition"
    if [[ -f "$marker" ]]; then
        RUN_ID="$(cat "$marker")"
        echo "== resuming $condition -> $RUN_ID =="
    else
        RUN_ID="$(date +%Y%m%d-%H%M%S)-e1-$condition"
        echo "$RUN_ID" > "$marker"
        echo "== $condition -> $RUN_ID =="
    fi
    RUN_DIR="$E1_ROOT/runs/$RUN_ID"

    python -m harness.run --benchmark "$BENCH" \
        --model "$BASE_MODEL" \
        --condition "$condition" --problem-ids all --n-samples 1 --greedy \
        --max-new-tokens "$MAX_TOKENS" --quantize --output-dir "$RUN_DIR"

    # 3. Resume-safety dedup, grade, zip checkpoint.
    python colab/dedup_generations.py --run-dir "$RUN_DIR"
    python -m harness.grade --run-dir "$RUN_DIR"
    if [[ "$SKIP_ZIP" == 0 ]]; then
        python - "$RUN_DIR" "$E1_ROOT/zips" <<'PY'
import shutil
import sys
from pathlib import Path

run_dir, zips = Path(sys.argv[1]), Path(sys.argv[2])
zips.mkdir(parents=True, exist_ok=True)
stem = zips / run_dir.name
shutil.make_archive(str(stem), "zip", root_dir=run_dir.parent,
                    base_dir=run_dir.name)
print(f"zip: {stem}.zip ({stem.with_suffix('.zip').stat().st_size / 1e6:.1f} MB)")
PY
    fi
    echo "-> $condition done: $RUN_DIR"
done

echo "---- done ----"
echo "Artifacts: $E1_ROOT/runs/<id>-e1-<condition>/{config,generations,predictions,metrics}.json*"
echo "Zip backups: $E1_ROOT/zips/"
echo "Direct arm (reuse, do NOT regenerate): graded run 20260826-direct-q4"
echo "  (Qwen/Qwen2.5-1.5B base half), seeds are deterministic so it is"
echo "  bit-identical to a fresh direct run."
echo "Analysis (local, no GPU):"
echo "  python icl-why/e1_label_corruption/analyze_e1.py --run-dir <direct> <few_shot> <few_shot_random_label> <few_shot_format_only>"
