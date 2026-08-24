"""Shared utilities: deterministic seed policy, JSONL IO, hashing, timestamps.

Seed policy (frozen, documented in benchmark_manifest.json):
    seed = sha256(f"{base_seed}|{benchmark_version}|{model_id}|{condition}|{problem_id}|{sample_index}")
    -> first 8 hex digits as a 32-bit integer.

Every generation gets its own deterministic seed, so rerunning a single
(problem, model, condition, sample_index) cell reproduces it exactly without
rerunning anything else. Python's built-in hash() is never used (not stable
across processes).
"""

import hashlib
import json
from datetime import datetime, timezone


def sample_seed(base_seed, benchmark_version, model_id, condition, problem_id, sample_index):
    key = f"{base_seed}|{benchmark_version}|{model_id}|{condition}|{problem_id}|{sample_index}"
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def append_jsonl(path, record):
    """Append one record and flush, so a crash loses at most the current row."""
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


def read_jsonl(path):
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
