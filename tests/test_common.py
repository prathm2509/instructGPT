"""Tests for harness.common JSONL IO, in particular the truncated-final-line
tolerance that keeps resume + grading alive after a hard kill (Colab disconnect).

No torch / GPU needed.
"""

import unittest
from pathlib import Path

from harness.common import append_jsonl, read_jsonl

TMP_BASE = Path(__file__).resolve().parent.parent / ".test-tmp"


class JsonlIoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        TMP_BASE.mkdir(exist_ok=True)

    def test_roundtrip(self):
        path = TMP_BASE / "jsonl_roundtrip.jsonl"
        append_jsonl(path, {"a": 1, "b": "x"})
        append_jsonl(path, {"a": 2, "b": "y"})
        self.assertEqual([r["a"] for r in read_jsonl(path)], [1, 2])
        path.unlink()

    def test_truncated_final_line_is_skipped(self):
        path = TMP_BASE / "jsonl_truncated.jsonl"
        append_jsonl(path, {"a": 1})
        with path.open("a", encoding="utf-8") as handle:
            handle.write('{"a": 2, "partial")')  # hard kill mid-write
        records = read_jsonl(path)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["a"], 1)
        path.unlink()

    def test_blank_lines_ignored(self):
        path = TMP_BASE / "jsonl_blank.jsonl"
        path.write_text('{"a": 1}\n\n{"a": 2}\n', encoding="utf-8")
        self.assertEqual(len(read_jsonl(path)), 2)
        path.unlink()


if __name__ == "__main__":
    unittest.main()
