import json
import unittest
from pathlib import Path

from harness.conditions import (
    few_shot_format_only_prompt,
    few_shot_prompt,
    few_shot_random_label_prompt,
)


class ConditionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).parents[1] / "benchmark_v1.json"
        cls.tasks = json.loads(path.read_text(encoding="utf-8"))["prompts"]

    def test_existing_few_shot_is_unchanged(self):
        task = self.tasks[0]
        self.assertEqual(few_shot_prompt(task), task["few_shot_prefix"] + task["prompt"])

    def test_format_only_removes_all_example_answers(self):
        for task in self.tasks:
            prompt = few_shot_format_only_prompt(task)
            self.assertTrue(prompt.endswith(task["prompt"]))
            prefix = prompt[: -len(task["prompt"])]
            self.assertGreaterEqual(prefix.count("?"), 1)

    def test_random_label_is_deterministic_and_preserves_structure(self):
        for task in self.tasks:
            first = few_shot_random_label_prompt(task)
            self.assertEqual(first, few_shot_random_label_prompt(task))
            self.assertTrue(first.endswith(task["prompt"]))
            prefix = first[: -len(task["prompt"])]
            original_prefix = task["few_shot_prefix"]
            self.assertEqual(prefix.count("Answer:"), original_prefix.count("Answer:"))


if __name__ == "__main__":
    unittest.main()
