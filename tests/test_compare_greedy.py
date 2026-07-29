from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "benchmark" / "compare_greedy.py"
SPEC = importlib.util.spec_from_file_location("compare_greedy", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class CompareGreedyTests(unittest.TestCase):
    def test_compare_accepts_identical_greedy_signatures(self) -> None:
        response = {
            "ok": True,
            "signature": {
                "content": "ready.",
                "finish_reason": "stop",
                "completion_tokens": 3,
            },
        }
        self.assertEqual(MODULE.compare_responses([response], [response]), [])

    def test_compare_reports_changed_output(self) -> None:
        expected = [{"ok": True, "signature": {"content": "A"}}]
        actual = [{"ok": True, "signature": {"content": "B"}}]
        mismatches = MODULE.compare_responses(expected, actual)
        self.assertEqual(len(mismatches), 1)
        self.assertIn("prompt 1", mismatches[0])

    def test_fingerprint_changes_with_token_budget(self) -> None:
        prompts = [[{"role": "user", "content": "hello"}]]
        self.assertNotEqual(
            MODULE.prompt_fingerprint(prompts, 16),
            MODULE.prompt_fingerprint(prompts, 32),
        )


if __name__ == "__main__":
    unittest.main()
