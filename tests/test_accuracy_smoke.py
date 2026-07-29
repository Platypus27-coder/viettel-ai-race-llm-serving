from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "benchmark" / "test_accuracy.py"
SPEC = importlib.util.spec_from_file_location("accuracy_smoke", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class AccuracySmokeTests(unittest.TestCase):
    def test_extract_choice_does_not_take_a_from_answer_label(self) -> None:
        self.assertEqual(MODULE.extract_choice("Answer: C"), "C")

    def test_extract_choice_accepts_common_single_choice_forms(self) -> None:
        self.assertEqual(MODULE.extract_choice("(b)"), "B")
        self.assertEqual(MODULE.extract_choice("The correct option is D."), "D")

    def test_extract_choice_rejects_letters_embedded_in_words(self) -> None:
        self.assertEqual(MODULE.extract_choice("because"), "")
        self.assertEqual(MODULE.extract_choice("ABCD"), "")


if __name__ == "__main__":
    unittest.main()
