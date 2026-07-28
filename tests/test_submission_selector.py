from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "select_submission.py"
SPEC = importlib.util.spec_from_file_location("select_submission", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

RECORD_SCRIPT = Path(__file__).parents[1] / "scripts" / "record_submission.py"
RECORD_SPEC = importlib.util.spec_from_file_location(
    "record_submission", RECORD_SCRIPT
)
assert RECORD_SPEC and RECORD_SPEC.loader
RECORD_MODULE = importlib.util.module_from_spec(RECORD_SPEC)
RECORD_SPEC.loader.exec_module(RECORD_MODULE)


class SubmissionSelectorTests(unittest.TestCase):
    def test_recovery_slot(self) -> None:
        rendered = MODULE.render_compose(3, "bf16", 4096, 64)
        self.assertNotIn("--quantization", rendered)
        self.assertIn("--max-num-batched-tokens=4096", rendered)
        self.assertIn("--max-num-seqs=64", rendered)

    def test_fp8_slot_does_not_enable_fp8_kv(self) -> None:
        rendered = MODULE.render_compose(5, "fp8", 4096, 64)
        self.assertIn("--quantization=fp8", rendered)
        self.assertNotIn("--kv-cache-dtype", rendered)
        self.assertNotIn("--calculate-kv-scales", rendered)

    def test_slots_three_and_four_are_fixed_bf16_candidates(self) -> None:
        slot3 = MODULE.resolve_configuration(
            MODULE.build_parser().parse_args(
                ["--slot", "3", "--output", "slot3.yml"]
            )
        )
        slot4 = MODULE.resolve_configuration(
            MODULE.build_parser().parse_args(
                ["--slot", "4", "--output", "slot4.yml"]
            )
        )
        self.assertEqual(slot3, ("bf16", 4096, 64))
        self.assertEqual(slot4, ("bf16", 2048, 64))

    def test_fp8_requires_accuracy_and_complete_workload(self) -> None:
        parser = MODULE.build_parser()
        valid = parser.parse_args(
            [
                "--slot",
                "5",
                "--variant",
                "fp8",
                "--batch-tokens",
                "4096",
                "--accuracy",
                "0.32",
                "--successful-requests",
                "420",
                "--output",
                "slot5.yml",
            ]
        )
        self.assertEqual(
            MODULE.resolve_configuration(valid), ("fp8", 4096, 64)
        )

        invalid = parser.parse_args(
            [
                "--slot",
                "5",
                "--variant",
                "fp8",
                "--batch-tokens",
                "4096",
                "--accuracy",
                "0.31",
                "--successful-requests",
                "420",
                "--output",
                "slot5.yml",
            ]
        )
        with self.assertRaisesRegex(ValueError, "accuracy >= 0.32"):
            MODULE.resolve_configuration(invalid)

    def test_slot_five_bf16_fallback_uses_48_sequences(self) -> None:
        args = MODULE.build_parser().parse_args(
            [
                "--slot",
                "5",
                "--variant",
                "seqs48",
                "--batch-tokens",
                "2048",
                "--output",
                "slot5.yml",
            ]
        )
        self.assertEqual(
            MODULE.resolve_configuration(args), ("bf16", 2048, 48)
        )

    def test_selection_policy_uses_accuracy_inside_ers_tie(self) -> None:
        records = [
            {
                "slot": 1,
                "ers": 0.900,
                "accuracy": 0.36,
                "p95_ttft_ms": 20,
                "successful_requests": 420,
            },
            {
                "slot": 2,
                "ers": 0.905,
                "accuracy": 0.33,
                "p95_ttft_ms": 10,
                "successful_requests": 420,
            },
        ]
        best = RECORD_MODULE.choose_best(records)
        self.assertIsNotNone(best)
        self.assertEqual(best["slot"], 1)

    def test_selection_rejects_low_accuracy(self) -> None:
        records = [
            {
                "slot": 1,
                "ers": 0.99,
                "accuracy": 0.31,
                "successful_requests": 420,
            },
            {
                "slot": 2,
                "ers": 0.80,
                "accuracy": 0.35,
                "successful_requests": 420,
            },
        ]
        best = RECORD_MODULE.choose_best(records)
        self.assertIsNotNone(best)
        self.assertEqual(best["slot"], 2)

    def test_selection_accepts_portal_confirmed_full_penalty_factor(self) -> None:
        records = [
            {
                "slot": 1,
                "ers": 60.02,
                "accuracy": None,
                "f_delta": 1,
                "successful_requests": 414,
                "portal_valid": True,
            },
            {
                "slot": 2,
                "ers": 48.2,
                "accuracy": 0.4,
                "successful_requests": 420,
                "portal_valid": True,
            },
        ]
        best = RECORD_MODULE.choose_best(records)
        self.assertIsNotNone(best)
        self.assertEqual(best["slot"], 1)

    def test_selection_handles_mixed_ers_scales(self) -> None:
        records = [
            {
                "slot": 1,
                "ers": 60.02,
                "accuracy": 0.4,
                "portal_valid": True,
            },
            {
                "slot": 2,
                "ers": 0.61,
                "accuracy": 0.4,
                "portal_valid": True,
            },
        ]
        best = RECORD_MODULE.choose_best(records)
        self.assertIsNotNone(best)
        self.assertEqual(best["slot"], 2)


if __name__ == "__main__":
    unittest.main()
