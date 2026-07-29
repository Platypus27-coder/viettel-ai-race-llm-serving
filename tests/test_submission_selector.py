from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


SCRIPT = Path(__file__).parents[1] / "scripts" / "select_submission.py"
SPEC = importlib.util.spec_from_file_location("select_submission", SCRIPT)
assert SPEC and SPEC.loader
SELECTOR = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SELECTOR
SPEC.loader.exec_module(SELECTOR)

RECORD_SCRIPT = Path(__file__).parents[1] / "scripts" / "record_submission.py"
RECORD_SPEC = importlib.util.spec_from_file_location(
    "record_submission", RECORD_SCRIPT
)
assert RECORD_SPEC and RECORD_SPEC.loader
RECORDER = importlib.util.module_from_spec(RECORD_SPEC)
sys.modules[RECORD_SPEC.name] = RECORDER
RECORD_SPEC.loader.exec_module(RECORDER)


V6_COMPOSE = """services:
  model:
    image: vllm/vllm-openai:v0.22.1
    command:
      - --model=/model
      - --served-model-name=LFM2.5-1.2B-Instruct
      - --max-model-len=8192
      - --gpu-memory-utilization=0.97
      - --quantization=fp8
      - --kv-cache-dtype=fp8_e4m3
      - --enable-prefix-caching
"""

CUSTOM_DIGEST = "sha256:" + "a" * 64
CUSTOM_IMAGE = f"registry.example/lfm-shortconv@{CUSTOM_DIGEST}"


def incumbent(ers: float = 61.41) -> dict[str, object]:
    return {
        "candidate": "v6-incumbent",
        "ers": ers,
        "f_delta": 1.0,
        "portal_valid": True,
    }


def challenger(
    candidate: str = "shortconv-fp8", ers: float = 62.0, **overrides: object
) -> dict[str, object]:
    record: dict[str, object] = {
        "candidate": candidate,
        "ers": ers,
        "accuracy": 0.36,
        "gpqa": {"summary": {"task": "gpqa_diamond"}},
        "portal_valid": True,
        "healthcheck_passed": True,
        "preflight_successful_requests": 420,
        "preflight_expected_requests": 420,
    }
    record.update(overrides)
    return record


class SubmissionSelectorTests(unittest.TestCase):
    def test_shortconv_candidate_changes_only_the_image(self) -> None:
        rendered = SELECTOR.render_compose(
            "shortconv-fp8", V6_COMPOSE, CUSTOM_IMAGE
        )
        self.assertIn("# Controlled candidate: shortconv-fp8", rendered)
        self.assertIn(f"image: {CUSTOM_IMAGE}", rendered)
        for argument in SELECTOR.V6_REQUIRED_ARGUMENTS:
            self.assertIn(argument, rendered)
        self.assertNotIn("--speculative-config", rendered)
        self.assertNotIn("--max-num-batched-tokens", rendered)
        self.assertNotIn("--max-num-seqs", rendered)

    def test_speculative_candidate_has_only_its_declared_vllm_flag(self) -> None:
        rendered = SELECTOR.render_compose(
            "speculative-draft", V6_COMPOSE, CUSTOM_IMAGE
        )
        self.assertIn("--speculative-config=", rendered)
        self.assertIn('"model":"/opt/draft/LFM2.5-350M"', rendered)
        self.assertIn('"num_speculative_tokens":4', rendered)
        self.assertIn('"draft_tensor_parallel_size":1', rendered)
        self.assertIn('"max_model_len":8192', rendered)
        self.assertNotIn("--max-num-batched-tokens", rendered)
        self.assertNotIn("--max-num-seqs", rendered)

    def test_batch_candidates_change_only_batch_token_budget(self) -> None:
        for name, budget in (("batch1536", 1536), ("batch1024", 1024)):
            with self.subTest(candidate=name):
                rendered = SELECTOR.render_compose(name, V6_COMPOSE)
                self.assertIn(f"--max-num-batched-tokens={budget}", rendered)
                self.assertNotIn("--max-num-seqs", rendered)
                self.assertNotIn("--speculative-config", rendered)

    def test_custom_candidate_requires_an_immutable_digest(self) -> None:
        with self.assertRaisesRegex(ValueError, "pinned by digest"):
            SELECTOR.render_compose("shortconv-fp8", V6_COMPOSE)
        with self.assertRaisesRegex(ValueError, "immutable"):
            SELECTOR.render_compose(
                "shortconv-fp8", V6_COMPOSE, "registry.example/lfm:latest"
            )

    def test_selector_rejects_a_source_with_uncontrolled_scheduler_flags(self) -> None:
        invalid = V6_COMPOSE + "      - --max-num-seqs=64\n"
        with self.assertRaisesRegex(ValueError, "scheduler/performance"):
            SELECTOR.render_compose("batch1536", invalid)

    def test_current_root_incumbent_is_a_valid_selector_source(self) -> None:
        source = (Path(__file__).parents[1] / "docker-compose.yml").read_text(
            encoding="utf-8"
        )
        rendered = SELECTOR.render_compose("batch1536", source)
        self.assertIn("# Controlled candidate: batch1536", rendered)
        self.assertIn("--max-num-batched-tokens=1536", rendered)

    def test_strict_gain_over_v6_is_required_to_replace_it(self) -> None:
        best = RECORDER.choose_best([incumbent(), challenger(ers=61.41)])
        self.assertIsNotNone(best)
        self.assertEqual(best["candidate"], "v6-incumbent")

        best = RECORDER.choose_best([incumbent(), challenger(ers=61.42)])
        self.assertIsNotNone(best)
        self.assertEqual(best["candidate"], "shortconv-fp8")

    def test_challenger_needs_health_and_420_request_preflight(self) -> None:
        failed_preflight = challenger(
            ers=75.0,
            healthcheck_passed=False,
            preflight_successful_requests=419,
        )
        best = RECORDER.choose_best([incumbent(), failed_preflight])
        self.assertIsNotNone(best)
        self.assertEqual(best["candidate"], "v6-incumbent")

    def test_speculative_candidate_needs_matching_greedy_artifact(self) -> None:
        speculative = challenger("speculative-draft", 75.0)
        best = RECORDER.choose_best([incumbent(), speculative])
        self.assertIsNotNone(best)
        self.assertEqual(best["candidate"], "v6-incumbent")

        speculative["greedy_comparison"] = {
            "summary": {"matches_expected": True}
        }
        best = RECORDER.choose_best([incumbent(), speculative])
        self.assertIsNotNone(best)
        self.assertEqual(best["candidate"], "speculative-draft")

    def test_accuracy_breaks_ties_between_valid_challengers(self) -> None:
        lower_accuracy = challenger("batch1536", 63.0, accuracy=0.33)
        higher_accuracy = challenger("batch1024", 63.005, accuracy=0.36)
        best = RECORDER.choose_best([lower_accuracy, higher_accuracy])
        self.assertIsNotNone(best)
        self.assertEqual(best["candidate"], "batch1024")

    def test_manifest_summary_can_read_a_controlled_comparison_report(self) -> None:
        summary = RECORDER._summary_from_metrics(
            {
                "results": [
                    {
                        "candidate": "batch1536",
                        "runs": [{"ers": 0.61, "successful_requests": 420}],
                    },
                    {
                        "candidate": "batch1024",
                        "runs": [{"ers": 0.62, "successful_requests": 420}],
                    },
                ]
            },
            "batch1024",
        )
        self.assertEqual(summary, {"ers": 0.62, "successful_requests": 420})

    def test_manifest_record_tracks_required_artifacts(self) -> None:
        compose = Path(__file__).parents[1] / "docker-compose.yml"
        metrics_data = {
            "runs": [
                {
                    "ers": 0.7,
                    "successful_requests": 420,
                    "expected_requests": 420,
                    "ttft_ms": {"p95": 40},
                    "tpot_ms": {"p95": 3},
                }
            ]
        }
        gpqa_data = {"results": {"gpqa_diamond": {"acc,none": 0.36}}}

        def fake_json_artifact(
            path: Path | None, label: str
        ) -> dict[str, object] | None:
            if path is None:
                return None
            data = {
                "metrics": metrics_data,
                "GPQA": gpqa_data,
                "resolved vLLM config": {"max_num_seqs": 256, "chunked_prefill": True},
            }[label]
            return {"path": f"{label}.json", "sha256": label, "data": data}

        args = RECORDER.build_parser().parse_args(
            [
                "--candidate",
                "shortconv-fp8",
                "--compose",
                str(compose),
                "--image-reference",
                CUSTOM_IMAGE,
                "--image-digest",
                CUSTOM_DIGEST,
                "--resolved-vllm-config",
                "resolved.json",
                "--metrics",
                "metrics.json",
                "--gpqa",
                "gpqa.json",
                "--startup-log",
                "vllm.log",
                "--ers",
                "62.0",
                "--accuracy",
                "0.36",
                "--healthcheck-passed",
                "--preflight-successful-requests",
                "420",
            ]
        )
        with (
            patch.object(RECORDER, "_json_artifact", side_effect=fake_json_artifact),
            patch.object(
                RECORDER,
                "_file_artifact",
                return_value={"path": "vllm.log", "sha256": "log-sha"},
            ),
        ):
            record = RECORDER.build_record(args)

        self.assertEqual(record["image_digest"], CUSTOM_DIGEST)
        self.assertEqual(record["compose_sha256"], RECORDER.sha256(compose))
        self.assertEqual(record["resolved_vllm_config"]["data"]["max_num_seqs"], 256)
        self.assertEqual(record["metrics"]["summary"]["successful_requests"], 420)
        self.assertEqual(record["gpqa"]["summary"]["task"], "gpqa_diamond")
        self.assertNotIn("data", record["metrics"])
        self.assertNotIn("data", record["gpqa"])
        self.assertEqual(record["startup_log"]["sha256"], "log-sha")


if __name__ == "__main__":
    unittest.main()
