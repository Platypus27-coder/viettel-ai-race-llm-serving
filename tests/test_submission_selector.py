from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
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
CUSTOM_IMAGE = f"exampleuser/viettel-ai-vllm@{CUSTOM_DIGEST}"


def benchmark_summary(
    speculative_decoding: dict[str, object] | None = None,
) -> dict[str, object]:
    summary: dict[str, object] = {
        "expected_requests": 420,
        "successful_requests": 420,
        "failed_requests": 0,
    }
    if speculative_decoding is not None:
        summary["speculative_decoding"] = speculative_decoding
    return summary


def measured_speculative_evidence(mean_acceptance_length: float = 3.6) -> dict[str, object]:
    return {
        "available": True,
        "counter_scope": "benchmark_delta",
        "acceptance_status": "measured",
        "counter_reset_detected": False,
        "counters": {"num_drafts": 100.0},
        "mean_acceptance_length": mean_acceptance_length,
    }


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
        "metrics": {"summary": benchmark_summary()},
        "portal_valid": True,
        "healthcheck_passed": True,
        "preflight_successful_requests": 420,
        "preflight_expected_requests": 420,
    }
    record.update(overrides)
    if candidate in RECORDER.CUSTOM_IMAGE_CANDIDATES:
        record.setdefault("image_reference", CUSTOM_IMAGE)
        record.setdefault("image_digest", CUSTOM_DIGEST)
    if candidate in RECORDER.SPECULATIVE_DRAFT_CANDIDATES:
        record.setdefault("compose_sha256", "d" * 64)
        record.setdefault(
            "run_manifest",
            {
                "summary": {
                    "repository_commit": "b" * 40,
                    "profile": "speculative-draft-v6-fp8-smoke",
                    "portal_candidate": {
                        "candidate": candidate,
                        "image_reference": record["image_reference"],
                        "image_digest": record["image_digest"],
                        "compose_sha256": record["compose_sha256"],
                        "source_equivalent_preflight": True,
                    },
                    "workload": {
                        "expected_requests": 420,
                        "seed": 42,
                        "request_rate": "inf",
                        "output_tokens": 300,
                        "trace_sha256": "c" * 64,
                    },
                }
            },
        )
    if "gpqa" not in overrides:
        record["gpqa"] = {
            "summary": {
                "task": "gpqa_diamond",
                "metrics": {"acc,none": float(record["accuracy"])},
            }
        }
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
        self.assertIn("# Parent candidate: v6-incumbent", rendered)
        self.assertIn("--speculative-config=", rendered)
        self.assertIn('"method":"draft_model"', rendered)
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

    def test_speculative_scheduler_children_inherit_the_pinned_parent(self) -> None:
        parent = SELECTOR.render_compose(
            "speculative-draft", V6_COMPOSE, CUSTOM_IMAGE
        )
        parent_arguments = SELECTOR._command_arguments(parent)
        for name, budget in (
            ("speculative-draft-batch1536", 1536),
            ("speculative-draft-batch1024", 1024),
        ):
            with self.subTest(candidate=name):
                rendered = SELECTOR.render_compose(name, parent)
                rendered_arguments = SELECTOR._command_arguments(rendered)
                self.assertIn(f"# Controlled candidate: {name}", rendered)
                self.assertIn("# Parent candidate: speculative-draft", rendered)
                self.assertIn(
                    f"# Parent Compose SHA-256: {SELECTOR.sha256_text(parent)}",
                    rendered,
                )
                self.assertEqual(rendered.count("# Controlled candidate:"), 1)
                self.assertIn(f"image: {CUSTOM_IMAGE}", rendered)
                self.assertIn(SELECTOR.SPECULATIVE_DRAFT_ARGUMENT, rendered_arguments)
                self.assertEqual(
                    set(rendered_arguments) - set(parent_arguments),
                    {f"--max-num-batched-tokens={budget}"},
                )
                self.assertEqual(set(parent_arguments) - set(rendered_arguments), set())

    def test_speculative_scheduler_children_reject_non_speculative_or_unpinned_parent(
        self,
    ) -> None:
        for name in (
            "speculative-draft-batch1536",
            "speculative-draft-batch1024",
        ):
            with self.subTest(candidate=name, parent="v6"):
                with self.assertRaisesRegex(ValueError, "renderer-produced"):
                    SELECTOR.render_compose(name, V6_COMPOSE)

        parent = SELECTOR.render_compose(
            "speculative-draft", V6_COMPOSE, CUSTOM_IMAGE
        )
        unpinned = parent.replace(
            f"image: {CUSTOM_IMAGE}", "image: vllm/vllm-openai:v0.22.1"
        )
        with self.assertRaisesRegex(ValueError, "Docker Hub"):
            SELECTOR.render_compose("speculative-draft-batch1536", unpinned)

    def test_speculative_scheduler_children_reject_mutated_parent_or_image_override(
        self,
    ) -> None:
        parent = SELECTOR.render_compose(
            "speculative-draft", V6_COMPOSE, CUSTOM_IMAGE
        )
        mutated_config = parent.replace('"num_speculative_tokens":4', '"num_speculative_tokens":5')
        with self.assertRaisesRegex(ValueError, "approved 4-token"):
            SELECTOR.render_compose("speculative-draft-batch1536", mutated_config)
        with self.assertRaisesRegex(ValueError, "inherit the image"):
            SELECTOR.render_compose(
                "speculative-draft-batch1536", parent, CUSTOM_IMAGE
            )

    def test_legacy_batch_candidates_cannot_silently_use_speculative_parent(self) -> None:
        parent = SELECTOR.render_compose(
            "speculative-draft", V6_COMPOSE, CUSTOM_IMAGE
        )
        with self.assertRaisesRegex(ValueError, "not the v6 incumbent"):
            SELECTOR.render_compose("batch1536", parent)

    def test_custom_candidate_requires_an_immutable_digest(self) -> None:
        with self.assertRaisesRegex(ValueError, "pinned by digest"):
            SELECTOR.render_compose("shortconv-fp8", V6_COMPOSE)
        with self.assertRaisesRegex(ValueError, "Docker Hub"):
            SELECTOR.render_compose(
                "shortconv-fp8", V6_COMPOSE, "registry.example/lfm:latest"
            )

    def test_custom_candidate_rejects_non_docker_hub_or_malformed_reference(self) -> None:
        for reference in (
            f"ghcr.io/example/viettel-ai-vllm@{CUSTOM_DIGEST}",
            f"registry.example/viettel-ai-vllm@{CUSTOM_DIGEST}",
            f"exampleuser/viettel-ai-vllm@{CUSTOM_DIGEST}\n    command: bad",
        ):
            with self.subTest(reference=reference), self.assertRaisesRegex(
                ValueError, "Docker Hub"
            ):
                SELECTOR.render_compose("shortconv-fp8", V6_COMPOSE, reference)

        rendered = SELECTOR.render_compose(
            "shortconv-fp8", V6_COMPOSE, f"docker.io/{CUSTOM_IMAGE}"
        )
        self.assertIn(f"image: docker.io/{CUSTOM_IMAGE}", rendered)

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

        artifact_failure = challenger(
            ers=75.0,
            metrics={
                "summary": {
                    **benchmark_summary(),
                    "failed_requests": 1,
                    "successful_requests": 419,
                }
            },
        )
        best = RECORDER.choose_best([incumbent(), artifact_failure])
        self.assertIsNotNone(best)
        self.assertEqual(best["candidate"], "v6-incumbent")

    def test_challenger_uses_gpqa_artifact_not_manual_accuracy_claim(self) -> None:
        bad_gpqa = challenger(
            ers=75.0,
            accuracy=0.99,
            gpqa={
                "summary": {
                    "task": "gpqa_diamond",
                    "metrics": {"acc,none": 0.20},
                }
            },
        )
        best = RECORDER.choose_best([incumbent(), bad_gpqa])
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
        self.assertEqual(best["candidate"], "v6-incumbent")

        speculative["metrics"] = {
            "summary": benchmark_summary(measured_speculative_evidence())
        }
        best = RECORDER.choose_best([incumbent(), speculative])
        self.assertIsNotNone(best)
        self.assertEqual(best["candidate"], "speculative-draft")

        speculative.pop("run_manifest")
        best = RECORDER.choose_best([incumbent(), speculative])
        self.assertIsNotNone(best)
        self.assertEqual(best["candidate"], "v6-incumbent")

    def test_speculative_candidate_rejects_weak_or_reset_acceptance_evidence(self) -> None:
        speculative = challenger(
            "speculative-draft",
            75.0,
            greedy_comparison={"summary": {"matches_expected": True}},
        )
        for evidence in (
            measured_speculative_evidence(3.49),
            {
                **measured_speculative_evidence(),
                "counter_reset_detected": True,
            },
            {
                **measured_speculative_evidence(),
                "acceptance_status": "no_drafts_observed",
            },
        ):
            speculative["metrics"] = {"summary": benchmark_summary(evidence)}
            with self.subTest(evidence=evidence):
                best = RECORDER.choose_best([incumbent(), speculative])
                self.assertIsNotNone(best)
                self.assertEqual(best["candidate"], "v6-incumbent")

    def test_accuracy_breaks_ties_between_valid_challengers(self) -> None:
        lower_accuracy = challenger("batch1536", 63.0, accuracy=0.33)
        higher_accuracy = challenger("batch1024", 63.005, accuracy=0.36)
        best = RECORDER.choose_best([lower_accuracy, higher_accuracy])
        self.assertIsNotNone(best)
        self.assertEqual(best["candidate"], "batch1024")

    def test_accuracy_breaks_a_near_tie_against_the_incumbent(self) -> None:
        baseline = incumbent()
        baseline["gpqa"] = {
            "summary": {
                "task": "gpqa_diamond",
                "metrics": {"acc,none": 0.40},
            }
        }
        lower_accuracy = challenger("shortconv-fp8", 61.415, accuracy=0.32)
        best = RECORDER.choose_best([baseline, lower_accuracy])
        self.assertIsNotNone(best)
        self.assertEqual(best["candidate"], "v6-incumbent")

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
                    "failed_requests": 0,
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

    def test_speculative_scheduler_record_requires_bound_preflight_manifest(self) -> None:
        parent = SELECTOR.render_compose("speculative-draft", V6_COMPOSE, CUSTOM_IMAGE)
        child = SELECTOR.render_compose("speculative-draft-batch1536", parent)
        metrics_data = {
            "runs": [
                {
                    "expected_requests": 420,
                    "successful_requests": 420,
                    "failed_requests": 0,
                    "speculative_decoding": measured_speculative_evidence(),
                }
            ]
        }
        gpqa_data = {"results": {"gpqa_diamond": {"acc,none": 0.36}}}
        greedy_data = {"matches_expected": True, "responses": []}
        resolved_data = {"max_num_seqs": 256, "chunked_prefill": True}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            compose = root / "candidate.yml"
            metrics = root / "metrics.json"
            gpqa = root / "gpqa.json"
            greedy = root / "greedy.json"
            resolved = root / "resolved.json"
            startup_log = root / "vllm.log"
            run_manifest = root / "run_manifest.json"
            compose.write_text(child, encoding="utf-8")
            metrics.write_text(json.dumps(metrics_data), encoding="utf-8")
            gpqa.write_text(json.dumps(gpqa_data), encoding="utf-8")
            greedy.write_text(json.dumps(greedy_data), encoding="utf-8")
            resolved.write_text(json.dumps(resolved_data), encoding="utf-8")
            startup_log.write_text("resolved settings", encoding="utf-8")
            manifest_data = {
                "repository_commit": "b" * 40,
                "profile": "speculative-draft-v6-fp8-smoke",
                "portal_candidate": {
                    "candidate": "speculative-draft-batch1536",
                    "image_reference": CUSTOM_IMAGE,
                    "image_digest": CUSTOM_DIGEST,
                    "compose_sha256": RECORDER.sha256(compose),
                    "source_equivalent_preflight": True,
                },
                "workload": {
                    "expected_requests": 420,
                    "seed": 42,
                    "request_rate": "inf",
                    "output_tokens": 300,
                    "trace_sha256": "c" * 64,
                },
                "artifact_sha256": {
                    "metrics": RECORDER.sha256(metrics),
                    "gpqa": RECORDER.sha256(gpqa),
                    "greedy_comparison": RECORDER.sha256(greedy),
                    "resolved_vllm_config": RECORDER.sha256(resolved),
                    "startup_log": RECORDER.sha256(startup_log),
                },
            }
            run_manifest.write_text(json.dumps(manifest_data), encoding="utf-8")
            args = RECORDER.build_parser().parse_args(
                [
                    "--candidate", "speculative-draft-batch1536",
                    "--compose", str(compose),
                    "--metrics", str(metrics),
                    "--gpqa", str(gpqa),
                    "--greedy-comparison", str(greedy),
                    "--resolved-vllm-config", str(resolved),
                    "--startup-log", str(startup_log),
                    "--run-manifest", str(run_manifest),
                    "--ers", "75",
                    "--healthcheck-passed",
                    "--preflight-successful-requests", "420",
                ]
            )
            record = RECORDER.build_record(args)
            self.assertEqual(record["candidate"], "speculative-draft-batch1536")
            self.assertEqual(
                record["run_manifest"]["summary"]["portal_candidate"]["image_digest"],
                CUSTOM_DIGEST,
            )

            manifest_data["portal_candidate"]["compose_sha256"] = "0" * 64
            run_manifest.write_text(json.dumps(manifest_data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Compose SHA-256"):
                RECORDER.build_record(args)

    def test_manifest_record_rejects_non_docker_hub_custom_image(self) -> None:
        compose = Path(__file__).parents[1] / "docker-compose.yml"
        args = RECORDER.build_parser().parse_args(
            [
                "--candidate",
                "shortconv-fp8",
                "--compose",
                str(compose),
                "--image-reference",
                f"ghcr.io/example/viettel-ai-vllm@{CUSTOM_DIGEST}",
                "--image-digest",
                CUSTOM_DIGEST,
                "--ers",
                "62.0",
            ]
        )
        with self.assertRaisesRegex(ValueError, "Docker Hub"):
            RECORDER.build_record(args)


if __name__ == "__main__":
    unittest.main()
