"""Record a reproducible portal result and recommend only validated winners."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


EXPECTED_WORKLOAD_REQUESTS = 420
CANONICAL_WORKLOAD_TRACE_SHA256 = (
    "09deb30b9a136403af819dee53531342a4bdb6d00bff16aaf9aa6a00cbd47e3c"
)
CANONICAL_WORKLOAD_TRACE = {
    "num_conversations": 70,
    "user_turns_per_conversation": 6,
    "shared_system_prefix_tokens": 1000,
    "per_conversation_prefix_tokens": 1000,
    "new_user_tokens_per_turn": 150,
    "output_tokens_per_turn_pinned": 300,
}
CANONICAL_WORKLOAD_ARRIVAL = {
    "kind": "poisson",
    "seed": 42,
    "request_rate": "inf",
}
CANONICAL_OUTPUT_TOKENS = 300
GPQA_DIAMOND_TASK = "gpqa_diamond"
OFFLINE_SERVING_ENV = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
MANIFEST_SCHEMA_VERSION = 4
MINIMUM_ACCURACY = 0.32
PREFERRED_ACCURACY = 0.35
SPECULATIVE_MIN_MEAN_ACCEPTANCE_LENGTH = 3.5
ACCURACY_ARTIFACT_TOLERANCE = 0.005
ERS_TIE_THRESHOLD = 0.01
INCUMBENT = "v6-incumbent"
CANDIDATES = (
    INCUMBENT,
    "shortconv-fp8",
    "speculative-draft",
    "speculative-draft-batch1536",
    "speculative-draft-batch1024",
    "batch1536",
    "batch1024",
)
SPECULATIVE_DRAFT_CANDIDATES = frozenset(
    {
        "speculative-draft",
        "speculative-draft-batch1536",
        "speculative-draft-batch1024",
    }
)
# Scheduler children retain the draft image and therefore must retain every
# provenance, equivalence, and measured-acceptance requirement of their
# speculative parent.  Keeping them explicit avoids silently treating a
# custom-image child as an ordinary v6 scheduler sweep.
CUSTOM_IMAGE_CANDIDATES = {"shortconv-fp8", *SPECULATIVE_DRAFT_CANDIDATES}
DIGEST = re.compile(r"^sha256:[0-9a-fA-F]{64}$")
# A custom contest image must be served publicly from Docker Hub.  This is a
# syntactic fail-closed check; the publishing workflow separately verifies an
# anonymous registry pull before a digest is considered for promotion.
DOCKER_HUB_DIGEST_IMAGE = re.compile(
    r"^(?:(?:docker\.io|index\.docker\.io)/)?"
    r"[a-z0-9](?:[a-z0-9_-]*[a-z0-9])?/"
    r"[a-z0-9](?:[a-z0-9._-]*[a-z0-9])?"
    r"(?::[A-Za-z0-9_][A-Za-z0-9_.-]*)?"
    r"@sha256:[0-9a-fA-F]{64}$"
)
IMAGE_LINE = re.compile(
    r"^[ \t]*image:[ \t]*(?P<value>.+?)[ \t]*$", re.MULTILINE
)
COMMIT_SHA = re.compile(r"^[0-9a-fA-F]{40}$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_ers(record: dict[str, Any]) -> float:
    """Return ERS on the official 0..1 scale from either portal format."""
    value = float(record["ers"])
    return value / 100.0 if value > 1.0 else value


def _artifact_summary(record: dict[str, Any], name: str) -> dict[str, Any] | None:
    artifact = record.get(name)
    if not isinstance(artifact, dict):
        return None
    summary = artifact.get("summary")
    return summary if isinstance(summary, dict) else None


def _gpqa_accuracy_from_summary(summary: dict[str, Any] | None) -> float | None:
    """Return the authoritative GPQA score from an lm-eval results artifact."""
    if not isinstance(summary, dict):
        return None
    if summary.get("task") != GPQA_DIAMOND_TASK:
        return None
    metrics = summary.get("metrics")
    if not isinstance(metrics, dict):
        return None
    value = metrics.get("acc,none")
    if value is None:
        # lm-eval task versions can use a different aggregation suffix.  Never
        # accept stderr fields as an accuracy value.
        for key, candidate in metrics.items():
            if str(key).startswith("acc,") and not str(key).endswith("_stderr"):
                value = candidate
                break
    try:
        accuracy = float(value)
    except (TypeError, ValueError):
        return None
    return accuracy if 0.0 <= accuracy <= 1.0 else None


def artifact_gpqa_accuracy(record: dict[str, Any]) -> float | None:
    return _gpqa_accuracy_from_summary(_artifact_summary(record, "gpqa"))


def _benchmark_summary(record: dict[str, Any]) -> dict[str, Any] | None:
    return _artifact_summary(record, "metrics")


def _is_exact_int(value: Any) -> bool:
    """Reject bools and numeric strings in evidence that must be exact counts."""
    return isinstance(value, int) and not isinstance(value, bool)


def _single_benchmark_run(data: Any, candidate: str | None) -> dict[str, Any]:
    """Return the one raw benchmark run bound to ``candidate``.

    Portal promotion only accepts the one-run artifact emitted by the repository
    benchmark.  A comparison report is accepted only when it has one matching
    candidate and that candidate has exactly one run.
    """
    if not isinstance(data, dict):
        raise ValueError("--metrics must be a benchmark JSON object")

    runs: Any
    comparisons = data.get("results")
    if isinstance(comparisons, list):
        matches = [
            item
            for item in comparisons
            if isinstance(item, dict) and item.get("candidate") == candidate
        ]
        if len(matches) != 1:
            raise ValueError(
                "--metrics must contain exactly one comparison result for the candidate"
            )
        runs = matches[0].get("runs")
    else:
        runs = data.get("runs")

    if not isinstance(runs, list) or len(runs) != 1 or not isinstance(runs[0], dict):
        raise ValueError("--metrics must contain exactly one completed raw benchmark run")
    return runs[0]


def _require_canonical_trace(trace: Any) -> None:
    if not isinstance(trace, dict):
        raise ValueError("--metrics raw run is missing its trace")
    for key, expected in CANONICAL_WORKLOAD_TRACE.items():
        if trace.get(key) != expected:
            raise ValueError(f"--metrics trace {key} is not the canonical workload")
    arrival = trace.get("arrival")
    if not isinstance(arrival, dict):
        raise ValueError("--metrics trace arrival is missing")
    for key, expected in CANONICAL_WORKLOAD_ARRIVAL.items():
        if arrival.get(key) != expected:
            raise ValueError(
                f"--metrics trace arrival {key} is not the canonical workload"
            )


def _canonical_workload_evidence(data: Any, candidate: str | None) -> dict[str, Any]:
    """Validate raw request evidence for the published 70x6 workload.

    Aggregate counters alone are insufficient: require all 420 unique
    conversation/turn records, successful completion, and exactly 300 output
    tokens for every record.  The returned compact marker is stored with the
    submission manifest after the original, hashed benchmark JSON is archived.
    """
    run = _single_benchmark_run(data, candidate)
    _require_canonical_trace(run.get("trace"))

    expected_counts = {
        "expected_requests": EXPECTED_WORKLOAD_REQUESTS,
        "observed_requests": EXPECTED_WORKLOAD_REQUESTS,
        "successful_requests": EXPECTED_WORKLOAD_REQUESTS,
        "failed_requests": 0,
    }
    for key, expected in expected_counts.items():
        value = run.get(key)
        if not _is_exact_int(value) or value != expected:
            raise ValueError(
                f"--metrics raw run {key} must equal {expected} for the canonical workload"
            )
    success_rate = run.get("success_rate")
    if (
        isinstance(success_rate, bool)
        or not isinstance(success_rate, (int, float))
        or float(success_rate) != 1.0
    ):
        raise ValueError("--metrics raw run success_rate must equal 1.0")

    failures = run.get("failures")
    if not isinstance(failures, list) or failures:
        raise ValueError("--metrics raw run failures must be an empty list")
    requests = run.get("requests")
    if not isinstance(requests, list) or len(requests) != EXPECTED_WORKLOAD_REQUESTS:
        raise ValueError("--metrics raw run must retain all 420 request records")

    expected_coordinates = {
        (conversation_id, turn)
        for conversation_id in range(CANONICAL_WORKLOAD_TRACE["num_conversations"])
        for turn in range(
            1, CANONICAL_WORKLOAD_TRACE["user_turns_per_conversation"] + 1
        )
    }
    observed_coordinates: set[tuple[int, int]] = set()
    for index, request in enumerate(requests):
        if not isinstance(request, dict):
            raise ValueError(f"--metrics request {index} must be an object")
        conversation_id = request.get("conversation_id")
        turn = request.get("turn")
        if not _is_exact_int(conversation_id) or not _is_exact_int(turn):
            raise ValueError(f"--metrics request {index} has invalid conversation_id/turn")
        coordinate = (conversation_id, turn)
        if coordinate not in expected_coordinates or coordinate in observed_coordinates:
            raise ValueError(
                f"--metrics request {index} has a duplicate or out-of-range conversation/turn"
            )
        observed_coordinates.add(coordinate)
        if request.get("success") is not True:
            raise ValueError(f"--metrics request {index} is not successful")
        output_tokens = request.get("output_tokens")
        if not _is_exact_int(output_tokens) or output_tokens != CANONICAL_OUTPUT_TOKENS:
            raise ValueError(
                f"--metrics request {index} must contain exactly "
                f"{CANONICAL_OUTPUT_TOKENS} output tokens"
            )
    if observed_coordinates != expected_coordinates:
        raise ValueError("--metrics raw request records do not cover the canonical 70x6 trace")

    request_fingerprint = hashlib.sha256(
        json.dumps(
            sorted(observed_coordinates), separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()
    raw_request_fingerprint = hashlib.sha256(
        json.dumps(requests, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
            "utf-8"
        )
    ).hexdigest()
    return {
        "trace_sha256": CANONICAL_WORKLOAD_TRACE_SHA256,
        "raw_run_count": 1,
        "request_count": EXPECTED_WORKLOAD_REQUESTS,
        "successful_request_count": EXPECTED_WORKLOAD_REQUESTS,
        "output_tokens_per_request": CANONICAL_OUTPUT_TOKENS,
        "request_coordinates_sha256": request_fingerprint,
        "raw_request_records_sha256": raw_request_fingerprint,
    }


def _passes_canonical_workload_evidence(evidence: Any) -> bool:
    if not isinstance(evidence, dict):
        return False
    return (
        evidence.get("trace_sha256") == CANONICAL_WORKLOAD_TRACE_SHA256
        and evidence.get("raw_run_count") == 1
        and evidence.get("request_count") == EXPECTED_WORKLOAD_REQUESTS
        and evidence.get("successful_request_count") == EXPECTED_WORKLOAD_REQUESTS
        and evidence.get("output_tokens_per_request") == CANONICAL_OUTPUT_TOKENS
        and isinstance(evidence.get("request_coordinates_sha256"), str)
        and re.fullmatch(r"[0-9a-f]{64}", evidence["request_coordinates_sha256"])
        is not None
        and isinstance(evidence.get("raw_request_records_sha256"), str)
        and re.fullmatch(r"[0-9a-f]{64}", evidence["raw_request_records_sha256"])
        is not None
    )


def _require_compact_hashes(hashes: Any, *names: str) -> bool:
    return isinstance(hashes, dict) and all(
        isinstance(hashes.get(name), str)
        and re.fullmatch(r"[0-9a-f]{64}", hashes[name]) is not None
        for name in names
    )


def passes_accuracy_gate(record: dict[str, Any]) -> bool:
    if candidate_name(record) != INCUMBENT:
        accuracy = artifact_gpqa_accuracy(record)
        return accuracy is not None and accuracy >= MINIMUM_ACCURACY
    accuracy = record.get("accuracy")
    if accuracy is not None:
        return float(accuracy) >= MINIMUM_ACCURACY
    f_delta = record.get("f_delta")
    return f_delta is not None and float(f_delta) == 1.0


def candidate_name(record: dict[str, Any]) -> str:
    """Infer the old slot-one record as the incumbent for manifest migration."""
    if record.get("candidate"):
        return str(record["candidate"])
    return INCUMBENT if int(record.get("slot", -1)) == 1 else "legacy"


def passes_preflight(record: dict[str, Any]) -> bool:
    """New candidates need a full, healthy 420-request preflight run."""
    if candidate_name(record) == INCUMBENT:
        return True
    if record.get("healthcheck_passed") is not True:
        return False
    expected = int(record.get("preflight_expected_requests") or EXPECTED_WORKLOAD_REQUESTS)
    successful = record.get("preflight_successful_requests")
    summary = _benchmark_summary(record)
    if not isinstance(summary, dict):
        return False
    try:
        artifact_expected = int(summary.get("expected_requests"))
        artifact_successful = int(summary.get("successful_requests"))
        artifact_failed = int(summary.get("failed_requests"))
    except (TypeError, ValueError):
        return False
    return (
        expected == EXPECTED_WORKLOAD_REQUESTS
        and successful == expected
        and artifact_expected == EXPECTED_WORKLOAD_REQUESTS
        and artifact_successful == artifact_expected
        and artifact_failed == 0
        and _passes_canonical_workload_evidence(summary.get("workload_evidence"))
    )


def passes_greedy_equivalence(record: dict[str, Any]) -> bool:
    """Speculative decoding must preserve the captured greedy parent output."""
    if candidate_name(record) not in SPECULATIVE_DRAFT_CANDIDATES:
        return True
    artifact = record.get("greedy_comparison")
    if not isinstance(artifact, dict):
        return False
    summary = artifact.get("summary")
    return isinstance(summary, dict) and summary.get("matches_expected") is True


def passes_speculative_evidence(record: dict[str, Any]) -> bool:
    """Require run-scoped acceptance evidence before selecting draft decoding.

    A server that exposed no counters, restarted them, or executed no draft
    tokens cannot demonstrate the mechanism needed for the 75-point attempt.
    The threshold is a preflight gate, not a claim that T4 latency predicts the
    H200 portal score.
    """
    if candidate_name(record) not in SPECULATIVE_DRAFT_CANDIDATES:
        return True
    benchmark = _benchmark_summary(record)
    if not isinstance(benchmark, dict):
        return False
    evidence = benchmark.get("speculative_decoding")
    if not isinstance(evidence, dict):
        return False
    if (
        evidence.get("available") is not True
        or evidence.get("counter_scope") != "benchmark_delta"
        or evidence.get("acceptance_status") != "measured"
        or evidence.get("counter_reset_detected") is not False
    ):
        return False
    counters = evidence.get("counters")
    if not isinstance(counters, dict):
        return False
    try:
        drafts = float(counters.get("num_drafts"))
        mean_acceptance_length = float(evidence.get("mean_acceptance_length"))
    except (TypeError, ValueError):
        return False
    return drafts > 0 and mean_acceptance_length >= SPECULATIVE_MIN_MEAN_ACCEPTANCE_LENGTH


def passes_speculative_run_manifest(record: dict[str, Any]) -> bool:
    """Require compact identity evidence even for a hand-edited manifest.

    ``build_record`` verifies the full hashed artifact. This second check keeps
    the selection layer fail-closed when it is given a pre-existing JSON
    manifest rather than a freshly recorded submission.
    """
    candidate = candidate_name(record)
    if candidate not in SPECULATIVE_DRAFT_CANDIDATES:
        return True
    artifact = record.get("run_manifest")
    if not isinstance(artifact, dict):
        return False
    summary = artifact.get("summary")
    if not isinstance(summary, dict):
        return False
    portal_candidate = summary.get("portal_candidate")
    workload = summary.get("workload")
    artifact_hashes = summary.get("artifact_sha256")
    if not isinstance(portal_candidate, dict) or not isinstance(workload, dict):
        return False
    if not isinstance(artifact_hashes, dict):
        return False
    if (
        portal_candidate.get("candidate") != candidate
        or portal_candidate.get("source_equivalent_preflight") is not True
        or portal_candidate.get("image_reference") != record.get("image_reference")
        or portal_candidate.get("image_digest") != record.get("image_digest")
        or portal_candidate.get("compose_sha256") != record.get("compose_sha256")
    ):
        return False
    repository_commit = summary.get("repository_commit")
    if not isinstance(repository_commit, str) or not COMMIT_SHA.fullmatch(repository_commit):
        return False
    return (
        isinstance(summary.get("profile"), str)
        and summary["profile"].startswith("speculative-draft")
        and summary.get("offline_serving") == OFFLINE_SERVING_ENV
        and workload.get("expected_requests") == EXPECTED_WORKLOAD_REQUESTS
        and workload.get("seed") == 42
        and workload.get("request_rate") == "inf"
        and workload.get("output_tokens") == 300
        and workload.get("trace_sha256") == CANONICAL_WORKLOAD_TRACE_SHA256
        and _require_compact_hashes(
            artifact_hashes,
            "raw_workload_evidence",
            "source_equivalent_command",
        )
    )


def passes_custom_image_provenance(record: dict[str, Any]) -> bool:
    """Keep hand-edited manifests from bypassing Docker Hub/digest validation."""
    if candidate_name(record) not in CUSTOM_IMAGE_CANDIDATES:
        return True
    reference = record.get("image_reference")
    digest = record.get("image_digest")
    if not isinstance(reference, str) or not DOCKER_HUB_DIGEST_IMAGE.fullmatch(reference):
        return False
    return isinstance(digest, str) and _digest_from_reference(reference) == digest


def _eligible(record: dict[str, Any]) -> bool:
    return (
        record.get("ers") is not None
        and record.get("portal_valid", True)
        and passes_accuracy_gate(record)
        and passes_preflight(record)
        and passes_greedy_equivalence(record)
        and passes_speculative_evidence(record)
        and passes_speculative_run_manifest(record)
        and passes_custom_image_provenance(record)
    )


def _choose_with_ties(records: list[dict[str, Any]]) -> dict[str, Any]:
    records.sort(key=normalized_ers, reverse=True)
    top_ers = normalized_ers(records[0])
    tied = [
        item
        for item in records
        if top_ers - normalized_ers(item) < ERS_TIE_THRESHOLD
    ]

    def accuracy_for_tie_break(item: dict[str, Any]) -> float:
        artifact_accuracy = artifact_gpqa_accuracy(item)
        if artifact_accuracy is not None:
            return artifact_accuracy
        try:
            return float(item["accuracy"])
        except (KeyError, TypeError, ValueError):
            return -1.0

    tied.sort(
        key=lambda item: (
            -accuracy_for_tie_break(item),
            float(item.get("p95_ttft_ms") or float("inf")),
        )
    )
    return tied[0]


def choose_best(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Choose a validated improvement; preserve v6 when no one beats it.

    The portal result is the ranking signal.  A non-incumbent candidate also
    needs a complete preflight run and must strictly improve on the best v6
    portal ERS before it can replace v6.
    """
    eligible = [record for record in records if _eligible(record)]
    if not eligible:
        return None

    incumbents = [record for record in eligible if candidate_name(record) == INCUMBENT]
    incumbent = _choose_with_ties(incumbents) if incumbents else None
    challengers = [record for record in eligible if candidate_name(record) != INCUMBENT]
    if incumbent:
        incumbent_ers = normalized_ers(incumbent)
        challengers = [
            record
            for record in challengers
            if normalized_ers(record) > incumbent_ers
        ]
    if not challengers:
        return incumbent
    # The published decision policy applies the <0.01 ERS tie-break against
    # the incumbent too.  This is meaningful once the required baseline GPQA
    # artifact is recorded; otherwise the historical incumbent's legacy
    # accuracy field remains the conservative fallback.
    return _choose_with_ties(([incumbent] if incumbent else []) + challengers)


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path.resolve())


def _json_artifact(path: Path | None, label: str) -> dict[str, Any] | None:
    if path is None:
        return None
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} JSON file does not exist: {resolved}")
    try:
        data = json.loads(resolved.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} must be valid JSON: {resolved}") from exc
    return {"path": _display_path(resolved), "sha256": sha256(resolved), "data": data}


def _file_artifact(path: Path | None, label: str) -> dict[str, str] | None:
    if path is None:
        return None
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} file does not exist: {resolved}")
    return {"path": _display_path(resolved), "sha256": sha256(resolved)}


def _require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _require_equal(value: Any, expected: Any, label: str) -> None:
    if value != expected:
        raise ValueError(f"{label} does not match the submitted candidate")


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{label} is missing or invalid")
    return value


def _validate_raw_workload_evidence_artifact(
    data: Any,
    *,
    metrics_sha256: str | None,
    raw_request_records_sha256: str,
) -> None:
    evidence = _require_mapping(data, "--raw-workload-evidence")
    _require_equal(evidence.get("required"), True, "raw workload evidence required")
    _require_equal(evidence.get("passed"), True, "raw workload evidence passed")
    _require_equal(
        evidence.get("raw_benchmark_sha256"),
        metrics_sha256,
        "raw workload evidence benchmark SHA-256",
    )
    workload = _require_mapping(evidence.get("workload"), "raw workload evidence workload")
    for key, expected in {
        "expected_requests": EXPECTED_WORKLOAD_REQUESTS,
        "seed": 42,
        "request_rate": "inf",
        "output_tokens": CANONICAL_OUTPUT_TOKENS,
        "trace_sha256": CANONICAL_WORKLOAD_TRACE_SHA256,
    }.items():
        _require_equal(workload.get(key), expected, f"raw workload evidence {key}")
    for key, expected in {
        "expected_requests": EXPECTED_WORKLOAD_REQUESTS,
        "observed_requests": EXPECTED_WORKLOAD_REQUESTS,
        "successful_requests": EXPECTED_WORKLOAD_REQUESTS,
        "failed_requests": 0,
    }.items():
        _require_equal(evidence.get(key), expected, f"raw workload evidence {key}")
    _require_equal(
        evidence.get("request_records_sha256"),
        raw_request_records_sha256,
        "raw workload evidence request-records SHA-256",
    )
    records = evidence.get("per_request_completion_evidence")
    if not isinstance(records, list) or len(records) != EXPECTED_WORKLOAD_REQUESTS:
        raise ValueError("raw workload evidence must retain all 420 completion records")
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"raw workload evidence completion {index} must be an object")
        if record.get("success") is not True or record.get("output_tokens") != CANONICAL_OUTPUT_TOKENS:
            raise ValueError(
                f"raw workload evidence completion {index} is not an exact successful "
                f"{CANONICAL_OUTPUT_TOKENS}-token request"
            )


def _validate_source_equivalent_command_artifact(data: Any) -> None:
    command = _require_mapping(data, "--source-equivalent-command")
    _require_equal(
        command.get("captured_before_server_start"),
        True,
        "source-equivalent command captured_before_server_start",
    )
    _require_equal(
        command.get("source_equivalent_preflight"),
        True,
        "source-equivalent command source_equivalent_preflight",
    )
    _require_equal(
        command.get("offline_serving"),
        OFFLINE_SERVING_ENV,
        "source-equivalent command offline_serving",
    )
    if not isinstance(command.get("command"), list) or not command["command"]:
        raise ValueError("source-equivalent command must retain the vLLM command list")


def _validate_speculative_run_manifest(
    data: Any,
    *,
    candidate: str,
    compose_sha256: str,
    image_reference: str | None,
    image_digest: str | None,
    metrics_sha256: str | None,
    gpqa_sha256: str | None,
    greedy_sha256: str | None,
    resolved_config_sha256: str | None,
    startup_log_sha256: str | None,
    raw_workload_evidence_sha256: str | None,
    source_equivalent_command_sha256: str | None,
) -> None:
    """Reject evidence that is not bound to the submitted draft Compose.

    Colab is source-equivalent rather than a Docker execution environment, so
    the run manifest must explicitly bind its reproducible source preflight to
    the immutable image and Compose generated by the release workflow.
    """
    manifest = _require_mapping(data, "--run-manifest")
    portal_candidate = _require_mapping(
        manifest.get("portal_candidate"), "run manifest portal_candidate"
    )
    _require_equal(portal_candidate.get("candidate"), candidate, "run manifest candidate")
    _require_equal(
        portal_candidate.get("source_equivalent_preflight"),
        True,
        "run manifest source_equivalent_preflight",
    )
    _require_equal(
        portal_candidate.get("image_reference"), image_reference, "run manifest image reference"
    )
    _require_equal(
        portal_candidate.get("image_digest"), image_digest, "run manifest image digest"
    )
    _require_equal(
        portal_candidate.get("compose_sha256"), compose_sha256, "run manifest Compose SHA-256"
    )
    _require_equal(
        manifest.get("offline_serving"),
        OFFLINE_SERVING_ENV,
        "run manifest offline_serving",
    )

    repository_commit = manifest.get("repository_commit")
    if not isinstance(repository_commit, str) or not COMMIT_SHA.fullmatch(repository_commit):
        raise ValueError("run manifest must contain a 40-character repository_commit")
    profile = manifest.get("profile")
    if not isinstance(profile, str) or not profile.startswith("speculative-draft"):
        raise ValueError("run manifest must use a speculative-draft Colab profile")

    workload = _require_mapping(manifest.get("workload"), "run manifest workload")
    required_workload = {
        "expected_requests": EXPECTED_WORKLOAD_REQUESTS,
        "seed": 42,
        "request_rate": "inf",
        "output_tokens": 300,
    }
    for key, expected in required_workload.items():
        _require_equal(workload.get(key), expected, f"run manifest workload {key}")
    _require_equal(
        workload.get("trace_sha256"),
        CANONICAL_WORKLOAD_TRACE_SHA256,
        "run manifest workload trace_sha256",
    )

    artifact_hashes = _require_mapping(
        manifest.get("artifact_sha256"), "run manifest artifact_sha256"
    )
    expected_hashes = {
        "metrics": metrics_sha256,
        "gpqa": gpqa_sha256,
        "greedy_comparison": greedy_sha256,
        "resolved_vllm_config": resolved_config_sha256,
        "startup_log": startup_log_sha256,
        "raw_workload_evidence": raw_workload_evidence_sha256,
        "source_equivalent_command": source_equivalent_command_sha256,
    }
    for key, expected in expected_hashes.items():
        if expected is None:
            raise ValueError(f"{candidate} requires {key} evidence")
        _require_equal(artifact_hashes.get(key), expected, f"run manifest {key} SHA-256")


def _summary_from_run_manifest(data: Any) -> dict[str, Any] | None:
    if not isinstance(data, dict):
        return None
    portal_candidate = data.get("portal_candidate")
    workload = data.get("workload")
    hashes = data.get("artifact_sha256")
    return {
        "repository_commit": data.get("repository_commit"),
        "profile": data.get("profile"),
        "offline_serving": data.get("offline_serving"),
        "portal_candidate": portal_candidate if isinstance(portal_candidate, dict) else None,
        "workload": workload if isinstance(workload, dict) else None,
        "artifact_sha256": hashes if isinstance(hashes, dict) else None,
    }


def _summary_from_metrics(
    data: Any, candidate: str | None = None
) -> dict[str, Any] | None:
    """Keep benchmark evidence compact while the original JSON stays hashed."""
    if not isinstance(data, dict):
        return None
    runs = data.get("runs")
    summary: Any = runs[-1] if isinstance(runs, list) and runs else data
    comparisons = data.get("results")
    if isinstance(comparisons, list):
        matches = [
            item
            for item in comparisons
            if isinstance(item, dict)
            and (candidate is None or item.get("candidate") == candidate)
        ]
        if matches:
            candidate_runs = matches[-1].get("runs")
            summary = (
                candidate_runs[-1]
                if isinstance(candidate_runs, list) and candidate_runs
                else matches[-1]
            )
    if not isinstance(summary, dict):
        return None
    keys = (
        "ers",
        "score_if_accuracy_safe",
        "expected_requests",
        "successful_requests",
        "failed_requests",
        "success_rate",
        "ttft_ms",
        "tpot_ms",
        "server_metrics",
        "speculative_decoding",
        "config_fingerprint",
    )
    compact = {key: summary[key] for key in keys if key in summary}
    try:
        compact["workload_evidence"] = _canonical_workload_evidence(data, candidate)
    except ValueError:
        # ``build_record`` turns this into a hard failure for every challenger.
        # Keeping summary extraction tolerant lets selection read historical v6
        # records without pretending that aggregate-only data is sufficient.
        pass
    return compact


def _summary_from_gpqa(data: Any) -> dict[str, Any] | None:
    if not isinstance(data, dict):
        return None
    results = data.get("results")
    if isinstance(results, dict):
        metrics = results.get(GPQA_DIAMOND_TASK)
        if isinstance(metrics, dict):
            return {"task": GPQA_DIAMOND_TASK, "metrics": metrics}
    return {"keys": sorted(data.keys())}


def _summary_from_greedy(data: Any) -> dict[str, Any] | None:
    if not isinstance(data, dict):
        return None
    responses = data.get("responses")
    return {
        "matches_expected": data.get("matches_expected") is True,
        "response_count": len(responses) if isinstance(responses, list) else None,
        "prompt_fingerprint": data.get("prompt_fingerprint"),
    }


def _image_reference(compose: Path) -> str | None:
    text = compose.read_text(encoding="utf-8")
    matches = IMAGE_LINE.findall(text)
    if len(matches) != 1:
        return None
    value = matches[0].strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _digest_from_reference(reference: str | None) -> str | None:
    if not reference or "@" not in reference:
        return None
    candidate = reference.rsplit("@", maxsplit=1)[1]
    return candidate if DIGEST.fullmatch(candidate) else None


def build_record(args: argparse.Namespace) -> dict[str, Any]:
    compose = args.compose.resolve() if args.compose else None
    if compose is None or not compose.is_file():
        raise FileNotFoundError("A valid --compose file is required")
    compose_sha256 = sha256(compose)

    reference = args.image_reference or _image_reference(compose)
    reference_digest = _digest_from_reference(reference)
    digest = args.image_digest or reference_digest
    if digest is not None and not DIGEST.fullmatch(digest):
        raise ValueError("--image-digest must use sha256:<64 hex chars>")
    if args.candidate in CUSTOM_IMAGE_CANDIDATES:
        if not reference or not DOCKER_HUB_DIGEST_IMAGE.fullmatch(reference):
            raise ValueError(
                f"{args.candidate} requires a Docker Hub namespace/repository "
                "image reference pinned by @sha256:<64 hex chars>"
            )
        if digest is None:
            raise ValueError(
                f"{args.candidate} requires the custom image digest in the manifest"
            )
        if reference_digest != digest:
            raise ValueError(
                "--image-digest must match the digest embedded in --image-reference"
            )

    if args.candidate != INCUMBENT:
        required = {
            "--metrics": args.metrics,
            "--gpqa": args.gpqa,
            "--resolved-vllm-config": args.resolved_vllm_config,
            "--startup-log": args.startup_log,
        }
        missing = [flag for flag, value in required.items() if value is None]
        if missing:
            raise ValueError(
                f"{args.candidate} requires " + ", ".join(missing)
            )
    if args.candidate in SPECULATIVE_DRAFT_CANDIDATES and args.greedy_comparison is None:
        raise ValueError(f"{args.candidate} requires --greedy-comparison")
    if args.candidate in SPECULATIVE_DRAFT_CANDIDATES and args.run_manifest is None:
        raise ValueError(f"{args.candidate} requires --run-manifest")
    if args.candidate in SPECULATIVE_DRAFT_CANDIDATES and args.raw_workload_evidence is None:
        raise ValueError(f"{args.candidate} requires --raw-workload-evidence")
    if args.candidate in SPECULATIVE_DRAFT_CANDIDATES and args.source_equivalent_command is None:
        raise ValueError(f"{args.candidate} requires --source-equivalent-command")

    metrics = _json_artifact(args.metrics, "metrics")
    gpqa = _json_artifact(args.gpqa, "GPQA")
    greedy_comparison = _json_artifact(args.greedy_comparison, "greedy comparison")
    resolved_config = _json_artifact(args.resolved_vllm_config, "resolved vLLM config")
    startup_log = _file_artifact(args.startup_log, "startup log")
    run_manifest = _json_artifact(args.run_manifest, "run manifest")
    raw_workload_evidence = _json_artifact(
        args.raw_workload_evidence, "raw workload evidence"
    )
    source_equivalent_command = _json_artifact(
        args.source_equivalent_command, "source-equivalent command"
    )
    workload_evidence: dict[str, Any] | None = None
    if args.candidate != INCUMBENT:
        if not metrics:
            raise ValueError(f"{args.candidate} is missing required benchmark evidence")
        workload_evidence = _canonical_workload_evidence(metrics["data"], args.candidate)
    if args.candidate in SPECULATIVE_DRAFT_CANDIDATES:
        if not all(
            (
                metrics,
                gpqa,
                greedy_comparison,
                resolved_config,
                startup_log,
                run_manifest,
                raw_workload_evidence,
                source_equivalent_command,
            )
        ):
            raise ValueError(f"{args.candidate} is missing required preflight evidence")
        _validate_raw_workload_evidence_artifact(
            raw_workload_evidence["data"],
            metrics_sha256=metrics["sha256"],
            raw_request_records_sha256=workload_evidence["raw_request_records_sha256"],
        )
        _validate_source_equivalent_command_artifact(source_equivalent_command["data"])
        _validate_speculative_run_manifest(
            run_manifest["data"],
            candidate=args.candidate,
            compose_sha256=compose_sha256,
            image_reference=reference,
            image_digest=digest,
            metrics_sha256=metrics["sha256"],
            gpqa_sha256=gpqa["sha256"],
            greedy_sha256=greedy_comparison["sha256"],
            resolved_config_sha256=resolved_config["sha256"],
            startup_log_sha256=startup_log["sha256"],
            raw_workload_evidence_sha256=raw_workload_evidence["sha256"],
            source_equivalent_command_sha256=source_equivalent_command["sha256"],
        )
    if metrics:
        metrics["summary"] = _summary_from_metrics(metrics["data"], args.candidate)
        del metrics["data"]
    if gpqa:
        gpqa["summary"] = _summary_from_gpqa(gpqa["data"])
        del gpqa["data"]
    if greedy_comparison:
        greedy_comparison["summary"] = _summary_from_greedy(
            greedy_comparison["data"]
        )
        del greedy_comparison["data"]
    if run_manifest:
        run_manifest["summary"] = _summary_from_run_manifest(run_manifest["data"])
        del run_manifest["data"]
    if raw_workload_evidence:
        del raw_workload_evidence["data"]
    if source_equivalent_command:
        del source_equivalent_command["data"]

    recorded_accuracy = args.accuracy
    if args.candidate != INCUMBENT:
        gpqa_accuracy = _gpqa_accuracy_from_summary(
            gpqa["summary"] if gpqa else None
        )
        if gpqa_accuracy is None:
            raise ValueError(
                "--gpqa must contain the exact lm-eval gpqa_diamond task with "
                "an accuracy metric such as acc,none"
            )
        if (
            args.accuracy is not None
            and abs(float(args.accuracy) - gpqa_accuracy)
            > ACCURACY_ARTIFACT_TOLERANCE
        ):
            raise ValueError(
                "--accuracy differs from the GPQA artifact by more than "
                f"{ACCURACY_ARTIFACT_TOLERANCE:.3f}"
            )
        recorded_accuracy = gpqa_accuracy

        preflight_record = {
            "candidate": args.candidate,
            "healthcheck_passed": args.healthcheck_passed,
            "preflight_successful_requests": args.preflight_successful_requests,
            "preflight_expected_requests": args.preflight_expected_requests,
            "metrics": metrics,
        }
        if not passes_preflight(preflight_record):
            raise ValueError(
                "--metrics must prove a healthy 420/420 benchmark with zero failures, "
                "and the matching --preflight-successful-requests must be 420"
            )
        if args.candidate in SPECULATIVE_DRAFT_CANDIDATES:
            speculative_record = dict(preflight_record)
            speculative_record["metrics"] = metrics
            if not passes_speculative_evidence(speculative_record):
                raise ValueError(
                    f"{args.candidate} requires benchmark-delta speculative metrics "
                    "with no reset, observed drafts, and mean acceptance length >= "
                    f"{SPECULATIVE_MIN_MEAN_ACCEPTANCE_LENGTH:.1f}"
                )
            greedy_record = {
                "candidate": args.candidate,
                "greedy_comparison": greedy_comparison,
            }
            if not passes_greedy_equivalence(greedy_record):
                raise ValueError(
                    f"{args.candidate} requires a greedy comparison that matches its parent"
                )

    return {
        "candidate": args.candidate,
        "submission_id": args.submission_id,
        "compose_file": _display_path(compose),
        "compose_sha256": compose_sha256,
        "image_reference": reference,
        "image_digest": digest,
        "resolved_vllm_config": resolved_config,
        "metrics": metrics,
        "gpqa": gpqa,
        "greedy_comparison": greedy_comparison,
        "startup_log": startup_log,
        "run_manifest": run_manifest,
        "raw_workload_evidence": raw_workload_evidence,
        "source_equivalent_command": source_equivalent_command,
        "ers": args.ers,
        "ers_normalized": normalized_ers({"ers": args.ers}),
        "accuracy": recorded_accuracy,
        "accuracy_reported": args.accuracy,
        "accuracy_drop": args.accuracy_drop,
        "f_delta": args.f_delta,
        "penalty": args.penalty,
        "final_score": args.final_score,
        "p50_ttft_ms": args.p50_ttft_ms,
        "p95_ttft_ms": args.p95_ttft_ms,
        "p95_tpot_ms": args.p95_tpot_ms,
        "tbt_median_ms": args.tbt_median_ms,
        "successful_requests": args.successful_requests,
        "failed_requests": args.failed_requests,
        "healthcheck_passed": args.healthcheck_passed,
        "preflight_successful_requests": args.preflight_successful_requests,
        "preflight_expected_requests": args.preflight_expected_requests,
        "portal_valid": args.portal_valid,
        "status": args.status,
        "notes": args.notes,
    }


def _new_manifest() -> dict[str, Any]:
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "selection_policy": {
            "minimum_accuracy": MINIMUM_ACCURACY,
            "preferred_accuracy": PREFERRED_ACCURACY,
            "ers_tie_threshold": ERS_TIE_THRESHOLD,
            "replacement_requires": [
                "portal_valid",
                "accuracy_gate",
                "healthcheck",
                "420_of_420_preflight",
                "strict_portal_ers_gain_over_v6",
            ],
            "speculative_draft_requires": {
                "greedy_equivalence": True,
                "metrics_scope": "benchmark_delta",
                "counter_reset_detected": False,
                "minimum_mean_acceptance_length": (
                    SPECULATIVE_MIN_MEAN_ACCEPTANCE_LENGTH
                ),
                "bound_run_manifest": True,
            },
            "tie_breakers": ["higher_accuracy", "lower_p95_ttft_ms"],
        },
        "submissions": [],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", choices=CANDIDATES, required=True)
    parser.add_argument("--submission-id")
    parser.add_argument("--compose", type=Path, default=Path("docker-compose.yml"))
    parser.add_argument("--image-reference")
    parser.add_argument("--image-digest")
    parser.add_argument("--resolved-vllm-config", type=Path)
    parser.add_argument("--metrics", type=Path)
    parser.add_argument("--gpqa", type=Path)
    parser.add_argument("--greedy-comparison", type=Path)
    parser.add_argument(
        "--run-manifest",
        type=Path,
        help="Colab source-equivalent preflight manifest required by draft candidates",
    )
    parser.add_argument(
        "--raw-workload-evidence",
        type=Path,
        help="Colab raw per-request workload evidence required by draft candidates",
    )
    parser.add_argument(
        "--source-equivalent-command",
        type=Path,
        help="Colab pre-start source-equivalent vLLM command required by draft candidates",
    )
    parser.add_argument("--startup-log", "--log", dest="startup_log", type=Path)
    parser.add_argument("--ers", type=float, required=True)
    parser.add_argument("--accuracy", type=float)
    parser.add_argument("--accuracy-drop", type=float)
    parser.add_argument("--f-delta", type=float)
    parser.add_argument("--penalty", type=float)
    parser.add_argument("--final-score", type=float)
    parser.add_argument("--p50-ttft-ms", type=float)
    parser.add_argument("--p95-ttft-ms", type=float)
    parser.add_argument("--p95-tpot-ms", type=float)
    parser.add_argument("--tbt-median-ms", type=float)
    parser.add_argument("--successful-requests", type=int)
    parser.add_argument("--failed-requests", type=int)
    parser.add_argument(
        "--healthcheck-passed",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--preflight-successful-requests", type=int)
    parser.add_argument(
        "--preflight-expected-requests",
        type=int,
        default=EXPECTED_WORKLOAD_REQUESTS,
    )
    parser.add_argument(
        "--portal-valid",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--status")
    parser.add_argument("--notes")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("benchmark/submission_results.json"),
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        record = build_record(args)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))

    manifest_path = args.manifest.resolve()
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = _new_manifest()
    manifest["schema_version"] = max(
        int(manifest.get("schema_version", 0)), MANIFEST_SCHEMA_VERSION
    )
    default_policy = _new_manifest()["selection_policy"]
    policy = manifest.setdefault("selection_policy", {})
    if not isinstance(policy, dict):
        raise ValueError("manifest selection_policy must be an object")
    for key, value in default_policy.items():
        policy.setdefault(key, value)
    submissions = manifest.setdefault("submissions", [])
    manifest["submissions"] = [
        item for item in submissions if candidate_name(item) != args.candidate
    ]
    manifest["submissions"].append(record)
    manifest["submissions"].sort(key=lambda item: candidate_name(item))

    best = choose_best(manifest["submissions"])
    manifest["recommended_candidate"] = best["candidate"] if best else None
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(manifest_path)
    if best:
        print(
            f"Recommended candidate: {best['candidate']} "
            f"(ERS={best['ers']}, accuracy={best.get('accuracy')}, "
            f"f_delta={best.get('f_delta')})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
