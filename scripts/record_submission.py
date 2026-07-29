"""Record a reproducible portal result and recommend only validated winners."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


EXPECTED_WORKLOAD_REQUESTS = 420
MINIMUM_ACCURACY = 0.32
PREFERRED_ACCURACY = 0.35
ERS_TIE_THRESHOLD = 0.01
INCUMBENT = "v6-incumbent"
CANDIDATES = (
    INCUMBENT,
    "shortconv-fp8",
    "speculative-draft",
    "batch1536",
    "batch1024",
)
CUSTOM_IMAGE_CANDIDATES = {"shortconv-fp8", "speculative-draft"}
DIGEST = re.compile(r"^sha256:[0-9a-fA-F]{64}$")
IMAGE_LINE = re.compile(
    r"^[ \t]*image:[ \t]*(?P<value>.+?)[ \t]*$", re.MULTILINE
)


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


def passes_accuracy_gate(record: dict[str, Any]) -> bool:
    accuracy = record.get("accuracy")
    if candidate_name(record) != INCUMBENT:
        return record.get("gpqa") is not None and accuracy is not None and float(accuracy) >= MINIMUM_ACCURACY
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
    return expected == EXPECTED_WORKLOAD_REQUESTS and successful == expected


def passes_greedy_equivalence(record: dict[str, Any]) -> bool:
    """Speculative decoding must preserve the captured greedy parent output."""
    if candidate_name(record) != "speculative-draft":
        return True
    artifact = record.get("greedy_comparison")
    if not isinstance(artifact, dict):
        return False
    summary = artifact.get("summary")
    return isinstance(summary, dict) and summary.get("matches_expected") is True


def _eligible(record: dict[str, Any]) -> bool:
    return (
        record.get("ers") is not None
        and record.get("portal_valid", True)
        and passes_accuracy_gate(record)
        and passes_preflight(record)
        and passes_greedy_equivalence(record)
    )


def _choose_with_ties(records: list[dict[str, Any]]) -> dict[str, Any]:
    records.sort(key=normalized_ers, reverse=True)
    top_ers = normalized_ers(records[0])
    tied = [
        item
        for item in records
        if top_ers - normalized_ers(item) < ERS_TIE_THRESHOLD
    ]
    tied.sort(
        key=lambda item: (
            -(float(item["accuracy"]) if item.get("accuracy") is not None else -1.0),
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
    winner = _choose_with_ties(challengers)
    return winner


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
        "config_fingerprint",
    )
    return {key: summary[key] for key in keys if key in summary}


def _summary_from_gpqa(data: Any) -> dict[str, Any] | None:
    if not isinstance(data, dict):
        return None
    results = data.get("results")
    if isinstance(results, dict):
        for task_name, metrics in results.items():
            if "gpqa" in str(task_name).lower() and isinstance(metrics, dict):
                return {"task": task_name, "metrics": metrics}
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

    reference = args.image_reference or _image_reference(compose)
    digest = args.image_digest or _digest_from_reference(reference)
    if digest is not None and not DIGEST.fullmatch(digest):
        raise ValueError("--image-digest must use sha256:<64 hex chars>")
    if args.candidate in CUSTOM_IMAGE_CANDIDATES and digest is None:
        raise ValueError(
            f"{args.candidate} requires the custom image digest in the manifest"
        )

    if args.candidate != INCUMBENT:
        required = {
            "--metrics": args.metrics,
            "--gpqa": args.gpqa,
            "--resolved-vllm-config": args.resolved_vllm_config,
            "--startup-log": args.startup_log,
            "--accuracy": args.accuracy,
        }
        missing = [flag for flag, value in required.items() if value is None]
        if missing:
            raise ValueError(
                f"{args.candidate} requires " + ", ".join(missing)
            )
    if args.candidate == "speculative-draft" and args.greedy_comparison is None:
        raise ValueError("speculative-draft requires --greedy-comparison")

    metrics = _json_artifact(args.metrics, "metrics")
    gpqa = _json_artifact(args.gpqa, "GPQA")
    greedy_comparison = _json_artifact(args.greedy_comparison, "greedy comparison")
    resolved_config = _json_artifact(args.resolved_vllm_config, "resolved vLLM config")
    startup_log = _file_artifact(args.startup_log, "startup log")
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

    return {
        "candidate": args.candidate,
        "submission_id": args.submission_id,
        "compose_file": _display_path(compose),
        "compose_sha256": sha256(compose),
        "image_reference": reference,
        "image_digest": digest,
        "resolved_vllm_config": resolved_config,
        "metrics": metrics,
        "gpqa": gpqa,
        "greedy_comparison": greedy_comparison,
        "startup_log": startup_log,
        "ers": args.ers,
        "ers_normalized": normalized_ers({"ers": args.ers}),
        "accuracy": args.accuracy,
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
        "schema_version": 2,
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
    manifest.setdefault("schema_version", 2)
    manifest.setdefault("selection_policy", _new_manifest()["selection_policy"])
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
