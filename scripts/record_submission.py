"""Append a reproducible portal submission record and recommend the best slot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_ers(record: dict[str, Any]) -> float:
    """Return ERS on the rules' 0..1 scale.

    The portal has exposed ERS both as a normalized value and as points out of
    100, so manifests accept either representation.
    """
    value = float(record["ers"])
    return value / 100.0 if value > 1.0 else value


def passes_accuracy_gate(record: dict[str, Any]) -> bool:
    accuracy = record.get("accuracy")
    if accuracy is not None:
        return float(accuracy) >= 0.32
    f_delta = record.get("f_delta")
    return f_delta is not None and float(f_delta) == 1.0


def choose_best(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    eligible = [
        record
        for record in records
        if record.get("ers") is not None
        and record.get("portal_valid", True)
        and passes_accuracy_gate(record)
    ]
    if not eligible:
        return None
    eligible.sort(key=normalized_ers, reverse=True)
    top_ers = normalized_ers(eligible[0])
    tied = [
        item for item in eligible if top_ers - normalized_ers(item) < 0.01
    ]
    tied.sort(
        key=lambda item: (
            -(float(item["accuracy"]) if item.get("accuracy") is not None else -1.0),
            float(item.get("p95_ttft_ms") or float("inf")),
        )
    )
    return tied[0]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slot", type=int, choices=range(1, 6), required=True)
    parser.add_argument("--submission-id")
    parser.add_argument("--compose", type=Path)
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
    parser.add_argument("--successful-requests", type=int, default=420)
    parser.add_argument("--failed-requests", type=int, default=0)
    parser.add_argument("--warmup-count", type=int, default=0)
    parser.add_argument(
        "--portal-valid",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--status")
    parser.add_argument("--notes")
    parser.add_argument("--log")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("benchmark/submission_results.json"),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    compose = args.compose.resolve() if args.compose else None
    if compose is not None and not compose.is_file():
        raise FileNotFoundError(compose)

    manifest_path = args.manifest.resolve()
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {
            "selection_policy": {
                "minimum_accuracy": 0.32,
                "preferred_accuracy": 0.35,
                "ers_tie_threshold": 0.01,
                "tie_breakers": ["higher_accuracy", "lower_p95_ttft_ms"],
            },
            "submissions": [],
        }

    record = {
        "slot": args.slot,
        "submission_id": args.submission_id,
        "compose_file": str(compose) if compose else None,
        "compose_sha256": sha256(compose) if compose else None,
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
        "warmup_count": args.warmup_count,
        "portal_valid": args.portal_valid,
        "status": args.status,
        "notes": args.notes,
        "log": args.log,
    }
    manifest["submissions"] = [
        item
        for item in manifest.get("submissions", [])
        if int(item.get("slot", -1)) != args.slot
    ]
    manifest["submissions"].append(record)
    manifest["submissions"].sort(key=lambda item: int(item["slot"]))

    best = choose_best(manifest["submissions"])
    manifest["recommended_slot"] = best["slot"] if best else None
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(manifest_path)
    if best:
        print(
            f"Recommended slot: {best['slot']} "
            f"(ERS={best['ers']}, accuracy={best.get('accuracy')}, "
            f"f_delta={best.get('f_delta')})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
