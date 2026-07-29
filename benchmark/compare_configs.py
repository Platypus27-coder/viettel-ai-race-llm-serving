"""Run explicit, controlled Compose candidates against the same workload.

Examples:
    conda run -n viettel python benchmark/compare_configs.py \
        --compose incumbent=docker-compose.yml \
        --compose shortconv=artifacts/shortconv-fp8.yml

No profile name resolves to a deleted Compose file.  Every comparison target is
spelled out on the command line, with the root v6 Compose as the default.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


sys.path.insert(0, str(Path(__file__).parent))
from benchmark_ers import (  # noqa: E402
    TraceConfig,
    load_tokenizer,
    load_trace,
    parse_rate,
    reset_prefix_cache,
    run_benchmark,
)


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_TRACE = (
    PROJECT_DIR
    / "019e649f-4e27-74db-82da-920f57b13786"
    / "grading-workload-spec.json"
)


@dataclass(frozen=True)
class ComposeTarget:
    name: str
    path: Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_reference(path: Path) -> str | None:
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("image:"):
            return stripped.split(":", maxsplit=1)[1].strip().strip("'\"")
    return None


def parse_target(value: str) -> ComposeTarget:
    """Parse NAME=PATH and reject ambiguous implicit profile names."""
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "--compose must use NAME=PATH, for example incumbent=docker-compose.yml"
        )
    name, raw_path = value.split("=", maxsplit=1)
    if not name or not raw_path:
        raise argparse.ArgumentTypeError("--compose requires a non-empty NAME and PATH")
    path = Path(raw_path).resolve()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"Compose file does not exist: {path}")
    return ComposeTarget(name=name, path=path)


async def test_target(
    target: ComposeTarget,
    trace: TraceConfig,
    tokenizer: Any,
    base_url: str,
    runs: int,
) -> dict[str, Any]:
    """Start one explicit Compose file, benchmark it, then always stop it."""
    print(f"\n{'#' * 60}\nTesting: {target.name}\nCompose: {target.path}\n{'#' * 60}")
    subprocess.run(
        ["docker", "compose", "-f", str(target.path), "config", "--quiet"],
        cwd=PROJECT_DIR,
        check=True,
    )
    subprocess.run(
        ["docker", "compose", "-f", str(target.path), "up", "-d"],
        cwd=PROJECT_DIR,
        check=True,
    )
    summaries: list[dict[str, Any]] = []
    try:
        for index in range(runs):
            if index:
                await reset_prefix_cache(base_url)
            summary = await run_benchmark(base_url, trace, tokenizer)
            summary["run_index"] = index + 1
            summaries.append(summary)
    finally:
        print(f"Stopping: {target.name}")
        subprocess.run(
            ["docker", "compose", "-f", str(target.path), "down"],
            cwd=PROJECT_DIR,
            check=False,
        )

    return {
        "candidate": target.name,
        "compose_file": str(target.path),
        "compose_sha256": sha256(target.path),
        "image_reference": image_reference(target.path),
        "runs": summaries,
    }


def _latest_summary(result: dict[str, Any]) -> dict[str, Any]:
    runs = result.get("runs") or []
    return runs[-1] if runs else {}


def print_comparison(results: list[dict[str, Any]]) -> None:
    print(f"\n{'=' * 92}")
    print("CONTROLLED CONFIGURATION COMPARISON")
    print(f"{'=' * 92}")
    print(
        f"{'Candidate':<22} {'ERS':>8} {'TTFT p95(ms)':>14} "
        f"{'TPOT p95(ms)':>14} {'Success':>12}"
    )
    print("-" * 92)
    for result in results:
        summary = _latest_summary(result)
        ttft = summary.get("ttft_ms", {})
        tpot = summary.get("tpot_ms", {})
        print(
            f"{result['candidate']:<22} {summary.get('ers', float('nan')):>8.4f} "
            f"{ttft.get('p95', float('nan')):>14.2f} "
            f"{tpot.get('p95', float('nan')):>14.3f} "
            f"{summary.get('successful_requests', 0)}/"
            f"{summary.get('expected_requests', 0):<7}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--compose",
        action="append",
        type=parse_target,
        help="Explicit NAME=PATH candidate. Repeat for each candidate.",
    )
    parser.add_argument("--trace", default=str(DEFAULT_TRACE))
    parser.add_argument("--request-rate", default="inf")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument(
        "--tokenizer-path", default="LiquidAI/LFM2.5-1.2B-Instruct"
    )
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark/comparison_results.json"),
    )
    return parser


async def async_main(args: argparse.Namespace) -> int:
    if args.runs < 1:
        raise ValueError("--runs must be at least 1")
    targets = args.compose or [
        ComposeTarget("v6-incumbent", (PROJECT_DIR / "docker-compose.yml").resolve())
    ]
    names = [target.name for target in targets]
    if len(names) != len(set(names)):
        raise ValueError("Each --compose NAME must be unique")

    trace = load_trace(
        args.trace,
        request_rate=parse_rate(args.request_rate),
        seed=args.seed,
    )
    tokenizer = load_tokenizer(args.tokenizer_path)
    results = []
    for target in targets:
        results.append(
            await test_target(target, trace, tokenizer, args.base_url, args.runs)
        )
    print_comparison(results)

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "generated_at_unix": time.time(),
        "trace": {
            "path": str(Path(args.trace).resolve()),
            "request_rate": args.request_rate,
            "seed": args.seed,
            "runs_per_candidate": args.runs,
        },
        "results": results,
    }
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Results saved to: {output}")
    return 0


def main() -> int:
    args = build_parser().parse_args()
    try:
        return asyncio.run(async_main(args))
    except ValueError as exc:
        raise SystemExit(f"error: {exc}") from exc


if __name__ == "__main__":
    raise SystemExit(main())
