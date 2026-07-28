"""
Viettel AI Race 2026 — A/B Config Tester
==========================================
Compares different docker-compose configurations to find the optimal setup.

Usage:
    python compare_configs.py --configs baseline recovery tbt
"""

import asyncio
import json
import subprocess
import time
import sys
import os
import argparse

# Add parent dir to path for benchmark import
sys.path.insert(0, os.path.dirname(__file__))
from benchmark_ers import (
    ArrivalConfig,
    TraceConfig,
    load_tokenizer,
    load_trace,
    run_benchmark,
)


COMPOSE_DIR = os.path.dirname(os.path.dirname(__file__))

CONFIG_PROFILES = {
    "baseline": {
        "file": "configs/docker-compose.baseline.yml",
        "description": "BTC baseline (BF16, prefix caching, 0.95 mem)",
    },
    "recovery": {
        "file": "docker-compose.yml",
        "description": "Submission 3 recovery (BF16, batch 4096, seqs 64)",
    },
    "tbt": {
        "file": "configs/docker-compose.slot-04-bf16-batch2048-seqs64.yml",
        "description": "Submission 4 TBT priority (BF16, batch 2048, seqs 64)",
    },
}


async def test_config(
    config_name: str,
    compose_file: str,
    trace: TraceConfig,
    tokenizer,
    base_url: str = "http://localhost:8000",
) -> dict:
    """Test a single configuration and return results."""
    print(f"\n{'#'*60}")
    print(f"  Testing: {config_name}")
    print(f"  File: {compose_file}")
    print(f"{'#'*60}\n")

    compose_path = os.path.join(COMPOSE_DIR, compose_file)

    # Start the container
    print("Starting container...")
    subprocess.run(
        ["docker", "compose", "-f", compose_path, "up", "-d"],
        cwd=COMPOSE_DIR,
        check=True,
    )

    try:
        summary = await run_benchmark(base_url, trace, tokenizer)
        return {
            "config": config_name,
            "ers": summary["ers"],
            "score_max": summary["score_if_accuracy_safe"],
            "successful_requests": summary["successful_requests"],
            "total_requests": summary["expected_requests"],
            "mean_ttft_ms": summary["ttft_ms"].get("mean"),
            "mean_tpot_ms": summary["tpot_ms"].get("mean"),
            "p95_ttft_ms": summary["ttft_ms"].get("p95"),
            "p95_tpot_ms": summary["tpot_ms"].get("p95"),
            "config_fingerprint": summary["config_fingerprint"],
        }
    finally:
        # Stop the container
        print(f"\nStopping container for {config_name}...")
        subprocess.run(
            ["docker", "compose", "-f", compose_path, "down"],
            cwd=COMPOSE_DIR,
        )
        await asyncio.sleep(5)


async def compare_all(configs: list[str], trace: TraceConfig, tokenizer):
    """Compare multiple configurations."""
    results = []

    for config_name in configs:
        if config_name not in CONFIG_PROFILES:
            print(f"Unknown config: {config_name}")
            continue

        profile = CONFIG_PROFILES[config_name]
        result = await test_config(
            config_name, profile["file"], trace, tokenizer
        )
        results.append(result)

    # Print comparison table
    print(f"\n{'='*80}")
    print(f"  📊 CONFIGURATION COMPARISON")
    print(f"{'='*80}")
    print(f"  {'Config':<15} {'ERS':>8} {'Score':>8} {'TTFT(ms)':>10} {'TPOT(ms)':>10} {'Success':>10}")
    print(f"  {'─'*73}")

    best_ers = max((r["ers"] for r in results), default=0)
    for r in results:
        marker = " 🏆" if r["ers"] == best_ers and len(results) > 1 else ""
        mean_ttft = r["mean_ttft_ms"] if r["mean_ttft_ms"] is not None else float("nan")
        mean_tpot = r["mean_tpot_ms"] if r["mean_tpot_ms"] is not None else float("nan")
        print(
            f"  {r['config']:<15} {r['ers']:>8.4f} {r['score_max']:>8.2f} "
            f"{mean_ttft:>10.2f} {mean_tpot:>10.3f} "
            f"{r['successful_requests']}/{r['total_requests']}{marker}"
        )
    print()

    # Save comparison
    out_path = os.path.join(os.path.dirname(__file__), "comparison_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  📁 Results saved to: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Viettel AI Race 2026 — Config Comparison"
    )
    parser.add_argument(
        "--configs", nargs="+", default=["baseline", "recovery", "tbt"],
        choices=list(CONFIG_PROFILES.keys()),
        help="Configurations to compare"
    )
    parser.add_argument("--conversations", type=int, default=5)
    parser.add_argument("--turns", type=int, default=3)
    parser.add_argument("--trace")
    parser.add_argument("--request-rate", type=float, default=float("inf"))
    parser.add_argument(
        "--tokenizer-path", default="LiquidAI/LFM2.5-1.2B-Instruct"
    )
    args = parser.parse_args()

    if args.trace:
        trace = load_trace(args.trace, request_rate=args.request_rate)
    else:
        trace = TraceConfig(
            num_conversations=args.conversations,
            user_turns_per_conversation=args.turns,
            arrival=ArrivalConfig(request_rate=args.request_rate),
        )
    tokenizer = load_tokenizer(args.tokenizer_path)

    asyncio.run(compare_all(args.configs, trace, tokenizer))


if __name__ == "__main__":
    main()
