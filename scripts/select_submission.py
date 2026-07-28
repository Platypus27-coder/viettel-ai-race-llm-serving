"""Render one standalone Compose file for the three remaining portal runs."""

from __future__ import annotations

import argparse
from pathlib import Path


def render_compose(
    slot: int,
    precision: str,
    batch_tokens: int,
    max_num_seqs: int,
) -> str:
    quantization = ""
    if precision == "fp8":
        quantization = "      - --quantization=fp8\n"

    variant = f"slot{slot}-{precision}-batch{batch_tokens}-seqs{max_num_seqs}"
    return f"""# Generated submission: {variant}
services:
  model:
    image: vllm/vllm-openai:v0.22.1
    entrypoint:
      - python3
      - -m
      - vllm.entrypoints.openai.api_server
    command:
      - --model=/model
      - --served-model-name=LFM2.5-1.2B-Instruct
      - --host=0.0.0.0
      - --port=8000
      - --tensor-parallel-size=1
      - --gpu-memory-utilization=0.95
      - --max-model-len=8192
{quantization}      - --enable-prefix-caching
      - --enable-chunked-prefill
      - --max-num-batched-tokens={batch_tokens}
      - --max-num-seqs={max_num_seqs}
    ports:
      - "8000:8000"
    shm_size: "2g"
    environment:
      - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
      - OMP_NUM_THREADS=1
      - MKL_NUM_THREADS=1
      - VLLM_NO_USAGE_STATS=1
      - DO_NOT_TRACK=1
      - VLLM_LOGGING_LEVEL=WARNING
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
"""


def resolve_configuration(args: argparse.Namespace) -> tuple[str, int, int]:
    if args.slot == 3:
        return "bf16", 4096, 64
    if args.slot == 4:
        return "bf16", 2048, 64

    if not args.variant:
        raise ValueError("slot 5 requires --variant fp8 or seqs48")
    if args.batch_tokens not in {2048, 4096}:
        raise ValueError(
            "slot 5 requires --batch-tokens 2048 or 4096 "
            "from the slot 3/4 winner"
        )
    if args.variant == "fp8":
        if args.accuracy is None or args.accuracy < 0.32:
            raise ValueError("FP8 is allowed only with --accuracy >= 0.32")
        if args.successful_requests != 420:
            raise ValueError(
                "FP8 is allowed only with --successful-requests 420"
            )
        return "fp8", args.batch_tokens, 64
    return "bf16", args.batch_tokens, 48


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slot", type=int, choices=(3, 4, 5), required=True)
    parser.add_argument("--variant", choices=("fp8", "seqs48"))
    parser.add_argument("--batch-tokens", type=int)
    parser.add_argument("--accuracy", type=float)
    parser.add_argument("--successful-requests", type=int)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        precision, batch_tokens, max_num_seqs = resolve_configuration(args)
    except ValueError as exc:
        parser.error(str(exc))

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        render_compose(args.slot, precision, batch_tokens, max_num_seqs),
        encoding="utf-8",
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
