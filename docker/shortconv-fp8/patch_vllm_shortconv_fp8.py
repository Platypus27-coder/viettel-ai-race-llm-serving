#!/usr/bin/env python3
"""Apply and verify the minimal ShortConv FP8 wiring for vLLM 0.22.1.

This intentionally uses only the standard library. It fails closed when the
installed source does not exactly match vLLM 0.22.1, avoiding a silent patch to
a newer or otherwise incompatible release.
"""

from __future__ import annotations

import argparse
import ast
import importlib.metadata
import sys
from pathlib import Path


EXPECTED_VERSION = "0.22.1"
SHORT_CONV_RELATIVE_PATH = Path("model_executor/layers/mamba/short_conv.py")
LFM2_RELATIVE_PATH = Path("model_executor/models/lfm2.py")


class PatchError(RuntimeError):
    """The installed vLLM source is not the expected patch target."""


def _replace_once(
    source: str,
    before: str,
    after: str,
    *,
    description: str,
) -> str:
    """Replace one known v0.22.1 anchor, tolerating an already patched file."""
    patched_count = source.count(after)
    if patched_count == 1:
        # The unpatched anchor can be a prefix of its patched replacement (the
        # import insertion is one such case). Remove the patched occurrence
        # before checking whether a second, truly unpatched anchor remains.
        remaining_source = source.replace(after, "", 1)
        if before not in remaining_source:
            return source

    count = source.count(before)
    if count == 1 and patched_count == 0:
        return source.replace(before, after, 1)
    raise PatchError(
        f"Expected exactly one {description} anchor; found {count}. "
        "Refusing to patch an unexpected vLLM source tree."
    )


def _patch_short_conv(source: str) -> str:
    source = _replace_once(
        source,
        "from vllm.model_executor.layers.linear import (\n"
        "    ColumnParallelLinear,\n"
        "    MergedColumnParallelLinear,\n"
        "    RowParallelLinear,\n"
        ")\n"
        "from vllm.model_executor.layers.mamba.abstract import MambaBase\n",
        "from vllm.model_executor.layers.linear import (\n"
        "    ColumnParallelLinear,\n"
        "    MergedColumnParallelLinear,\n"
        "    RowParallelLinear,\n"
        ")\n"
        "from vllm.model_executor.layers.mamba.abstract import MambaBase\n"
        "from vllm.model_executor.layers.quantization import QuantizationConfig\n",
        description="ShortConv QuantizationConfig import",
    )
    source = _replace_once(
        source,
        "        model_config: ModelConfig | None = None,\n"
        "        cache_config: CacheConfig | None = None,\n"
        "        prefix: str = \"\",\n",
        "        model_config: ModelConfig | None = None,\n"
        "        cache_config: CacheConfig | None = None,\n"
        "        quant_config: QuantizationConfig | None = None,\n"
        "        prefix: str = \"\",\n",
        description="ShortConv constructor signature",
    )
    source = _replace_once(
        source,
        "        self.in_proj = MergedColumnParallelLinear(\n"
        "            input_size=dim,\n"
        "            output_sizes=[dim] * 3,\n"
        "            bias=self.bias,\n"
        "            prefix=f\"{prefix}.in_proj\",\n"
        "        )\n",
        "        self.in_proj = MergedColumnParallelLinear(\n"
        "            input_size=dim,\n"
        "            output_sizes=[dim] * 3,\n"
        "            bias=self.bias,\n"
        "            quant_config=quant_config,\n"
        "            prefix=f\"{prefix}.in_proj\",\n"
        "        )\n",
        description="ShortConv in_proj",
    )
    return _replace_once(
        source,
        "        self.out_proj = RowParallelLinear(\n"
        "            input_size=dim,\n"
        "            output_size=dim,\n"
        "            bias=self.bias,\n"
        "            prefix=f\"{prefix}.out_proj\",\n"
        "        )\n",
        "        self.out_proj = RowParallelLinear(\n"
        "            input_size=dim,\n"
        "            output_size=dim,\n"
        "            bias=self.bias,\n"
        "            quant_config=quant_config,\n"
        "            prefix=f\"{prefix}.out_proj\",\n"
        "        )\n",
        description="ShortConv out_proj",
    )


def _patch_lfm2(source: str) -> str:
    return _replace_once(
        source,
        "        self.short_conv = ShortConv(\n"
        "            config=config,\n"
        "            dim=config.conv_dim,\n"
        "            layer_idx=layer_idx,\n"
        "            model_config=model_config,\n"
        "            cache_config=cache_config,\n"
        "            prefix=f\"{prefix}.conv\",\n"
        "        )\n",
        "        self.short_conv = ShortConv(\n"
        "            config=config,\n"
        "            dim=config.conv_dim,\n"
        "            layer_idx=layer_idx,\n"
        "            model_config=model_config,\n"
        "            cache_config=cache_config,\n"
        "            quant_config=quant_config,\n"
        "            prefix=f\"{prefix}.conv\",\n"
        "        )\n",
        description="Lfm2ShortConvDecoderLayer ShortConv construction",
    )


def _assert_contains_once(source: str, expected: str, *, description: str) -> None:
    count = source.count(expected)
    if count != 1:
        raise PatchError(f"Expected one {description}; found {count}.")


def _verify(short_conv_source: str, lfm2_source: str) -> None:
    """Verify scope as well as syntax: only ShortConv GEMM projections change."""
    ast.parse(short_conv_source)
    ast.parse(lfm2_source)

    _assert_contains_once(
        short_conv_source,
        "from vllm.model_executor.layers.quantization import QuantizationConfig\n",
        description="ShortConv QuantizationConfig import",
    )
    _assert_contains_once(
        short_conv_source,
        "        quant_config: QuantizationConfig | None = None,\n",
        description="ShortConv quant_config parameter",
    )
    _assert_contains_once(
        short_conv_source,
        "        self.in_proj = MergedColumnParallelLinear(\n"
        "            input_size=dim,\n"
        "            output_sizes=[dim] * 3,\n"
        "            bias=self.bias,\n"
        "            quant_config=quant_config,\n"
        "            prefix=f\"{prefix}.in_proj\",\n"
        "        )\n",
        description="quantized ShortConv in_proj",
    )
    _assert_contains_once(
        short_conv_source,
        "        self.out_proj = RowParallelLinear(\n"
        "            input_size=dim,\n"
        "            output_size=dim,\n"
        "            bias=self.bias,\n"
        "            quant_config=quant_config,\n"
        "            prefix=f\"{prefix}.out_proj\",\n"
        "        )\n",
        description="quantized ShortConv out_proj",
    )
    conv_block_start = short_conv_source.index("        self.conv = ColumnParallelLinear(\n")
    conv_block_end = short_conv_source.index("        self.conv.weight.data", conv_block_start)
    if "quant_config=" in short_conv_source[conv_block_start:conv_block_end]:
        raise PatchError("conv1d must remain outside the FP8 patch scope.")

    _assert_contains_once(
        lfm2_source,
        "        self.short_conv = ShortConv(\n"
        "            config=config,\n"
        "            dim=config.conv_dim,\n"
        "            layer_idx=layer_idx,\n"
        "            model_config=model_config,\n"
        "            cache_config=cache_config,\n"
        "            quant_config=quant_config,\n"
        "            prefix=f\"{prefix}.conv\",\n"
        "        )\n",
        description="quant_config threading into Lfm2ShortConvDecoderLayer",
    )


def _resolve_vllm_root(root_argument: str | None) -> Path:
    if root_argument is not None:
        return Path(root_argument).resolve()

    try:
        distribution_version = importlib.metadata.version("vllm")
        import vllm
    except Exception as exc:  # pragma: no cover - used only in image build
        raise PatchError("Cannot import the installed vLLM package.") from exc

    if distribution_version != EXPECTED_VERSION:
        raise PatchError(
            f"vLLM {EXPECTED_VERSION} is required, found {distribution_version}."
        )
    return Path(vllm.__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        help="vLLM package root; defaults to the installed vLLM package.",
    )
    parser.add_argument("--apply", action="store_true", help="apply the patch")
    parser.add_argument("--verify", action="store_true", help="verify the patch")
    args = parser.parse_args()

    if not args.apply and not args.verify:
        parser.error("choose --apply, --verify, or both")

    try:
        root = _resolve_vllm_root(args.root)
        short_conv_path = root / SHORT_CONV_RELATIVE_PATH
        lfm2_path = root / LFM2_RELATIVE_PATH
        short_conv_source = short_conv_path.read_text(encoding="utf-8")
        lfm2_source = lfm2_path.read_text(encoding="utf-8")

        if args.apply:
            short_conv_source = _patch_short_conv(short_conv_source)
            lfm2_source = _patch_lfm2(lfm2_source)
            short_conv_path.write_text(short_conv_source, encoding="utf-8")
            lfm2_path.write_text(lfm2_source, encoding="utf-8")

        if args.verify:
            _verify(short_conv_source, lfm2_source)
    except (OSError, PatchError, SyntaxError) as exc:
        print(f"ShortConv FP8 patch failed: {exc}", file=sys.stderr)
        return 1

    print(f"ShortConv FP8 patch verified under {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
