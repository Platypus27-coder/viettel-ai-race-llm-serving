"""Static unit tests for the self-contained vLLM 0.22.1 ShortConv patch."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


PATCH_PATH = Path(__file__).with_name("patch_vllm_shortconv_fp8.py")
PATCH_SPEC = importlib.util.spec_from_file_location("shortconv_fp8_patch", PATCH_PATH)
assert PATCH_SPEC is not None and PATCH_SPEC.loader is not None
patch = importlib.util.module_from_spec(PATCH_SPEC)
sys.modules[PATCH_SPEC.name] = patch
PATCH_SPEC.loader.exec_module(patch)


SHORT_CONV_V0221 = '''\
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.mamba.abstract import MambaBase


class ShortConv(MambaBase):
    def __init__(
        self,
        config,
        dim: int,
        layer_idx: int,
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        prefix: str = "",
    ):
        self.bias = config.conv_bias
        self.conv = ColumnParallelLinear(
            input_size=self.L_cache,
            output_size=dim,
            bias=self.bias,
            prefix=f"{prefix}.conv1d",
        )
        self.conv.weight.data = self.conv.weight.data.unsqueeze(1)
        self.in_proj = MergedColumnParallelLinear(
            input_size=dim,
            output_sizes=[dim] * 3,
            bias=self.bias,
            prefix=f"{prefix}.in_proj",
        )
        self.out_proj = RowParallelLinear(
            input_size=dim,
            output_size=dim,
            bias=self.bias,
            prefix=f"{prefix}.out_proj",
        )
'''

LFM2_V0221 = '''\
class Lfm2ShortConvDecoderLayer:
    def __init__(
        self,
        config: Lfm2Config,
        layer_idx: int,
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        self.short_conv = ShortConv(
            config=config,
            dim=config.conv_dim,
            layer_idx=layer_idx,
            model_config=model_config,
            cache_config=cache_config,
            prefix=f"{prefix}.conv",
        )
'''


class ShortConvFp8PatchTests(unittest.TestCase):
    def test_apply_verify_and_repeat_are_idempotent(self) -> None:
        short_conv = patch._patch_short_conv(SHORT_CONV_V0221)
        lfm2 = patch._patch_lfm2(LFM2_V0221)

        patch._verify(short_conv, lfm2)
        self.assertEqual(patch._patch_short_conv(short_conv), short_conv)
        self.assertEqual(patch._patch_lfm2(lfm2), lfm2)

    def test_patch_scope_excludes_conv1d(self) -> None:
        short_conv = patch._patch_short_conv(SHORT_CONV_V0221)
        conv_start = short_conv.index("        self.conv = ColumnParallelLinear(\n")
        conv_end = short_conv.index("        self.conv.weight.data", conv_start)
        self.assertNotIn("quant_config=", short_conv[conv_start:conv_end])
        self.assertEqual(short_conv.count("quant_config=quant_config"), 2)

    def test_unexpected_source_fails_closed(self) -> None:
        with self.assertRaises(patch.PatchError):
            patch._patch_short_conv("class ShortConv:\n    pass\n")


if __name__ == "__main__":
    unittest.main()
