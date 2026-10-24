"""Unit tests for the block-quantized MoE shard compatibility check."""

import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from torch import nn

from sglang.srt.configs.model_config import ModelImpl
from sglang.srt.model_executor.model_runner_components.moe_ep_setup import (
    check_quantized_moe_compatibility,
)
from sglang.srt.models.registry import ModelRegistry
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_DEFAULT_ARCH = "FakeBlockFp8MoeForCausalLM"
_OPT_IN_ARCH = "FakeRequantizedMoeForCausalLM"


class _DefaultMoeModel(nn.Module):
    pass


class _PartialBlockMoeModel(nn.Module):
    supports_quantized_moe_partial_block_shards = True


def _model_config(arch: str, moe_intermediate_size: int):
    hf_config = SimpleNamespace(
        architectures=[arch],
        quantization_config={"quant_method": "fp8", "weight_block_size": [128, 128]},
    )
    return SimpleNamespace(
        hf_config=hf_config,
        hf_text_config=SimpleNamespace(moe_intermediate_size=moe_intermediate_size),
        quantization="fp8",
        model_impl=ModelImpl.AUTO,
    )


class TestCheckQuantizedMoeCompatibility(CustomTestCase):
    def setUp(self):
        registry_patch = patch.dict(
            ModelRegistry.models,
            {_DEFAULT_ARCH: _DefaultMoeModel, _OPT_IN_ARCH: _PartialBlockMoeModel},
        )
        registry_patch.start()
        self.addCleanup(registry_patch.stop)

    def _check(self, arch: str, moe_intermediate_size: int, tp_size: int):
        check_quantized_moe_compatibility(
            model_config=_model_config(arch, moe_intermediate_size),
            tp_size=tp_size,
            moe_ep_size=1,
            moe_dp_size=1,
        )

    def test_partial_block_shards_rejected_by_default(self):
        # 2048 / 64 = 32 per rank, below the 128 block.
        with self.assertRaisesRegex(ValueError, "weight_block_size_n=128"):
            self._check(_DEFAULT_ARCH, 2048, tp_size=64)

    def test_partial_block_shards_allowed_for_opted_in_model(self):
        self._check(_OPT_IN_ARCH, 2048, tp_size=64)

    def test_opt_in_keeps_divisibility_check(self):
        with self.assertRaisesRegex(ValueError, "must be divisible by moe_tp_size"):
            self._check(_OPT_IN_ARCH, 2048, tp_size=3)

    def test_aligned_shards_pass_without_resolving_model(self):
        with patch(
            "sglang.srt.model_loader.utils.get_model_architecture",
            side_effect=AssertionError("should not resolve the model class"),
        ):
            for arch in (_DEFAULT_ARCH, _OPT_IN_ARCH):
                self._check(arch, 2048, tp_size=16)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
