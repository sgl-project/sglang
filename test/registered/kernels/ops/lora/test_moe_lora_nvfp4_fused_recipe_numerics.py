"""The fused per-token epilogue down-B (the second TMEM accumulator, the raw
BF16 bridge) under every 4over6 activation recipe: the fused prefill test of
the provider file run with a provider that quantizes both GEMM inputs under
the recipe and a reference that quantizes them with the recipe file's float64
reference, over the eight recipes (MAE / MSE error, e4m3_max 448 / 256, the
fast and the strict error) at ranks 16 and 8. The fused epilogue reads the
recipe's stored row scales (row_amax / 1536 under 256); the gates are the
fused test's: every pattern against the reference, the fused and the unfused
path within 4e-3 of each other, the tokens without an adapter and the batch
without adapters bitwise the unfused path under the same recipe, the tiny
tokens' deltas visible, one captured prefill graph replayed over the patterns
bitwise with eager. Coverage in this tree: the recipe file's quantizer tests
run all eight recipes, its unfused provider gate runs MAE-256-strict and
MSE-448-fast in both phases, and this file runs all eight on the fused
prefill path."""

from __future__ import annotations

import msgspec
import pytest
import test_moe_lora_blockscaled_provider_numerics as P
import torch
from test_moe_lora_nvfp4_recipe_numerics import (
    _RECIPE_IDS,
    _RECIPES,
    _quant_nvfp4_recipe_ref,
)

from sglang.test.ci.ci_register import register_cuda_ci

# Every test here needs SM100 (block-scaled tcgen05 MMA): on the H100 lane the
# file is collected and skipped whole, so it is registered on the SM100 lane
# only. It runs on one GPU: 4-gpu-b200 is the kernel lane's only SM100 pool, so
# its four GPUs are allocation granularity, not multi-GPU coverage.
register_cuda_ci(est_time=40, stage="base-b-kernel-unit", runner_config="4-gpu-b200")


@pytest.mark.parametrize("rank", [16, 8])
@pytest.mark.parametrize("recipe", _RECIPES, ids=_RECIPE_IDS)
def test_fused_per_token_epilogue_under_a_recipe(
    monkeypatch: pytest.MonkeyPatch, recipe, rank: int
) -> None:
    quant_info = P._quant_info

    def quant_info_under_recipe(family, w13, w2):
        info, w13_eff, w2_eff, elem = quant_info(family, w13, w2)
        assert elem == "nvfp4_pt"
        info = msgspec.structs.replace(info, recipe=recipe)
        assert info.activation_denominator == 6.0 * recipe.e4m3_max
        return info, w13_eff, w2_eff, elem

    def quantize_under_recipe(x):
        return _quant_nvfp4_recipe_ref(x, recipe)[3].to(torch.float32)

    monkeypatch.setattr(P, "_quant_info", quant_info_under_recipe)
    monkeypatch.setattr(P, "_quant_nvfp4_per_token", quantize_under_recipe)
    P.test_fused_per_token_epilogue_plan_matches_reference(monkeypatch, rank)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
