from types import SimpleNamespace
from unittest import mock

import pytest

import sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_config as cfg_mod
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_EXEC = SimpleNamespace(
    deterministic=SimpleNamespace(enable_deterministic_inference=False)
)


def _default_config(M, E, topk, gfx95, block_shape=None):
    with (
        mock.patch.object(cfg_mod, "get_exec", return_value=_EXEC),
        mock.patch.object(cfg_mod, "_is_gfx95", gfx95),
        mock.patch.object(cfg_mod, "_use_low_smem_fp8_default", return_value=False),
    ):
        return cfg_mod.get_default_config(
            M, E, 704, 2816, topk, "fp8_w8a8", False, block_shape
        )


@pytest.mark.parametrize(
    "M,E,topk,block_m",
    [
        (1, 128, 8, 16),  # 0.06 rows per expert
        (127, 128, 8, 16),  # 7.9
        (128, 128, 8, 32),  # 8
        (383, 128, 8, 32),  # 23.9
        (384, 128, 8, 64),  # 24
        (767, 128, 8, 64),  # 47.9
        (768, 128, 8, 128),  # 48
        (4096, 512, 10, 128),  # 80
    ],
)
def test_gfx95_fp8_tile_follows_rows_per_expert(M, E, topk, block_m):
    config = _default_config(M, E, topk, gfx95=True)
    assert config["BLOCK_SIZE_M"] == block_m


def test_gfx95_fp8_tiles_keep_block_k():
    # BLOCK_SIZE_K fixes the K reduction order, so keeping it at the value of the
    # replaced branches keeps the results bit-identical.
    for _, config in cfg_mod._GFX95_FP8_MOE_TILES:
        assert config["BLOCK_SIZE_K"] == 128


def test_other_paths_keep_previous_defaults():
    assert _default_config(16, 128, 8, gfx95=False)["BLOCK_SIZE_M"] == 64
    assert _default_config(4096, 128, 8, gfx95=False)["BLOCK_SIZE_N"] == 256
    blockwise = _default_config(16, 128, 8, gfx95=True, block_shape=[128, 128])
    assert blockwise["BLOCK_SIZE_N"] == 128 and blockwise["BLOCK_SIZE_K"] == 128
