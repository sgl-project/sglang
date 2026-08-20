import unittest
from unittest.mock import patch

from sglang.srt.configs.mamba_utils import Mamba2CacheParams, Mamba2StateShape
from sglang.srt.environ import envs
from sglang.srt.mem_cache.memory_pool import MambaPool
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

MAMBA_LAYER_IDS = [0, 1]


def _speculative_pool(device: str) -> MambaPool:
    shape = Mamba2StateShape.create(
        tp_world_size=1,
        intermediate_size=64,
        n_groups=1,
        num_heads=2,
        head_dim=16,
        state_size=8,
        conv_kernel=4,
    )
    with envs.SGLANG_MAMBA_SSM_DTYPE.override("bfloat16"):
        cache_params = Mamba2CacheParams(shape=shape, layers=MAMBA_LAYER_IDS)
    return MambaPool(
        size=2,
        spec_state_size=2,
        cache_params=cache_params,
        mamba_layer_ids=MAMBA_LAYER_IDS,
        device=device,
        speculative_num_draft_tokens=3,
        speculative_eagle_topk=1,
    )


class TestMambaPoolSpeculativeDevice(unittest.TestCase):
    """The speculative scratch buffers were allocated with a hardcoded
    device="cuda" while every sibling allocation honored the `device` argument,
    so building the pool on any other device raised "Torch not compiled with
    CUDA enabled" before the first forward. Both intermediate layouts are
    covered because they allocate through separate code paths."""

    def test_dense_layout_buffers_follow_the_pool_device(self):
        with patch("sglang.srt.mem_cache.memory_pool._is_cpu", True):
            pool = _speculative_pool("cpu")

        self.assertEqual(pool.mamba_cache.intermediate_ssm.device.type, "cpu")
        for window in pool.mamba_cache.intermediate_conv_window:
            self.assertEqual(window.device.type, "cpu")

    def test_deduplicated_layout_buffers_follow_the_pool_device(self):
        with patch("sglang.srt.mem_cache.memory_pool._is_cpu", False):
            pool = _speculative_pool("cpu")

        self.assertEqual(pool.mamba_cache.intermediate_ssm.device.type, "cpu")
        for phys in pool._intermediate_conv_window_phys:
            self.assertEqual(phys.device.type, "cpu")

    def test_layouts_agree_on_the_logical_window_shape(self):
        """The dedup view is an as_strided alias, so a layout change that drops
        a step or a window column would still allocate and only corrupt the
        conv rollback at commit time."""
        with patch("sglang.srt.mem_cache.memory_pool._is_cpu", True):
            dense = _speculative_pool("cpu")
        with patch("sglang.srt.mem_cache.memory_pool._is_cpu", False):
            dedup = _speculative_pool("cpu")

        self.assertEqual(
            [tuple(w.shape) for w in dense.mamba_cache.intermediate_conv_window],
            [tuple(w.shape) for w in dedup.mamba_cache.intermediate_conv_window],
        )


if __name__ == "__main__":
    unittest.main()
