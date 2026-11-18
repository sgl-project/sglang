"""K-tail-safe tiling of aiter's MXFP4 gemm_afp4wfp4 (triton-lang/triton#11256), on faked archs (no GPU)."""

import unittest
from types import SimpleNamespace
from unittest import mock

import sglang.kernels.arch as arch_module
from sglang.kernels.arch._hip import HipArch
from sglang.kernels.ops.gemm import afp4wfp4_k_tail_gfx950 as k_tail
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _hip_arch(gcn_arch_name: str) -> HipArch:
    arch = HipArch()
    arch.props = SimpleNamespace(gcnArchName=gcn_arch_name)
    return arch


class TestKTailSafeTiling(CustomTestCase):
    def test_narrows_a_tail_in_an_unfilled_stage(self):
        # Two K blocks in three stages: the tail lands in a never-filled stage.
        self.assertEqual(
            k_tail.k_tail_safe_tiling(1024, k=1536, num_stages=3), (512, 3)
        )
        # No power of two >= 128 divides 2880; six blocks of 512 put the tail in a filled stage.
        self.assertEqual(
            k_tail.k_tail_safe_tiling(1024, k=2880, num_stages=3), (512, 3)
        )
        # A block wider than K is shrunk to K's next power of two first, as the kernel wrapper does.
        self.assertEqual(k_tail.k_tail_safe_tiling(2048, k=768, num_stages=3), (256, 3))

    def test_keeps_a_tail_in_a_filled_stage(self):
        # Kimi-K3 TP8 dense down_proj: five blocks in three stages keep AITER's block (128 is 1.7x slower).
        self.assertEqual(
            k_tail.k_tail_safe_tiling(1024, k=4224, num_stages=3), (1024, 3)
        )
        # The boundary: one block more than stages.
        self.assertEqual(
            k_tail.k_tail_safe_tiling(1024, k=3136, num_stages=3), (1024, 3)
        )
        self.assertEqual(k_tail.k_tail_safe_tiling(512, k=1088, num_stages=2), (512, 2))

    def test_drops_stages_when_the_minimum_block_is_not_enough(self):
        # Three blocks of the minimum 128 in three stages.
        self.assertEqual(k_tail.k_tail_safe_tiling(256, k=320, num_stages=3), (128, 2))
        # One masked block cannot share a stage with a full one.
        self.assertIsNone(k_tail.k_tail_safe_tiling(256, k=96, num_stages=3))


class TestKTailSafeConfig(CustomTestCase):
    # AITER's untuned decode config for 8 < M <= 32.
    _DECODE_CONFIG = {
        "BLOCK_SIZE_M": 32,
        "BLOCK_SIZE_K": 1024,
        "num_stages": 3,
        "NUM_KSPLIT": 1,
    }

    def _config(self, get_config, gcn_arch_name="gfx950:sramecc+:xnack-", k_packed=768):
        k_tail.k_tail_safe_config.cache_clear()
        try:
            with (
                mock.patch.object(k_tail, "_aiter_get_config", get_config),
                mock.patch.object(
                    arch_module, "_DEVICE_ARCH", _hip_arch(gcn_arch_name)
                ),
            ):
                return k_tail.k_tail_safe_config(m=16, n=7168, k_packed=k_packed)
        finally:
            k_tail.k_tail_safe_config.cache_clear()

    def _decode_config(self):
        return mock.Mock(return_value=(dict(self._DECODE_CONFIG), False))

    def test_narrows_the_kernels_own_config(self):
        get_config = self._decode_config()

        config = self._config(get_config, k_packed=768)

        # gemm_afp4wfp4's _get_config takes packed K, as gemm_afp4wfp4 passes it.
        get_config.assert_called_once_with(16, 7168, 768)
        self.assertEqual(config, {**self._DECODE_CONFIG, "BLOCK_SIZE_K": 512})

    def test_other_archs_keep_aiters_config(self):
        # The workaround was measured on gfx950 only; any other gfx (in or out of gfx95x) is left alone.
        for gcn_arch_name in ("gfx942:sramecc+:xnack-", "gfx951", "gfx1250"):
            with self.subTest(arch=gcn_arch_name):
                get_config = self._decode_config()
                self.assertIsNone(self._config(get_config, gcn_arch_name))
                get_config.assert_not_called()

    def test_tail_in_a_filled_stage_keeps_aiters_config(self):
        # Kimi-K3 TP8 dense down_proj, K=4224.
        self.assertIsNone(self._config(self._decode_config(), k_packed=2112))

    def test_no_config_keeps_the_default_path(self):
        for error in (KeyError("no M bucket"), AssertionError("no DEFAULT.json")):
            with self.subTest(error=type(error).__name__):
                self.assertIsNone(self._config(mock.Mock(side_effect=error)))

    def test_a_changed_private_getter_raises(self):
        # _get_config is private aiter API: a changed signature fails loudly, not silently unfixed.
        with self.assertRaises(TypeError):
            self._config(mock.Mock(side_effect=TypeError("changed signature")))

    def test_split_k_config_is_left_to_aiter(self):
        split_k = {"BLOCK_SIZE_K": 512, "num_stages": 2, "NUM_KSPLIT": 4}
        self.assertIsNone(self._config(mock.Mock(return_value=(split_k, False))))


if __name__ == "__main__":
    unittest.main()
