"""SGLANG_AITER_MLA_PERSIST=0 on gfx942 with an fp8 KV cache and 16 heads.

Decode on gfx942 sends a bf16 query, and aiter's gfx942 MLA table has the
bf16-q / fp8-KV / 16-head decode kernel only in its persistent form. With
persist off the server failed at its first decode with "cannot get heuristic
kernel ... q_type:bf16 kv_type:fp8 gqa:16 ps:0". The backend now keeps the
persistent kernel there; every other configuration keeps the user's choice.
"""

import unittest
from unittest import mock

import torch

from sglang.kernels.ops.quantization.fp8_kernel import fp8_dtype
from sglang.srt.layers.attention import aiter_backend
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

# (use_ps, fast_mode, intra_batch_mode) as the module sets them for PERSIST=0.
_PERSIST_OFF = (False, False, False)
_PERSIST_ON = (True, False, True)


def _resolve(**overrides):
    kwargs = dict(
        use_ps=False,
        fast_mode=False,
        intra_batch_mode=False,
        kv_cache_dtype=fp8_dtype,
        kernel_num_head=16,
        head_pad_mode="none",
        dcp_world_size=1,
    )
    kwargs.update(overrides)
    return aiter_backend._keep_mla_ps_without_nps_kernel(**kwargs)


class TestMlaPsWithoutNpsKernel(CustomTestCase):
    def _arch(self, *, hip=True, gfx95=False, gfx1250=False):
        for name, value in (
            ("is_hip", hip),
            ("is_gfx95_supported", gfx95),
            ("is_gfx1250_supported", gfx1250),
        ):
            patcher = mock.patch.object(aiter_backend, name, return_value=value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_gfx942_fp8_kv_16_heads_uses_persistent_kernel(self):
        self._arch()
        for pad_mode in ("none", "repeat"):
            with (
                self.subTest(pad_mode),
                self.assertLogs(aiter_backend.logger, level="WARNING") as logs,
            ):
                self.assertEqual(_resolve(head_pad_mode=pad_mode), _PERSIST_ON)
            self.assertIn("--kv-cache-dtype auto", logs.output[0])

    def test_other_configurations_keep_persist_off(self):
        cases = (
            ("gfx950", dict(gfx95=True), {}),
            ("gfx1250", dict(gfx1250=True), {}),
            ("cuda", dict(hip=False), {}),
            ("bf16 kv", {}, dict(kv_cache_dtype=torch.bfloat16)),
            ("128 heads", {}, dict(kernel_num_head=128)),
            ("zero pad (h12, gluon)", {}, dict(head_pad_mode="zero")),
            ("dcp", {}, dict(dcp_world_size=2)),
        )
        for name, arch, overrides in cases:
            with (
                self.subTest(name),
                mock.patch.object(aiter_backend.logger, "warning") as warning,
            ):
                with (
                    mock.patch.object(
                        aiter_backend, "is_hip", return_value=arch.get("hip", True)
                    ),
                    mock.patch.object(
                        aiter_backend,
                        "is_gfx95_supported",
                        return_value=arch.get("gfx95", False),
                    ),
                    mock.patch.object(
                        aiter_backend,
                        "is_gfx1250_supported",
                        return_value=arch.get("gfx1250", False),
                    ),
                ):
                    self.assertEqual(_resolve(**overrides), _PERSIST_OFF)
                warning.assert_not_called()

    def test_persist_on_is_left_alone(self):
        self._arch()
        state = (True, True, False)
        self.assertEqual(
            _resolve(use_ps=True, fast_mode=True, intra_batch_mode=False), state
        )


if __name__ == "__main__":
    unittest.main()
