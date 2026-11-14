import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.communication import k3_ar_fusion, k3_sp_collective
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.models import kimi_k3
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _Group:
    def __init__(self, world_size=8, rank=3):
        self.world_size = world_size
        self.rank_in_group = rank
        self.ca_comm = None

    def reduce_scatter_tensor(self, out, x):
        rows = out.shape[0]
        out.copy_(
            x[self.rank_in_group * rows : (self.rank_in_group + 1) * rows]
            * self.world_size
        )

    def all_gather_into_tensor(self, out, x):
        out.copy_(x.repeat(self.world_size, *([1] * (x.dim() - 1))))


def _parallel(world_size=8, rank=3):
    return SimpleNamespace(tp_group=_Group(world_size, rank), tp_size=world_size)


class _Model:
    _prefill_sp_first_layer = 1
    _prefill_sp_begin = kimi_k3.KimiK3LinearModel._prefill_sp_begin


def _batch(mode=ForwardMode.EXTEND):
    return SimpleNamespace(forward_mode=mode)


class _ExpertsSkippingTheQuant:
    quant_method = SimpleNamespace(hidden_size=64)

    def __call__(self, x, topk_output):
        return torch.full((x.shape[0], 8), 2.0)


class TestK31PrefillSPTail(CustomTestCase):
    def _begin(
        self, num_tokens, mode=ForwardMode.EXTEND, capture=False, max_rows=16384
    ):
        with (
            patch.object(kimi_k3, "get_parallel", return_value=_parallel()),
            patch.object(kimi_k3, "get_is_capture_mode", return_value=capture),
            patch.object(kimi_k3, "is_in_breakable_cuda_graph", return_value=False),
            patch.object(kimi_k3, "is_cp_active", return_value=False),
            patch.object(k3_ar_fusion, "symm_buffer_max_rows", return_value=max_rows),
        ):
            return _Model()._prefill_sp_begin(_batch(mode), torch.empty(num_tokens, 8))

    def test_eager_extend_at_or_above_the_floor(self):
        self.assertEqual(envs.SGLANG_K31_PREFILL_SP_TAIL_MIN_TOKENS.get(), 4097)
        for n in (4112, 12288, 16384):
            sp = self._begin(n)
            self.assertIsNotNone(sp, n)
            self.assertEqual((sp.num_tokens, sp.padded, sp.local), (n, n, n // 8))
            self.assertEqual(sp.rows, slice(3 * n // 8, 4 * n // 8))
            self.assertEqual(sp.first_layer, 1)
        self.assertIsNone(self._begin(4096))
        self.assertIsNone(self._begin(2048))

    def test_padding_to_whole_latent_rows(self):
        sp = self._begin(12345)
        self.assertEqual((sp.padded, sp.local), (12352, 1544))
        self.assertEqual(sp.rows, slice(3 * 1544, 4 * 1544))
        self.assertIsNone(self._begin(16380, max_rows=16376))

    def test_other_modes_graphs_and_kill_switch(self):
        for mode in (
            ForwardMode.DECODE,
            ForwardMode.TARGET_VERIFY,
            ForwardMode.MIXED,
            ForwardMode.IDLE,
        ):
            self.assertIsNone(self._begin(16384, mode=mode), mode)
        self.assertIsNone(self._begin(16384, capture=True))
        with envs.SGLANG_K31_PREFILL_SP_TAIL.override(False):
            self.assertIsNone(self._begin(16384))
        with envs.SGLANG_K31_PREFILL_SP_TAIL_MIN_TOKENS.override(2048):
            self.assertIsNotNone(self._begin(2048))

    def test_nccl_fallback(self):
        group = _Group(world_size=8, rank=5)
        parallel = SimpleNamespace(tp_group=group)
        x = torch.arange(16 * 6, dtype=torch.float32).view(16, 6)
        residual = torch.ones(2, 6)
        with (
            patch.object(k3_sp_collective, "get_parallel", return_value=parallel),
            patch.object(k3_sp_collective, "prefill_sp_comm", return_value=None),
        ):
            out = k3_sp_collective.prefill_sp_reduce_scatter(
                x, torch.empty(2, 6), residual, num_blocks=8
            )
            torch.testing.assert_close(out, x[10:12] * 8 + 1)
            local = torch.randn(2, 6)
            full = k3_sp_collective.prefill_sp_all_gather(local, num_blocks=8)
            self.assertEqual(full.shape, (16, 6))
            torch.testing.assert_close(full[10:12], local)

    def test_runner_that_skips_the_gathered_quant_fails(self):
        with self.assertRaises(RuntimeError):
            kimi_k3.KimiK3MoE._forward_routed_gathered(
                SimpleNamespace(experts=_ExpertsSkippingTheQuant()),
                torch.ones(3, 2),
                torch.zeros(3, 2, dtype=torch.int32),
                torch.empty(3, 64, dtype=torch.float8_e4m3fn),
                torch.zeros(3, 2, dtype=torch.uint8),
                torch.zeros(3, 8),
            )


if __name__ == "__main__":
    unittest.main()
