import contextlib
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from sglang.srt.layers import layernorm_sp
from sglang.srt.models import qwen4_exp
from sglang.srt.runtime_context import get_forward, reset_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _Group:
    world_size = 2
    rank_in_group = 0

    def all_gather_into_tensor(self, output, local):
        output.copy_(torch.cat((local, local + 10)))

    def reduce_scatter_tensor(self, output, full):
        output.copy_(full.chunk(self.world_size)[self.rank_in_group])


class _Boundary:
    def prepare(self, hidden_states, forward_batch):
        return hidden_states

    def finish(self, hidden_states, forward_batch):
        return hidden_states

    @contextlib.contextmanager
    def exit(self, forward_batch):
        yield SimpleNamespace(defer_moe_finalize=True, finish=lambda value: value)


class TestQwen4ExpFullRowFallbacks(CustomTestCase):
    def tearDown(self):
        reset_context()

    def _parallel(self):
        return patch.object(
            layernorm_sp,
            "get_parallel",
            return_value=SimpleNamespace(tp_group=_Group()),
        )

    def test_gdn_runs_on_full_rows_and_returns_to_the_shard(self):
        seen = []
        layer = SimpleNamespace(
            _prepare_attn_stage=lambda hidden, batch, ple: hidden,
            linear_attn=lambda hidden, batch: (
                seen.append((hidden.shape[0], get_forward().sp_active)) or hidden
            ),
            attn_boundary=_Boundary(),
            _run_ffn_stage=lambda hidden, batch: hidden,
        )
        batch = SimpleNamespace(forward_mode=SimpleNamespace(is_idle=lambda: False))
        with self._parallel(), get_forward().scoped(sp_active=True):
            output = qwen4_exp.Qwen4ExpLinearDecoderLayer.forward(
                layer, torch.ones(3, 2), forward_batch=batch, ple_batch=None
            )
        self.assertEqual(seen, [(6, False)])
        self.assertEqual(output.shape, (3, 2))

    def test_qsa_indexer_receives_full_rows_once(self):
        seen = []
        layer = qwen4_exp.Qwen4ExpAttentionDecoderLayer.__new__(
            qwen4_exp.Qwen4ExpAttentionDecoderLayer
        )
        torch.nn.Module.__init__(layer)
        layer.is_qsa = True
        layer.alt_stream = None
        layer._prepare_attn_stage = lambda hidden, batch, ple: hidden
        layer.attn_boundary = _Boundary()
        layer._run_ffn_stage = lambda hidden, batch: hidden
        layer._prepare_qkv_gate = lambda positions, hidden_states, forward_batch: (
            hidden_states,
            hidden_states,
            hidden_states,
            None,
        )
        layer._compute_qsa_topk_indices = lambda hidden, positions, batch: (
            seen.append((hidden.shape[0], get_forward().sp_active))
            or torch.zeros(hidden.shape[0], 1, dtype=torch.long)
        )
        layer.attn = lambda q, k, v, batch, **kwargs: q
        layer.o_proj = lambda hidden: (hidden, None)
        batch = SimpleNamespace(forward_mode=SimpleNamespace(is_idle=lambda: False))
        with (
            self._parallel(),
            get_forward().scoped(sp_active=True),
            patch.object(qwen4_exp, "get_is_capture_mode", return_value=False),
        ):
            output = qwen4_exp.Qwen4ExpAttentionDecoderLayer.forward(
                layer,
                torch.arange(5),
                torch.ones(3, 2),
                batch,
                ple_batch=None,
            )
        self.assertEqual(seen, [(5, False)])
        self.assertEqual(output.shape, (3, 2))

    def test_moe_runs_on_full_rows_without_deferred_finalize(self):
        calls = []

        class FakeMoe:
            def __call__(self, hidden, batch, defer_finalize):
                calls.append((hidden.shape[0], defer_finalize, get_forward().sp_active))
                return hidden

        layer = SimpleNamespace(ffn_boundary=_Boundary(), mlp=FakeMoe())
        with (
            self._parallel(),
            get_forward().scoped(sp_active=True),
            patch.object(qwen4_exp, "Qwen2MoeSparseMoeBlock", FakeMoe),
        ):
            output = qwen4_exp.Qwen4ExpLayerExtensionMixin._run_ffn_stage(
                layer, torch.ones(3, 2), SimpleNamespace()
            )
        self.assertEqual(calls, [(6, False, False)])
        self.assertEqual(output.shape, (3, 2))


if __name__ == "__main__":
    unittest.main()
