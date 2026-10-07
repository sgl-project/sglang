"""DeepSeek-V4.1 mHC: in wide target prefill (4096 to 65536 rows) each sublayer's
triplet runs on the stats stream beside that sublayer's TP all-reduce. The stream
forks right before the collective, not before the combine; decode and verify keep
forking before the combine, and other prefill widths, batch-invariant mode and
the excluded layouts compute the triplet in line. Runs the real
forward_hc_pre_from_prev, posts and mix_stats on a mocked layer with recorded
streams, stopping at the kernels."""

import unittest
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest import mock

import torch

import sglang.srt.models.deepseek_v4 as deepseek_v4
import sglang.srt.models.deepseek_v4_mhc as deepseek_v4_mhc
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.models.deepseek_v2 import MoEOutput
from sglang.srt.runtime_context import override_platform
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

HC, HIDDEN = 4, 5120
WIDE_ROWS = (4096, 8192, 65536)
NARROW_ROWS = (6, 4095, 65537)
PREFILL_MODES = (ForwardMode.EXTEND, ForwardMode.MIXED, ForwardMode.SPLIT_PREFILL)


class _Stream:
    def __init__(self, name, events):
        self.name = name
        self.events = events

    def wait_stream(self, other):
        self.events.append(("wait", self.name, other.name))


class _Pieces(MoEOutput):
    """A MoE output whose merge is recorded instead of run."""

    __slots__ = ()

    def get_merged(self):
        self.experts.events.append(("merge", self.experts.current()))
        return self.routed


def _config(cp_prefill=False):
    return deepseek_v4_mhc.HcConfig(
        mult=HC,
        sinkhorn_iters=20,
        eps=1e-6,
        rms_eps=1e-6,
        hidden=HIDDEN,
        pre_from_prev=True,
        cp_prefill=cp_prefill,
    )


class _Run:
    """One forward_hc_pre_from_prev over a mocked layer, recording stream order."""

    def __init__(self, rows, mode, *, honor_handover=True, shared=None):
        self.rows = rows
        self.events = []
        self.main = _Stream("main", self.events)
        self.side = _Stream("side", self.events)
        self.stack = [self.main]
        self.honor_handover = honor_handover
        self.shared = shared
        self.residual = mock.Mock(is_cuda=True, shape=(rows, HC, HIDDEN))
        self.forward_batch = SimpleNamespace(forward_mode=mode)
        self.attn_partial = mock.Mock(name="attn_partial")
        self.attn_reduced = mock.Mock(name="attn_reduced")
        self.attn_plain = mock.Mock(name="attn_plain")
        self.moe_routed = mock.Mock(name="moe_routed")
        self.moe_reduced = mock.Mock(name="moe_reduced")
        self.moe_plain = mock.Mock(name="moe_plain")
        self.post_inputs = {}
        self.attn_defer = []
        self.moe_pieces = []
        self.coefficients = []

    def current(self):
        return self.stack[-1].name

    @contextmanager
    def _on(self, stream):
        self.stack.append(stream)
        try:
            yield
        finally:
            self.stack.pop()

    def _hc(self, name):
        return SimpleNamespace(name=name, cfg=None)

    def _mix_stats_impl(self, hc, x, **kwargs):
        assert x is self.residual
        self.events.append(("stats", hc.name, self.current()))
        coefficients = tuple(mock.Mock() for _ in range(3))
        self.coefficients.extend(coefficients)
        return coefficients

    def _combine(self, hc, state, quantized=None):
        self.events.append(("combine", hc.name, self.current()))
        return mock.Mock(shape=(self.rows, HIDDEN))

    def _attention(self, *, defer_all_reduce, **kwargs):
        self.attn_defer.append(defer_all_reduce)
        self.events.append(("attention", self.current()))
        if defer_all_reduce and self.honor_handover:
            return deepseek_v4_mhc.AttnOutput(self.attn_partial)
        return self.attn_plain

    def _moe(self, x, forward_batch, *, return_moe_output, **kwargs):
        self.moe_pieces.append(return_moe_output)
        self.events.append(("moe", self.current()))
        if not return_moe_output:
            return self.moe_plain
        return _Pieces(
            routed=self.moe_routed,
            shared=self.shared,
            experts=self,
            routed_scaling_factor=1.0,
            shared_is_replicated=self.shared is not None,
        )

    def _attn_all_reduce(self, partial):
        assert partial is self.attn_partial
        self.events.append(("all_reduce", "attn", self.current()))
        return self.attn_reduced

    def _moe_all_reduce(self, merged):
        assert merged is self.moe_routed
        self.events.append(("all_reduce", "moe", self.current()))
        return self.moe_reduced

    def _post_fusion(self, hc, y, residual, coefficients, next):
        assert residual is self.residual
        self.post_inputs[hc.name] = y
        self.events.append(("post", hc.name, self.current()))
        return mock.Mock(residual=self.residual)

    def layer(self, *, stats_stream=True):
        layer = mock.MagicMock()
        layer.hc_cfg = _config()
        layer.attn_hc = self._hc("attn")
        layer.ffn_hc = self._hc("ffn")
        layer.hc_stats_stream = self.side if stats_stream else None
        layer._can_fuse_attn_mhc = False
        layer._can_fuse_ffn_mhc = False
        layer._can_overlap_attn_stats = stats_stream
        layer._can_overlap_ffn_stats = stats_stream
        layer.self_attn.accepts_mxfp8_swizzled_input.return_value = False
        layer.self_attn.maybe_use_decode_attn_tp.return_value = nullcontext()
        layer.self_attn.side_effect = self._attention
        layer._run_moe_ffn_dp_sync.side_effect = self._moe
        layer.mlp.tp_size = 4
        return layer

    def __call__(
        self,
        *,
        batch_invariant=False,
        is_blackwell=True,
        attn_dp_size=1,
        cp_prefill=False,
        stats_stream=True,
    ):
        layer = self.layer(stats_stream=stats_stream)
        layer.hc_cfg = _config(cp_prefill=cp_prefill)
        with (
            override_platform(is_blackwell=is_blackwell, is_sm90=False, is_sm100=False),
            mock.patch(
                "sglang.srt.batch_invariant_ops.is_batch_invariant_mode_enabled",
                return_value=batch_invariant,
            ),
            mock.patch.object(
                deepseek_v4_mhc,
                "get_parallel",
                return_value=SimpleNamespace(attn_dp_size=attn_dp_size),
            ),
            mock.patch.object(
                deepseek_v4_mhc,
                "get_forward",
                return_value=SimpleNamespace(sp_active=False),
            ),
            mock.patch.object(
                torch.cuda, "current_stream", side_effect=lambda: self.stack[-1]
            ),
            mock.patch.object(torch.cuda, "stream", side_effect=self._on),
            mock.patch.object(
                deepseek_v4_mhc, "_mix_stats_impl", side_effect=self._mix_stats_impl
            ),
            mock.patch.object(deepseek_v4_mhc, "combine", side_effect=self._combine),
            mock.patch.object(
                deepseek_v4_mhc, "_post_fusion", side_effect=self._post_fusion
            ),
            mock.patch(
                "sglang.srt.layers.dp_attention.attn_tp_all_reduce",
                side_effect=self._attn_all_reduce,
            ),
            mock.patch(
                "sglang.srt.layers.moe.post_experts_all_reduce",
                side_effect=self._moe_all_reduce,
            ),
            mock.patch(
                "sglang.srt.layers.moe.utils.should_add_replicated_moe_output",
                return_value=True,
            ),
        ):
            deepseek_v4.DeepseekV4DecoderLayer.forward_hc_pre_from_prev(
                layer,
                positions=object(),
                state=mock.Mock(residual=self.residual),
                input_ids=object(),
                forward_batch=self.forward_batch,
                input_ids_global=object(),
            )
        return self.events


# The wide-prefill order of #39704: the side stream forks after the sublayer's
# compute (and the MoE merge), right before its all-reduce, and the main stream
# joins it before the post.
OVERLAPPED = [
    ("combine", "attn", "main"),
    ("attention", "main"),
    ("wait", "side", "main"),
    ("stats", "attn", "side"),
    ("all_reduce", "attn", "main"),
    ("wait", "main", "side"),
    ("post", "attn", "main"),
    ("combine", "ffn", "main"),
    ("moe", "main"),
    ("merge", "main"),
    ("wait", "side", "main"),
    ("stats", "ffn", "side"),
    ("all_reduce", "moe", "main"),
    ("wait", "main", "side"),
    ("post", "ffn", "main"),
]

# Everything on the main stream, each triplet after its sublayer has reduced.
IN_LINE = [
    ("combine", "attn", "main"),
    ("attention", "main"),
    ("stats", "attn", "main"),
    ("post", "attn", "main"),
    ("combine", "ffn", "main"),
    ("moe", "main"),
    ("stats", "ffn", "main"),
    ("post", "ffn", "main"),
]

# Decode and verify: fork before each combine, as before this change.
DECODE = [
    ("wait", "side", "main"),
    ("combine", "attn", "main"),
    ("attention", "main"),
    ("stats", "attn", "side"),
    ("wait", "main", "side"),
    ("post", "attn", "main"),
    ("wait", "side", "main"),
    ("combine", "ffn", "main"),
    ("moe", "main"),
    ("stats", "ffn", "side"),
    ("wait", "main", "side"),
    ("post", "ffn", "main"),
]


class TestMhcPrefillStatsOverlap(CustomTestCase):
    def _assert_side_stream_lifetimes(self, run):
        # The residual is read on the stats stream and both triplets are read on
        # the main stream; each must be recorded there so the caching allocator
        # does not reuse its memory early.
        self.assertEqual(
            run.residual.record_stream.call_args_list, [mock.call(run.side)] * 2
        )
        self.assertEqual(len(run.coefficients), 6)
        for coefficient in run.coefficients:
            coefficient.record_stream.assert_called_once_with(run.main)

    def _assert_in_line(self, run, events):
        self.assertEqual(events, IN_LINE)
        run.residual.record_stream.assert_not_called()
        for coefficient in run.coefficients:
            coefficient.record_stream.assert_not_called()
        self.assertEqual(run.attn_defer, [False])
        self.assertEqual(run.moe_pieces, [False])
        self.assertIs(run.post_inputs["attn"], run.attn_plain)
        self.assertIs(run.post_inputs["ffn"], run.moe_plain)

    def test_wide_prefill_runs_stats_beside_each_all_reduce(self):
        for mode in PREFILL_MODES:
            for rows in WIDE_ROWS:
                with self.subTest(mode=mode.name, rows=rows):
                    run = _Run(rows, mode)
                    self.assertEqual(run(), OVERLAPPED)
                    self.assertEqual(run.attn_defer, [True])
                    self.assertEqual(run.moe_pieces, [True])
                    self.assertIs(run.post_inputs["attn"], run.attn_reduced)
                    self.assertIs(run.post_inputs["ffn"], run.moe_reduced)
                    self._assert_side_stream_lifetimes(run)

    def test_other_prefill_widths_compute_stats_in_line(self):
        for rows in NARROW_ROWS:
            with self.subTest(rows=rows):
                run = _Run(rows, ForwardMode.EXTEND)
                self._assert_in_line(run, run())

    def test_decode_and_verify_keep_forking_before_the_combine(self):
        for mode in (ForwardMode.DECODE, ForwardMode.TARGET_VERIFY):
            for rows in (6, 8192):
                with self.subTest(mode=mode.name, rows=rows):
                    run = _Run(rows, mode)
                    self.assertEqual(run(), DECODE)
                    self.assertEqual(run.attn_defer, [False])
                    self.assertEqual(run.moe_pieces, [False])
                    self._assert_side_stream_lifetimes(run)

    def test_batch_invariant_mode_disables_the_overlap(self):
        run = _Run(8192, ForwardMode.EXTEND)
        self._assert_in_line(run, run(batch_invariant=True))

    def test_excluded_layouts_compute_stats_in_line(self):
        cases = {
            "not_blackwell": dict(is_blackwell=False),
            "attention_dp": dict(attn_dp_size=2),
            "prefill_cp": dict(cp_prefill=True),
            "no_stats_stream": dict(stats_stream=False),
        }
        for name, kwargs in cases.items():
            with self.subTest(name):
                run = _Run(8192, ForwardMode.EXTEND)
                self._assert_in_line(run, run(**kwargs))

    def test_declined_handover_computes_stats_in_line(self):
        # The attention can return reduced rows even when asked to defer; the type
        # is the ground truth, and nothing is left to overlap.
        run = _Run(8192, ForwardMode.EXTEND, honor_handover=False)
        events = run()
        self.assertEqual(events[:4], IN_LINE[:4])
        self.assertIs(run.post_inputs["attn"], run.attn_plain)
        self.assertEqual(events[4:], OVERLAPPED[7:])

    def test_replicated_shared_expert_joins_after_the_reduction(self):
        shared = torch.full((2,), 3.0)
        run = _Run(8192, ForwardMode.EXTEND, shared=shared)
        run.moe_reduced = torch.ones(2)
        self.assertEqual(run(), OVERLAPPED)
        self.assertTrue(torch.equal(run.post_inputs["ffn"], torch.full((2,), 4.0)))


class TestMhcPrefillStatsOverlapGates(CustomTestCase):
    def _attn_layer(self, *, tp_size=4, attn_tp_size=4, reduce_results=True):
        return SimpleNamespace(
            hc_stats_stream=object(),
            self_attn=SimpleNamespace(
                attn_tp_size=attn_tp_size,
                wo_b=SimpleNamespace(reduce_results=reduce_results),
            ),
            _tp_size=tp_size,
        )

    def _attn_gate(self, layer):
        with mock.patch.object(
            deepseek_v4,
            "get_parallel",
            return_value=SimpleNamespace(tp_size=layer._tp_size),
        ):
            return deepseek_v4.DeepseekV4DecoderLayer._can_overlap_attn_stats.func(
                layer
            )

    def _ffn_gate(self, *, tp_size=4, a2a_none=True, reduce_results=True):
        layer = SimpleNamespace(
            hc_stats_stream=object(),
            mlp=SimpleNamespace(tp_size=tp_size, reduce_results=reduce_results),
        )
        backend = mock.Mock()
        backend.is_none.return_value = a2a_none
        with mock.patch.object(
            deepseek_v4, "get_moe_a2a_backend", return_value=backend
        ):
            return deepseek_v4.DeepseekV4DecoderLayer._can_overlap_ffn_stats.func(layer)

    def test_attention_gate(self):
        self.assertTrue(self._attn_gate(self._attn_layer()))
        self.assertFalse(self._attn_gate(self._attn_layer(tp_size=8, attn_tp_size=4)))
        self.assertFalse(self._attn_gate(self._attn_layer(tp_size=2, attn_tp_size=2)))
        self.assertFalse(self._attn_gate(self._attn_layer(reduce_results=False)))
        layer = self._attn_layer()
        layer.hc_stats_stream = None
        self.assertFalse(self._attn_gate(layer))

    def test_moe_gate(self):
        self.assertTrue(self._ffn_gate())
        self.assertFalse(self._ffn_gate(tp_size=2))
        self.assertFalse(self._ffn_gate(a2a_none=False))
        self.assertFalse(self._ffn_gate(reduce_results=False))


if __name__ == "__main__":
    unittest.main()
