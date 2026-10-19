"""DeepSeek-V4.1 mHC: in wide target prefill (4096 to 65536 rows) each sublayer's
triplet runs on the stats stream beside that sublayer's TP all-reduce. The boundary
transport forks the stream right before the sum it is about to complete (attention
or MoE), and the post joins it before reading the triplet; other prefill widths,
decode, batch-invariant mode and the excluded layouts compute the triplet in line.
Runs the real transports' prepare_output, the owed sums, HcUpdate.post, the posts
and mix_stats on a mocked layer with recorded streams, stopping at the kernels."""

import unittest
from contextlib import ExitStack, contextmanager
from types import SimpleNamespace
from unittest import mock

import torch

import sglang.srt.models.deepseek_v4 as deepseek_v4
import sglang.srt.models.deepseek_v4_mhc as deepseek_v4_mhc
from sglang.srt.layers.layer_boundary.layout import SumGroup
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.models import deepseek_v4_boundary as hc_boundary
from sglang.srt.models.deepseek_v2 import MoEOutput
from sglang.srt.runtime_context import override_platform
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

HC, HIDDEN = 4, 5120
WIDE_ROWS = (4096, 8192, 65536)
NARROW_ROWS = (6, 4095, 65537)
PREFILL_MODES = (ForwardMode.EXTEND, ForwardMode.MIXED, ForwardMode.SPLIT_PREFILL)


def _overlapped(name):
    return [
        ("wait", "side", "main"),
        ("stats", name, "side"),
        ("all_reduce", name, "main"),
        ("wait", "main", "side"),
        ("post", name, "main"),
    ]


def _in_line(name):
    return [
        ("all_reduce", name, "main"),
        ("stats", name, "main"),
        ("post", name, "main"),
    ]


class _Stream:
    def __init__(self, name, events):
        self.name = name
        self.events = events

    def wait_stream(self, other):
        self.events.append(("wait", self.name, other.name))


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


class _Pieces(MoEOutput):
    """Unmerged expert output, the shape the V4 MoE hands its transport."""

    __slots__ = ()

    def get_merged(self):
        return self.routed


class _Run:
    """One sublayer's transport exit and post, recording stream order."""

    def __init__(self, rows, mode, *, attention, merged_moe=False):
        self.rows = rows
        self.merged_moe = merged_moe
        self.attention = attention
        self.name = "attn" if attention else "ffn"
        self.events = []
        self.main = _Stream("main", self.events)
        self.side = _Stream("side", self.events)
        self.stack = [self.main]
        self.residual = mock.Mock(spec=torch.Tensor, is_cuda=True)
        self.residual.shape = (rows, HC, HIDDEN)
        self.state = hc_boundary.ReadState(self.residual, call=hc_boundary.LayerCall())
        self.batch = SimpleNamespace(
            forward_mode=mode, residual_stream=SimpleNamespace(residual=self.state)
        )
        self.partial = mock.Mock(name="partial", shape=(rows, HIDDEN))
        self.reduced = mock.Mock(name="reduced")
        self.coefficients = []
        self.posted = []

    def current(self):
        return self.stack[-1]

    @contextmanager
    def _on(self, stream):
        self.stack.append(stream)
        try:
            yield
        finally:
            self.stack.pop()

    def _mix_stats_impl(self, hc, x, **kwargs):
        assert x is self.residual
        self.events.append(("stats", hc.name, self.current().name))
        coefficients = tuple(mock.Mock() for _ in range(3))
        self.coefficients.extend(coefficients)
        return coefficients

    def _post_fusion(self, hc, y, residual, coefficients, next):
        assert y is self.reduced and residual is self.residual
        self.posted.append(coefficients)
        self.events.append(("post", hc.name, self.current().name))
        return mock.Mock()

    def _all_reduce(self, value):
        assert value is self.partial
        self.events.append(("all_reduce", self.name, self.current().name))
        return self.reduced

    def _layer(self, *, stats_stream, cp_prefill):
        return SimpleNamespace(
            hc_cfg=_config(cp_prefill=cp_prefill),
            attn_hc=SimpleNamespace(name="attn", cfg=None),
            ffn_hc=SimpleNamespace(name="ffn", cfg=None),
            hc_stats_stream=self.side if stats_stream else None,
            _can_overlap_attn_stats=stats_stream,
            _can_overlap_ffn_stats=stats_stream,
            boundary_pre_from_prev=True,
            self_attn=SimpleNamespace(attn_tp_size=4),
            mlp=SimpleNamespace(tp_size=4),
        )

    def _transport(self, layer):
        if self.attention:
            transport = hc_boundary.AttentionTransport.__new__(
                hc_boundary.AttentionTransport
            )
            transport.num_rows = self.rows
            transport.rows = SimpleNamespace(group=SumGroup.ATTN_TP, decode_group=None)
            transport.batch = self.batch
        else:
            transport = hc_boundary.MoeTransport.__new__(hc_boundary.MoeTransport)
            transport.rows = SimpleNamespace(group=SumGroup.MOE_OUTPUT)
            transport.mlp_reduce_scatter = False
            transport.batch = self.batch
        transport.layer = layer
        transport.defers_sum = False
        return transport

    def __call__(
        self,
        *,
        batch_invariant=False,
        is_blackwell=True,
        attn_dp_size=1,
        sp_active=False,
        cp_prefill=False,
        stats_stream=True,
    ):
        layer = self._layer(stats_stream=stats_stream, cp_prefill=cp_prefill)
        transport = self._transport(layer)
        if self.attention:
            output = deepseek_v4_mhc.AttnOutput(self.partial)
        elif self.merged_moe:
            output = self.partial
        else:
            output = _Pieces(self.partial, None, None, 1.0, False)
        with ExitStack() as stack:
            stack.enter_context(
                override_platform(is_blackwell=is_blackwell, is_sm90=False)
            )
            for target, name, value in (
                (deepseek_v4_mhc, "_mix_stats_impl", self._mix_stats_impl),
                (deepseek_v4_mhc, "_post_fusion", self._post_fusion),
                (
                    deepseek_v4_mhc,
                    "get_parallel",
                    lambda: SimpleNamespace(attn_dp_size=attn_dp_size),
                ),
                (
                    deepseek_v4_mhc,
                    "get_forward",
                    lambda: SimpleNamespace(sp_active=sp_active),
                ),
                (
                    hc_boundary,
                    "get_cp_decode_attn_tp_ctx",
                    lambda: SimpleNamespace(use_decode_attn_tp=False),
                ),
                (
                    hc_boundary,
                    "_sum_group",
                    lambda group: SimpleNamespace(all_reduce=self._all_reduce),
                ),
                (torch.cuda, "current_stream", self.current),
                (torch.cuda, "stream", self._on),
                (
                    "sglang.srt.batch_invariant_ops.is_batch_invariant_mode_enabled",
                    None,
                    lambda: batch_invariant,
                ),
            ):
                if name is None:
                    stack.enter_context(mock.patch(target, value))
                else:
                    stack.enter_context(mock.patch.object(target, name, value))
            if not self.attention:
                stack.enter_context(
                    mock.patch(
                        "sglang.srt.layers.moe.post_experts_all_reduce",
                        self._all_reduce,
                    )
                )
            owed = transport.prepare_output(output)
            reduced = owed.complete()
            hc_boundary.HcUpdate(layer, self.attention).post(reduced, self.state)
        return self.events


class TestMhcPrefillStatsOverlap(CustomTestCase):
    def _assert_side_stream_lifetimes(self, run):
        # The residual is read on the stats stream and the triplet on the main
        # stream; each must be recorded there so the caching allocator does not
        # reuse its memory early.
        self.assertEqual(
            run.residual.record_stream.call_args_list, [mock.call(run.side)]
        )
        self.assertEqual(len(run.coefficients), 3)
        for coefficient in run.coefficients:
            coefficient.record_stream.assert_called_once_with(run.main)
        self.assertEqual(run.posted, [tuple(run.coefficients)])

    def _assert_in_line(self, run, events):
        self.assertEqual(events, _in_line(run.name))
        run.residual.record_stream.assert_not_called()
        self.assertIsNone(run.state.stats_join)

    def test_wide_prefill_runs_stats_beside_each_all_reduce(self):
        sublayers = {"attention": {"attention": True}, "moe": {"attention": False}}
        sublayers["merged_moe"] = {"attention": False, "merged_moe": True}
        for name, kwargs in sublayers.items():
            for mode in PREFILL_MODES:
                for rows in WIDE_ROWS:
                    with self.subTest(sublayer=name, mode=mode.name, rows=rows):
                        run = _Run(rows, mode, **kwargs)
                        self.assertEqual(run(), _overlapped(run.name))
                        self._assert_side_stream_lifetimes(run)

    def test_other_prefill_widths_compute_stats_in_line(self):
        for attention in (True, False):
            for rows in NARROW_ROWS:
                with self.subTest(attention=attention, rows=rows):
                    run = _Run(rows, ForwardMode.EXTEND, attention=attention)
                    self._assert_in_line(run, run())

    def test_decode_and_verify_compute_stats_in_line_here(self):
        # The prefill overlap stays off; their fork before the combine
        # (use_stats_stream) is unchanged and not exercised here.
        for mode in (ForwardMode.DECODE, ForwardMode.TARGET_VERIFY):
            with self.subTest(mode=mode.name):
                run = _Run(8192, mode, attention=True)
                self._assert_in_line(run, run())

    def test_batch_invariant_mode_disables_the_overlap(self):
        run = _Run(8192, ForwardMode.EXTEND, attention=True)
        self._assert_in_line(run, run(batch_invariant=True))

    def test_excluded_layouts_compute_stats_in_line(self):
        cases = {
            "not_blackwell": dict(is_blackwell=False),
            "attention_dp": dict(attn_dp_size=2),
            "sequence_parallel": dict(sp_active=True),
            "prefill_cp": dict(cp_prefill=True),
            "no_stats_stream": dict(stats_stream=False),
        }
        for attention in (True, False):
            for name, kwargs in cases.items():
                with self.subTest(attention=attention, case=name):
                    run = _Run(8192, ForwardMode.EXTEND, attention=attention)
                    self._assert_in_line(run, run(**kwargs))

    def test_precomputed_stats_are_kept(self):
        run = _Run(8192, ForwardMode.EXTEND, attention=True)
        precomputed = tuple(mock.Mock() for _ in range(3))
        run.state.stats = precomputed
        events = run()
        self.assertEqual(
            events, [("all_reduce", "attn", "main"), ("post", "attn", "main")]
        )
        self.assertEqual(run.posted, [precomputed])


class TestMhcPrefillStatsOverlapGates(CustomTestCase):
    _cls = deepseek_v4.DeepseekV4DecoderLayerBase

    def _attn_gate(self, *, tp_size=4, attn_tp_size=4, stats_stream=True, lagged=True):
        layer = SimpleNamespace(
            boundary_pre_from_prev=lagged,
            hc_stats_stream=object() if stats_stream else None,
            self_attn=SimpleNamespace(attn_tp_size=attn_tp_size),
        )
        with mock.patch.object(
            deepseek_v4,
            "get_parallel",
            return_value=SimpleNamespace(tp_size=tp_size),
        ):
            return self._cls._can_overlap_attn_stats.func(layer)

    def _ffn_gate(self, *, tp_size=4, a2a_none=True, stats_stream=True, lagged=True):
        layer = SimpleNamespace(
            boundary_pre_from_prev=lagged,
            hc_stats_stream=object() if stats_stream else None,
            mlp=SimpleNamespace(tp_size=tp_size),
        )
        backend = mock.Mock()
        backend.is_none.return_value = a2a_none
        with mock.patch.object(
            deepseek_v4, "get_moe_a2a_backend", return_value=backend
        ):
            return self._cls._can_overlap_ffn_stats.func(layer)

    def test_attention_gate(self):
        self.assertTrue(self._attn_gate())
        self.assertFalse(self._attn_gate(tp_size=8, attn_tp_size=4))
        self.assertFalse(self._attn_gate(tp_size=2, attn_tp_size=2))
        self.assertFalse(self._attn_gate(stats_stream=False))
        self.assertFalse(self._attn_gate(lagged=False))

    def test_moe_gate(self):
        self.assertTrue(self._ffn_gate())
        self.assertFalse(self._ffn_gate(tp_size=2))
        self.assertFalse(self._ffn_gate(a2a_none=False))
        self.assertFalse(self._ffn_gate(stats_stream=False))
        self.assertFalse(self._ffn_gate(lagged=False))


if __name__ == "__main__":
    unittest.main()
