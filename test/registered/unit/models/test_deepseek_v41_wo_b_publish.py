"""CPU tests of the DSV4.1 wo_b publish glue (--enable-dsv41-low-latency-decode): when
the attention's wo_b publishes into the Lamport mailbox, that the published rows take
only the consumer, and that a publish never outlives its consumer."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import contextlib
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

import sglang.srt.models.deepseek_v4 as deepseek_v4
import sglang.srt.models.deepseek_v4_mhc as mhc
from sglang.srt.models.dsv41_low_latency import mxfp8_linear, switch, wo_b_publish
from sglang.test.test_utils import CustomTestCase

_CFG = mhc.HcConfig(
    mult=4,
    sinkhorn_iters=20,
    eps=1e-6,
    rms_eps=1e-6,
    hidden=8,
    pre_from_prev=True,
    cp_prefill=False,
)
_LAMPORT = "sglang.srt.models.dsv41_low_latency.lamport"


class _Gemm:
    """PublishingMxfp8Linear's host API: supports / prepare / call."""

    def __init__(self, supported=True, prepared=("a", "scale", (2, 8))):
        self.supported = supported
        self.prepared = prepared
        self.calls = []

    def supports(self, linear, num_tokens):
        return self.supported and 0 < num_tokens <= 8

    def prepare(self, linear, x):
        return self.prepared

    def __call__(self, linear, prepared):
        self.calls.append(prepared)
        return torch.full((2, 8), 3.0)


def _publisher(gemm=None):
    publisher = wo_b_publish.WoBPublisher(hidden=8, hc_mult=4, top_k=6)
    publisher._built = True
    publisher._gemm = gemm or _Gemm()
    publisher._all_reduce = MagicMock(
        return_value=(torch.ones(2, 4, 8), torch.ones(2, 8))
    )
    return publisher


def _eager():
    return patch.object(torch.cuda, "is_current_stream_capturing", return_value=False)


class TestWoBPublisher(CustomTestCase):
    def test_no_build_during_capture(self):
        """Symmetric memory cannot be allocated under capture: decline, build later."""
        publisher = wo_b_publish.WoBPublisher(hidden=8, hc_mult=4, top_k=6)
        with (
            patch.object(torch.cuda, "is_current_stream_capturing", return_value=True),
            patch.object(publisher, "_build") as build,
        ):
            self.assertIsNone(publisher.publish("wo_b", torch.zeros(2, 8)))
        build.assert_not_called()

    def test_builds_once_on_the_first_eager_publish(self):
        publisher = wo_b_publish.WoBPublisher(hidden=8, hc_mult=4, top_k=6)

        def build(wo_b):
            publisher._built = True
            publisher._gemm = _Gemm()

        with _eager(), patch.object(publisher, "_build", side_effect=build) as built:
            for _ in range(2):
                self.assertTrue(
                    publisher.take(publisher.publish("wo_b", torch.zeros(2, 8)))
                )
        built.assert_called_once()

    def test_a_decline_leaves_nothing_pending(self):
        for gemm in (_Gemm(supported=False), _Gemm(prepared=None)):
            publisher = _publisher(gemm)
            with _eager():
                self.assertIsNone(publisher.publish("wo_b", torch.zeros(2, 8)))
                self.assertEqual(gemm.calls, [])
                # A pending publish would make this raise.
                publisher._gemm = _Gemm()
                self.assertIsNotNone(publisher.publish("wo_b", torch.zeros(2, 8)))

    def test_publish_before_the_previous_rows_were_reduced_raises(self):
        """An unreduced publish desyncs the mailbox generations on every rank."""
        publisher = _publisher()
        with _eager():
            published = publisher.publish("wo_b", (torch.zeros(2, 8), "scales"))
            self.assertFalse(publisher.take(torch.zeros(2, 8)))
            with self.assertRaises(RuntimeError):
                publisher.publish("wo_b", torch.zeros(2, 8))
            self.assertTrue(publisher.take(published))
            self.assertFalse(publisher.take(published))
            publisher.publish("wo_b", torch.zeros(2, 8))

    def test_build_precompiles_under_the_pinned_tactics(self):
        """The first-forward precompile tunes wo_b inside an autotune pass; it must see
        only the pins, or the pinned pass finds an unpinned cache entry for bucket 8."""
        state = {"pinned": False, "seen": []}

        @contextlib.contextmanager
        def pins():
            state["pinned"] = True
            try:
                yield
            finally:
                state["pinned"] = False

        class _Linear:
            def __init__(self, all_reduce):
                self.all_reduce = all_reduce

            def precompile(self, linear, num_tokens):
                state["seen"].append((num_tokens, state["pinned"]))

        all_reduce = MagicMock()
        fakes = {
            f"{_LAMPORT}.all_reduce_mhc": SimpleNamespace(AllReduceMHC=all_reduce),
            f"{_LAMPORT}.publishing_gemm": SimpleNamespace(
                PublishingMxfp8Linear=_Linear
            ),
        }
        wo_b = SimpleNamespace(weight=SimpleNamespace(device="cuda:0"))
        publisher = wo_b_publish.WoBPublisher(hidden=5120, hc_mult=4, top_k=6)
        with (
            patch.dict(sys.modules, fakes),
            patch.object(mxfp8_linear, "pinned_tactics", pins),
        ):
            publisher._build(wo_b)
        self.assertEqual(state["seen"], [(n, True) for n in range(1, 9)])
        self.assertEqual(
            all_reduce.call_args.kwargs,
            dict(
                hidden_size=5120,
                hc_mult=4,
                max_num_tokens=switch.WO_B_MAILBOX_MAX_TOKENS,
                top_k=6,
                device="cuda:0",
            ),
        )


class TestLayerPublish(CustomTestCase):
    def test_publishes_only_when_the_layer_defers_the_sum(self):
        publisher = MagicMock()
        publisher.publish.return_value = "rows"
        layer = SimpleNamespace(defers_attn_sum=lambda rows: rows <= 6)
        with patch.object(wo_b_publish, "_publisher", publisher):
            bound = wo_b_publish._LayerPublish(layer)
            self.assertEqual(bound("wo_b", torch.zeros(6, 8)), "rows")
            # A quantized input counts the rows of its values.
            x = (torch.zeros(6, 8), torch.zeros(6))
            self.assertEqual(bound("wo_b", x), "rows")
            self.assertIsNone(bound("wo_b", torch.zeros(9, 8)))
        self.assertEqual(publisher.publish.call_count, 2)

    def test_install_shares_one_publisher_without_a_module_cycle(self):
        def layer(hidden):
            layer = torch.nn.Module()
            layer.self_attn = torch.nn.Module()
            layer.hc_cfg = SimpleNamespace(hidden=hidden, mult=4)
            layer.config = SimpleNamespace(num_experts_per_tok=6)
            return layer

        layers = [layer(5120), layer(5120)]
        with patch.object(wo_b_publish, "_publisher", None):
            for each in layers:
                wo_b_publish.install(each)
            shared = wo_b_publish._publisher
        self.assertEqual((shared._hidden, shared._hc_mult, shared._top_k), (5120, 4, 6))
        for each in layers:
            self.assertIsInstance(
                each.self_attn.wo_b_publish, wo_b_publish._LayerPublish
            )
            # A Module-valued attribute would make the layer its attention's child.
            self.assertEqual(len(list(each.modules())), 2)

    def test_take_without_a_publisher(self):
        with patch.object(wo_b_publish, "_publisher", None):
            self.assertFalse(wo_b_publish.take(torch.zeros(2, 8)))


class TestRunAttnPost(CustomTestCase):
    def _run(self, publisher, partial):
        rows = 2
        residual = torch.randn(rows, 4, 8)
        triplet = (torch.rand(rows, 4), torch.rand(rows, 4), torch.rand(rows, 4, 4))
        norm = SimpleNamespace(
            weight=torch.nn.Parameter(torch.ones(8)), variance_epsilon=1e-6
        )
        jit = MagicMock(
            return_value=(None, torch.zeros(rows, 4, 8), torch.zeros(rows, 8))
        )
        with (
            patch.object(wo_b_publish, "_publisher", publisher),
            patch(
                "sglang.kernels.ops.communication.all_reduce_mhc."
                "all_reduce_mhc_post_combine_norm",
                jit,
            ),
        ):
            state = mhc.run_attn_post(
                SimpleNamespace(cfg=_CFG),
                mhc.AttnOutput(partial),
                residual,
                stats_stream=None,
                next=SimpleNamespace(norm=norm),
                world_size=4,
                precomputed=triplet,
            )
        return state, jit, triplet, residual

    def test_published_rows_launch_only_the_consumer(self):
        publisher = _publisher()
        with _eager():
            partial = publisher.publish("wo_b", torch.zeros(2, 8))
        state, jit, (pre, post, comb), residual = self._run(publisher, partial)
        jit.assert_not_called()
        consumer = publisher._all_reduce
        args = consumer.call_args.args
        self.assertIs(args[0], partial)
        self.assertIs(args[1], residual)
        self.assertIs(args[2], post)
        self.assertIs(args[3], comb)
        self.assertIs(args[4], pre)
        self.assertFalse(args[5].requires_grad)
        self.assertIs(state.streams, consumer.return_value[0])
        self.assertIs(state.input.rows, consumer.return_value[1])
        self.assertFalse(publisher.take(partial))

    def test_unpublished_rows_keep_the_jit_kernel(self):
        for publisher in (None, _publisher()):
            with self.subTest(publisher=publisher is not None):
                state, jit, _, _ = self._run(publisher, torch.zeros(2, 8))
                jit.assert_called_once()
                self.assertIs(state.streams, jit.return_value[1])
                if publisher is not None:
                    publisher._all_reduce.assert_not_called()


class TestProjectWoB(CustomTestCase):
    def _attention(self, publish):
        attention = object.__new__(deepseek_v4.MQALayer)
        torch.nn.Module.__init__(attention)
        attention.attn_tp_size = 4
        attention.wo_b = MagicMock(return_value=(torch.ones(2, 8), None))
        attention.wo_b.reduce_results = True
        attention.wo_b.use_decode_attn_tp = False
        attention.wo_b_publish = publish
        return attention

    def _project(self, attention, defer=True):
        with (
            patch.object(
                deepseek_v4, "get_parallel", return_value=SimpleNamespace(tp_size=4)
            ),
            patch(
                "sglang.srt.layers.moe.utils.should_skip_mlp_all_reduce",
                return_value=False,
            ),
        ):
            return attention._project_wo_b(torch.zeros(2, 2, 4), defer)

    def test_published(self):
        rows = torch.full((2, 8), 3.0)
        publish = MagicMock(return_value=rows)
        attention = self._attention(publish)
        out = self._project(attention)
        self.assertIsInstance(out, mhc.AttnOutput)
        self.assertIs(out.partial, rows)
        self.assertEqual(publish.call_args.args[1].shape, (2, 8))
        attention.wo_b.assert_not_called()

    def test_declined_runs_wo_b_without_its_sum(self):
        attention = self._attention(MagicMock(return_value=None))
        out = self._project(attention)
        self.assertIsInstance(out, mhc.AttnOutput)
        attention.wo_b.assert_called_once()
        self.assertTrue(attention.wo_b.call_args.kwargs["skip_all_reduce"])

    def test_not_deferred_never_publishes(self):
        publish = MagicMock()
        out = self._project(self._attention(publish), defer=False)
        self.assertIsInstance(out, torch.Tensor)
        publish.assert_not_called()


if __name__ == "__main__":
    unittest.main()
