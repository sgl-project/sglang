"""Engram preparation, ordered collectives and per-layer joins on CPU."""

import unittest
from contextlib import contextmanager
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch
from torch import nn

from sglang.srt.environ import envs
from sglang.srt.layers.dp_attention import DpPaddingMode
from sglang.srt.layers.engram import Engram, EngramPrefetcher, EngramPrefetchFence
from sglang.srt.layers.moe import MoeA2ABackend
from sglang.srt.layers.quantization.fp8_utils import Mxfp8DenseGemmBackend
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.models.deepseek_v4 import DeepseekV4Model
from sglang.srt.runtime_context import get_context, get_flags
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _Table(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.arange(32, dtype=torch.float32).view(8, 4) / 32
        self.rows = []

    def forward(self, ids, batch, **kwargs):
        self.rows.append(ids.shape[0])
        return self.weight[ids]


class _Projection(nn.Linear):
    def forward(self, emb):
        if not emb.shape[0]:
            raise AssertionError("empty projection cannot run")
        return super().forward(emb), None


def _engram():
    # Avoid checkpoint/tokenizer loading; exercise real prepare/gate/forward.
    layer = Engram.__new__(Engram)
    nn.Module.__init__(layer)
    layer.embed = _Table()
    layer.wkv = _Projection(8, 12, bias=False)
    layer.wkv.quant_method = UnquantizedLinearMethod.__new__(UnquantizedLinearMethod)
    with torch.no_grad():
        layer.wkv.weight.copy_(torch.arange(96).view(12, 8).float() / 128)
    layer.q_weight = nn.Parameter(torch.ones(2, 4), requires_grad=False)
    layer.k_weight = nn.Parameter(torch.ones(2, 4), requires_grad=False)
    layer.eps, layer.clamp_value = 1e-6, 1e-6
    return layer


class TestEngramPrepare(CustomTestCase):
    def test_preparation_is_independent_of_the_later_residual(self):
        layer = _engram()
        ids = torch.tensor([[1, 2], [4, 7]])
        kv = layer.prepare(ids)
        for offset in (0.0, 3.0):
            x = torch.arange(16).float().view(2, 2, 4) / 16 + offset
            # Compute the gate equation independently, including per-HC RMS.
            key = kv[:, :8].view(2, 2, 4)
            value = kv[:, 8:]
            dot = (x * key).sum(-1) * 0.5
            dot *= torch.rsqrt(x.square().mean(-1) + layer.eps)
            dot *= torch.rsqrt(key.square().mean(-1) + layer.eps)
            gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot))
            expected = x + gate.unsqueeze(-1) * value.unsqueeze(1)
            torch.testing.assert_close(layer.apply_prepared(x, kv), expected)
            torch.testing.assert_close(layer(x, ids), expected)

    def test_empty_rows_still_lookup_but_do_not_project(self):
        layer = _engram()
        ids, x = torch.empty(0, 2, dtype=torch.long), torch.empty(0, 2, 4)
        self.assertIs(layer(x, ids), x)
        self.assertEqual(layer.embed.rows, [0])

    def test_only_stream_safe_projection_routes_are_admitted(self):
        layer = _engram()
        self.assertTrue(layer.supports_prefetch())
        for backend in Mxfp8DenseGemmBackend:
            for converted, marlin, view in (
                (False, False, True),
                (True, False, True),
                (False, True, True),
                (False, False, False),
            ):
                with self.subTest(backend=backend, converted=converted, marlin=marlin):
                    layer.wkv.quant_method = NS(
                        mxfp8_dense_backend=backend,
                        use_mxfp8=view,
                        convert_mxfp8_to_block=converted,
                        use_marlin=marlin,
                    )
                    self.assertEqual(
                        layer.supports_prefetch(),
                        backend == Mxfp8DenseGemmBackend.FLASHINFER_CUTEDSL
                        and not converted
                        and not marlin
                        and view,
                    )
        layer.wkv.quant_method = NS(
            mxfp8_dense_backend=Mxfp8DenseGemmBackend.FLASHINFER_CUTEDSL
        )
        layer.wkv.block_fp8_mxfp8_ready = True
        self.assertTrue(layer.supports_prefetch())


class TestEngramStreamDependencies(CustomTestCase):
    def test_staged_prefetch_rearms_fence_and_keeps_future_collectives_unqueued(self):
        from sglang.srt.layers.moe.mega_moe import _wait_for_moe_prefetch
        from sglang.srt.runtime_context import get_forward

        for capturing in (False, True):
            for rank in range(4):
                with self.subTest(capturing=capturing, rank=rank):
                    self._check_staged_dependencies(
                        capturing, rank, _wait_for_moe_prefetch, get_forward
                    )

    def _check_staged_dependencies(self, capturing, rank, wait_moe, get_forward):
        class Stream:
            def __init__(self):
                self.frontier, self.waits = set(), 0

            def wait_stream(self, other):
                self.frontier.update(other.frontier)
                self.waits += 1

            def wait_event(self, event):
                self.frontier.update(event.frontier)

        class Event:
            def record(self, stream):
                self.frontier = stream.frontier.copy()

        main, producer, moe = Stream(), Stream(), Stream()
        active, collectives = [main], []

        @contextmanager
        def stream_scope(stream):
            previous, active[0] = active[0], stream
            try:
                yield
            finally:
                active[0] = previous

        class Layer:
            def __init__(self, layer_id):
                self.layer_hash_index, self.layer_id = layer_id, layer_id

            def prepare(self, ids, batch):
                collectives.extend(
                    (f"gather{self.layer_id}", f"scatter{self.layer_id}")
                )
                active[0].frontier.add(f"kv{self.layer_id}")
                return ids + self.layer_id

            def apply_prepared(self, x, kv):
                if f"kv{self.layer_id}" not in active[0].frontier:
                    raise AssertionError("unfinished Engram projection")
                return x + kv.sum().item()

        # A rank with no real rows still queues the identical stage sequence.
        hashes = torch.ones(0 if rank == 3 else rank + 1, 4)
        layers = [(1, Layer(1)), (3, Layer(3))]
        with (
            patch("torch.cuda.Event", Event),
            patch("torch.cuda.current_stream", side_effect=lambda: active[0]),
            patch("torch.cuda.stream", side_effect=stream_scope),
            patch("torch.cuda.is_current_stream_capturing", return_value=capturing),
            patch.object(torch.Tensor, "record_stream"),
        ):
            prefetch = EngramPrefetcher(producer, layers)
            fence = prefetch.moe_fence()
            with get_forward().scoped(moe_prefetch_fence=fence):
                self.assertEqual(
                    prefetch.start(hashes, None, layer_ids=(), fence=fence), {}
                )
                self.assertEqual(producer.waits, 0)
                self.assertIsNone(fence.stream)
                prepared = prefetch.start(hashes, None, layer_ids=(1,), fence=fence)
                self.assertEqual(collectives, ["gather1", "scatter1"])
                with stream_scope(moe):
                    wait_moe()
                    wait_moe()
                self.assertIn("kv1", moe.frontier)
                self.assertNotIn("kv3", moe.frontier)
                self.assertEqual(moe.waits, 1)
                main.wait_stream(moe)
                output = prefetch.consume(1, layers[0][1], 5, prepared)
                self.assertEqual(output, 5 + 2 * hashes.shape[0])
                prepared.update(
                    prefetch.start(hashes, None, layer_ids=(3,), fence=fence)
                )
                self.assertNotIn("kv3", moe.frontier)
                with stream_scope(moe):
                    wait_moe()
                    wait_moe()
                self.assertIn("kv3", moe.frontier)
                self.assertEqual(moe.waits, 2)
                main.wait_stream(moe)
                output = prefetch.consume(3, layers[1][1], output * 2, prepared)
                self.assertEqual(
                    output, (5 + 2 * hashes.shape[0]) * 2 + 4 * hashes.shape[0]
                )
                self.assertFalse(prepared)
            self.assertIsNone(get_forward().moe_prefetch_fence)
        self.assertEqual(collectives, ["gather1", "scatter1", "gather3", "scatter3"])

    def test_moe_fence_joins_the_launch_stream_once_per_forward(self):
        """An unfenced alternate-stream MegaMoE can overlap pending NCCL;
        joining the caller's main stream instead does not prevent that race.
        """

        class Stream:
            def __init__(self, frontier=()):
                self.frontier = set(frontier)
                self.waits = 0

            def wait_stream(self, other):
                self.frontier.update(other.frontier)
                self.waits += 1

        from sglang.srt.layers.moe.mega_moe import _wait_for_moe_prefetch
        from sglang.srt.runtime_context import get_forward

        producer = Stream({"lookup", "reduce_scatter", "wkv"})
        for _ in range(2):
            launch = Stream({"route", "pre_dispatch"})
            with (
                patch("torch.cuda.current_stream", return_value=launch),
                get_forward().scoped(moe_prefetch_fence=EngramPrefetchFence(producer)),
            ):
                _wait_for_moe_prefetch()
                self.assertTrue(producer.frontier <= launch.frontier)
                _wait_for_moe_prefetch()
                self.assertEqual(launch.waits, 1)
            self.assertIsNone(get_forward().moe_prefetch_fence)

    def test_gate_joins_only_its_projection_and_never_a_later_layer(self):
        """A whole-stream join silently serializes look-ahead; missing the
        per-layer join reads unfinished KV. Emulate GPU stream frontiers.
        """
        for capturing in (False, True):
            with self.subTest(capturing=capturing):
                self._check_dependencies(capturing)

    def _check_dependencies(self, capturing):
        class Stream:
            def __init__(self):
                self.frontier = set()

            def wait_stream(self, other):
                self.frontier.update(other.frontier)

            def wait_event(self, event):
                self.frontier.update(event.frontier)

        class Event:
            def record(self, stream):
                self.frontier = stream.frontier.copy()

        main, side = Stream(), Stream()
        active, records, collectives = [main], [], []

        @contextmanager
        def stream_scope(stream):
            previous, active[0] = active[0], stream
            try:
                yield
            finally:
                active[0] = previous

        class Hashes:
            def __init__(self, value):
                self.value = value

            def record_stream(self, stream):
                records.append(("hash", stream))

            def __getitem__(self, index):
                if "hash" not in active[0].frontier:
                    raise AssertionError("hash used before the fork dependency")
                return self.value

        class Layer:
            def __init__(self, layer_id):
                self.layer_hash_index, self.layer_id = layer_id, layer_id

            def prepare(self, value, batch):
                collectives.extend(
                    (f"gather{self.layer_id}", f"scatter{self.layer_id}")
                )
                name = f"kv{self.layer_id}"
                active[0].frontier.add(name)
                return NS(
                    name=name,
                    value=value + self.layer_id,
                    record_stream=lambda stream: records.append((name, stream)),
                )

            def apply_prepared(self, x, kv):
                if active[0] is not main or kv.name not in main.frontier:
                    raise AssertionError("gate reads unfinished side-stream KV")
                return x + kv.value

        layers = [(1, Layer(1)), (3, Layer(3))]
        with (
            patch("torch.cuda.Event", Event),
            patch("torch.cuda.current_stream", side_effect=lambda: active[0]),
            patch("torch.cuda.stream", side_effect=stream_scope),
            patch("torch.cuda.is_current_stream_capturing", return_value=capturing),
        ):
            prefetch = EngramPrefetcher(side, layers)
            for value in (2, 17):
                main.frontier, side.frontier = {"hash"}, set()
                prepared = prefetch.start(Hashes(value), None)
                output = prefetch.consume(1, layers[0][1], 5, prepared)
                self.assertEqual(output, 5 + value + 1)
                self.assertNotIn("kv3", main.frontier)
                # The second gate uses a residual updated by intervening layers.
                output = prefetch.consume(3, layers[1][1], output * 2, prepared)
                self.assertEqual(output, (5 + value + 1) * 2 + value + 3)
                self.assertFalse(prepared)
        self.assertEqual(
            collectives, ["gather1", "scatter1", "gather3", "scatter3"] * 2
        )
        self.assertEqual(
            records,
            [] if capturing else [("hash", side), ("kv1", main), ("kv3", main)] * 2,
        )


class TestEngramPrefetchEligibility(CustomTestCase):
    def test_global_mode_padding_and_actual_moe_dispatch_control_all_ranks(self):
        model = NS(
            engram_prefetcher=NS(layers=[(1, NS(supports_prefetch=lambda: True))]),
            layers=[NS(mlp=object())],
            start_layer=0,
            end_layer=1,
        )
        hidden = torch.zeros(4, 2, 4)
        batch = NS(
            global_forward_mode=ForwardMode.DECODE,
            forward_mode=ForwardMode.DECODE,
            dp_padding_mode=DpPaddingMode.MAX_LEN,
            tbo_children=None,
        )
        with (
            get_context().override_server_args(
                tp_size=4, attn_dp_size=4, moe_a2a_backend="megamoe"
            ),
            envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_RESERVED_SMS.override(2),
            # Hardware and the external expert runner are dependency boundaries.
            patch(
                "sglang.srt.layers.moe.mega_moe.should_use_mega_moe", return_value=True
            ) as dispatch,
            get_flags().dp.override(use_world_group_for_gather=False),
            get_flags().moe.override(a2a_backend=MoeA2ABackend.MEGAMOE),
        ):
            for mode in (ForwardMode.DECODE, ForwardMode.TARGET_VERIFY):
                batch.global_forward_mode = mode
                for local in (mode, ForwardMode.IDLE):
                    batch.forward_mode = local
                    self.assertTrue(
                        DeepseekV4Model._can_prefetch_engram(model, batch, hidden)
                    )
            batch.global_forward_mode = ForwardMode.EXTEND
            self.assertFalse(DeepseekV4Model._can_prefetch_engram(model, batch, hidden))
            batch.global_forward_mode = None
            self.assertFalse(DeepseekV4Model._can_prefetch_engram(model, batch, hidden))
            batch.global_forward_mode = ForwardMode.DECODE
            batch.dp_padding_mode = DpPaddingMode.SUM_LEN
            self.assertFalse(DeepseekV4Model._can_prefetch_engram(model, batch, hidden))
            batch.dp_padding_mode = DpPaddingMode.MAX_LEN
            dispatch.return_value = False
            self.assertFalse(DeepseekV4Model._can_prefetch_engram(model, batch, hidden))
            dispatch.return_value = True
            with get_flags().dp.override(use_world_group_for_gather=True):
                self.assertFalse(
                    DeepseekV4Model._can_prefetch_engram(model, batch, hidden)
                )
            batch.tbo_children = []
            self.assertFalse(DeepseekV4Model._can_prefetch_engram(model, batch, hidden))
            batch.tbo_children = None
            model.engram_prefetcher = None
            self.assertFalse(DeepseekV4Model._can_prefetch_engram(model, batch, hidden))


if __name__ == "__main__":
    unittest.main()
