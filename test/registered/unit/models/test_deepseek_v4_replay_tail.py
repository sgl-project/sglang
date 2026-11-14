"""DeepSeek-V4.1 decoder-replay tail: the state hand-off of its prefill-graph break,
who gets the tail (CUDA always, ROCm breakable prefill graphs on opt-in), the break's
shape threshold, and the KV store's row cut inside a tail graph."""

import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.environ import envs
from sglang.srt.models import deepseek_v4_replay_tail as replay_tail
from sglang.srt.models.deepseek_v4_mhc import HcNormed, HcPending, HcState
from sglang.srt.models.deepseek_v4_replay_tail import (
    DecoderReplayTail,
    _flatten_state,
    _tail_kv_store,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestDecoderReplayTailState(CustomTestCase):
    def test_hip_pending_state_round_trip(self):
        """A HIP prefill-graph tail break must preserve a deferred mHC post."""
        tensors = [torch.full((2, 3), value) for value in range(1, 9)]
        state = HcState(
            HcPending(*tensors[:4]),
            tensors[4],
            HcNormed(tensors[5]),
            tuple(tensors[5:8]),
        )

        leaves, has_pre, rebuild = _flatten_state(state)
        rebuilt = rebuild([tensor + 10 for tensor in leaves])

        self.assertTrue(has_pre)
        self.assertEqual(len(leaves), 5)
        self.assertIsInstance(rebuilt.streams, HcPending)
        for actual, expected in zip(rebuilt.streams, tensors[:4]):
            torch.testing.assert_close(actual, expected + 10)
        torch.testing.assert_close(rebuilt.pre, tensors[4] + 10)
        self.assertIsNone(rebuilt.input)
        self.assertIsNone(rebuilt.stats)

    def test_tensor_state_drops_the_normed_input_and_stats(self):
        streams, pre, normed = (torch.full((2, 3), v) for v in (1, 2, 3))
        stats = tuple(torch.full((2, 4), v) for v in (4, 5, 6))
        leaves, has_pre, rebuild = _flatten_state(
            HcState(streams, pre, HcNormed(normed), stats)
        )
        self.assertTrue(has_pre)
        self.assertEqual(
            [t.data_ptr() for t in leaves], [streams.data_ptr(), pre.data_ptr()]
        )
        rebuilt = rebuild([t * 2 for t in leaves])
        torch.testing.assert_close(rebuilt.streams, streams * 2)
        torch.testing.assert_close(rebuilt.pre, pre * 2)
        self.assertIsNone(rebuilt.input)
        self.assertIsNone(rebuilt.stats)

        leaves, has_pre, rebuild = _flatten_state(HcState(streams))
        self.assertFalse(has_pre)
        self.assertEqual(len(leaves), 1)
        self.assertIsNone(rebuild(leaves).pre)


def _model(tail_writes_source_kv=True):
    return SimpleNamespace(
        tail_writes_source_kv=tail_writes_source_kv, late_layer_start=20, end_layer=40
    )


class TestMaybeCreate(CustomTestCase):
    def _create(
        self,
        *,
        cuda,
        breakable=True,
        hip_opt_in=False,
        borrow=False,
        max_rows=2048,
        model=None,
    ):
        with ExitStack() as stack:
            stack.enter_context(patch.object(replay_tail, "is_cuda", return_value=cuda))
            stack.enter_context(
                patch.object(replay_tail, "is_hip", return_value=not cuda)
            )
            graph_backend = stack.enter_context(
                patch.object(
                    replay_tail, "check_cuda_graph_backend", return_value=breakable
                )
            )
            stack.enter_context(
                envs.SGLANG_OPT_HIP_PREFILL_GRAPH_EXACT_TAIL.override(hip_opt_in)
            )
            stack.enter_context(envs.SGLANG_ENABLE_GRAPH_POOL_BORROW.override(borrow))
            stack.enter_context(
                envs.SGLANG_DSV4_DECODER_TAIL_GRAPH_MAX_ROWS.override(max_rows)
            )
            tail = DecoderReplayTail.maybe_create(model or _model())
        return tail, graph_backend

    def test_hip_opt_in_is_off_by_default(self):
        self.assertIs(envs.SGLANG_OPT_HIP_PREFILL_GRAPH_EXACT_TAIL.default, False)

    def test_hip_needs_the_opt_in_and_a_breakable_prefill_graph(self):
        for hip_opt_in, breakable in ((False, True), (True, False), (False, False)):
            with self.subTest(hip_opt_in=hip_opt_in, breakable=breakable):
                tail, _ = self._create(
                    cuda=False, hip_opt_in=hip_opt_in, breakable=breakable
                )
                self.assertIsNone(tail)
        tail, _ = self._create(cuda=False, hip_opt_in=True)
        self.assertIsInstance(tail, DecoderReplayTail)
        self.assertEqual(tail._layer_ids, range(20, 40))
        self.assertEqual(tail.runner.max_rows, 2048)

    def test_cuda_ignores_the_hip_opt_in(self):
        for hip_opt_in in (False, True):
            with self.subTest(hip_opt_in=hip_opt_in):
                tail, graph_backend = self._create(
                    cuda=True, hip_opt_in=hip_opt_in, breakable=False
                )
                self.assertIsInstance(tail, DecoderReplayTail)
                graph_backend.assert_not_called()

    def test_graph_pool_borrow_refuses_the_tail_on_every_platform(self):
        for cuda in (True, False):
            with self.subTest(cuda=cuda):
                tail, _ = self._create(cuda=cuda, hip_opt_in=True, borrow=True)
                self.assertIsNone(tail)

    def test_no_tail_without_source_kv_or_tail_graphs(self):
        for cuda in (True, False):
            with self.subTest(cuda=cuda):
                tail, _ = self._create(cuda=cuda, hip_opt_in=True, max_rows=0)
                self.assertIsNone(tail)
                tail, _ = self._create(cuda=cuda, hip_opt_in=True, model=_model(False))
                self.assertIsNone(tail)


class TestBreaksOutOfPrefillGraph(CustomTestCase):
    def _breaks(
        self, num_tokens, *, cp_metadata=None, request_window=None, unified=False
    ):
        backend = SimpleNamespace(
            token_to_kv_pool=SimpleNamespace(request_window=request_window)
        )
        tail = object.__new__(DecoderReplayTail)
        with (
            patch.object(replay_tail, "get_attn_backend", return_value=backend),
            patch(
                "sglang.srt.layers.attention.dsv4.unified_kv_gate.is_unified_kv_triton",
                return_value=unified,
            ),
            envs.SGLANG_DSV4_DECODER_TAIL_GRAPH_MIN_PREFILL_TOKENS.override(4096),
        ):
            return tail.breaks_out_of_prefill_graph(
                SimpleNamespace(cp_metadata=cp_metadata), num_tokens=num_tokens
            )

    def test_threshold_is_inclusive(self):
        self.assertFalse(self._breaks(4095))
        self.assertTrue(self._breaks(4096))
        self.assertTrue(self._breaks(16384))

    def test_refused_paths_keep_the_tail_in_the_graph(self):
        self.assertFalse(self._breaks(8192, cp_metadata=object()))
        self.assertFalse(self._breaks(8192, request_window=object()))
        self.assertFalse(self._breaks(8192, unified=True))


class TestTailKvStore(CustomTestCase):
    def _store(self, num_rows, q_rope, x=None):
        attention = MagicMock()
        forward_batch = SimpleNamespace(global_num_token_non_padded_cpu=num_rows)
        backend = object()
        rows, positions, qkv_a = (torch.arange(16).view(8, 2) + k for k in range(3))
        x = rows if x is None else x
        with (
            patch.object(
                replay_tail,
                "get_tc_piecewise_forward_context",
                return_value=SimpleNamespace(forward_batch=forward_batch),
            ),
            patch.object(replay_tail, "get_attn_backend", return_value=backend),
        ):
            _tail_kv_store(attention, x, positions, qkv_a, q_rope)
        return attention._compute_kv_to_cache, (
            x,
            positions,
            qkv_a,
            forward_batch,
            backend,
        )

    def test_stores_and_ropes_only_the_live_rows(self):
        q = torch.arange(24).view(8, 3)
        call, (x, positions, qkv_a, forward_batch, backend) = self._store(3, q)
        call.assert_called_once()
        args, kwargs = call.call_args
        torch.testing.assert_close(args[0], x[:3])
        torch.testing.assert_close(args[1], positions[:3])
        self.assertIs(args[2], forward_batch)
        self.assertIs(args[3], backend)
        torch.testing.assert_close(kwargs["qkv_a"], qkv_a[:3])
        torch.testing.assert_close(kwargs["q_rope"], q[:3])
        self.assertEqual(kwargs["q_rope"].data_ptr(), q.data_ptr())

    def test_without_fused_q_rope_or_live_rows(self):
        call, _ = self._store(3, None)
        self.assertIsNone(call.call_args.kwargs["q_rope"])
        call, _ = self._store(0, torch.zeros(8, 3))
        call.assert_not_called()

    def test_slices_a_quantized_input_by_its_rows(self):
        from sglang.kernels.ops.quantization.mxfp8_amd_gfx95 import (
            Fp8GridActivation,
            Mxfp8Activation,
        )

        q, scale = torch.arange(16).view(8, 2), torch.arange(8).view(8, 1)
        call, _ = self._store(3, None, x=Mxfp8Activation(q, scale))
        live = call.call_args.args[0]
        self.assertIsInstance(live, Mxfp8Activation)
        torch.testing.assert_close(live.q, q[:3])
        torch.testing.assert_close(live.scale, scale[:3])
        call, _ = self._store(3, None, x=Fp8GridActivation(q))
        live = call.call_args.args[0]
        self.assertIsInstance(live, Fp8GridActivation)
        torch.testing.assert_close(live.x, q[:3])


if __name__ == "__main__":
    unittest.main()
