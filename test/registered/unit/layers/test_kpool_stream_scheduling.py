"""CPU scheduling checks: stream dependencies, cache contracts and gate math."""

import unittest
from contextlib import contextmanager, nullcontext
from types import MethodType, SimpleNamespace
from unittest.mock import Mock, patch

import torch
import torch.nn.functional as F

from sglang.kernels.ops.attention.dsa import triton_kernel
from sglang.srt.layers.attention.dsa import dsa_indexer_kpool as indexer_module
from sglang.srt.layers.attention.dsa.dsa_indexer_kpool import IndexerKPool
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _eager(method):
    return getattr(method, "_torchdynamo_orig_callable", method)


class _Stream:
    def __init__(self, name, trace):
        self.name = name
        self.trace = trace

    def wait_stream(self, other):
        self.trace.append(("wait", self.name, other.name))


class _Streams:
    def __init__(self):
        self.trace = []
        self.current = _Stream("main", self.trace)
        self.alt = _Stream("alt", self.trace)
        self.gate = _Stream("gate", self.trace)

    @contextmanager
    def use(self, stream):
        previous = self.current
        self.current = stream
        try:
            yield
        finally:
            self.current = previous

    def record(self, operation):
        self.trace.append((operation, self.current.name))


class TestKPoolStreamScheduling(unittest.TestCase):
    def test_prefill_overlap_excludes_cp_and_graph_capture(self):
        for mode in (ForwardMode.EXTEND, ForwardMode.DECODE, ForwardMode.TARGET_VERIFY):
            for has_stream in (False, True):
                for capture, breakable, cp in (
                    (False, False, False),
                    (True, False, False),
                    (False, True, False),
                    (False, False, True),
                ):
                    with (
                        self.subTest(
                            mode=mode,
                            stream=has_stream,
                            capture=capture,
                            breakable=breakable,
                            cp=cp,
                        ),
                        patch.object(
                            indexer_module, "get_is_capture_mode", return_value=capture
                        ),
                        patch.object(
                            indexer_module,
                            "is_in_breakable_cuda_graph",
                            return_value=breakable,
                        ),
                        patch.object(
                            indexer_module, "dsa_use_prefill_cp", return_value=cp
                        ),
                    ):
                        indexer = SimpleNamespace(
                            alt_stream=object() if has_stream else None
                        )
                        actual = IndexerKPool._can_overlap_prefill(
                            indexer,
                            SimpleNamespace(forward_mode=mode),
                            return_indices=True,
                        )
                        self.assertEqual(
                            actual,
                            mode == ForwardMode.EXTEND
                            and has_stream
                            and not (capture or breakable or cp),
                        )

    def test_precomputed_head_gate_matches_original_math(self):
        torch.manual_seed(19)
        for dtype in (torch.bfloat16, torch.float16, torch.float32):
            x = torch.randn(11, 32).to(dtype)
            matrix = torch.randn(8, 32)
            q_scale = torch.rand(11, 8, 1)
            indexer = SimpleNamespace(
                weights_proj=lambda value: (F.linear(value, matrix), None),
                n_heads=8,
                softmax_scale=128**-0.5,
            )
            expected = _eager(IndexerKPool._get_logits_head_gate)(indexer, x, q_scale)
            projected = _eager(IndexerKPool._project_and_scale_head_gates)(indexer, x)
            actual = _eager(IndexerKPool._apply_q_scale_and_softmax_scale)(
                indexer, projected, q_scale
            )
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_projection_reordering_preserves_rope_and_third_stream(self):
        for skip_rope in (False, True):
            streams = _Streams()
            x = torch.arange(32, dtype=torch.float32).reshape(4, 8)
            indexer = SimpleNamespace(
                alt_stream=streams.alt,
                compress_gate_stream=streams.gate,
                half_device_sm_count=8,
                head_dim=4,
                rope_head_dim=2,
                skip_rope=skip_rope,
                index_kpool_compress_gate=torch.ones(4, 8),
            )

            def project_q(value):
                streams.record("project_q")
                return value.clone(), None

            def project_k(value):
                streams.record("project_k")
                return value[:, :4].clone(), None

            def head_gate(value):
                streams.record("head_gate")
                return value[:, :2].clone()

            def rotate(value):
                streams.record("rotate")
                return value.flip(-1)

            def rope(positions, q, k):
                streams.record("rope")
                return q + 1, k + 2

            indexer.wq_b = project_q
            indexer.wk = project_k
            indexer.k_norm = lambda value: value
            indexer.rotary_emb = rope
            indexer._project_and_scale_head_gates = head_gate
            with (
                patch.object(
                    torch.cuda, "current_stream", side_effect=lambda: streams.current
                ),
                patch.object(torch.cuda, "stream", side_effect=streams.use),
                patch.object(
                    indexer_module.deep_gemm_wrapper,
                    "configure_deep_gemm_num_sms",
                    return_value=nullcontext(),
                ),
                patch.object(indexer_module, "rotate_activation", side_effect=rotate),
            ):
                actual = IndexerKPool._get_q_k_bf16(
                    indexer,
                    x,
                    x,
                    torch.arange(4),
                    True,
                    None,
                    precompute_compress_gate=True,
                    precompute_head_gate=True,
                )
                trace = streams.trace[:]
                expected = IndexerKPool._get_q_k_bf16(
                    indexer, x, x, torch.arange(4), False, None
                )
            torch.testing.assert_close(actual[0], expected[0])
            torch.testing.assert_close(actual[1], expected[1])
            torch.testing.assert_close(
                actual[2], F.linear(x, indexer.index_kpool_compress_gate)
            )
            self.assertIsNotNone(actual[3])
            self.assertIn(("wait", "gate", "main"), trace)
            join = trace.index(("wait", "main", "alt"))
            self.assertLess(trace.index(("head_gate", "main")), join)
            if skip_rope:
                self.assertLess(trace.index(("rotate", "main")), join)
                self.assertNotIn(("rope", "main"), trace)
            else:
                self.assertGreater(trace.index(("rope", "main")), join)
                self.assertGreater(trace.index(("rotate", "main")), join)

    def test_prefill_waits_before_topk_and_preserves_deferred_cache(self):
        for has_plan in (False, True):
            for return_indices in (False, True):
                for cp in (False, True):
                    for num_tokens in (0, 128, 8192):
                        with self.subTest(
                            plan=has_plan,
                            indices=return_indices,
                            cp=cp,
                            tokens=num_tokens,
                        ):
                            self._run_prefill(has_plan, return_indices, cp, num_tokens)

    def _run_prefill(self, has_plan, return_indices, cp, num_tokens):
        streams = _Streams()
        x = torch.ones(num_tokens, 8)
        compressed = object()
        calls = []
        metadata = SimpleNamespace(
            attn_metadata=SimpleNamespace(
                kpool_extend_plan=object() if has_plan else None
            )
        )
        batch = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            seq_lens_cpu=torch.tensor([8192 + num_tokens]),
        )
        prepare_qk = Mock(return_value=(x, x, None, None))
        indexer = SimpleNamespace(
            alt_stream=streams.alt,
            compress_gate_stream=streams.gate,
            index_topk=16,
            index_kpool=4,
            index_kpool_compress=True,
            block_size=128,
            scale_fmt=None,
            _get_q_k_bf16=prepare_qk,
        )
        indexer._can_overlap_prefill = MethodType(
            IndexerKPool._can_overlap_prefill, indexer
        )

        def compress(**kwargs):
            streams.record("compress")
            calls.append(kwargs)
            return compressed

        def quant(*args):
            streams.record("quant")
            return x, torch.ones(num_tokens, 1, 1)

        def head(*args):
            streams.record("head")
            return x

        def topk(*args, **kwargs):
            streams.record("topk")
            if not has_plan:
                self.assertIs(kwargs["kpool_extend_cache"], compressed)
            return x

        indexer._compress_write = compress
        indexer._resolve_head_gate_weights = head
        indexer._get_topk_ragged = topk
        indexer._get_topk_ragged_kpool_plan = topk
        with (
            patch.object(indexer_module, "is_cuda", return_value=True),
            patch.object(indexer_module, "is_hip", return_value=False),
            patch.object(indexer_module, "is_npu", return_value=False),
            patch.object(indexer_module, "get_is_capture_mode", return_value=False),
            patch.object(
                indexer_module, "is_in_breakable_cuda_graph", return_value=False
            ),
            patch.object(indexer_module, "dsa_use_prefill_cp", return_value=cp),
            patch.object(
                indexer_module,
                "get_attn_backend",
                return_value=SimpleNamespace(
                    get_indexer_metadata=lambda *args: metadata
                ),
            ),
            patch.object(
                torch.cuda, "current_stream", side_effect=lambda: streams.current
            ),
            patch.object(torch.cuda, "stream", side_effect=streams.use),
            patch.object(triton_kernel, "act_quant", side_effect=quant),
        ):
            actual = IndexerKPool._forward_cuda_impl(
                indexer, x, x, torch.arange(num_tokens), batch, 0, return_indices
            )
        self.assertEqual(calls[0]["return_compressed"], return_indices)
        self.assertEqual(calls[0]["write_cache"], has_plan or not return_indices)
        self.assertFalse(prepare_qk.call_args.args[3])
        self.assertFalse(prepare_qk.call_args.kwargs["precompute_head_gate"])
        overlap = not cp and return_indices
        self.assertIn(("compress", "alt" if overlap else "main"), streams.trace)
        if overlap:
            self.assertLess(
                streams.trace.index(("wait", "alt", "main")),
                streams.trace.index(("compress", "alt")),
            )
            join = streams.trace.index(("wait", "main", "alt"))
            self.assertGreater(join, streams.trace.index(("compress", "alt")))
            self.assertGreater(join, streams.trace.index(("quant", "main")))
            self.assertGreater(join, streams.trace.index(("head", "main")))
            self.assertLess(join, streams.trace.index(("topk", "main")))
        else:
            self.assertFalse(any(event[0] == "wait" for event in streams.trace))
        if return_indices:
            self.assertIs(actual, x)
            self.assertIn(("quant", "main"), streams.trace)
        else:
            self.assertIsNone(actual)
            self.assertNotIn(("quant", "main"), streams.trace)
            self.assertNotIn(("head", "main"), streams.trace)
            self.assertNotIn(("topk", "main"), streams.trace)


if __name__ == "__main__":
    unittest.main()
