"""Correctness coverage for the TLX vectorized-5D decode adapter."""

import os
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.utils.common import is_hip
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=90, suite="stage-b-test-1-gpu-small-amd")


@unittest.skipUnless(is_hip(), "TLX paged decode requires ROCm")
class TestTlxPagedDecodeAmd(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        try:
            from triton.language.extra.tlx.ops.amd_pa_decode import build_inputs

            import sglang.srt.layers.attention.tlx_utils as tlx_utils
            from sglang.srt.layers.attention.aiter_utils import (
                forward_decode_vectorized_5d,
            )
            from sglang.srt.layers.attention.tlx_utils import (
                tlx_pa_decode_available,
            )
        except ImportError as error:
            raise unittest.SkipTest(f"TLX-enabled Triton is not installed: {error}")
        if not tlx_pa_decode_available():
            raise unittest.SkipTest("TLX paged-decode operator is unavailable")
        cls.build_inputs = staticmethod(build_inputs)
        cls.forward_decode = staticmethod(forward_decode_vectorized_5d)
        cls.tlx_utils = tlx_utils

    def _run_wrapper(self, mode, backend, q, layer, batch, k_cache, v_cache):
        previous = os.environ.get("SGLANG_AITER_5D_DECODE_BACKEND")
        os.environ["SGLANG_AITER_5D_DECODE_BACKEND"] = mode
        try:
            output = torch.empty_like(q)
            self.forward_decode(
                backend, q, layer, batch, k_cache, v_cache, output, None
            )
            torch.cuda.synchronize()
            return output
        finally:
            if previous is None:
                os.environ.pop("SGLANG_AITER_5D_DECODE_BACKEND", None)
            else:
                os.environ["SGLANG_AITER_5D_DECODE_BACKEND"] = previous

    def test_tlx_matches_gluon_and_supports_graph_capture(self):
        for page_size in (16, 64):
            with self.subTest(page_size=page_size):
                batch_size, context, head_dim = 1, 8192, 64
                num_kv_heads, query_group_size = 8, 8
                q, k_cache, v_cache, context_lens, block_tables = self.build_inputs(
                    batch_size,
                    [context] * batch_size,
                    num_kv_heads * query_group_size,
                    num_kv_heads,
                    head_dim,
                    page_size,
                    cache_layout="5d",
                )
                backend = SimpleNamespace(
                    input_dtype=q.dtype,
                    kv_cache_dtype=k_cache.dtype,
                    k_scale=None,
                    v_scale=None,
                    forward_metadata=SimpleNamespace(
                        kv_indices=block_tables,
                        swa_page_table=None,
                        max_kv_len=context,
                    ),
                )
                layer = SimpleNamespace(
                    tp_k_head_num=num_kv_heads,
                    tp_q_head_num=num_kv_heads * query_group_size,
                    qk_head_dim=head_dim,
                    v_head_dim=head_dim,
                    scaling=head_dim**-0.5,
                    sliding_window_size=None,
                    k_scale=None,
                    v_scale=None,
                )
                forward_batch = SimpleNamespace(
                    batch_size=batch_size, seq_lens=context_lens
                )

                expected = self._run_wrapper(
                    "gluon", backend, q, layer, forward_batch, k_cache, v_cache
                )
                actual = self._run_wrapper(
                    "tlx", backend, q, layer, forward_batch, k_cache, v_cache
                )
                repeated = self._run_wrapper(
                    "tlx", backend, q, layer, forward_batch, k_cache, v_cache
                )

                torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
                self.assertTrue(torch.equal(actual, repeated))
                workspaces = list(backend._tlx_pa_decode_workspaces.values())
                self.assertEqual(len(workspaces), 1)
                self.assertEqual(workspaces[0][0].stride(-2), head_dim + 4)
                previous = os.environ.get("SGLANG_AITER_5D_DECODE_BACKEND")
                os.environ["SGLANG_AITER_5D_DECODE_BACKEND"] = "tlx"
                try:
                    captured = torch.empty_like(q)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        self.forward_decode(
                            backend,
                            q,
                            layer,
                            forward_batch,
                            k_cache,
                            v_cache,
                            captured,
                            None,
                        )
                    graph.replay()
                    torch.cuda.synchronize()
                finally:
                    if previous is None:
                        os.environ.pop("SGLANG_AITER_5D_DECODE_BACKEND", None)
                    else:
                        os.environ["SGLANG_AITER_5D_DECODE_BACKEND"] = previous
                torch.testing.assert_close(captured, expected, rtol=2e-2, atol=2e-2)

    def test_backend_mode_validation_and_fallback(self):
        q = torch.empty((1, 1, 64), dtype=torch.bfloat16, device="cuda")
        k_cache = torch.empty((1, 1, 8, 16, 8), dtype=q.dtype, device=q.device)
        v_cache = torch.empty((1, 1, 2, 64, 8), dtype=q.dtype, device=q.device)
        backend = SimpleNamespace(
            kv_cache_dtype=q.dtype,
            forward_metadata=SimpleNamespace(max_kv_len=16),
        )
        layer = SimpleNamespace(
            qk_head_dim=64,
            v_head_dim=64,
            sliding_window_size=None,
        )
        forward_batch = SimpleNamespace(batch_size=1)

        with self.assertRaisesRegex(ValueError, "expected gluon, tlx, or auto"):
            self.tlx_utils.should_use_tlx_decode(
                "unknown",
                backend,
                q,
                layer,
                forward_batch,
                k_cache,
                v_cache,
                None,
            )
        with mock.patch.object(
            self.tlx_utils, "can_forward_decode_vectorized_5d_tlx", return_value=False
        ):
            with self.assertRaisesRegex(RuntimeError, "was forced"):
                self.tlx_utils.should_use_tlx_decode(
                    "tlx",
                    backend,
                    q,
                    layer,
                    forward_batch,
                    k_cache,
                    v_cache,
                    None,
                )
        backend._tlx_pa_decode_eligibility = {}
        with mock.patch.object(
            self.tlx_utils, "can_forward_decode_vectorized_5d_tlx", return_value=True
        ):
            self.assertFalse(
                self.tlx_utils.should_use_tlx_decode(
                    "auto",
                    backend,
                    q,
                    layer,
                    forward_batch,
                    k_cache,
                    v_cache,
                    None,
                )
            )
