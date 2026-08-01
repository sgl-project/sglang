# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for the Triton sparse-MLA prefill adapter and its capability
check. The kernel is mocked, so these guard the wiring rather than the numerics
(which live in
``test/registered/kernels/ops/attention/test_dsa_triton_sparse_mla_prefill.py``):

- argument marshalling between the DSA backend and the kernel entry point,
- the two fast-path switches being off unless asked for, at both the CLI layer
  and the backend layer,
- the validator's accept/reject boundaries,
- that registering this backend does not change which backend SM120 selects on
  its own.
"""

import unittest
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestTritonSparseMLAValidator(CustomTestCase):
    """Boundaries of ``_validate_triton_sparse_mla_backend``."""

    def _validate(self, **kwargs):
        from sglang.srt.layers.attention.dsa_backend import (
            _validate_triton_sparse_mla_backend,
        )

        defaults = dict(device_sm_major=12, num_q_heads=8, union=0)
        defaults.update(kwargs)
        return _validate_triton_sparse_mla_backend(**defaults)

    def test_sm_major_boundary_is_hopper(self):
        # The kernel needs an SM90+ MMA for the [16, D_V] head tile; SM80 and
        # below must be refused at startup rather than failing mid-request.
        with self.assertRaisesRegex(ValueError, "SM90"):
            self._validate(device_sm_major=8)
        for sm in (9, 10, 12):
            self.assertIsNone(self._validate(device_sm_major=sm))

    def test_union_group_size_contract(self):
        for group in (0, 2, 4):
            self.assertIsNone(self._validate(union=group))
        with self.assertRaisesRegex(ValueError, "must be 0, 2 or 4"):
            self._validate(union=3)

    def test_union_tile_capacity(self):
        # The union tile holds 32 rows total, shared as num_q_heads * union.
        # Exceeding it would silently drop heads, so it is a startup error.
        with self.assertRaisesRegex(ValueError, "union"):
            self._validate(num_q_heads=16, union=4)
        self.assertIsNone(self._validate(num_q_heads=16, union=2))


class TestTritonSparseMLAAdapter(CustomTestCase):
    """The backend method forwards exactly what the kernel expects."""

    def _call_forward(self, *, union=0, dense=False, capturing=False):
        from sglang.srt.layers.attention.dsa_backend import DeepseekSparseAttnBackend

        captured = {}

        def _fake_kernel(q, kv, indices, sm_scale, d_v=512, **kwargs):
            captured.update(
                q=q, kv=kv, indices=indices, sm_scale=sm_scale, d_v=d_v, **kwargs
            )
            return torch.zeros(q.shape[0], q.shape[1], d_v, dtype=torch.bfloat16)

        backend = DeepseekSparseAttnBackend.__new__(DeepseekSparseAttnBackend)
        backend.dsa_triton_union = union
        backend.dsa_triton_dense_prefix = dense

        with patch(
            "sglang.kernels.ops.attention.dsa.triton_sparse_mla_prefill.sparse_mla_prefill",
            _fake_kernel,
        ), patch(
            "sglang.srt.model_executor.runner_utils.capture_mode.get_is_capture_mode",
            lambda: capturing,
        ):
            out = backend._forward_triton_sparse_mla(
                q_all=torch.zeros(4, 8, 576, dtype=torch.bfloat16),
                kv_cache=torch.zeros(64, 576, dtype=torch.bfloat16),
                page_table_1=torch.zeros(4, 16, dtype=torch.int32),
                sm_scale=0.0625,
                v_head_dim=512,
            )
        return captured, out

    def test_forwards_tensors_and_scale(self):
        captured, out = self._call_forward()
        self.assertEqual(tuple(captured["q"].shape), (4, 8, 576))
        self.assertEqual(tuple(captured["kv"].shape), (64, 576))
        self.assertEqual(tuple(captured["indices"].shape), (4, 16))
        self.assertEqual(captured["sm_scale"], 0.0625)
        self.assertEqual(captured["d_v"], 512)
        self.assertEqual(tuple(out.shape), (4, 8, 512))

    def test_fast_paths_off_unless_requested(self):
        captured, _ = self._call_forward()
        self.assertEqual(captured["union"], 0)
        self.assertFalse(captured["dense"])

    def test_fast_paths_are_plumbed_through(self):
        captured, _ = self._call_forward(union=4, dense=True)
        self.assertEqual(captured["union"], 4)
        self.assertTrue(captured["dense"])

    def test_union_is_disabled_under_cuda_graph_capture(self):
        # The union path reads the index range back to the host to size its
        # scratch, which cannot be captured. It must degrade to the per-token
        # path (same result) rather than break capture.
        captured, _ = self._call_forward(union=4, dense=True, capturing=True)
        self.assertEqual(captured["union"], 0)
        self.assertTrue(captured["dense"], "dense has no host sync; keep it on")


class TestTritonSparseMLATopkTransformRouting(CustomTestCase):
    """The prefill top-k transform must be RAGGED for this backend.

    Regression: the backend dequantizes the KV cache inside the RAGGED branch of
    `forward_extend`, and its kernel is bf16-only. When `get_topk_transform_method`
    left it on PAGED, the branch was skipped and the raw packed FP8 pool reached
    the kernel, which died on `Unsupported rhs dtype fp8e4nv` mid-forward. Only an
    end-to-end run caught it: the dispatch branch reads correct in isolation.
    """

    def _method(self, prefill_impl, *, store_fp8=True, mode=None):
        from sglang.srt.layers.attention.dsa_backend import DeepseekSparseAttnBackend
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        backend = DeepseekSparseAttnBackend.__new__(DeepseekSparseAttnBackend)
        backend.dsa_kv_cache_store_fp8 = store_fp8
        backend.dsa_prefill_impl = prefill_impl
        return backend.get_topk_transform_method(
            ForwardMode.EXTEND if mode is None else mode
        )

    def test_extend_uses_ragged_for_bf16_kv_backends(self):
        from sglang.srt.layers.attention.dsa_backend import TopkTransformMethod

        for impl in ("flashmla_sparse", "flashmla_sparse_q8", "triton_sparse_mla"):
            with self.subTest(impl=impl):
                self.assertEqual(self._method(impl), TopkTransformMethod.RAGGED, impl)

    def test_other_backends_keep_paged(self):
        from sglang.srt.layers.attention.dsa_backend import TopkTransformMethod

        self.assertEqual(
            self._method("flashinfer_sparse_mla"), TopkTransformMethod.PAGED
        )


class TestTritonSparseMLARegistration(CustomTestCase):
    """Selectable from the CLI, and opt-in only."""

    def test_choice_is_registered(self):
        from sglang.srt.server_args import DSA_CHOICES

        self.assertIn("triton_sparse_mla", DSA_CHOICES)

    def test_defaults_do_not_select_this_backend(self):
        from sglang.srt.server_args import ServerArgs

        args = ServerArgs(model_path="dummy")
        self.assertNotEqual(args.dsa_prefill_backend, "triton_sparse_mla")
        self.assertEqual(args.dsa_triton_union, 0)
        self.assertFalse(args.dsa_triton_dense_prefix)


if __name__ == "__main__":
    unittest.main()
