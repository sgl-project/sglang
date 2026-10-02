# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for the Triton sparse-MLA prefill adapter and its capability
check. The kernel is mocked, so these guard the wiring rather than the numerics
(which live in
``test/registered/kernels/ops/attention/test_dsa_triton_sparse_mla_prefill.py``):

- argument marshalling between the DSA backend and the kernel entry point,
- the union switch being off unless asked for, at both the CLI layer and the
  backend layer, forced off while the stream is capturing or under deterministic
  inference, and left on inside a breakable-graph replay,
- the validator's accept/reject boundaries,
- that registering this backend does not change which backend SM120 selects on
  its own.
"""

import unittest
from types import SimpleNamespace
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

        defaults = dict(device_sm_major=12, num_q_heads=8, union=0, index_kpool=1)
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

    def test_union_tile_must_be_a_power_of_two_of_at_least_16(self):
        # Regression: h=4 with union 2 (GLM-5 at TP16) and h=12 with union 2
        # passed the old `<= 32` check, then failed to compile on the first long
        # prefill: `tl.arange` needs a power of two and `tl.dot` needs M >= 16.
        # The launcher only catches OutOfResources, so the compile error reached
        # the request. Both shapes must be refused at startup instead.
        for heads, group in ((4, 2), (12, 2), (6, 4)):
            with self.subTest(heads=heads, group=group):
                with self.assertRaisesRegex(ValueError, "power of two"):
                    self._validate(num_q_heads=heads, union=group)
        for heads, group in ((4, 4), (8, 2), (8, 4), (16, 2)):
            with self.subTest(heads=heads, group=group):
                self.assertIsNone(self._validate(num_q_heads=heads, union=group))

    def test_head_count_is_capped_at_measured_range(self):
        # BLOCK_H = next_pow2(h) is never reduced by the smem step-down, so at
        # h=64 / 128 the [BLOCK_H, 512] fp32 accumulator spills or misses the
        # SM120 budget. Refuse what has not been measured rather than promise it.
        self.assertIsNone(self._validate(num_q_heads=32))
        with self.assertRaisesRegex(ValueError, "32 query heads"):
            self._validate(num_q_heads=33)
        with self.assertRaisesRegex(ValueError, "32 query heads"):
            self._validate(num_q_heads=128)

    def test_index_kpool_is_rejected_at_construction(self):
        # Regression: only `flashmla_sparse` is rerouted for the pooled indexer
        # tail, so this backend reached `_check_kpool_tail_backend` and raised
        # NotImplementedError on the first prefill. Reject it at startup.
        with self.assertRaisesRegex(ValueError, "index_kpool"):
            self._validate(index_kpool=2)
        self.assertIsNone(self._validate(index_kpool=1))


class TestTritonSparseMLAUnionResolution(CustomTestCase):
    """``_resolve_dsa_triton_union``: deterministic inference forces union off.

    Union makes a token's output depend on which tokens share its group and on
    the ``T % G`` tail, which breaks the batch invariance deterministic inference
    promises. The requested value must survive unchanged otherwise.
    """

    def _resolve(self, **kwargs):
        from sglang.srt.layers.attention.dsa_backend import _resolve_dsa_triton_union

        return _resolve_dsa_triton_union(**kwargs)

    def test_deterministic_forces_union_off(self):
        with self.assertLogs("sglang.srt.layers.attention.dsa_backend", "WARNING"):
            self.assertEqual(self._resolve(union=4, deterministic=True), 0)

    def test_requested_value_kept_otherwise(self):
        for union in (0, 2, 4):
            self.assertEqual(self._resolve(union=union, deterministic=False), union)
        self.assertEqual(self._resolve(union=0, deterministic=True), 0)


class TestTritonSparseMLAAdapter(CustomTestCase):
    """The backend method forwards exactly what the kernel expects."""

    def _call_forward(
        self, *, union=0, capturing=False, stream_capturing=False, topk_length=None
    ):
        from sglang.srt.layers.attention.dsa_backend import DeepseekSparseAttnBackend

        captured = {}

        def _fake_kernel(q, kv, indices, sm_scale, d_v=512, **kwargs):
            captured.update(
                q=q, kv=kv, indices=indices, sm_scale=sm_scale, d_v=d_v, **kwargs
            )
            return torch.zeros(q.shape[0], q.shape[1], d_v, dtype=torch.bfloat16)

        backend = DeepseekSparseAttnBackend.__new__(DeepseekSparseAttnBackend)
        backend.dsa_triton_union = union

        # `is_available` is patched too so the stream check is reached on the
        # CPU CI lane, where it would otherwise short-circuit.
        with (
            patch(
                "sglang.kernels.ops.attention.dsa.triton_sparse_mla_prefill.sparse_mla_prefill",
                _fake_kernel,
            ),
            patch(
                "sglang.srt.model_executor.runner_utils.capture_mode.get_is_capture_mode",
                lambda: capturing,
            ),
            patch("torch.cuda.is_available", lambda: True),
            patch("torch.cuda.is_current_stream_capturing", lambda: stream_capturing),
        ):
            out = backend._forward_triton_sparse_mla(
                q_all=torch.zeros(4, 8, 576, dtype=torch.bfloat16),
                kv_cache=torch.zeros(64, 576, dtype=torch.bfloat16),
                page_table_1=torch.zeros(4, 16, dtype=torch.int32),
                sm_scale=0.0625,
                v_head_dim=512,
                topk_length=topk_length,
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

    def test_union_off_unless_requested(self):
        captured, _ = self._call_forward()
        self.assertEqual(captured["union"], 0)

    def test_union_is_plumbed_through(self):
        captured, _ = self._call_forward(union=4)
        self.assertEqual(captured["union"], 4)

    def test_union_stays_on_when_only_the_breakable_graph_flag_is_set(self):
        # Regression: `get_is_capture_mode()` is true for the whole breakable
        # prefill graph replay, during which attention runs eagerly between the
        # graph segments. Gating on it switched union off for every replayed
        # prefill batch, i.e. for every bucket the default prefill graph covers.
        captured, _ = self._call_forward(union=4, capturing=True)
        self.assertEqual(captured["union"], 4)

    def test_union_is_disabled_when_the_stream_itself_is_capturing(self):
        # Under `--cuda-graph-backend-prefill full` the attention runs inside
        # stream capture; the union path has not been validated there, so the
        # per-token path (same result) must run instead.
        captured, _ = self._call_forward(
            union=4, capturing=False, stream_capturing=True
        )
        self.assertEqual(captured["union"], 0)

    def test_topk_length_is_forwarded_when_rows_match(self):
        # Without it the kernel rebuilds the per-row valid count on every layer
        # (~64 MB of temporaries at T=8192, topk=2048) from data the backend
        # already holds in `dsa_cache_seqlens_int32`.
        lengths = torch.full((4,), 16, dtype=torch.int32)
        captured, _ = self._call_forward(topk_length=lengths)
        self.assertIs(captured["topk_length"], lengths)

    def test_topk_length_is_dropped_when_rows_diverge(self):
        # Same contract as `_forward_flashmla_sparse`: metadata rows that do not
        # match q rows fall back to full-width compute instead of misindexing.
        captured, _ = self._call_forward(
            topk_length=torch.full((5,), 16, dtype=torch.int32)
        )
        self.assertIsNone(captured["topk_length"])


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
        import argparse

        from sglang.srt.server_args import ServerArgs

        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)
        actions_by_option = {
            option: action
            for action in parser._actions
            for option in action.option_strings
        }
        self.assertIn(
            "triton_sparse_mla", actions_by_option["--dsa-prefill-backend"].choices
        )

    def test_union_defaults_to_off(self):
        from sglang.srt.server_args import ServerArgs

        self.assertEqual(ServerArgs(model_path="dummy").dsa_triton_union, 0)

    def test_sm120_glm_fp8_still_resolves_to_flashinfer_on_its_own(self):
        # Registering this backend must not change what SM120 selects by itself:
        # the GLM FP8-KV arm of the resolver declares flashinfer_sparse_mla for
        # both phases unless the user set one, and a user-set prefill backend
        # is kept as given.
        from sglang.srt.arg_groups import overrides

        hf_config = SimpleNamespace(
            architectures=["GlmMoeDsaForCausalLM"], learnable_sink=False
        )
        platform = SimpleNamespace(is_npu=False, is_xpu=False, is_hip=False)

        def resolve(prefill):
            view = SimpleNamespace(
                dsa_prefill_backend=prefill,
                dsa_decode_backend=None,
                kv_cache_dtype="fp8_e4m3",
                enable_hisparse=False,
            )
            with (
                patch.object(
                    overrides,
                    "model_config_of",
                    return_value=SimpleNamespace(hf_config=hf_config),
                ),
                patch.object(overrides, "get_platform", return_value=platform),
                patch(
                    "sglang.srt.configs.model_config.is_deepseek_dsa",
                    return_value=True,
                ),
                patch("torch.cuda.get_device_capability", return_value=(12, 0)),
            ):
                return overrides._dsa_split_backend_resolution(view)

        self.assertEqual(
            resolve(None),
            {
                "dsa_prefill_backend": "flashinfer_sparse_mla",
                "dsa_decode_backend": "flashinfer_sparse_mla",
            },
        )
        self.assertEqual(
            resolve("triton_sparse_mla"),
            {"dsa_decode_backend": "flashinfer_sparse_mla"},
        )


if __name__ == "__main__":
    unittest.main()
