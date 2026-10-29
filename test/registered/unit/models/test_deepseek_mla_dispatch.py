"""Hermetic unit tests for DeepSeek MLA attention-method dispatch.

`_dispatch_mla_subtype` picks the forward method for MLA attention. On ROCm the
fused-decode-MLA + fused-RoPE fast path (`MLA_FUSED_ROPE_ROCM`) is only correct
for the aiter attention backend; taking it under the triton backend GPU-faults
on gfx95 (MI355). This test pins the dispatch table so the triton MLA path stays
on the plain `MLA` method.

Pure Python (no GPU, no model weights): `_is_hip` is patched and `attn` /
`forward_batch` are lightweight fakes. Runs on any PR-CI lane.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.srt.layers.cp import base as cp_base
from sglang.srt.layers.cp.zigzag import ZigzagCPStrategy
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.models.deepseek_common import attention_backend_handler as abh
from sglang.srt.models.deepseek_common.attention_forward_methods.forward_methods import (
    AttnForwardMethod,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


def _fake_forward_batch(is_decode: bool):
    return SimpleNamespace(forward_mode=SimpleNamespace(is_decode=lambda: is_decode))


def _fake_attn(backend: str, rocm_fused_decode_mla: bool = True):
    return SimpleNamespace(
        current_attention_backend=backend,
        rocm_fused_decode_mla=rocm_fused_decode_mla,
    )


class TestDispatchMLASubtype(CustomTestCase):
    def test_hip_aiter_decode_takes_fused_rope(self):
        # aiter + fused-decode + decode -> fused ROPE fast path (unchanged).
        with mock.patch.object(abh, "_is_hip", True):
            method = abh._dispatch_mla_subtype(
                _fake_attn("aiter"), _fake_forward_batch(is_decode=True)
            )
        self.assertEqual(method, AttnForwardMethod.MLA_FUSED_ROPE_ROCM)

    def test_hip_triton_decode_stays_plain_mla(self):
        # The fix: triton backend must NOT take the aiter-only fused path even
        # with rocm_fused_decode_mla set -- that path GPU-faults on gfx95.
        with mock.patch.object(abh, "_is_hip", True):
            method = abh._dispatch_mla_subtype(
                _fake_attn("triton"), _fake_forward_batch(is_decode=True)
            )
        self.assertEqual(method, AttnForwardMethod.MLA)

    def test_hip_aiter_extend_stays_plain_mla(self):
        # Fused path is decode-only; extend/prefill uses plain MLA.
        with mock.patch.object(abh, "_is_hip", True):
            method = abh._dispatch_mla_subtype(
                _fake_attn("aiter"), _fake_forward_batch(is_decode=False)
            )
        self.assertEqual(method, AttnForwardMethod.MLA)


class TestResolveRocmForwardMethod(CustomTestCase):
    """The generic MHA/MLA methods must never reach the CUDA forward paths on
    ROCm: those were stripped of their AMD branches when the AITER kernels moved
    into forward_mha_rocm.py / forward_mla_rocm.py."""

    def test_hip_routes_shared_methods_to_rocm(self):
        with mock.patch.object(abh, "_is_hip", True):
            self.assertEqual(
                abh.resolve_rocm_forward_method(AttnForwardMethod.MHA),
                AttnForwardMethod.MHA_ROCM,
            )
            self.assertEqual(
                abh.resolve_rocm_forward_method(AttnForwardMethod.MHA_ONE_SHOT),
                AttnForwardMethod.MHA_ONE_SHOT_ROCM,
            )
            self.assertEqual(
                abh.resolve_rocm_forward_method(AttnForwardMethod.MLA),
                AttnForwardMethod.MLA_ROCM,
            )

    def test_hip_leaves_platform_specific_methods_alone(self):
        with mock.patch.object(abh, "_is_hip", True):
            self.assertEqual(
                abh.resolve_rocm_forward_method(AttnForwardMethod.MLA_FUSED_ROPE_ROCM),
                AttnForwardMethod.MLA_FUSED_ROPE_ROCM,
            )

    def test_non_hip_is_identity(self):
        with mock.patch.object(abh, "_is_hip", False):
            for method in AttnForwardMethod:
                self.assertEqual(abh.resolve_rocm_forward_method(method), method)


class TestCPMLADispatch(CustomTestCase):
    def test_strategy_cp_uses_absorbed_mla_without_legacy_flags(self):
        # Normal MHA writes rank-local KV against full out_cache_loc before
        # the CP backend can gather it. Both one-shot and chunked MHA must
        # therefore be bypassed for an active strategy-based CP batch.
        attn = SimpleNamespace(
            chunked_prefix_cache_threshold=0,
            disable_chunked_prefix_cache=False,
            flashinfer_mla_disable_ragged=False,
        )
        with (
            mock.patch.object(abh, "_is_hip", False),
            mock.patch.object(cp_base, "_STRATEGY", ZigzagCPStrategy(cp_size=4)),
        ):
            for prefix in (0, 32):
                for capacity in (0, 8192):
                    for num_tokens in (1, 3952):
                        with self.subTest(
                            prefix=prefix, capacity=capacity, num_tokens=num_tokens
                        ):
                            batch = SimpleNamespace(
                                forward_mode=ForwardMode.EXTEND,
                                input_ids=range(num_tokens),
                                attn_cp_metadata=None,
                                extend_prefix_lens_cpu=[prefix],
                                extend_seq_lens_cpu=[num_tokens],
                                seq_lens_cpu=[prefix + num_tokens],
                                get_max_chunk_capacity=lambda: capacity,
                            )
                            expected = (
                                AttnForwardMethod.MLA
                                if num_tokens == 3952
                                else (
                                    AttnForwardMethod.MHA_ONE_SHOT
                                    if capacity == 8192
                                    else AttnForwardMethod.MHA_CHUNKED_KV
                                )
                            )
                            self.assertEqual(
                                abh._handle_attention_backend(attn, batch, "fa3"),
                                expected,
                            )


class TestKVShardedMLADispatch(CustomTestCase):
    def setUp(self):
        super().setUp()
        self.attn = SimpleNamespace(
            chunked_prefix_cache_threshold=0,
            disable_chunked_prefix_cache=False,
            flashinfer_mla_disable_ragged=False,
        )
        self.exec_config = SimpleNamespace(
            deterministic=SimpleNamespace(enable_deterministic_inference=False)
        )
        for patcher in (
            mock.patch.object(abh, "_is_hip", False),
            mock.patch.object(
                abh,
                "get_parallel",
                return_value=SimpleNamespace(enable_kv_cache_sharding=True),
            ),
            mock.patch.object(abh, "get_exec", return_value=self.exec_config),
            mock.patch.object(abh, "is_in_tc_piecewise_cuda_graph", return_value=False),
            mock.patch.object(abh, "is_in_breakable_cuda_graph", return_value=False),
            mock.patch.object(cp_base, "_STRATEGY", None),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def _batch(self, prefix=64, capacity=8192):
        num_tokens = 3952
        return SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            input_ids=range(num_tokens),
            attn_cp_metadata=None,
            extend_prefix_lens_cpu=[prefix],
            extend_seq_lens_cpu=[num_tokens],
            seq_lens_cpu=[prefix + num_tokens],
            get_max_chunk_capacity=lambda: capacity,
        )

    def test_fa3_disabled_chunked_prefix_cache_uses_absorbed_mla(self):
        # A cached prefix or the second prompt chunk must not enter FA3's
        # chunked-prefix path, which asserts prefix caching is enabled.
        self.attn.disable_chunked_prefix_cache = True
        self.assertEqual(
            abh.handle_attention_fa3(self.attn, self._batch()),
            AttnForwardMethod.MLA,
        )

    def test_fa3_deterministic_inference_uses_absorbed_mla(self):
        self.exec_config.deterministic.enable_deterministic_inference = True
        for prefix in (0, 64):
            with self.subTest(prefix=prefix):
                self.assertEqual(
                    abh.handle_attention_fa3(self.attn, self._batch(prefix=prefix)),
                    AttnForwardMethod.MLA,
                )

    def test_fa3_preserves_capacity_based_mha_dispatch(self):
        # Both readers can use sharded scratch. Retain one-shot attention
        # when the complete sequence fits, and chunked MHA otherwise.
        for prefix in (0, 64):
            for capacity, expected in (
                (8192, AttnForwardMethod.MHA_ONE_SHOT),
                (0, AttnForwardMethod.MHA_CHUNKED_KV),
            ):
                with self.subTest(prefix=prefix, capacity=capacity):
                    self.assertEqual(
                        abh.handle_attention_fa3(
                            self.attn, self._batch(prefix=prefix, capacity=capacity)
                        ),
                        expected,
                    )

    def test_fa3_short_prefix_uses_absorbed_mla(self):
        self.attn.chunked_prefix_cache_threshold = 4096
        self.assertEqual(
            abh.handle_attention_fa3(self.attn, self._batch()),
            AttnForwardMethod.MLA,
        )

    def test_fa3_active_cp_uses_absorbed_mla(self):
        # CP must gather rank-local latent activations before the pool write.
        with mock.patch.object(cp_base, "_STRATEGY", ZigzagCPStrategy(cp_size=4)):
            for prefix in (0, 64):
                for capacity in (0, 8192):
                    with self.subTest(prefix=prefix, capacity=capacity):
                        self.assertEqual(
                            abh.handle_attention_fa3(
                                self.attn,
                                self._batch(prefix=prefix, capacity=capacity),
                            ),
                            AttnForwardMethod.MLA,
                        )

    def test_trtllm_mla_keeps_chunked_mha_for_prefill(self):
        # TRTLLM has a separate dispatcher and requires prefix caching for
        # sharding; changing FA3 dispatch must not change its supported path.
        for prefix in (0, 64):
            with self.subTest(prefix=prefix):
                self.assertEqual(
                    abh.handle_attention_trtllm_mla(
                        self.attn, self._batch(prefix=prefix)
                    ),
                    AttnForwardMethod.MHA_CHUNKED_KV,
                )


if __name__ == "__main__":
    unittest.main()
