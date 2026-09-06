"""Hermetic unit tests for DeepSeek MLA attention-method dispatch on ROCm.

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
from sglang.srt.layers.cp import utils as cp_utils
from sglang.srt.layers.cp.zigzag import ZigzagCPStrategy
from sglang.srt.layers.utils import cp_utils as platform_cp_utils
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.models.deepseek_common import attention_backend_handler as abh
from sglang.srt.models.deepseek_common.attention_forward_methods.forward_methods import (
    AttnForwardMethod,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


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
            mock.patch.object(cp_utils, "enable_cp_v2", return_value=True),
            mock.patch.object(cp_base, "_STRATEGY", ZigzagCPStrategy(cp_size=4)),
            mock.patch.object(
                platform_cp_utils,
                "get_parallel",
                return_value=SimpleNamespace(enable_prefill_context_parallel=False),
            ),
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


if __name__ == "__main__":
    unittest.main()
