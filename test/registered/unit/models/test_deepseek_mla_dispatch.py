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

import torch

from sglang.srt.models.deepseek_common import attention_backend_handler as abh
from sglang.srt.models.deepseek_common.attention_forward_methods.forward_methods import (
    AttnForwardMethod,
)
from sglang.srt.model_executor.forward_batch_deepseek_mha_mixin import (
    ForwardBatchDeepSeekMHAMixin,
)
from sglang.srt.runtime_context import get_context, reset_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

# Unbound so SimpleNamespace fakes can exercise the real predicate.
_mixin_has_dp_padding_rows = ForwardBatchDeepSeekMHAMixin.has_dp_padding_rows

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _fake_forward_batch(is_decode: bool):
    return SimpleNamespace(forward_mode=SimpleNamespace(is_decode=lambda: is_decode))


def _fake_attn(backend: str, rocm_fused_decode_mla: bool = True):
    return SimpleNamespace(
        current_attention_backend=backend,
        rocm_fused_decode_mla=rocm_fused_decode_mla,
    )


def _fake_one_shot_attn():
    return SimpleNamespace(
        flashinfer_mla_disable_ragged=False,
        chunked_prefix_cache_threshold=0,
        disable_chunked_prefix_cache=False,
    )


def _fake_one_shot_forward_batch(extend_seq_lens, num_input_rows=None):
    """Fake extend ForwardBatch for the flashinfer one-shot dispatch tests.

    ``num_input_rows`` emulates the padded input row count (positions.shape[0]
    after DP-attention padding); None means "no padding rows" (equal to the
    sum of extend_seq_lens).
    """
    if num_input_rows is None:
        num_input_rows = sum(extend_seq_lens)

    def has_dp_padding_rows():
        # Same predicate as ForwardBatchDeepSeekMHAMixin.has_dp_padding_rows.
        return sum(extend_seq_lens) != num_input_rows

    return SimpleNamespace(
        forward_mode=SimpleNamespace(is_extend_without_speculative=lambda: True),
        # None short-circuits mla_use_prefill_cp before any CP metadata read.
        attn_cp_metadata=None,
        extend_prefix_lens_cpu=[0] * len(extend_seq_lens),
        extend_seq_lens_cpu=extend_seq_lens,
        seq_lens_cpu=extend_seq_lens,
        has_dp_padding_rows=has_dp_padding_rows,
        get_max_chunk_capacity=lambda: 128 * 1024,
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


class TestMhaOneShotDpAttentionPadding(CustomTestCase):
    def _dispatch(self, extend_seq_lens, num_input_rows=None):
        # Publish a default config so the real dispatch path can read parallel
        # config leaves (mla_use_prefill_cp -> enable_prefill_context_parallel).
        # Hermetic: restore() reinstates the pre-test context, reset_context()
        # clears a leaked empty publish.
        override = get_context().override_server_args()
        override.install()
        self.addCleanup(override.restore)
        self.addCleanup(reset_context)
        return abh.handle_attention_flashinfer(
            _fake_one_shot_attn(),
            _fake_one_shot_forward_batch(extend_seq_lens, num_input_rows),
        )

    def test_padded_extend_falls_back_to_mla(self):
        # TP8/DP2 gives attn_tp_size=4. A one-token request is padded to four
        # input rows (attn_tp alignment, or MAX_LEN under the prefill breakable
        # CUDA graph), so the ragged ONE_SHOT wrapper cannot use its one-token
        # indptr safely.
        method = self._dispatch([1], num_input_rows=4)
        self.assertEqual(method, AttnForwardMethod.MLA)

    def test_padding_free_extend_keeps_one_shot(self):
        method = self._dispatch([4])
        self.assertEqual(method, AttnForwardMethod.MHA_ONE_SHOT)

    def test_per_request_aligned_but_padded_falls_back_to_mla(self):
        # MAX_LEN padding pads a whole rank even when every request's extend
        # length is individually aligned: DP0=[4] vs DP1=[8] pads DP0 to 8
        # rows. The per-request alignment check misses this; the padding-row
        # check must not.
        method = self._dispatch([4], num_input_rows=8)
        self.assertEqual(method, AttnForwardMethod.MLA)


class TestHasDpPaddingRows(CustomTestCase):
    """Direct tests for the padding predicate both dispatch consumers read.

    Guards the fail-closed contract: unknown padding state (gpu_only batches
    without host extend lengths) must read as padded, not as padding-free.
    """

    def _batch(self, extend_seq_lens_cpu, positions):
        return SimpleNamespace(
            extend_seq_lens_cpu=extend_seq_lens_cpu, positions=positions
        )

    def test_no_padding_when_rows_match(self):
        fb = self._batch([2, 6], torch.empty(8))
        self.assertFalse(_mixin_has_dp_padding_rows(fb))

    def test_padding_when_extra_rows(self):
        fb = self._batch([2, 6], torch.empty(12))
        self.assertTrue(_mixin_has_dp_padding_rows(fb))

    def test_unknown_lengths_fail_closed(self):
        fb = self._batch(None, torch.empty(8))
        self.assertTrue(_mixin_has_dp_padding_rows(fb))

    def test_missing_positions_fail_closed(self):
        fb = self._batch([8], None)
        self.assertTrue(_mixin_has_dp_padding_rows(fb))


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


if __name__ == "__main__":
    unittest.main()
