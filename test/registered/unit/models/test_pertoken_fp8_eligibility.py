"""Unit tests for the fused per-token FP8 eligibility predicate.

Covers ``_is_per_channel_dynamic_fp8`` in
``srt/models/deepseek_common/utils.py``, the gate that decides whether a
projection's activation quant may be folded into the upstream RMSNorm and its
``(fp8, per-token scale)`` tuple fed straight to ``gemm_a8w8_bpreshuffle``.

The contract is read from the layer's own state: an explicit
``_fp8_weight_preshuffled`` marker (set by the quant method at load, not inferred
from the global gfx95 flag), ``input_scale is None`` (dynamic), and a
per-channel ``weight_scale`` whose ``numel`` equals the projection's output size
(``output_size_per_partition`` else ``output_size``), stored 1-D ``[N]`` or 2-D
``[N, 1]``.

This guards the real regressions: per-tensor / static-input-scale / block-scale
/ non-preshuffled fp8 must NOT be folded; a scale sized to the input dim ``K``
must NOT be accepted; a per-channel scale (1-D or 2-D) MUST be folded.

Pure logic, no server/engine: runs on CPU CI.
"""

import types
import unittest
from unittest.mock import patch

import torch

from sglang.srt.models.deepseek_common.utils import (
    _is_block_scale_fp8,
    _is_per_channel_dynamic_fp8,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_PRED = "sglang.srt.models.deepseek_common.utils._use_aiter_bpreshuffle_gfx95"

# 2112 = q_lora_rank(1536) + kv_lora_rank(512) + qk_rope_head_dim(64): the real
# fused entry-proj (fused_qkv_a_proj_with_mqa) output width for this checkpoint.
N = 2112
K = 256


def _make_proj(
    weight_scale,
    input_scale=None,
    weight_dtype=torch.float8_e4m3fn,
    preshuffled=True,
    out_features=N,
):
    """Build a minimal proj-like object with just the attrs the predicate reads.

    Defaults describe an eligible per-channel dynamic fp8 proj: preshuffled
    marker set, no static input_scale, output size == N. Individual tests flip
    one attribute to exercise a rejection path.
    """
    weight = torch.empty((N, K), dtype=weight_dtype)
    return types.SimpleNamespace(
        weight=weight,
        weight_scale=weight_scale,
        input_scale=input_scale,
        output_size=out_features,
        _fp8_weight_preshuffled=preshuffled,
    )


class TestPerChannelDynamicFp8Eligibility(CustomTestCase):
    @patch(_PRED, True)
    def test_per_channel_scale_1d_is_eligible(self):
        # This checkpoint's layout: one scale per output row, stored 1-D [N].
        proj = _make_proj(weight_scale=torch.empty(N, dtype=torch.float32))
        self.assertTrue(_is_per_channel_dynamic_fp8(proj))

    @patch(_PRED, True)
    def test_per_channel_scale_2d_is_eligible(self):
        # [N, 1] must also be accepted.
        proj = _make_proj(weight_scale=torch.empty((N, 1), dtype=torch.float32))
        self.assertTrue(_is_per_channel_dynamic_fp8(proj))

    @patch(_PRED, True)
    def test_per_tensor_scalar_scale_is_rejected(self):
        # Per-tensor weight scale (numel == 1) is not the per-channel contract.
        for ws in (torch.empty(1), torch.empty(())):
            proj = _make_proj(weight_scale=ws)
            self.assertFalse(_is_per_channel_dynamic_fp8(proj))

    @patch(_PRED, True)
    def test_scale_matching_input_dim_is_rejected(self):
        # A scale sized to the input dim K (not output channels N) must be
        # rejected -- exact output-channel match, not "some weight dimension".
        proj = _make_proj(weight_scale=torch.empty(K, dtype=torch.float32))
        self.assertFalse(_is_per_channel_dynamic_fp8(proj))

    @patch(_PRED, True)
    def test_missing_preshuffle_marker_is_rejected(self):
        # The explicit preshuffle marker must be set by the quant method at load;
        # without it (e.g. a non-aiter build) the fold must not fire.
        proj = _make_proj(
            weight_scale=torch.empty(N, dtype=torch.float32), preshuffled=False
        )
        self.assertFalse(_is_per_channel_dynamic_fp8(proj))

    @patch(_PRED, True)
    def test_static_input_scale_is_rejected(self):
        # A static per-tensor input_scale means the activation quant is not the
        # dynamic per-token path this fold requires.
        proj = _make_proj(
            weight_scale=torch.empty(N, dtype=torch.float32),
            input_scale=torch.empty(1, dtype=torch.float32),
        )
        self.assertFalse(_is_per_channel_dynamic_fp8(proj))

    @patch(_PRED, True)
    def test_block_scale_is_rejected(self):
        # Block-scale (2-D with >1 columns) is handled by _is_block_scale_fp8,
        # not this predicate.
        proj = _make_proj(weight_scale=torch.empty((N, K // 128), dtype=torch.float32))
        self.assertFalse(_is_per_channel_dynamic_fp8(proj))
        self.assertTrue(_is_block_scale_fp8(proj))

    @patch(_PRED, True)
    def test_non_fp8_weight_is_rejected(self):
        proj = _make_proj(
            weight_scale=torch.empty(N, dtype=torch.float32),
            weight_dtype=torch.bfloat16,
        )
        self.assertFalse(_is_per_channel_dynamic_fp8(proj))

    @patch(_PRED, True)
    def test_missing_weight_scale_is_rejected(self):
        proj = _make_proj(weight_scale=None)
        self.assertFalse(_is_per_channel_dynamic_fp8(proj))

    @patch(_PRED, False)
    def test_disabled_when_bpreshuffle_unavailable(self):
        # Even a valid per-channel proj must fall back when the aiter bpreshuffle
        # hardware path is not available.
        proj = _make_proj(weight_scale=torch.empty(N, dtype=torch.float32))
        self.assertFalse(_is_per_channel_dynamic_fp8(proj))

    @patch(_PRED, True)
    def test_block_and_per_channel_are_mutually_exclusive(self):
        # Invariant relied on by the caller ordering (block checked first).
        per_channel = _make_proj(weight_scale=torch.empty(N, dtype=torch.float32))
        block = _make_proj(weight_scale=torch.empty((N, K // 128), dtype=torch.float32))
        self.assertNotEqual(
            _is_per_channel_dynamic_fp8(per_channel), _is_block_scale_fp8(per_channel)
        )
        self.assertNotEqual(
            _is_per_channel_dynamic_fp8(block), _is_block_scale_fp8(block)
        )


if __name__ == "__main__":
    unittest.main()
