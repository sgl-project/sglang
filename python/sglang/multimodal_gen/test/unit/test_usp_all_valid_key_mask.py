"""An all-valid key mask must not force the dense-mask SDPA path.

Outside the FA backend, USPAttention serves a masked call with a dense SDPA
mask, which on sm100 pins PyTorch's cutlassF kernel even when the mask masks
nothing (Ideogram4 passes such a mask on every step).
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F

from sglang.multimodal_gen.runtime.layers.attention.layer import (
    USPAttention,
    build_varlen_mask_meta,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum

_LAYER = "sglang.multimodal_gen.runtime.layers.attention.layer"


class _CountingSdpa:
    """Stands in for the backend kernel and counts calls."""

    def __init__(self, scale: float):
        self.scale = scale
        self.calls = 0

    def forward(self, q, k, v, _ctx):
        self.calls += 1
        return _sdpa(q, k, v, self.scale)


def _sdpa(q, k, v, scale, key_mask=None):
    return F.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        attn_mask=None if key_mask is None else key_mask[:, None, None, :],
        scale=scale,
    ).transpose(1, 2)


class TestAllValidKeyMask(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        obj = USPAttention.__new__(USPAttention)
        obj.causal = False
        # sm100 default: the masked varlen fast path only serves FA
        obj.backend = AttentionBackendEnum.DYNAMIC_CUDNN_SDPA
        obj.softmax_scale = 8**-0.5
        obj.attn_impl = _CountingSdpa(obj.softmax_scale)
        obj.allow_cudnn_sdp = False
        obj.skip_sequence_parallel = False
        obj.sp_attention_mode = "ulysses"
        obj.sp_attention_mode_is_auto = False
        self.attn = obj

    def _forward(self, *args, **kwargs):
        with (
            patch(
                f"{_LAYER}.get_forward_context",
                return_value=SimpleNamespace(attn_metadata=None),
            ),
            patch(f"{_LAYER}.get_sequence_parallel_world_size", return_value=1),
        ):
            return self.attn.forward(*args, **kwargs)

    def test_all_valid_mask_runs_the_backend_kernel(self):
        q, k, v = (torch.randn(2, 6, 2, 8) for _ in range(3))
        mask = torch.ones(2, 6, dtype=torch.bool)

        out = self._forward(
            q, k, v, attn_mask=mask, attn_mask_meta=build_varlen_mask_meta(mask)
        )

        self.assertEqual(self.attn.attn_impl.calls, 1)
        torch.testing.assert_close(out, _sdpa(q, k, v, self.attn.softmax_scale))

    def test_padding_mask_keeps_the_masked_path(self):
        q, k, v = (torch.randn(1, 6, 2, 8) for _ in range(3))
        mask = torch.ones(1, 6, dtype=torch.bool)
        mask[:, 4:] = False

        out = self._forward(
            q, k, v, attn_mask=mask, attn_mask_meta=build_varlen_mask_meta(mask)
        )

        self.assertEqual(self.attn.attn_impl.calls, 0)
        torch.testing.assert_close(
            out, _sdpa(q, k, v, self.attn.softmax_scale, key_mask=mask)
        )

    def test_segmented_prefix_reaches_the_backend_kernel_concatenated(self):
        q, k, v = (torch.randn(1, 4, 2, 8) for _ in range(3))
        q_prefix, k_prefix, v_prefix = (torch.randn(1, 2, 2, 8) for _ in range(3))
        mask = torch.ones(1, 6, dtype=torch.bool)

        out = self._forward(
            q,
            k,
            v,
            attn_mask=mask,
            attn_mask_meta=build_varlen_mask_meta(mask),
            q_prefix=q_prefix,
            k_prefix=k_prefix,
            v_prefix=v_prefix,
        )

        self.assertEqual(self.attn.attn_impl.calls, 1)
        expected = _sdpa(
            torch.cat([q_prefix, q], dim=1),
            torch.cat([k_prefix, k], dim=1),
            torch.cat([v_prefix, v], dim=1),
            self.attn.softmax_scale,
        )
        torch.testing.assert_close(out, expected)

    def test_gap_meta_without_indices_keeps_the_masked_path(self):
        q, k, v = (torch.randn(1, 6, 2, 8) for _ in range(3))
        mask = torch.ones(1, 6, dtype=torch.bool)
        mask[:, 4:] = False

        out = self._forward(
            q, k, v, attn_mask=mask, attn_mask_meta={"pad_start": 4, "pad_end": 6}
        )

        self.assertEqual(self.attn.attn_impl.calls, 0)
        torch.testing.assert_close(
            out, _sdpa(q, k, v, self.attn.softmax_scale, key_mask=mask)
        )


if __name__ == "__main__":
    unittest.main()
