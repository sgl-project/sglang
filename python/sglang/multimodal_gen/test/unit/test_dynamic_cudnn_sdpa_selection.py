"""Which kernel DynamicCudnnSDPAImpl picks on Blackwell."""

import unittest
from unittest.mock import patch

import torch

from sglang.multimodal_gen.runtime.layers.attention.backends.sdpa import (
    DynamicCudnnSDPAImpl,
)

_CUDA = torch.cuda.is_available()


def _impl(capability: tuple[int, int], head_size: int = 128) -> DynamicCudnnSDPAImpl:
    # the constructor also pins the process-wide FA version; keep it untouched
    with (
        patch("torch.cuda.get_device_capability", return_value=capability),
        patch(
            "sglang.multimodal_gen.runtime.layers.attention.backends.flash_attn.set_fa_ver"
        ),
    ):
        return DynamicCudnnSDPAImpl(
            num_heads=2, head_size=head_size, causal=False, softmax_scale=0.125
        )


@unittest.skipUnless(_CUDA, "CUDA required")
class TestDynamicCudnnSelection(unittest.TestCase):
    def _qkv(self, sq: int, skv: int, d: int = 128):
        q = torch.empty(1, sq, 2, d, device="cuda", dtype=torch.bfloat16)
        kv = torch.empty(1, skv, 2, d, device="cuda", dtype=torch.bfloat16)
        return q, kv, kv

    def test_sm103_long_self_attention_takes_fa4(self):
        impl = _impl((10, 3))
        self.assertFalse(impl._use_cudnn_sdpa(*self._qkv(16384, 16384)))
        self.assertFalse(impl._use_cudnn_sdpa(*self._qkv(75600, 75600)))

    def test_sm103_short_cross_and_hd64_keep_cudnn(self):
        impl = _impl((10, 3))
        self.assertTrue(impl._use_cudnn_sdpa(*self._qkv(8192, 8192)))
        self.assertTrue(impl._use_cudnn_sdpa(*self._qkv(37800, 512)))
        impl64 = _impl((10, 3), head_size=64)
        self.assertTrue(impl64._use_cudnn_sdpa(*self._qkv(18900, 18900, d=64)))

    def test_sm100_keeps_cudnn(self):
        impl = _impl((10, 0))
        self.assertTrue(impl._use_cudnn_sdpa(*self._qkv(75600, 75600)))


if __name__ == "__main__":
    unittest.main()
