"""Unit tests for MRotaryEmbedding cos/sin cache dtype handling.

The base RotaryEmbedding deliberately keeps ``cos_sin_cache`` in fp32 on CUDA
for numerical stability (see the NOTE in ``RotaryEmbedding.__init__`` and
``SGLANG_ROPE_CACHE_FP32``). MRotaryEmbedding must not silently downcast that
cache to the query dtype: a bf16 cache shifts the rotary phase at the large
position values multimodal models use.

No server, no model loading. CPU-only: kernel launches are replaced with
recorders, so only the Python-side dtype handling is exercised.
"""

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
maybe_stub_sgl_kernel()

import unittest
from unittest.mock import MagicMock, patch

import torch

import sglang.srt.layers.rotary_embedding.mrope as mrope_module
from sglang.kernels.ops.attention import rotary_triton
from sglang.srt.layers.rotary_embedding.mrope import MRotaryEmbedding
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

HEAD_SIZE = 64
NUM_Q_HEADS = 4
NUM_KV_HEADS = 1
MROPE_SECTION = [16, 8, 8]  # sums to rotary_dim // 2


def _make_mrope() -> MRotaryEmbedding:
    mrope = MRotaryEmbedding(
        head_size=HEAD_SIZE,
        rotary_dim=HEAD_SIZE,
        max_position_embeddings=4096,
        base=10000,
        is_neox_style=True,
        dtype=torch.bfloat16,
        mrope_section=MROPE_SECTION,
    )
    # On CUDA the constructor keeps the cache in fp32; on CPU it casts to the
    # model dtype. Restore the CUDA-init state so the test reproduces what a
    # CUDA host sees.
    mrope.cos_sin_cache = mrope.cos_sin_cache.float()
    return mrope


class TestMRopeCacheDtype(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))

    def test_match_cos_sin_cache_dtype_keeps_fp32(self):
        """The dtype-match helper must only move the cache across devices,
        never downcast it to the (bf16) query dtype."""
        mrope = _make_mrope()
        query = torch.randn(4, NUM_Q_HEADS * HEAD_SIZE, dtype=torch.bfloat16)

        mrope._match_cos_sin_cache_dtype(query)

        self.assertEqual(
            mrope.cos_sin_cache.dtype,
            torch.float32,
            "multimodal RoPE cache was downcast to the query dtype; it must "
            "stay fp32 like the base RotaryEmbedding cache",
        )

    def test_forward_cuda_passes_fp32_cache_to_fused_kernel(self):
        """forward_cuda with 2D (multimodal) positions must hand the fp32
        cache to the fused mrope kernel and leave q/k dtypes untouched."""
        mrope = _make_mrope()
        num_tokens = 4
        positions = torch.zeros(3, num_tokens, dtype=torch.int64)
        query = torch.randn(num_tokens, NUM_Q_HEADS * HEAD_SIZE, dtype=torch.bfloat16)
        key = torch.randn(num_tokens, NUM_KV_HEADS * HEAD_SIZE, dtype=torch.bfloat16)

        seen = {}

        def fake_triton_mrope_fused(q, k, cos_sin_cache, *args):
            seen["cache_dtype"] = cos_sin_cache.dtype
            seen["q_dtype"] = q.dtype
            seen["k_dtype"] = k.dtype

        with patch.object(mrope_module, "triton_mrope_fused", fake_triton_mrope_fused):
            q_out, k_out = mrope.forward_cuda(positions, query, key)

        self.assertEqual(seen["cache_dtype"], torch.float32)
        self.assertEqual(seen["q_dtype"], torch.bfloat16)
        self.assertEqual(seen["k_dtype"], torch.bfloat16)
        # The fused kernel rotates q/k in place; output dtype is unchanged.
        self.assertEqual(q_out.dtype, torch.bfloat16)
        self.assertEqual(k_out.dtype, torch.bfloat16)

    def test_ernie45_fused_wrapper_keeps_fp32_cache(self):
        """The Ernie4.5 fused-rope wrapper must launch the kernel with the
        fp32 cache as-is (no per-call downcast, no copy on a matching
        device)."""
        num_tokens = 4
        q = torch.randn(num_tokens, NUM_Q_HEADS * HEAD_SIZE, dtype=torch.bfloat16)
        k = torch.randn(num_tokens, NUM_KV_HEADS * HEAD_SIZE, dtype=torch.bfloat16)
        cos_sin_cache = torch.randn(4096, HEAD_SIZE, dtype=torch.float32)
        positions = torch.zeros(3, num_tokens, dtype=torch.int64)

        with patch.object(
            rotary_triton, "_triton_ernie45_rope_qk_fused", MagicMock()
        ) as kernel:
            rotary_triton.triton_ernie45_rope_fused_inplace(
                q=q,
                k=k,
                cos_sin_cache=cos_sin_cache,
                positions=positions,
                mrope_section=[12, 12, 8],
                head_size=HEAD_SIZE,
                rotary_dim=HEAD_SIZE,
                is_neox_style=True,
            )

        launch = kernel.__getitem__.return_value
        launched_cache = launch.call_args.args[2]
        self.assertEqual(
            launched_cache.dtype,
            torch.float32,
            "Ernie4.5 fused rope downcast cos_sin_cache to the query dtype",
        )
        self.assertIs(launched_cache, cos_sin_cache)


if __name__ == "__main__":
    unittest.main()
