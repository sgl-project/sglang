# SPDX-License-Identifier: Apache-2.0
"""Ring admission is a backend capability, not a name whitelist."""

import importlib.util
import unittest
from unittest.mock import patch

import torch

from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
    AttentionBackend,
)
from sglang.multimodal_gen.runtime.layers.attention.backends.flash_attn import (
    FlashAttentionBackend,
)
from sglang.multimodal_gen.runtime.layers.attention.backends.sdpa import SDPABackend
from sglang.multimodal_gen.runtime.layers.attention.layer import USPAttention
from sglang.multimodal_gen.runtime.layers.usp import (
    _merge_attention_partials,
    _ring_merge_attention,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.runtime.server_args.server_args import (
    RING_CAPABLE_ATTENTION_BACKENDS,
)


class TestRingAdmission(unittest.TestCase):
    def test_default_is_not_ring_capable(self):
        self.assertFalse(AttentionBackend.supports_ring_rotation())
        self.assertFalse(SDPABackend.supports_ring_rotation())

    def test_lse_backends_declare_support(self):
        self.assertTrue(FlashAttentionBackend.supports_ring_rotation())

    def test_server_args_names_match_capabilities(self):
        # the name-level list gates before backend classes are importable on
        # every platform; keep it consistent with the classes it mirrors
        self.assertIn(
            FlashAttentionBackend.get_enum().name.lower(),
            RING_CAPABLE_ATTENTION_BACKENDS,
        )
        self.assertNotIn(
            SDPABackend.get_enum().name.lower(), RING_CAPABLE_ATTENTION_BACKENDS
        )
        self.assertIn(
            AttentionBackendEnum.AITER.name.lower(), RING_CAPABLE_ATTENTION_BACKENDS
        )

    def test_local_usp_backend_does_not_require_ring_capability(self):
        layer_module = "sglang.multimodal_gen.runtime.layers.attention.layer"
        with (
            patch(f"{layer_module}.get_compute_dtype", return_value=torch.float16),
            patch(f"{layer_module}.get_attn_backend", return_value=SDPABackend),
            patch(f"{layer_module}.get_ring_parallel_world_size", return_value=2),
        ):
            attention = USPAttention(
                num_heads=2,
                head_size=64,
                skip_sequence_parallel=True,
            )

        self.assertEqual(attention.backend, SDPABackend.get_enum())


@unittest.skipUnless(
    importlib.util.find_spec("aiter") is not None and torch.cuda.is_available(),
    "requires aiter on a ROCm GPU",
)
class TestAITerRingKernels(unittest.TestCase):
    """aiter returns the softmax LSE the ring merge needs: two KV chunks merged
    give one call over all keys, on the packed and the dense path."""

    def test_kv_chunks_merge_to_full_attention(self):
        from sglang.multimodal_gen.runtime.layers.attention.backends.aiter import (
            AITerBackend,
            AITerImpl,
        )

        self.assertTrue(AITerBackend.supports_ring_rotation())
        torch.manual_seed(0)
        impl = AITerImpl(num_heads=4, head_size=128, softmax_scale=128**-0.5)
        q, k, v = (
            torch.randn(384, 4, 128, device="cuda", dtype=torch.bfloat16)
            for _ in range(3)
        )
        chunks = (slice(0, 256), slice(256, 384))
        bounds = torch.tensor([0, 384], dtype=torch.int32, device="cuda")
        full = impl.forward_varlen(q, k, v, cu_seqlens=bounds, max_seqlen=384)
        out = lse = None
        for rows in chunks:
            step_out, step_lse = impl.forward_ring_kv_chunk(q, k[rows], v[rows])
            out, lse = _ring_merge_attention(out, lse, step_out, step_lse)
        # the merge works on bf16 partial outputs: one bf16 rounding apart
        torch.testing.assert_close(out.to(q.dtype), full, atol=1e-2, rtol=1e-2)

        q4, k4, v4 = q[None], k[None], v[None]
        parts = [
            impl.forward(q4, k4[:, rows], v4[:, rows], return_softmax_lse=True)
            for rows in chunks
        ]
        merged = _merge_attention_partials(*parts[0], *parts[1])
        torch.testing.assert_close(
            merged.to(q.dtype), impl.forward(q4, k4, v4), atol=1e-2, rtol=1e-2
        )


if __name__ == "__main__":
    unittest.main()
