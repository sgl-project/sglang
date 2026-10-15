# SPDX-License-Identifier: Apache-2.0
"""Ring admission is a backend capability, not a name whitelist."""

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
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.server_args.server_args import (
    RING_CAPABLE_ATTENTION_BACKENDS,
)


def _ring_capable_backends():
    """Ring-capable backend classes, as importable on this platform."""
    backends = [FlashAttentionBackend]
    if current_platform.is_rocm():
        from sglang.multimodal_gen.runtime.layers.attention.backends.aiter import (
            AITerBackend,
        )

        backends.append(AITerBackend)
    return backends


class TestRingAdmission(unittest.TestCase):
    def test_default_is_not_ring_capable(self):
        self.assertFalse(AttentionBackend.supports_ring_rotation())
        self.assertFalse(SDPABackend.supports_ring_rotation())
        self.assertFalse(SDPABackend.supports_ring_kv_chunk())

    def test_lse_backends_declare_support(self):
        self.assertTrue(FlashAttentionBackend.supports_ring_rotation())

    def test_ring_backends_expose_a_kv_chunk_kernel(self):
        # the masked tail-pad path dispatches on supports_ring_kv_chunk rather
        # than on the backend name, so a ring-capable backend that never got the
        # kernel would be admitted and then raise inside the per-hop merge
        for backend in _ring_capable_backends():
            with self.subTest(backend=backend.get_enum().name):
                self.assertTrue(backend.supports_ring_kv_chunk())

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

    @unittest.skipUnless(
        current_platform.is_rocm(), "the aiter package only imports on ROCm"
    )
    def test_aiter_names_match_capabilities(self):
        from sglang.multimodal_gen.runtime.layers.attention.backends.aiter import (
            AITerBackend,
        )

        self.assertTrue(AITerBackend.supports_ring_rotation())
        self.assertIn(
            AITerBackend.get_enum().name.lower(), RING_CAPABLE_ATTENTION_BACKENDS
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


if __name__ == "__main__":
    unittest.main()
