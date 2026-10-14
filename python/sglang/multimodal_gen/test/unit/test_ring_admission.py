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
