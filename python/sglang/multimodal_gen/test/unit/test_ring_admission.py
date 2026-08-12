# SPDX-License-Identifier: Apache-2.0
"""Ring admission is a backend capability, not a name whitelist."""

import unittest

from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
    AttentionBackend,
)
from sglang.multimodal_gen.runtime.layers.attention.backends.flash_attn import (
    FlashAttentionBackend,
)
from sglang.multimodal_gen.runtime.layers.attention.backends.sdpa import SDPABackend
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


if __name__ == "__main__":
    unittest.main()
