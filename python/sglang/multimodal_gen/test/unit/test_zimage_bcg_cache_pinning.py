"""Z-Image BCG: cache values consumed under capture must outlive slot replacement.

Regression for sgl-project/sglang#34183: the Z-Image forward keeps
single-slot shape-keyed caches (rope cos/sin caches, varlen mask metas)
whose keys change with every BCG text bucket — the rope keys even include
``data_ptr()`` of the per-signature static buffers. Capturing the next
bucket therefore replaces each slot and frees the old tensors, while the
previously captured graph still has their device addresses baked in;
replaying it dereferenced freed memory (illegal memory access or a hang at
the first replayed segment). ``_pin_for_active_capture`` keeps every value
a capture consumes alive for the module's lifetime.
"""

import unittest
import weakref
from unittest.mock import patch

import torch

import sglang.multimodal_gen.runtime.models.dits.zimage as zimage_mod
from sglang.multimodal_gen.runtime.models.dits.zimage import ZImageTransformer2DModel


def _bare_model() -> ZImageTransformer2DModel:
    # The helper only touches ``_bcg_pinned_cache_values``; skip heavy init.
    return ZImageTransformer2DModel.__new__(ZImageTransformer2DModel)


class TestZImageBcgCachePinning(unittest.TestCase):
    def test_pinned_value_survives_cache_slot_replacement(self):
        model = _bare_model()
        with patch.object(zimage_mod, "is_in_breakable_cuda_graph", lambda: True):
            value = (torch.randn(4), torch.arange(4))
            ref = weakref.ref(value[0])
            out = model._pin_for_active_capture(value)
            self.assertIs(out, value)
            # Simulate the single-slot cache dropping this entry for the next
            # bucket: the only remaining reference is the pin.
            del value, out
            self.assertIsNotNone(ref(), "captured cache value must stay alive")
            self.assertEqual(len(model._bcg_pinned_cache_values), 1)

    def test_nothing_pinned_outside_capture(self):
        model = _bare_model()
        with patch.object(zimage_mod, "is_in_breakable_cuda_graph", lambda: False):
            if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
                self.skipTest("unexpected active capture")
            value = (torch.randn(4),)
            ref = weakref.ref(value[0])
            out = model._pin_for_active_capture(value)
            self.assertIs(out, value)
            del value, out
            self.assertIsNone(ref(), "eager path must not grow pinned state")
            self.assertFalse(hasattr(model, "_bcg_pinned_cache_values"))


if __name__ == "__main__":
    unittest.main()
