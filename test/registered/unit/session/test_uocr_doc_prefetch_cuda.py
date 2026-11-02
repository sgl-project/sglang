"""A parked document window's pages are encoded on a side stream ahead of prefill."""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.managers import mm_schedule
from sglang.srt.models.unlimited_ocr_v2_support import document_prefetch as p
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")


def _encode(items):
    """Vision tower stand-in: one embedding row per placeholder token."""
    return [it.feature.float().repeat(1, 1) for it in items]


class TestEarlyEncode(CustomTestCase):
    def test_parked_window_features_move_like_prefill_and_are_held(self):
        device = torch.device("cuda", torch.cuda.current_device())
        early = p.EarlyEmbeddings(encode=_encode)
        prefetch = p.DocPrefetch(
            early=early,
            device=device,
            forward_stream=torch.cuda.Stream(device),
            tp_size=1,
        )
        item = SimpleNamespace(
            hash=7, cache_key=b"a" * 32, feature=torch.ones(3, 4), offsets=[(2, 4)]
        )
        req = SimpleNamespace(
            rid="w1",
            sampling_params=SimpleNamespace(
                custom_params={p.KEY_AFTER: "w0", p.KEY_BUDGET: 10},
                max_new_tokens=10,
            ),
            multimodal_inputs=SimpleNamespace(mm_items=[item]),
            finished=lambda: False,
            uocr_doc=None,
        )

        calls = []
        real = mm_schedule._move_items_to_device

        def spy(items, dev, data_embedding_func):
            calls.append((list(items), dev, data_embedding_func))
            return real(items, dev, data_embedding_func)

        with mock.patch.object(p, "_move_items_to_device", spy):
            self.assertTrue(prefetch.park(req))
        torch.cuda.synchronize()

        (moved, dev, func), *rest = calls
        self.assertEqual(rest, [])
        self.assertEqual(moved, [item])
        self.assertEqual(dev, device)
        self.assertIs(func, _encode)
        self.assertEqual(item.feature.device.type, "cuda")
        emb = early.prefetched(item.cache_key, 3)
        self.assertEqual(tuple(emb.shape), (3, 4))
        prefetch.release_early("w1")
        self.assertFalse(early.has_prefetched())


if __name__ == "__main__":
    unittest.main()
