"""CPU tests for ``SRTPlatform.get_token_allocator_cls`` (page_size == 1)."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.mem_cache.allocator import swa as swa_allocator
from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator
from sglang.srt.mem_cache.base_swa_memory_pool import BaseSWAKVPool
from sglang.srt.platforms.interface import SRTPlatform
from sglang.test.test_utils import CustomTestCase


class _RecordingAllocator(TokenToKVPoolAllocator):
    built = []

    def __init__(self, size, dtype, device, kvcache, need_sort):
        type(self).built.append(size)
        super().__init__(size, dtype, device, kvcache, need_sort)


def _platform(token_cls):
    return SimpleNamespace(
        get_token_allocator_cls=lambda: token_cls,
        get_paged_allocator_cls=lambda: None,
    )


def _build_swa():
    kvcache = MagicMock(spec=BaseSWAKVPool)
    kvcache.full_kv_pool = MagicMock()
    kvcache.swa_kv_pool = MagicMock()
    kvcache.swa_req_ring_size = None
    return swa_allocator.SWATokenToKVPoolAllocator(
        size=16,
        size_swa=8,
        page_size=1,
        dtype=torch.int64,
        device="cpu",
        kvcache=kvcache,
        need_sort=False,
    )


class TestPlatformTokenAllocatorHook(CustomTestCase):
    def test_default_is_none(self):
        self.assertIsNone(SRTPlatform.get_token_allocator_cls(MagicMock()))

    def test_swa_page1_uses_platform_class(self):
        _RecordingAllocator.built = []
        with patch.object(
            swa_allocator, "current_platform", _platform(_RecordingAllocator)
        ):
            alloc = _build_swa()
        self.assertIsInstance(alloc.full_attn_allocator, _RecordingAllocator)
        self.assertIsInstance(alloc.swa_attn_allocator, _RecordingAllocator)
        self.assertEqual(_RecordingAllocator.built, [16, 8])

    def test_swa_page1_defaults_to_in_tree_class(self):
        with patch.object(swa_allocator, "current_platform", _platform(None)):
            alloc = _build_swa()
        self.assertIs(type(alloc.full_attn_allocator), TokenToKVPoolAllocator)
        self.assertIs(type(alloc.swa_attn_allocator), TokenToKVPoolAllocator)


if __name__ == "__main__":
    unittest.main()
