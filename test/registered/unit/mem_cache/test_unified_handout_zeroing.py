from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=4, stage="base-b", runner_config="1-gpu-small")

import unittest

import torch

from sglang.srt.mem_cache.multi_ended_allocator import MultiEndedAllocator
from sglang.srt.mem_cache.unified_memory_pool import (
    MambaSubPoolSpec,
    MLASubPoolSpec,
    UnifiedKVPool,
    UnifiedMLATokenToKVPool,
)

BF16_NAN = 0x7FC1  # LE bf16 NaN bit pattern, as SGLANG_DEBUG_POISON_POOL fills


def _build(device, page_size=1, kernel_page_multiplier=None):
    """A tiny MLA+mamba unified pool + full-side allocator.

    Mirrors init_unified_mamba_pools' construction just enough for the
    allocator hand-out path (the piece under test)."""
    layer_num = 2
    full_spec = MLASubPoolSpec(
        name="full",
        layer_num=layer_num,
        grow_direction="up",
        kv_lora_rank=16,
        qk_rope_head_dim=8,
        store_dtype=torch.bfloat16,
    )
    mamba_spec = MambaSubPoolSpec(
        name="mamba",
        layer_num=1,
        grow_direction="down",
        conv_state_shapes=((8, 3),),
        conv_dtype=torch.bfloat16,
        temporal_state_shape=(2, 4, 4),
        temporal_dtype=torch.float32,
    )
    total_bytes = 4096 * full_spec.entry_bytes()
    buf = UnifiedKVPool(
        total_bytes=total_bytes,
        sub_pool_specs=[full_spec, mamba_spec],
        device=device,
        enable_memory_saver=False,
        page_size=page_size,
        view_tail_pad_bytes=page_size * full_spec.entry_bytes(),
    )
    kvcache = UnifiedMLATokenToKVPool(
        unified_buffer=buf,
        sub_pool_name="full",
        kv_cache_dtype=torch.bfloat16,
        page_size=page_size,
    )
    allocator = MultiEndedAllocator(
        kvcache=kvcache,
        unified_buffer=buf,
        sub_pool_name="full",
        device=device,
        is_id_owner=True,
        page_size=page_size,
        kernel_page_multiplier=(
            layer_num if kernel_page_multiplier is None else kernel_page_multiplier
        ),
    )
    return buf, kvcache, allocator


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required (fused alloc kernel)")
class TestUnifiedHandoutZeroing(unittest.TestCase):
    """Root-cause guard: pages must leave the allocator ZEROED.

    The trtllm MLA kernel arithmetically masks (NaN-unsafe) the unwritten
    tail rows of a request's last partial page, so recycled / fresh page
    bytes must never carry NaN bit patterns. Static pools get this from
    torch.zeros; the unified pool must re-establish it at every hand-out."""

    def _poison(self, buf):
        buf._raw.view(torch.int16).fill_(BF16_NAN)

    def _env(self, buf, kvcache):
        return buf._raw[: kvcache._num_pages * kvcache._page_bytes].view(
            kvcache._num_pages, kvcache._page_bytes
        )

    def _phys_pages(self, allocator, virt_tokens):
        return (allocator.translate_kv_loc(virt_tokens) // allocator.page_size).unique()

    def test_fresh_and_recycled_pages_zeroed(self):
        buf, kvcache, allocator = _build("cuda")
        env = self._env(buf, kvcache)

        # Fresh hand-out over a poisoned pool (the deterministic form of
        # "freed GPU heap happened to contain NaN patterns").
        self._poison(buf)
        out = allocator.alloc(16)
        self.assertIsNotNone(out)
        pages = self._phys_pages(allocator, out)
        self.assertTrue((env[pages] == 0).all().item())
        # Untouched pages must still be poisoned, else the assert above is
        # vacuous (a whole-pool memset would also pass it).
        wm_page = int(pages.max().item()) + 2
        self.assertFalse((env[wm_page] == 0).all().item())

        # Recycle: free, re-poison the raw bytes (data only; v2p bookkeeping
        # is separate storage), re-alloc — recycled pages must be zeroed too.
        allocator.free(out)
        self._poison(buf)
        out2 = allocator.alloc(16)
        self.assertIsNotNone(out2)
        pages2 = self._phys_pages(allocator, out2)
        self.assertTrue((env[pages2] == 0).all().item())

    def test_zeroing_enabled_for_single_layer_multiplier(self):
        # A shard owning exactly ONE full-attention MLA layer has
        # kernel_page_multiplier == 1 but its pool is still
        # UnifiedMLATokenToKVPool with the same NaN-unsafe partial-page
        # reads — zeroing must key on the pool type, not on multiplier > 1.
        buf, kvcache, allocator = _build("cuda", kernel_page_multiplier=1)
        self.assertTrue(allocator._zero_pages_on_alloc)
        self._poison(buf)
        out = allocator.alloc(8)
        self.assertIsNotNone(out)
        env = self._env(buf, kvcache)
        pages = self._phys_pages(allocator, out)
        self.assertTrue((env[pages] == 0).all().item())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
