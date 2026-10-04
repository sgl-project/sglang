from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=9, stage="base-b", runner_config="1-gpu-small")
# Backend-specific: AMD lowers unified page zeroing through ROCm Triton,
# catching HIP-only codegen, launch, or stale-page failures.
register_amd_ci(est_time=9, suite="stage-b-test-1-gpu-small-amd")

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.allocator.unified_hybrid_swa import (
    UnifiedMambaSWATokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.allocator.unified_sub_pool import MultiEndedAllocator
from sglang.srt.mem_cache.unified_memory_pool import (
    MambaSubPoolSpec,
    MHASubPoolSpec,
    MLASubPoolSpec,
    UnifiedKVPool,
    UnifiedMHATokenToKVPool,
    UnifiedMLATokenToKVPool,
)

BF16_NAN = 0x7FC1  # LE bf16 NaN bit pattern, as SGLANG_DEBUG_POISON_POOL fills


def _build(device, page_size=1):
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
    )
    return buf, kvcache, allocator


def _build_mha(device, page_size=8):
    """The same pool with an MHA full side, as a GDN hybrid (Qwen3.5) builds it:
    its mamba neighbour's fp32 state shares the bytes the KV pages reuse."""
    full_spec = MHASubPoolSpec(
        name="full",
        layer_num=2,
        grow_direction="up",
        head_num=2,
        head_dim=8,
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
    buf = UnifiedKVPool(
        total_bytes=4096 * full_spec.entry_bytes(),
        sub_pool_specs=[full_spec, mamba_spec],
        device=device,
        enable_memory_saver=False,
        page_size=page_size,
    )
    kvcache = UnifiedMHATokenToKVPool(
        unified_buffer=buf, sub_pool_name="full", page_size=page_size
    )
    allocator = MultiEndedAllocator(
        kvcache=kvcache,
        unified_buffer=buf,
        sub_pool_name="full",
        device=device,
        is_id_owner=True,
        page_size=page_size,
    )
    return buf, kvcache, allocator


def _build_tri(device, page_size=4):
    """A tri-pool, [mamba (up) | swa (float) | full (down)], with MHA pools on
    both KV sides, as a mamba + sliding-window hybrid builds it."""
    full_spec = MHASubPoolSpec(
        name="full",
        layer_num=2,
        grow_direction="down",
        head_num=2,
        head_dim=8,
        store_dtype=torch.bfloat16,
    )
    swa_spec = MHASubPoolSpec(
        name="swa",
        layer_num=2,
        grow_direction="float",
        head_num=2,
        head_dim=8,
        store_dtype=torch.bfloat16,
    )
    mamba_spec = MambaSubPoolSpec(
        name="mamba",
        layer_num=1,
        grow_direction="up",
        conv_state_shapes=((3, 8),),
        conv_dtype=torch.bfloat16,
        temporal_state_shape=(0, 0, 0),
        temporal_dtype=torch.float32,
    )
    n_full, n_swa = 256, 128
    buf = UnifiedKVPool(
        total_bytes=n_full * full_spec.entry_bytes()
        + n_swa * swa_spec.entry_bytes()
        + 16 * mamba_spec.entry_bytes(),
        sub_pool_specs=[full_spec, swa_spec, mamba_spec],
        device=device,
        enable_memory_saver=False,
        page_size=page_size,
    )
    kvcache = SimpleNamespace(
        full_kv_pool=UnifiedMHATokenToKVPool(
            unified_buffer=buf, sub_pool_name="full", page_size=page_size
        ),
        swa_kv_pool=UnifiedMHATokenToKVPool(
            unified_buffer=buf, sub_pool_name="swa", page_size=page_size
        ),
        attach_allocators=lambda **_: None,
    )
    allocator = UnifiedMambaSWATokenToKVPoolAllocator(
        unified_buffer=buf,
        kvcache=kvcache,
        mamba_kvcache=SimpleNamespace(move_kv_cache=lambda dst, src: None),
        device=device,
        full_max_total_num_tokens=n_full,
        swa_max_total_num_tokens=n_swa,
        page_size=page_size,
    )
    return buf, kvcache.swa_kv_pool, allocator


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required (fused alloc kernel)")
class TestUnifiedHandoutZeroing(unittest.TestCase):
    """Root-cause guard: pages must leave the allocator ZEROED.

    The trtllm MLA kernel arithmetically masks (NaN-unsafe) the unwritten
    tail rows of a request's last partial page, so recycled / fresh page
    bytes must never carry NaN bit patterns. Static pools get this from
    torch.zeros; the unified pool must re-establish it at every hand-out."""

    build = staticmethod(_build)

    def _poison(self, buf):
        buf._raw.view(torch.int16).fill_(BF16_NAN)

    def _env(self, buf, kvcache):
        return buf._raw[: kvcache._num_pages * kvcache._page_bytes].view(
            kvcache._num_pages, kvcache._page_bytes
        )

    def _phys_pages(self, allocator, virt_tokens):
        return (allocator.translate_kv_loc(virt_tokens) // allocator.page_size).unique()

    def test_fresh_and_recycled_pages_zeroed(self):
        buf, kvcache, allocator = self.build("cuda")
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

    def test_zeroing_keys_on_pool_type(self):
        # UnifiedMLATokenToKVPool has NaN-unsafe partial-page reads whatever
        # its geometry; zeroing must key on the pool type, not on any
        # id-space scale.
        buf, kvcache, allocator = self.build("cuda")
        self.assertTrue(allocator._zero_pages_on_alloc)
        self._poison(buf)
        out = allocator.alloc(8)
        self.assertIsNotNone(out)
        env = self._env(buf, kvcache)
        pages = self._phys_pages(allocator, out)
        self.assertTrue((env[pages] == 0).all().item())


class TestUnifiedMHAHandoutZeroing(TestUnifiedHandoutZeroing):
    """The same contract for an MHA pool: trtllm_mha's fmha_v2 prefill loads a
    request's last partial page whole and multiplies its unwritten V rows by a
    zero probability, so one NaN there turns the request's output into NaN."""

    build = staticmethod(_build_mha)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required (fused alloc kernel)")
class TestTriPoolSWAHandoutZeroing(unittest.TestCase):
    """A tri-pool hands its sliding-window pages out through the float middle's
    own allocator; they must leave it zeroed like every other MHA page."""

    def test_fresh_and_recycled_swa_pages_zeroed(self):
        buf, swa, allocator = _build_tri("cuda")
        sa = allocator.swa_attn_allocator
        self.assertTrue(sa._zero_pages_on_alloc)
        env = buf._raw[: swa._num_pages * swa._page_bytes].view(
            swa._num_pages, swa._page_bytes
        )
        for _ in range(2):  # fresh, then recycled
            buf._raw.view(torch.int16).fill_(BF16_NAN)
            out = allocator.alloc(16)
            self.assertIsNotNone(out)
            pages = (sa.translate_kv_loc(out) // sa.page_size).unique()
            self.assertTrue((env[pages] == 0).all().item())
            allocator.free(out)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
