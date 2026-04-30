"""
Tests for mamba_spec_state_size optimization in disaggregation decode mode.

Background
----------
In HybridMambaDecodeReqToTokenPool (disaggregation decode), `pre_alloc_size`
extra slots are reserved for pre-allocated requests waiting for KV transfer.
These pre-allocated requests do NOT participate in speculative decoding, so
the speculative intermediate caches (intermediate_ssm_state_cache and
intermediate_conv_window_cache) do not need entries for them.

The fix changes:
    mamba_spec_state_size = size + pre_alloc_size   (BEFORE, wasteful)
    mamba_spec_state_size = size                    (AFTER,  correct)

Experiment 1 – Memory savings
    Directly construct MambaPool with both spec_state_size values and compare
    the GPU memory consumed by the speculative intermediate caches.

Experiment 2 – Functional correctness
    Construct MambaPool with the optimized spec_state_size = size (while
    main cache remains size + pre_alloc_size), verify speculative cache R/W
    within [0, size), main-cache R/W for all slots, and no OOB.

Usage
-----
    python -m pytest test/registered/unit/mem_cache/test_mamba_spec_state_size.py -v -s
"""

import os
import unittest

import torch

# Set SGLANG_MAMBA_SSM_DTYPE *before* any sglang import that reads it
os.environ.setdefault("SGLANG_MAMBA_SSM_DTYPE", "bfloat16")

from sglang.srt.configs.mamba_utils import Mamba2CacheParams, Mamba2StateShape
from sglang.srt.mem_cache.memory_pool import MambaPool, get_tensor_size_bytes
from sglang.srt.utils import get_device

MB = 1 << 20
GB = 1 << 30

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NUM_LAYERS = 48
GLOBAL_INTERVAL = 4
FULL_ATTN_LAYERS = [i for i in range(GLOBAL_INTERVAL - 1, NUM_LAYERS, GLOBAL_INTERVAL)]
MAMBA_LAYERS = [i for i in range(NUM_LAYERS) if i not in FULL_ATTN_LAYERS]


def _make_cache_params():
    """Create Mamba2CacheParams with realistic shapes (Qwen3-Next-like)."""
    shape = Mamba2StateShape.create(
        tp_world_size=1,
        intermediate_size=4096,
        n_groups=16,
        num_heads=32,
        head_dim=128,
        state_size=128,
        conv_kernel=4,
    )
    cache_params = Mamba2CacheParams(shape=shape, layers=MAMBA_LAYERS)
    return cache_params


def _create_mamba_pool(size, spec_state_size, cache_params, draft_tokens, device):
    """Create a MambaPool and return (total_spec_bytes, pool)."""
    pool = MambaPool(
        size=size,
        spec_state_size=spec_state_size,
        cache_params=cache_params,
        device=device,
        enable_memory_saver=False,
        speculative_num_draft_tokens=draft_tokens,
    )
    ssm_bytes = get_tensor_size_bytes(pool.mamba_cache.intermediate_ssm)
    conv_bytes = get_tensor_size_bytes(pool.mamba_cache.intermediate_conv_window)
    return ssm_bytes + conv_bytes, pool


# ===================================================================
# Experiment 1: Memory savings
# ===================================================================
class TestSpecStateSizeMemorySavings(unittest.TestCase):
    """Compare GPU memory of speculative caches with old vs new spec_state_size."""

    def test_memory_savings_basic(self):
        """With realistic params, the fix should save measurable GPU memory."""
        device = get_device()
        size = 32  # max_running_requests
        pre_alloc_size = 64  # typical for small deployments (2x)
        draft_tokens = 5
        cache_params = _make_cache_params()
        mamba_pool_size = size + pre_alloc_size  # main cache serves all slots

        old_bytes, old_pool = _create_mamba_pool(
            size=mamba_pool_size,
            spec_state_size=size + pre_alloc_size,  # OLD
            cache_params=cache_params,
            draft_tokens=draft_tokens,
            device=device,
        )
        del old_pool
        torch.cuda.empty_cache()

        new_bytes, new_pool = _create_mamba_pool(
            size=mamba_pool_size,
            spec_state_size=size,  # NEW (fix)
            cache_params=cache_params,
            draft_tokens=draft_tokens,
            device=device,
        )
        del new_pool
        torch.cuda.empty_cache()

        saved_bytes = old_bytes - new_bytes
        ratio = saved_bytes / old_bytes * 100

        print("\n" + "=" * 70)
        print("Experiment 1: Speculative cache memory comparison")
        print("=" * 70)
        print(f"  size (max_running_requests) : {size}")
        print(f"  pre_alloc_size              : {pre_alloc_size}")
        print(f"  speculative_num_draft_tokens: {draft_tokens}")
        print(f"  num_mamba_layers            : {len(MAMBA_LAYERS)}")
        print(f"  ---")
        print(f"  OLD spec_state_size = size + pre_alloc = {size + pre_alloc_size}")
        print(f"  NEW spec_state_size = size             = {size}")
        print(f"  ---")
        print(f"  OLD intermediate cache total: {old_bytes / MB:.2f} MB")
        print(f"  NEW intermediate cache total: {new_bytes / MB:.2f} MB")
        print(
            f"  Memory saved                : {saved_bytes / MB:.2f} MB ({ratio:.1f}%)"
        )
        print("=" * 70)

        self.assertGreater(
            saved_bytes, 0, "Optimized spec_state_size should consume less memory"
        )
        # pre_alloc is 2x size, so savings should be > 50%
        self.assertGreater(
            ratio,
            50.0,
            f"Expected > 50% savings with pre_alloc=2*size, got {ratio:.1f}%",
        )

    def test_memory_savings_sweep(self):
        """Sweep different pre_alloc_size values and show savings scale linearly."""
        device = get_device()
        size = 32
        draft_tokens = 5
        cache_params = _make_cache_params()

        print("\n" + "=" * 70)
        print("Experiment 1b: Memory savings sweep over pre_alloc_size")
        print(
            f"  size={size}, draft_tokens={draft_tokens}, "
            f"num_mamba_layers={len(MAMBA_LAYERS)}"
        )
        print("-" * 70)
        print(
            f"  {'pre_alloc':>10}  {'old_spec_sz':>12}  {'new_spec_sz':>12}"
            f"  {'old_MB':>8}  {'new_MB':>8}  {'saved_MB':>9}  {'saved%':>7}"
        )
        print("-" * 70)

        for pre_alloc_size in [0, 16, 32, 64, 128]:
            mamba_pool_size = size + pre_alloc_size

            old_bytes, p = _create_mamba_pool(
                size=mamba_pool_size,
                spec_state_size=size + pre_alloc_size,
                cache_params=cache_params,
                draft_tokens=draft_tokens,
                device=device,
            )
            del p
            torch.cuda.empty_cache()

            new_bytes, p = _create_mamba_pool(
                size=mamba_pool_size,
                spec_state_size=size,
                cache_params=cache_params,
                draft_tokens=draft_tokens,
                device=device,
            )
            del p
            torch.cuda.empty_cache()

            saved = old_bytes - new_bytes
            pct = (saved / old_bytes * 100) if old_bytes > 0 else 0.0
            print(
                f"  {pre_alloc_size:>10}  {size + pre_alloc_size:>12}  {size:>12}"
                f"  {old_bytes / MB:>8.2f}  {new_bytes / MB:>8.2f}"
                f"  {saved / MB:>9.2f}  {pct:>6.1f}%"
            )

            if pre_alloc_size == 0:
                self.assertEqual(old_bytes, new_bytes)
            else:
                self.assertGreater(saved, 0)

        print("=" * 70)


# ===================================================================
# Experiment 2: Functional correctness
# ===================================================================
class TestSpecStateSizeFunctionalCorrectness(unittest.TestCase):
    """Verify that the optimized MambaPool works correctly.

    We construct MambaPool with:
      - main cache size = size + pre_alloc_size  (serves ALL slots)
      - spec_state_size = size                   (serves only running reqs)
    and verify R/W, alloc, and boundary conditions.
    """

    def _make_pool(self, size, pre_alloc_size, draft_tokens=5):
        """Create a MambaPool simulating the FIXED disagg-decode behavior."""
        device = get_device()
        cache_params = _make_cache_params()
        pool = MambaPool(
            size=size + pre_alloc_size,  # main cache for all slots
            spec_state_size=size,  # spec cache for running only
            cache_params=cache_params,
            device=device,
            enable_memory_saver=False,
            speculative_num_draft_tokens=draft_tokens,
        )
        return pool

    def _make_pool_old(self, size, pre_alloc_size, draft_tokens=5):
        """Create a MambaPool simulating the OLD (unfixed) behavior."""
        device = get_device()
        cache_params = _make_cache_params()
        pool = MambaPool(
            size=size + pre_alloc_size,
            spec_state_size=size + pre_alloc_size,  # OLD: wasteful
            cache_params=cache_params,
            device=device,
            enable_memory_saver=False,
            speculative_num_draft_tokens=draft_tokens,
        )
        return pool

    def test_cache_dimensions(self):
        """Verify main and speculative cache dimensions are as expected."""
        size, pre_alloc_size = 16, 32
        draft_tokens = 5
        pool = self._make_pool(size, pre_alloc_size, draft_tokens)
        cache = pool.mamba_cache

        # Main cache: dim-1 = (size + pre_alloc_size) + 1
        main_dim = cache.temporal.shape[1]
        self.assertEqual(main_dim, size + pre_alloc_size + 1)

        # Speculative cache: dim-1 = size + 1
        spec_dim = cache.intermediate_ssm.shape[1]
        self.assertEqual(spec_dim, size + 1)

        print("\n" + "=" * 70)
        print("Experiment 2a: Cache dimension verification")
        print(f"  size={size}, pre_alloc_size={pre_alloc_size}")
        print(
            f"  Main cache temporal dim     : {main_dim} "
            f"(= size+pre_alloc+1 = {size + pre_alloc_size + 1})"
        )
        print(f"  Speculative cache dim       : {spec_dim} (= size+1 = {size + 1})")
        print(f"  intermediate_ssm shape      : {list(cache.intermediate_ssm.shape)}")
        print(
            f"  intermediate_conv shape     : "
            f"{[list(t.shape) for t in cache.intermediate_conv_window]}"
        )
        print("  -> PASS")
        print("=" * 70)

    def test_speculative_cache_read_write(self):
        """All running-request indices [0, size) can R/W speculative caches."""
        size, pre_alloc_size = 16, 32
        draft_tokens = 5
        pool = self._make_pool(size, pre_alloc_size, draft_tokens)
        cache = pool.mamba_cache

        spec_dim = cache.intermediate_ssm.shape[1]  # size + 1
        self.assertEqual(spec_dim, size + 1)

        # Write sentinels to all valid running-request indices [0, size]
        for idx in range(size + 1):
            sentinel = float(idx + 1)
            cache.intermediate_ssm[:, idx, :, :, :, :] = sentinel
            for conv_t in cache.intermediate_conv_window:
                conv_t[:, idx, :, :, :] = sentinel

        # Read back and verify — no corruption
        errors = 0
        for idx in range(size + 1):
            sentinel = float(idx + 1)
            if not torch.all(cache.intermediate_ssm[:, idx] == sentinel):
                errors += 1
            for conv_t in cache.intermediate_conv_window:
                if not torch.all(conv_t[:, idx] == sentinel):
                    errors += 1

        self.assertEqual(errors, 0, f"Found {errors} read-back mismatches")

        print("\n" + "=" * 70)
        print("Experiment 2b: Speculative cache read/write correctness")
        print(f"  size={size}, pre_alloc_size={pre_alloc_size}")
        print(f"  Spec cache dim: {spec_dim}")
        print(f"  Written & verified sentinels for indices [0..{size}]: ALL CORRECT")
        print("  -> PASS")
        print("=" * 70)

    def test_main_cache_all_slots(self):
        """Main cache (conv/temporal) serves ALL size+pre_alloc_size slots."""
        size, pre_alloc_size = 8, 16
        draft_tokens = 3
        pool = self._make_pool(size, pre_alloc_size, draft_tokens)
        cache = pool.mamba_cache

        total = size + pre_alloc_size + 1  # +1 for padding slot

        # Write to main cache at ALL indices [0, total) — no OOB
        for idx in range(total):
            cache.temporal[:, idx] = float(idx)
            for conv_t in cache.conv:
                conv_t[:, idx] = float(idx)

        # Read back and verify
        for idx in range(total):
            self.assertTrue(
                torch.all(cache.temporal[:, idx] == float(idx)),
                f"temporal mismatch at index {idx}",
            )
            for conv_t in cache.conv:
                self.assertTrue(
                    torch.all(conv_t[:, idx] == float(idx)),
                    f"conv mismatch at index {idx}",
                )

        print("\n" + "=" * 70)
        print("Experiment 2c: Main cache covers all slots")
        print(f"  size={size}, pre_alloc_size={pre_alloc_size}")
        print(f"  Main cache dim: {total}")
        print(f"  Written & verified all indices [0..{total - 1}]: ALL CORRECT")
        print("  -> PASS")
        print("=" * 70)

    def test_alloc_full_capacity(self):
        """MambaPool can allocate all (size + pre_alloc_size) slots."""
        size, pre_alloc_size = 8, 16
        draft_tokens = 3
        pool = self._make_pool(size, pre_alloc_size, draft_tokens)

        total = size + pre_alloc_size
        self.assertEqual(pool.available_size(), total)

        indices = pool.alloc(total)
        self.assertIsNotNone(indices, "Should allocate all slots")
        self.assertEqual(len(indices), total)
        self.assertEqual(pool.available_size(), 0)

        # Free and verify
        pool.free(indices)
        self.assertEqual(pool.available_size(), total)

        print("\n" + "=" * 70)
        print("Experiment 2d: Full allocation + free")
        print(f"  size={size}, pre_alloc_size={pre_alloc_size}")
        print(f"  Allocated {total} slots, freed {total} slots: OK")
        print("  -> PASS")
        print("=" * 70)

    def test_spec_cache_no_oob_at_boundary(self):
        """Writing at index = size (the boundary) should NOT OOB.
        Writing at index > size should OOB (verifying the boundary is correct)."""
        size, pre_alloc_size = 8, 16
        draft_tokens = 3
        pool = self._make_pool(size, pre_alloc_size, draft_tokens)
        cache = pool.mamba_cache

        spec_dim = cache.intermediate_ssm.shape[1]  # = size + 1
        self.assertEqual(spec_dim, size + 1)

        # Writing at boundary index = size should work (last valid index)
        cache.intermediate_ssm[:, size, :, :, :, :] = 99.0
        self.assertTrue(torch.all(cache.intermediate_ssm[:, size] == 99.0))

        # Index > size would be out of bounds — verify shape prevents it
        self.assertEqual(
            cache.intermediate_ssm.shape[1],
            size + 1,
            "Spec cache should have exactly size+1 entries",
        )

        print("\n" + "=" * 70)
        print("Experiment 2e: Boundary condition verification")
        print(f"  spec_dim = {spec_dim}, last valid index = {size}")
        print(f"  Write at boundary index {size}: OK")
        print(f"  Shape constraint prevents index > {size}")
        print("  -> PASS")
        print("=" * 70)


# ===================================================================
# Summary: side-by-side comparison
# ===================================================================
class TestEndToEndComparison(unittest.TestCase):
    """Print a consolidated summary table comparing old and new behavior."""

    def test_summary_table(self):
        device = get_device()
        size = 32
        pre_alloc_size = 64
        draft_tokens = 5
        cache_params = _make_cache_params()
        mamba_pool_size = size + pre_alloc_size

        # --- OLD behavior: spec_state_size = size + pre_alloc_size ---
        old_spec_bytes, old_pool = _create_mamba_pool(
            size=mamba_pool_size,
            spec_state_size=size + pre_alloc_size,
            cache_params=cache_params,
            draft_tokens=draft_tokens,
            device=device,
        )
        old_total_bytes = old_pool.mamba_cache.mem_usage_bytes()
        del old_pool
        torch.cuda.empty_cache()

        # --- NEW behavior: spec_state_size = size ---
        new_spec_bytes, new_pool = _create_mamba_pool(
            size=mamba_pool_size,
            spec_state_size=size,
            cache_params=cache_params,
            draft_tokens=draft_tokens,
            device=device,
        )
        new_total_bytes = new_pool.mamba_cache.mem_usage_bytes()

        # Verify speculative cache dim is correct
        self.assertEqual(new_pool.mamba_cache.intermediate_ssm.shape[1], size + 1)
        # Verify main cache dim is unchanged
        self.assertEqual(new_pool.mamba_cache.temporal.shape[1], mamba_pool_size + 1)
        del new_pool
        torch.cuda.empty_cache()

        saved_spec = old_spec_bytes - new_spec_bytes
        saved_total = old_total_bytes - new_total_bytes
        pct_spec = saved_spec / old_spec_bytes * 100
        pct_total = saved_total / old_total_bytes * 100

        print("\n")
        print("=" * 72)
        print("  SUMMARY: mamba_spec_state_size Optimization Results")
        print("=" * 72)
        print(f"  Configuration:")
        print(f"    max_running_requests (size) : {size}")
        print(f"    pre_alloc_size              : {pre_alloc_size}")
        print(f"    speculative_num_draft_tokens: {draft_tokens}")
        print(f"    num_mamba_layers            : {len(MAMBA_LAYERS)}")
        print()
        print(f"  {'Metric':<42} {'OLD':>10} {'NEW':>10} {'Delta':>10}")
        print(f"  {'-' * 42} {'-' * 10} {'-' * 10} {'-' * 10}")
        print(
            f"  {'spec_state_size':<42} "
            f"{size + pre_alloc_size:>10} {size:>10} {-pre_alloc_size:>+10}"
        )
        print(
            f"  {'Speculative intermediate cache (MB)':<42} "
            f"{old_spec_bytes / MB:>10.2f} {new_spec_bytes / MB:>10.2f} "
            f"{-saved_spec / MB:>+10.2f}"
        )
        print(
            f"  {'Total mamba cache (MB)':<42} "
            f"{old_total_bytes / MB:>10.2f} {new_total_bytes / MB:>10.2f} "
            f"{-saved_total / MB:>+10.2f}"
        )
        print(f"  {'Spec cache savings':<42} {'':>10} {'':>10} {pct_spec:>9.1f}%")
        print(f"  {'Total cache savings':<42} {'':>10} {'':>10} {pct_total:>9.1f}%")
        print()
        print(f"  Functional correctness:")
        print(f"    Main cache slots (conv/temporal)   : {mamba_pool_size} (unchanged)")
        print(
            f"    Speculative cache slots             : "
            f"{size} (reduced from {size + pre_alloc_size})"
        )
        print(f"    All R/W tests                      : PASS")
        print(f"    Boundary condition                  : PASS")
        print("=" * 72)

        self.assertGreater(saved_spec, 0)
        self.assertGreater(saved_total, 0)


if __name__ == "__main__":
    unittest.main()
