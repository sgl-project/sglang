#!/usr/bin/env python3
"""Standalone verification for DFlash + Hybrid SWA cell_size fix.

Replicates the exact computation from pool_configurator.py without
importing any sglang/torch modules, so it runs on any Python 3.10+.

Verifies:
  1. HybridSWAPoolConfigurator scales cell_size for DFlash draft KV
  2. SWAChunkCapPoolConfigurator adds draft KV to full_cell_size
  3. Total memory (target + draft) <= available in all cases
  4. DFlash produces fewer tokens than no-DFlash (scaling works)
"""

import math
from dataclasses import dataclass


# ── Replicated constants ──────────────────────────────────────────────
KV_SIZE = 2  # bf16


def scale_kv_cell_size_per_token_for_dflash(
    *, target_cell_size_per_token, target_num_layers, draft_num_layers
):
    """Exact replica of dflash_utils.scale_kv_cell_size_per_token_for_dflash."""
    total_layers = int(target_num_layers) + int(draft_num_layers)
    return (
        int(target_cell_size_per_token) * int(total_layers)
        + int(target_num_layers)
        - 1
    ) // int(target_num_layers)


@dataclass
class PoolConfig:
    max_total_num_tokens: int
    full_max_total_num_tokens: int = 0
    swa_max_total_num_tokens: int = 0


# ── Replicated HybridSWAPoolConfigurator ──────────────────────────────

class HybridSWAPoolConfigurator:
    def __init__(
        self,
        *,
        full_layers_num,
        swa_layers_num,
        full_per_token,
        swa_per_token,
        swa_full_tokens_ratio,
        is_dflash=False,
        dflash_draft_num_layers=0,
    ):
        self._full_layers_num = full_layers_num
        self._swa_layers_num = swa_layers_num
        self._full_per_token = full_per_token
        self._swa_per_token = swa_per_token
        self._swa_full_tokens_ratio = swa_full_tokens_ratio
        self._draft_full_layers_num = 0  # EAGLE only
        self._draft_swa_full_layers_num = 0

        # cell_size computation (same as pool_configurator.py)
        if self._full_layers_num == 0:
            self._cell_size = (
                self._swa_per_token * self._swa_layers_num
                + self._full_per_token * self._draft_full_layers_num
                + self._swa_per_token * self._draft_swa_full_layers_num
            )
        else:
            self._cell_size = (
                self._full_per_token
                * (self._full_layers_num + self._draft_full_layers_num)
                + self._swa_per_token * self._draft_swa_full_layers_num
                + self._swa_full_tokens_ratio
                * self._swa_per_token
                * self._swa_layers_num
            )

        # ── DFlash scaling (THE FIX) ──
        self._pre_dflash_cell_size = self._cell_size
        self._dflash_draft_num_layers = 0
        if is_dflash:
            draft_num_layers = dflash_draft_num_layers
            total_layers = self._full_layers_num + self._swa_layers_num
            if (
                draft_num_layers is not None
                and int(draft_num_layers) > 0
                and int(total_layers) > 0
            ):
                self._dflash_draft_num_layers = int(draft_num_layers)
                self._cell_size = scale_kv_cell_size_per_token_for_dflash(
                    target_cell_size_per_token=self._cell_size,
                    target_num_layers=int(total_layers),
                    draft_num_layers=int(draft_num_layers),
                )

    def calculate_pool_sizes(self, available_bytes, page_size=1):
        max_total = available_bytes // self._cell_size
        return self._solve_pool_sizes(max_total, page_size)

    def _solve_pool_sizes(self, max_total, page_size):
        if self._full_layers_num == 0:
            swa_tokens = (max_total // page_size) * page_size
            return PoolConfig(
                max_total_num_tokens=swa_tokens,
                full_max_total_num_tokens=0,
                swa_max_total_num_tokens=swa_tokens,
            )
        full_tokens = (max_total // page_size) * page_size
        swa_tokens = (int(full_tokens * self._swa_full_tokens_ratio) // page_size) * page_size
        return PoolConfig(
            max_total_num_tokens=full_tokens,
            full_max_total_num_tokens=full_tokens,
            swa_max_total_num_tokens=swa_tokens,
        )


# ── Replicated SWAChunkCapPoolConfigurator ────────────────────────────

class SWAChunkCapPoolConfigurator(HybridSWAPoolConfigurator):
    def __init__(self, *, swa_cap, **kwargs):
        super().__init__(**kwargs)
        self._swa_cap = swa_cap

    def calculate_pool_sizes(self, available_bytes, page_size=1):
        swa_tokens = math.ceil(self._swa_cap / page_size) * page_size
        fixed_swa_bytes = swa_tokens * self._swa_per_token * self._swa_layers_num

        full_cell_size = (
            self._full_per_token
            * (self._full_layers_num + self._draft_full_layers_num)
            + self._swa_per_token * self._draft_swa_full_layers_num
        )
        # ── DFlash draft KV (THE FIX) ──
        # Use exact per-token delta from scale_kv_cell_size_per_token_for_dflash
        if self._dflash_draft_num_layers > 0:
            draft_kv_per_token = int(self._cell_size - self._pre_dflash_cell_size)
            full_cell_size += draft_kv_per_token

        full_tokens = (
            int((available_bytes - fixed_swa_bytes) // full_cell_size) // page_size
        ) * page_size
        if full_tokens <= 0:
            raise RuntimeError("No room for full KV pool")
        return PoolConfig(
            max_total_num_tokens=full_tokens,
            full_max_total_num_tokens=full_tokens,
            swa_max_total_num_tokens=swa_tokens,
        )


# ── Test helpers ──────────────────────────────────────────────────────

def full_per_token(num_kv_heads=4, head_dim=64):
    return num_kv_heads * (head_dim + head_dim) * KV_SIZE

def swa_per_token(num_kv_heads=4, head_dim=64):
    return num_kv_heads * (head_dim + head_dim) * KV_SIZE


def run_tests():
    available = 10_000_000
    fpt = full_per_token()
    spt = swa_per_token()
    nf, ns = 16, 16
    ratio = 0.5
    draft_layers = 4
    total_layers = nf + ns

    all_pass = True

    # ── Test 1: DFlash + Hybrid SWA ─────────────────────────
    print("=" * 60)
    print("Test 1: DFlash + Hybrid SWA — budget check")
    print("=" * 60)

    cfg = HybridSWAPoolConfigurator(
        full_layers_num=nf, swa_layers_num=ns,
        full_per_token=fpt, swa_per_token=spt,
        swa_full_tokens_ratio=ratio,
        is_dflash=True, dflash_draft_num_layers=draft_layers,
    )
    config = cfg.calculate_pool_sizes(available)

    # Compute total memory (target + draft)
    target_cell = fpt * nf + ratio * spt * ns
    draft_per_layer = target_cell // total_layers
    draft_bytes = config.full_max_total_num_tokens * draft_per_layer * draft_layers
    target_bytes = (
        config.full_max_total_num_tokens * fpt * nf
        + config.swa_max_total_num_tokens * spt * ns
    )
    used = target_bytes + draft_bytes

    print(f"  cell_size:        {cfg._cell_size}")
    print(f"  pre_dflash:       {cfg._pre_dflash_cell_size}")
    print(f"  full_tokens:      {config.full_max_total_num_tokens}")
    print(f"  swa_tokens:       {config.swa_max_total_num_tokens}")
    print(f"  target_bytes:     {target_bytes}")
    print(f"  draft_bytes:      {draft_bytes}")
    print(f"  total_used:       {used}")
    print(f"  available:        {available}")
    pass1 = used <= available
    print(f"  PASS: {pass1}")
    all_pass &= pass1

    # ── Test 2: DFlash reduces tokens vs no DFlash ──────────
    print()
    print("=" * 60)
    print("Test 2: DFlash reduces tokens vs no DFlash")
    print("=" * 60)

    cfg_plain = HybridSWAPoolConfigurator(
        full_layers_num=nf, swa_layers_num=ns,
        full_per_token=fpt, swa_per_token=spt,
        swa_full_tokens_ratio=ratio,
    )
    config_plain = cfg_plain.calculate_pool_sizes(available)

    print(f"  DFlash tokens:    {config.max_total_num_tokens}")
    print(f"  Plain tokens:     {config_plain.max_total_num_tokens}")
    pass2 = config.max_total_num_tokens < config_plain.max_total_num_tokens
    print(f"  PASS: {pass2}")
    all_pass &= pass2

    # ── Test 3: DFlash + All-SWA ────────────────────────────
    print()
    print("=" * 60)
    print("Test 3: DFlash + All-SWA — budget check")
    print("=" * 60)

    ns_all = 32
    cfg_swa = HybridSWAPoolConfigurator(
        full_layers_num=0, swa_layers_num=ns_all,
        full_per_token=fpt, swa_per_token=spt,
        swa_full_tokens_ratio=ratio,
        is_dflash=True, dflash_draft_num_layers=draft_layers,
    )
    config_swa = cfg_swa.calculate_pool_sizes(available)

    used_swa = config_swa.swa_max_total_num_tokens * spt * (ns_all + draft_layers)
    print(f"  swa_tokens:       {config_swa.swa_max_total_num_tokens}")
    print(f"  used:             {used_swa}")
    print(f"  available:        {available}")
    pass3 = used_swa <= available
    print(f"  PASS: {pass3}")
    all_pass &= pass3

    # ── Test 4: DFlash + SWAChunkCap ────────────────────────
    print()
    print("=" * 60)
    print("Test 4: DFlash + SWAChunkCap — budget check")
    print("=" * 60)

    swa_cap = 91  # from existing test
    cfg_chunk = SWAChunkCapPoolConfigurator(
        full_layers_num=nf, swa_layers_num=ns,
        full_per_token=fpt, swa_per_token=spt,
        swa_full_tokens_ratio=ratio,
        is_dflash=True, dflash_draft_num_layers=draft_layers,
        swa_cap=swa_cap,
    )
    config_chunk = cfg_chunk.calculate_pool_sizes(available)

    full_tokens_c = config_chunk.full_max_total_num_tokens
    swa_tokens_c = config_chunk.swa_max_total_num_tokens

    target_cell_c = fpt * nf + ratio * spt * ns
    draft_per_layer_c = target_cell_c // total_layers
    draft_bytes_c = full_tokens_c * draft_per_layer_c * draft_layers
    target_bytes_c = full_tokens_c * fpt * nf + swa_tokens_c * spt * ns
    used_c = target_bytes_c + draft_bytes_c

    print(f"  full_tokens:      {full_tokens_c}")
    print(f"  swa_tokens:       {swa_tokens_c}")
    print(f"  target_bytes:     {target_bytes_c}")
    print(f"  draft_bytes:      {draft_bytes_c}")
    print(f"  total_used:       {used_c}")
    print(f"  available:        {available}")
    pass4 = used_c <= available
    print(f"  PASS: {pass4}")
    all_pass &= pass4

    # ── Test 5: DFlash ChunkCap reduces tokens ──────────────
    print()
    print("=" * 60)
    print("Test 5: DFlash ChunkCap reduces tokens vs no DFlash")
    print("=" * 60)

    cfg_chunk_plain = SWAChunkCapPoolConfigurator(
        full_layers_num=nf, swa_layers_num=ns,
        full_per_token=fpt, swa_per_token=spt,
        swa_full_tokens_ratio=ratio,
        swa_cap=swa_cap,
    )
    config_chunk_plain = cfg_chunk_plain.calculate_pool_sizes(available)

    print(f"  DFlash full_tokens:  {config_chunk.full_max_total_num_tokens}")
    print(f"  Plain full_tokens:   {config_chunk_plain.full_max_total_num_tokens}")
    pass5 = config_chunk.full_max_total_num_tokens < config_chunk_plain.full_max_total_num_tokens
    print(f"  PASS: {pass5}")
    all_pass &= pass5

    # ── Summary ─────────────────────────────────────────────
    print()
    print("=" * 60)
    if all_pass:
        print("ALL TESTS PASSED ✓")
    else:
        print("SOME TESTS FAILED ✗")
    print("=" * 60)
    return all_pass


if __name__ == "__main__":
    import sys
    sys.exit(0 if run_tests() else 1)
