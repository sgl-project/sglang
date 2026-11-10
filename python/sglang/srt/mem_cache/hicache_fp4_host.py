"""NVFP4 host copy of an FP8 device KV pool for HiCache (default off).

Switch: ``SGLANG_HICACHE_FP4_HOST_KV=1`` (exactly "1", read once per process). It engages for
an unquantized FP8-E4M3 MHA device pool (the full-attention pool of a hybrid model) whose
HiCache uses ``io_backend=kernel`` and the ``page_first`` host layout, with no L3 storage.
The device pool, the attention backends and the PD transfer stay FP8; only the host (L2)
copy of the target KV is NVFP4: E2M1 values with one E4M3 scale per 16 values and a fixed
global scale of 1/4 (a power of two, so exact; it keeps block scales of small blocks out of
E4M3's subnormals, and FP8's 448 still fits). Backup quantizes the device rows, load-back
dequantizes them to FP8 (``pool_host/mha_fp8_nvfp4.py``). FP8 MTP draft layers keep their
bytes on host.

The copy kernels are persistent and capped at a CTA quota: load-back at
``SGLANG_HICACHE_LOAD_BLOCK_QUOTA`` (the KV load quota of the stock path, default 8 here),
backup at ``SGLANG_HICACHE_FP4_HOST_BACKUP_BLOCK_QUOTA`` (default 64). A load-back overlaps
the forward that consumes it, so its quota stays at 8 or below. Under ``write_back`` a backup
runs inside eviction, behind the forward fence while the scheduler waits for it, so its time is
exposed; at 64 CTAs it takes 0.96-1.13x the stock staged FP8 backup's time, and beside a
forward it slows the MoE no more than the stock one does (mb-h530-backup).

Sizing: the whole host tier stays within ``--hicache-size``. The stock proportional split
gives KV + index one share; the KV host tokens are sized with the NVFP4 host bytes per token
plus the index's. ``SGLANG_HICACHE_FP4_HOST_MAMBA_SHARE`` (0 to 1, default 0.36) moves that
fraction of the bytes FP4 saves, against an FP8 host copy of the same share, to the Mamba
host state pool: 0 spends them all on more KV tokens, 1 keeps the FP8 token count and gives
every saved byte to Mamba states; 0.36 grows host KV tokens and Mamba states by about the same
factor (x1.39 at the c120 split), as both host pools bind at per-path 1. The split is taken
over ``--hicache-size`` less ``SGLANG_HICACHE_FP4_HOST_MARGIN_GB`` (default 0.5 GB), since every
host pool rounds its size up by a padding page or slot; once all host pools exist,
``check_host_tier`` logs their physical bytes, each pool once, and refuses a tier over
``--hicache-size``.

``SGLANG_HICACHE_FP4_HOST_TMA_LOAD=1`` (default off) moves the load-back to one fused TMA kernel
per layer: host pages come in through a shared-memory ring and are dequantized there by table
lookup, giving the same FP8 bytes as the Triton load at the same CTA quota.
``SGLANG_HICACHE_FP4_HOST_FUSED_BACKUP=1`` (default off) moves the backup to one fused kernel per op
at the backup quota: one warp quantizes a token's rows of every layer and writes its host span, packed
bytes and scales, as whole lines, giving the same NVFP4 bytes as the Triton backup.

Each engaged path logs one ``HICACHE_FP4_HOST <key>: ...`` line per process.
"""

import logging
import os
from typing import Optional, Sequence

logger = logging.getLogger(__name__)

ENV = "SGLANG_HICACHE_FP4_HOST_KV"
ENABLED = os.environ.get(ENV, "0") == "1"

MAMBA_SHARE_ENV = "SGLANG_HICACHE_FP4_HOST_MAMBA_SHARE"
MAMBA_SHARE = float(os.environ.get(MAMBA_SHARE_ENV, "0.36") or 0.36)
if not 0.0 <= MAMBA_SHARE <= 1.0:
    raise ValueError(f"{MAMBA_SHARE_ENV}={MAMBA_SHARE}: expected a fraction in [0, 1]")

# The padding page or slot each host pool adds is at most one Mamba state (59 MB at c144);
# 0.5 GB covers it with room, so the physical tier lands near hicache-size - 0.5 GB.
MARGIN_ENV = "SGLANG_HICACHE_FP4_HOST_MARGIN_GB"
MARGIN_GB = float(os.environ.get(MARGIN_ENV, "0.5") or 0.5)
if MARGIN_GB < 0.0:
    raise ValueError(f"{MARGIN_ENV}={MARGIN_GB}: expected a non-negative GB count")


def _quota(env: str, default: int) -> int:
    quota = int(os.environ.get(env, "") or default)
    if quota <= 0:
        raise ValueError(f"{env}={quota}: expected a positive CTA count")
    return quota


# Load 8: the image's TRT-LLM MoE reserves 8 SMs for overlapping kernels; 16 load CTAs slow a
# 16K-token MoE by 38% (mb-h530-load). Backup 64: 0.96-1.13x the stock staged backup's time at
# 4.7K-123K tokens on VR200, against 2.1-2.5x at 8 (mb-h530-backup).
LOAD_BLOCK_QUOTA_ENV = "SGLANG_HICACHE_LOAD_BLOCK_QUOTA"
BACKUP_BLOCK_QUOTA_ENV = "SGLANG_HICACHE_FP4_HOST_BACKUP_BLOCK_QUOTA"
LOAD_BLOCK_QUOTA = _quota(LOAD_BLOCK_QUOTA_ENV, 8)
BACKUP_BLOCK_QUOTA = _quota(BACKUP_BLOCK_QUOTA_ENV, 64)
if ENABLED and LOAD_BLOCK_QUOTA > 8:
    # The load-back runs beside the forward that consumes it. Its CTAs are persistent: above the
    # MoE's 8 reserved SMs they stall its cooperative launch. Backups run between forwards.
    logger.warning(
        "HICACHE_FP4_HOST: load quota %d exceeds 8 CTAs; the forward's MoE slows",
        LOAD_BLOCK_QUOTA,
    )

# Load-back through the fused TMA kernel (kvcacheio/hicache_nvfp4_load.cuh), default off. Its
# dequant table is built by the Triton load and checked on every (scale, byte) pair at init, and
# a startup self-test against the Triton load on the pool's own buffers must pass, else Triton stays.
TMA_LOAD_ENV = "SGLANG_HICACHE_FP4_HOST_TMA_LOAD"
TMA_LOAD = os.environ.get(TMA_LOAD_ENV, "0") == "1"

# Backup through the fused kernel (kvcacheio/hicache_nvfp4_backup.cuh), default off: the Triton
# backup's arithmetic, one warp per token writing its whole host span. A startup self-test against the
# Triton backup on every (block amax, value) pair and on random rows must pass, else Triton stays.
FUSED_BACKUP_ENV = "SGLANG_HICACHE_FP4_HOST_FUSED_BACKUP"
FUSED_BACKUP = os.environ.get(FUSED_BACKUP_ENV, "0") == "1"

# NVFP4: one byte per two values, one E4M3 scale per 16 values.
SCALE_BLOCK = 16

_logged: set = set()


def log_once(key: str, fmt: str, *args) -> None:
    if key in _logged:
        return
    _logged.add(key)
    logger.info("HICACHE_FP4_HOST %s: " + fmt, key, *args)


def fp8_device_kv_info(pool):
    """The MHA buffer info of ``pool`` when the switch is on and it is an unquantized
    FP8-E4M3 MHA pool with contiguous per-layer rows; else None."""
    if not ENABLED or pool is None:
        return None
    import torch

    from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool, MHATokenToKVPool

    full = pool.full_kv_pool if isinstance(pool, HybridLinearKVPool) else pool
    if not isinstance(full, MHATokenToKVPool) or full.is_quantized_kv_cache:
        return None
    if full.dtype != torch.float8_e4m3fn:
        return None
    info = pool.get_device_pool_infos()[0].buffer_info
    if info.scales is not None or info.transfer_buffers is not None:
        return None
    if info.head_dim != info.v_head_dim or info.head_dim % (2 * SCALE_BLOCK):
        return None
    row = info.head_num * info.head_dim
    if row & (row - 1):
        return None  # the copy kernels tile a token's row with power-of-two ranges
    for tensor in (*info.buffers.k, *info.buffers.v):
        if tensor.element_size() != 1 or tensor.stride(0) != row or not tensor.is_contiguous():
            return None
    return info


def host_bytes_per_token(info, draft_infos: Sequence = (), *, fp4: bool) -> int:
    """Host bytes per token of the target KV (FP8 or NVFP4 with scales) plus packed drafts."""
    layers = len(info.buffers.k)
    values = (info.head_dim + info.v_head_dim) * info.head_num * layers
    target = values // 2 + values // SCALE_BLOCK if fp4 else values * info.dtype.itemsize
    drafts = sum(
        (d.head_dim + d.v_head_dim) * d.head_num * len(d.buffers.k) * d.dtype.itemsize
        for d in draft_infos
    )
    return target + drafts


def size_host_tier(
    *,
    kv_pool,
    kv_share_gb: float,
    mamba_share_gb: float,
    index_bytes_per_token: float,
    draft_pools: Sequence = (),
    mamba_pinned: bool = False,
) -> Optional[tuple]:
    """(KV payload GB, Mamba GB) for an NVFP4 host copy, or None when the switch does not
    engage. ``kv_share_gb`` is the stock split's KV + index share and ``mamba_share_gb`` the
    Mamba share; the index host pool takes ``index_bytes_per_token`` per KV host token. Both
    shares shrink in proportion so they sum to hicache-size less MARGIN_GB. A pinned Mamba
    share (SGLANG_HICACHE_MAMBA_SIZE_GB) stays as it is: the margin and every byte FP4
    saves go to KV."""
    info = fp8_device_kv_info(kv_pool)
    if info is None:
        return None
    tier_gb = kv_share_gb + mamba_share_gb
    budget_gb = tier_gb - MARGIN_GB
    if budget_gb <= 0:
        raise ValueError(f"{MARGIN_ENV}={MARGIN_GB} leaves no host tier of {tier_gb:.2f} GB")
    if mamba_pinned:
        kv_share_gb = budget_gb - mamba_share_gb
        if kv_share_gb <= 0:
            raise ValueError(
                f"SGLANG_HICACHE_MAMBA_SIZE_GB={mamba_share_gb} leaves no KV host share "
                f"in {budget_gb:.2f} GB"
            )
    else:
        kv_share_gb *= budget_gb / tier_gb
        mamba_share_gb *= budget_gb / tier_gb
    draft_infos = [d.get_device_pool_infos()[0].buffer_info for d in draft_pools]
    fp8_bpt = host_bytes_per_token(info, draft_infos, fp4=False)
    fp4_bpt = host_bytes_per_token(info, draft_infos, fp4=True)
    fp8_tokens = kv_share_gb * 1e9 / (fp8_bpt + index_bytes_per_token)
    share = 0.0 if mamba_pinned else MAMBA_SHARE
    to_mamba_gb = share * fp8_tokens * (fp8_bpt - fp4_bpt) / 1e9
    kv_gb = kv_share_gb - to_mamba_gb
    payload_gb = kv_gb * fp4_bpt / (fp4_bpt + index_bytes_per_token)
    log_once(
        "sizing",
        "budget %.2f GB (hicache-size %.2f - %s %.2f); KV + index share %.2f GB -> NVFP4 KV "
        "%.2f GB (%d B/token, %.2fM tokens; FP8 would hold %.2fM at %d B/token) + index %.2f "
        "GB; Mamba %.2f -> %.2f GB (%s=%g%s)",
        budget_gb,
        tier_gb,
        MARGIN_ENV,
        MARGIN_GB,
        kv_share_gb,
        payload_gb,
        fp4_bpt,
        payload_gb * 1e9 / fp4_bpt / 1e6,
        fp8_tokens / 1e6,
        fp8_bpt,
        kv_gb - payload_gb,
        mamba_share_gb,
        mamba_share_gb + to_mamba_gb,
        MAMBA_SHARE_ENV,
        share,
        "; pinned by SGLANG_HICACHE_MAMBA_SIZE_GB" if mamba_pinned else "",
    )
    return payload_gb, mamba_share_gb + to_mamba_gb


def host_pool_bytes(pool) -> int:
    """Host bytes a HiCache host pool allocated: what its own 'Allocating' line reports."""
    from sglang.srt.mem_cache.memory_pool_host import DeepSeekV4PagedHostPool

    if isinstance(pool, DeepSeekV4PagedHostPool):  # the QSA indexer: per layer, per page
        return pool.layer_num * pool.num_host_pages * pool.item_bytes
    return pool.size * pool.size_per_token


def check_host_tier(*, pools: Sequence[tuple], hicache_size_gb: float) -> int:
    """Log the physical host tier once, every (name, host pool) counted once, and refuse a
    tier over hicache-size. Returns its bytes."""
    from sglang.srt.mem_cache.pool_host.mha_fp8_nvfp4 import MHATokenToKVPoolFP8NVFP4Host

    parts = []
    total = 0
    seen = set()
    for name, pool in pools:
        if id(pool) in seen:  # an entry sharing another's host pool owns no bytes of its own
            continue
        seen.add(id(pool))
        nbytes = host_pool_bytes(pool)
        total += nbytes
        detail = (
            pool.host_bytes_detail()
            if isinstance(pool, MHATokenToKVPoolFP8NVFP4Host)
            else ""
        )
        parts.append(f"{name} {nbytes / 1e9:.3f} GB{detail}")
    log_once(
        "tier",
        "physical host tier %.3f GB = %s; hicache-size %.2f GB, %s %.2f",
        total / 1e9,
        " + ".join(parts),
        hicache_size_gb,
        MARGIN_ENV,
        MARGIN_GB,
    )
    if total > hicache_size_gb * 1e9:
        raise ValueError(
            f"HICACHE_FP4_HOST: the host tier is {total / 1e9:.3f} GB, over hicache-size "
            f"{hicache_size_gb} GB; raise {MARGIN_ENV}"
        )
    return total
