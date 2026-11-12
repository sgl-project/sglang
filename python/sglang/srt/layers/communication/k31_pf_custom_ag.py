# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
# K3.1 prefill: the prefill forward's NCCL all-gathers on the direct
# multicast all-gather. Default off (SGLANG_K31_PF_CUSTOM_AG unset or 0 = NCCL, unchanged).
"""Route the prefill forward's NCCL all-gathers onto ``sp_collective.all_gather_direct``, the multicast-store
all-gather that the prefill SP tail already runs at every layer, through ONE persistent symmetric byte buffer:

* the DSA indexer top-k ``all_gatherv`` over attn-TP: rows are cost-balanced, so per-rank row counts differ.
  Each rank pads its block to max(sizes) rows, the gather moves world x max rows, and the real rows are copied
  into the caller's ``out`` in rank order. Skipped (NCCL) when padding would move more than PAD_MAX x the
  real rows (cold first chunks, where the split is most uneven);
* the DCP prefix gathers (MLA KV rows and index-K rows, ``dcp.comm._all_gather_dcp_kv_cache``): equal blocks
  already, so the gathered rows come back as a view of the buffer and the caller copies them out.

Both are pure byte copies, so the output is bitwise identical to NCCL's for every input. The buffer is
overwritten by the next call: a caller consumes (copies out) the view before the next gather. Stream order
gives that on one rank; across ranks the kernel's entry barrier does (no rank writes into a peer's buffer
before every rank has entered the kernel, i.e. finished its earlier stream work), the same argument as the
served SP-tail buffer. All decisions below depend only on inputs that are identical on every rank of the
group (sizes, byte counts, env), so all ranks take the same path. The plane setup checks that across the TP
group once (``rank_agreement``): a rank without a plane sends every rank to NCCL, and a configuration that
differs between ranks raises.

Knobs: SGLANG_K31_PF_CUSTOM_AG=1 enables; SGLANG_K31_PF_CUSTOM_AG_MAX_MB (default 768) caps the gathered
bytes per call (= the buffer size per GPU; larger gathers use NCCL); SGLANG_K31_PF_CUSTOM_AG_PAD_MAX
(default 1.5); SGLANG_K31_PF_CUSTOM_AG_BLOCKS (default 32 CTAs, the served SP-tail AG count).
"""

from __future__ import annotations

import logging
from typing import Callable, Optional, Sequence

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

_BUF_NAME = "k31_pf_custom_ag_flat"
_VEC = 16  # the kernel moves 16-byte vectors: every rank's block must be a multiple

_ENABLED: Optional[bool] = None
_PLANES: dict = {}
_SEEN: set = set()


def enabled() -> bool:
    global _ENABLED
    if _ENABLED is None:
        _ENABLED = envs.SGLANG_K31_PF_CUSTOM_AG.get()
    return _ENABLED


def max_bytes() -> int:
    return int(envs.SGLANG_K31_PF_CUSTOM_AG_MAX_MB.get() * 1024 * 1024)


def pad_max() -> float:
    return envs.SGLANG_K31_PF_CUSTOM_AG_PAD_MAX.get()


# ---------------------------------------------------------------------------------------------
# Core (no sglang state: the caller passes the buffer, its multicast map and the launch).
# ---------------------------------------------------------------------------------------------


def plan_rows(sizes: Sequence[int], pad_limit: float) -> Optional[int]:
    """Rows per rank block of a padded gather of per-rank row counts ``sizes``, or None (keep NCCL) when
    the padded volume len(sizes) * max(sizes) exceeds ``pad_limit`` x sum(sizes)."""
    if len(sizes) == 0 or any(int(s) < 0 for s in sizes):
        return None
    m, tot = max(int(s) for s in sizes), sum(int(s) for s in sizes)
    if m <= 0 or tot <= 0 or len(sizes) * m > pad_limit * tot:
        return None
    return m


def block_ok(nbytes_local: int, world: int, cap: int) -> bool:
    return nbytes_local > 0 and nbytes_local % _VEC == 0 and world * nbytes_local <= cap


def row_bytes(x: torch.Tensor) -> int:
    n = 1
    for d in x.shape[1:]:
        n *= int(d)
    return n * x.element_size()


def gather_into(
    world: int,
    x: torch.Tensor,
    region_u8: torch.Tensor,
    mc_of: Callable,
    launch: Callable,
) -> torch.Tensor:
    """[n, ...] -> [world * n, ...] rank-major, same dtype, as a view of the 1-D uint8 ``region_u8``."""
    xb = x.contiguous().view(torch.uint8).reshape(-1)
    out = region_u8[: world * xb.numel()]
    launch(world, xb.view(torch.bfloat16), out.view(torch.bfloat16), mc_of(out))
    return out.view(x.dtype).view((world * x.shape[0],) + tuple(x.shape[1:]))


def pad_local(x: torch.Tensor, m: int) -> torch.Tensor:
    if x.shape[0] == m:
        return x
    xp = x.new_empty((m,) + tuple(x.shape[1:]))
    xp[: x.shape[0]].copy_(x)
    return xp


def compact(
    g: torch.Tensor, m: int, sizes: Sequence[int], out: torch.Tensor
) -> torch.Tensor:
    """Rank r's first sizes[r] rows of the padded gather ``g`` ([world * m, ...]) into ``out``, rank order."""
    off = 0
    for r, s in enumerate(sizes):
        s = int(s)
        if s:
            out[off : off + s].copy_(g[r * m : r * m + s])
        off += s
    return out


def all_gatherv_rows_core(
    world, rank, x, sizes, out, pad_limit, cap, region_of, mc_of, launch
) -> bool:
    """``out[:sum(sizes)]`` = rank-ordered concatenation of every rank's first sizes[r] rows. ``x`` holds this
    rank's sizes[rank] rows (or already max(sizes) rows, the first sizes[rank] real). False = not done (NCCL)."""
    m = plan_rows(sizes, pad_limit)
    if m is None:
        return False
    if x.shape[0] not in (int(sizes[rank]), m):
        raise ValueError(
            f"all_gatherv: local rows {x.shape[0]} vs sizes[{rank}]={sizes[rank]} / padded {m}"
        )
    nb = m * row_bytes(x)
    if not block_ok(nb, world, cap):
        return False
    region = region_of(world * nb)
    if region is None:
        return False
    g = gather_into(world, pad_local(x, m), region, mc_of, launch)
    compact(g, m, sizes, out)
    return True


def all_gather_rows_core(
    world, x, cap, region_of, mc_of, launch
) -> Optional[torch.Tensor]:
    """Equal-block gather [n, ...] -> [world * n, ...] as a view of the buffer (consume before the next
    gather), or None (NCCL)."""
    nb = x.shape[0] * row_bytes(x)
    if not block_ok(nb, world, cap):
        return None
    region = region_of(world * nb)
    if region is None:
        return None
    return gather_into(world, x, region, mc_of, launch)


# ---------------------------------------------------------------------------------------------
# Served glue: the TP group's multicast plane (CustomAllReduceV2 registered by the prefill SP tail).
# ---------------------------------------------------------------------------------------------


def env_config() -> tuple:
    """This rank's custom all-gather knobs: (BLOCKS, MAX_MB, PAD_MAX)."""
    return (
        int(envs.SGLANG_K31_PF_CUSTOM_AG_BLOCKS.get()),
        float(envs.SGLANG_K31_PF_CUSTOM_AG_MAX_MB.get()),
        float(pad_max()),
    )


def rank_agreement(entries: Sequence) -> bool:
    """Rank-uniform decision from every TP rank's (env_config, plane_config) entry, plane_config None when that
    rank has no plane. True = custom path on every rank, False = NCCL on every rank (some rank has no plane).
    Raises when the knobs or the plane configurations differ between ranks: the kernel pairs CTA windows and
    buffer regions by index, so ranks with different CTA counts, caps or row widths would hang or copy rows
    that a peer has not stored yet."""
    envs_seen = {tuple(e[0]) for e in entries}
    if len(envs_seen) > 1:
        raise RuntimeError(
            "K31 PF CUSTOM AG: SGLANG_K31_PF_CUSTOM_AG_{BLOCKS,MAX_MB,PAD_MAX} differ between TP ranks: "
            f"{[tuple(e[0]) for e in entries]}"
        )
    planes = [e[1] for e in entries]
    if any(p is None for p in planes):
        return False
    planes_seen = {tuple(p) for p in planes}
    if len(planes_seen) > 1:
        raise RuntimeError(
            "K31 PF CUSTOM AG: plane configurations (blocks, cap bytes, row width) differ between TP ranks: "
            f"{[tuple(p) for p in planes]}"
        )
    return True


def _plane(group):
    """(world, rank, cap, region_of, mc_of, launch) for ``group`` when it is the TP group (same ranks, same
    order) and the prefill SP tail's multicast plane exists; else None."""
    key = None if group is None else tuple(getattr(group, "ranks", ()) or ())
    if key in _PLANES:
        return _PLANES[key]
    plane = None
    plane_config = None
    tp = None
    is_tp = False
    try:
        from sglang.kernels.ops.communication import sp_collective
        from sglang.srt.layers.communication import k3_ar_fusion, k3_sp_collective
        from sglang.srt.runtime_context import get_parallel

        tp = get_parallel().tp_group
        is_tp = (
            group is not None
            and list(group.ranks) == list(tp.ranks)
            and group.rank_in_group == tp.rank_in_group
            and group.world_size == tp.world_size
        )
        comm = k3_sp_collective.prefill_sp_comm()
        if comm is not None and is_tp:
            blocks = envs.SGLANG_K31_PF_CUSTOM_AG_BLOCKS.get()
            try:
                bcap = int(comm.config.num_pull_blocks)
            except Exception:
                bcap = blocks
            n_blocks = max(1, min(blocks, bcap))
            tuning = sp_collective.Tuning(n_blocks, 1024)
            cap = max_bytes()
            max_rows = int(k3_ar_fusion.symm_buffer_max_rows())
            width = -(-cap // (max_rows * 2))
            width = (
                -(-width // 8) * 8
            )  # bf16 elements per buffer row, a 16-byte multiple

            def region_of(nbytes: int):
                rows = -(-nbytes // (width * 2))
                if rows <= 0 or rows > max_rows:
                    return None
                buf = k3_ar_fusion.symm_buffer(_BUF_NAME, rows, width, torch.bfloat16)
                return buf.reshape(-1).view(torch.uint8)

            def launch(w, xb, ob, mc):
                sp_collective.all_gather_direct(
                    w, xb, ob, output_mc_ptr=mc, tuning=tuning
                )

            plane = (
                tp.world_size,
                tp.rank_in_group,
                min(cap, max_rows * width * 2),
                region_of,
                k3_ar_fusion.get_mc_ptr,
                launch,
            )
            plane_config = (n_blocks, plane[2], width)
    except Exception as e:  # no plane -> NCCL, logged once
        logger.info("K31 PF CUSTOM AG: no multicast plane (%s); using NCCL", e)
        plane = None
        plane_config = None
    if is_tp:
        # Every TP rank reaches this at the same call (same enable switch, same call sites); outside the try so a
        # mismatch raises on every rank instead of leaving one rank on NCCL while its peers spin.
        if not rank_agreement(tp.all_gather_object((env_config(), plane_config))):
            if plane is not None:
                logger.info(
                    "K31 PF CUSTOM AG: a TP rank has no multicast plane; using NCCL on every rank"
                )
            plane = None
    _PLANES[key] = plane
    return plane


def _note(site: str, ok: bool, detail: str) -> None:
    k = (site, ok)
    if k not in _SEEN:
        _SEEN.add(k)
        if ok:
            logger.info("K31 PF CUSTOM AG ENGAGED site=%s %s", site, detail)
        else:
            logger.info(
                "K31 PF CUSTOM AG fallback (NCCL) site=%s first of its kind: %s",
                site,
                detail,
            )


def _capturing() -> bool:
    try:
        return torch.cuda.is_current_stream_capturing()
    except Exception:
        return False


def try_all_gatherv_rows(
    group, x: torch.Tensor, sizes: Sequence[int], out: torch.Tensor, site: str
) -> bool:
    """Knob path of ``group.all_gatherv(x, sizes=sizes, output=out)``; False = caller runs NCCL."""
    if not enabled() or _capturing():
        return False
    p = _plane(group)
    if p is None:
        _note(site, False, "no plane / group is not the TP group")
        return False
    world, rank, cap, region_of, mc_of, launch = p
    ok = all_gatherv_rows_core(
        world, rank, x, sizes, out, pad_max(), cap, region_of, mc_of, launch
    )
    _note(
        site,
        ok,
        f"world={world} rows={sum(int(s) for s in sizes)} max={max(int(s) for s in sizes)} row_bytes={row_bytes(x)}",
    )
    return ok


def try_all_gather_rows(group, x: torch.Tensor, site: str) -> Optional[torch.Tensor]:
    """Knob path of an equal-block ``all_gather_into_tensor``; returns a view of the buffer (consume before
    the next gather) or None = caller runs NCCL."""
    if not enabled() or _capturing():
        return None
    p = _plane(group)
    if p is None:
        _note(site, False, "no plane / group is not the TP group")
        return None
    world, rank, cap, region_of, mc_of, launch = p
    g = all_gather_rows_core(world, x, cap, region_of, mc_of, launch)
    _note(
        site,
        g is not None,
        f"world={world} rows={x.shape[0]} row_bytes={row_bytes(x)} bytes={world * x.shape[0] * row_bytes(x)}",
    )
    return g


def aliases_buffer(t: torch.Tensor) -> bool:
    """True if ``t`` lies in a K3 named symmetric buffer (then the caller must copy it out)."""
    try:
        from sglang.srt.layers.communication import k3_ar_fusion

        return k3_ar_fusion.find_mc_ptr(t) is not None
    except Exception:
        return False
