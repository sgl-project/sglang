"""Fused Qwen3.5 full-attention prologue / epilogue for bs1 MTP verify (kernels in mka_fused.cu)."""

import hashlib
import threading
from pathlib import Path

import torch

_EXT = None
_LOCK = threading.Lock()


def _ext():
    global _EXT
    if _EXT is None:
        with _LOCK:
            if _EXT is None:
                from torch.utils.cpp_extension import load

                major, minor = torch.cuda.get_device_capability()
                source = Path(__file__).with_suffix(".cu")
                include = Path(__file__).resolve().parents[2] / "jit" / "include"
                hop_header = include / "sgl_kernel" / "distributed" / "q35_hop.cuh"
                # Two source trees at the same path must not share a cached build, so the name carries a digest
                # of the source and the hop header.
                digest = hashlib.sha256(
                    source.read_bytes() + hop_header.read_bytes()
                ).hexdigest()[:12]
                _EXT = load(
                    name=f"sglang_attn_fused_{digest}",
                    sources=[str(source)],
                    extra_cuda_cflags=[
                        "-O3",
                        "--std=c++17",
                        f"-gencode=arch=compute_{major}{minor}a,code=sm_{major}{minor}a",
                    ],
                    extra_include_paths=[str(include)],
                    verbose=False,
                )
    return _EXT


def ensure_built() -> None:
    """Compile (or load) the extension now, before any collective can time out on a JIT build."""
    if torch.cuda.is_available():
        _ext()


def _pack_fragments(w_nk: torch.Tensor, units: int, steps: int) -> torch.Tensor:
    # [units*256, K] e4m3 -> [unit][rank 8][warp 8][step][m16 block 2][lane 32][16 B] mma.m16n8k32 A
    # fragments; rank r covers K [r*steps*32, (r+1)*steps*32), warp w rows [32w, 32w + 32) of the unit.
    w = w_nk.contiguous().view(torch.uint8)
    assert w.shape[0] == units * 256 and w.shape[1] == 8 * steps * 32, w.shape
    v = w.view(units, 8, 2, 2, 8, 8, steps, 2, 4, 4)
    return v.permute(0, 5, 1, 6, 2, 4, 8, 7, 3, 9).contiguous().view(-1)


def _pack_rows32(w_nk: torch.Tensor) -> torch.Tensor:
    # [R, 4096] e4m3 -> [cta = R/32][warp 8][step 16][m16 block 2][lane 32][16 B]; warp w covers K
    # [512w, 512w + 512) of the CTA's 32 rows (mka_gate_kernel).
    w = w_nk.contiguous().view(torch.uint8)
    v = w.view(w.shape[0] // 32, 2, 2, 8, 8, 16, 2, 4, 4)
    return v.permute(0, 4, 5, 1, 3, 7, 6, 2, 8).contiguous().view(-1)


def split_gate(nq: int) -> bool:
    """Whether the prologue hands the gate rows to mka_gate_kernel (used from nq = 8, i.e. TP4)."""
    return nq >= 8


def pack_qkv(w_nk: torch.Tensor, nq: int) -> tuple[torch.Tensor, torch.Tensor]:
    """qkv(+gate) weight [N, 4096] e4m3 -> (prologue layout, gate-kernel layout or empty)."""
    main = _pack_fragments(w_nk, w_nk.shape[0] // 256, 16)
    if not split_gate(nq):
        return main, main.new_empty(0)
    gate_rows = (
        w_nk[: nq * 512]
        .reshape(nq, 2, 256, w_nk.shape[1])[:, 1]
        .reshape(nq * 256, w_nk.shape[1])
    )
    return main, _pack_rows32(gate_rows)


def pack_o(w_nk: torch.Tensor) -> torch.Tensor:
    """o_proj weight [4096, nq*256] e4m3 -> epilogue layout (K chunk nq*32 per cluster rank)."""
    return _pack_fragments(w_nk, 16, w_nk.shape[1] // 256)


def prologue(
    x,
    w_packed,
    sx,
    sw,
    q_norm_w,
    k_norm_w,
    cos_sin,
    positions,
    axis_map,
    cache_loc,
    k_cache,
    v_cache,
    k_scale,
    v_scale,
    q_out,
    gate_out,
    nq: int,
    eps: float,
):
    w_main, w_gate = w_packed
    _ext().prologue(
        x,
        w_main,
        sx,
        sw,
        q_norm_w,
        k_norm_w,
        cos_sin,
        positions,
        axis_map,
        cache_loc,
        k_cache,
        v_cache,
        k_scale,
        v_scale,
        q_out,
        gate_out,
        w_gate,
        nq,
        eps,
        True,
    )


def epilogue(attn, gate, w_packed, sx, sw, out, nq: int, site: list[int] | None = None):
    """site: [mc, uc, ctrl, rank, world, m_cap, timeout_ns] of the q35 hop workspace to push into."""
    _ext().epilogue(attn, gate, w_packed, sx, sw, out, nq, True, site or [])
