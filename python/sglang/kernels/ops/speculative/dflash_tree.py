"""DFlash tree verify kernels (NVRTC source dflash_tree.cuh): build an 8-node token tree per request from the
selector lattice, and the greedy accept that copies the accepted paths' K / V rows."""

import ctypes
import functools

import torch

from sglang.kernels.jit import muse_spark_cuda as cu

TREE_NODES = 8


@functools.cache
def compile_kernels():
    mod = cu.build_module(
        "dflash_tree.cu",
        cu.read_source("dflash_tree.cuh"),
        [
            b"--gpu-architecture=sm_%d%d" % torch.cuda.get_device_capability(),
            b"-std=c++17",
        ],
    )
    return {n: cu.function(mod, n) for n in ("dflash_tree_build", "dflash_tree_accept")}


def _launch(name, grid, block, *args):
    if grid:
        args = [
            ctypes.c_void_p(a.data_ptr() if a is not None else 0)
            if not isinstance(a, int)
            else ctypes.c_int(a)
            for a in args
        ]
        cu.launch(
            compile_kernels()[name],
            grid,
            block,
            0,
            torch.cuda.current_stream().cuda_stream,
            args,
        )


def dflash_tree_build(
    *, candidate_ids, scores, anchor, tok, parent, depth, anc
) -> None:
    """Fill rows [:B] of the [max_bs, TREE_NODES] tree arrays (``tok`` .. ``anc``) from the lattice."""
    B, S, K = candidate_ids.shape
    assert B <= tok.shape[0]
    i64 = torch.int64
    _launch(
        "dflash_tree_build",
        (B + 3) // 4,
        128,
        candidate_ids.to(i64).contiguous(),
        scores.float().contiguous(),
        anchor.to(i64).contiguous(),
        B,
        S,
        K,
        tok,
        parent,
        depth,
        anc,
    )


def dflash_tree_accept(
    *,
    tok,
    parent,
    target_predict,
    verify_cache_loc,
    swa_map,
    k_ptrs,
    v_ptrs,
    is_swa,
    row_bytes,
    perm,
):
    """Greedy accept over tree rows [:B] and the accepted paths' K / V copy (per layer ``k_ptrs`` / ``v_ptrs``),
    one launch -> (accepted tokens, commit lengths, bonus tokens); ``perm``: committed node order as hidden-row ids."""
    B, dev = verify_cache_loc.shape[0], target_predict.device
    out = torch.empty((B, TREE_NODES), dtype=torch.int64, device=dev)
    commit, bonus = (
        torch.empty((B,), dtype=torch.int32, device=dev),
        torch.empty((B,), dtype=torch.int64, device=dev),
    )
    _launch(
        "dflash_tree_accept",
        (B, k_ptrs.numel()),
        128,
        tok,
        parent,
        target_predict.to(torch.int64).contiguous(),
        verify_cache_loc.to(torch.int64).contiguous(),
        swa_map,
        k_ptrs,
        v_ptrs,
        is_swa,
        row_bytes,
        out,
        commit,
        bonus,
        perm,
    )
    return out, commit, bonus
