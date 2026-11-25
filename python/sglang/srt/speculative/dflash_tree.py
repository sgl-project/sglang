"""DFlash tree verify: all-greedy steps verify an 8-node token tree instead of a chain.

Grown best-first over the selector lattice (CandidateSelector.build_lattice) by summed log-softmax:
node 0 is the bonus token, each pop adds its top 4 children. Node j verifies at prefix + depth[j]
seeing anc[j] (bit i: node i is j or an ancestor). Accept keeps the longest root path the target's
argmax agrees with (lowest node on a tie), copying its K / V rows into the committed slots rather
than rewriting req_to_token (under overlap a finished request's slot may hold a new one). Needs
selector top_k in [4, 32], block size 8, page size 1, and a target whose supports_tree_verify() is true.
"""

import logging

import torch

from sglang.kernels.ops.speculative.dflash_tree import (
    TREE_NODES,
    compile_kernels,
    dflash_tree_accept,
    dflash_tree_build,
)


class TreeBuffers:
    """Static [max_bs, TREE_NODES] tree arrays."""

    def __init__(self, max_bs: int, device):
        rows = torch.arange(TREE_NODES, device=device)
        self.tok, self.perm = (
            torch.zeros((max_bs, TREE_NODES), dtype=torch.int64, device=device)
            for _ in range(2)
        )
        self.chain = (
            (rows - 1).int(),
            rows.to(torch.uint8),
            ((2 << rows) - 1).to(torch.uint8),
        )
        self.parent, self.depth, self.anc = (c.repeat(max_bs, 1) for c in self.chain)
        self.max_bs = None

    def set_chains(self, bs: int) -> None:
        for buf, c in zip((self.parent, self.depth, self.anc), self.chain):
            buf[:bs].copy_(c)


_BUFFERS = {}


def tree_buffers(device, max_bs=None):
    """The device's TreeBuffers, created by the first call that gives ``max_bs``."""
    device = torch.device(device)
    if device.type != "cuda":  # NVRTC kernels: other devices verify chains
        return None
    index = device.index
    index = torch.cuda.current_device() if index is None else index
    if index not in _BUFFERS and max_bs is not None:
        _BUFFERS[index] = TreeBuffers(max_bs, torch.device("cuda", index))
        compile_kernels()
    return _BUFFERS.get(index)


def tree_max_batch(model_runner) -> int:
    """Largest verify batch (also: verify blocks per extend forward) taking a tree (0: none); needs the backends."""
    tree = tree_buffers(model_runner.device)
    if tree is None:
        return 0
    if tree.max_bs is None:
        backends = [
            b
            for b in (
                model_runner.attn_backend,
                getattr(model_runner, "decode_attn_backend", None),
            )
            if b is not None
        ]
        from sglang.srt.runtime_context import get_exec, get_parallel

        m = (
            0
            if get_parallel().attn_dp_enabled
            else min(
                tree.tok.shape[0], model_runner.model.tree_verify_max_batch(backends)
            )
        )
        tree.max_bs = max(
            (b for b in get_exec().graph.cuda_graph_config.decode.bs if b <= m),
            default=0,
        )  # pads to a tree graph
        logging.getLogger(__name__).info(
            "DFLASH tree verify: %d-node trees up to batch %d", TREE_NODES, tree.max_bs
        )
    return tree.max_bs


def build_tree(*, candidate_ids, scores, anchor, out: TreeBuffers) -> None:
    """Fill ``out`` rows [:B] from the lattice; only all-greedy steps read them (other steps rewrite the chains)."""
    dflash_tree_build(
        candidate_ids=candidate_ids,
        scores=scores,
        anchor=anchor,
        tok=out.tok,
        parent=out.parent,
        depth=out.depth,
        anc=out.anc,
    )


def accept_tree(
    *, tree: TreeBuffers, target_predict, verify_cache_loc, kv_move: "TreeKVMove"
):
    """Greedy accept over ``tree`` rows [:B] and the accepted paths' K / V copy, one launch; ``tree.perm``: committed node order as hidden-row ids."""
    return dflash_tree_accept(
        tok=tree.tok,
        parent=tree.parent,
        target_predict=target_predict,
        verify_cache_loc=verify_cache_loc,
        swa_map=kv_move.swa_map,
        k_ptrs=kv_move.k_ptrs,
        v_ptrs=kv_move.v_ptrs,
        is_swa=kv_move.is_swa,
        row_bytes=kv_move.row_bytes,
        perm=tree.perm,
    )


class TreeKVMove:
    """Per-layer K / V buffer pointers of a (hybrid) KV pool."""

    def __init__(self, kv_pool, device):
        hybrid = hasattr(kv_pool, "swa_kv_pool")
        pools = (
            [(kv_pool.full_kv_pool, 0), (kv_pool.swa_kv_pool, 1)]
            if hybrid
            else [(kv_pool, 0)]
        )
        ptrs, row_bytes = [], set()
        for pool, swa in pools:
            for k, v in zip(pool.k_buffer, pool.v_buffer):
                assert k.is_contiguous() and v.is_contiguous()
                row_bytes |= {
                    k[0].numel() * k.element_size(),
                    v[0].numel() * v.element_size(),
                }
                ptrs.append((k.data_ptr(), v.data_ptr(), swa))
        (self.row_bytes,) = row_bytes
        assert self.row_bytes % 16 == 0 and self.row_bytes <= 512
        self.k_ptrs, self.v_ptrs, self.is_swa = (
            torch.tensor(c, device=device) for c in zip(*ptrs)
        )
        self.is_swa = self.is_swa.to(torch.uint8)
        self.swa_map = kv_pool.full_to_swa_index_mapping if hybrid else None
        assert self.swa_map is None or self.swa_map.dtype == torch.int64
