"""SUFFIX V2 linear-chain draft → EagleVerifyInput converter.

Phase 2.x design: SUFFIX V2 doesn't write its own verify path. Instead it
expresses its linear chain as a topk=1 EAGLE tree and reuses EAGLE V2's
`verify_tree_greedy` kernel + `prepare_for_v2_verify` / `sample` scaffold.

Layout for `K = num_verify_tokens` per req, `bs` total reqs:
  flat draft array       shape (bs*K,)         [bonus_0, sfx_0_1, sfx_0_2, ..., sfx_0_(K-1),
                                                bonus_1, sfx_1_1, ...]
  positions              shape (bs*K,)         positions[i*K+j] = seq_lens[i] + j
  retrieve_index         shape (bs, K) int64   retrieve_index[i, j] = i*K + j
  retrieve_next_token    shape (bs, K) int64   = i*K + j + 1 if j<K-1 else -1
  retrieve_next_sibling  shape (bs, K) int64   all -1 (no siblings in topk=1)

EAGLE's `verify_tree_greedy` kernel walks these arrays without caring whether
the tree is degenerate (linear) or branching. As long as next_sibling is -1
everywhere and next_token forms a single chain, the kernel produces the same
"accept the longest matching prefix" semantics that we'd get from a custom
NgramVerifyInput.greedy_verify, but with no custom kernel.
"""

from __future__ import annotations

from typing import Tuple

import torch


def build_linear_retrieve_arrays(
    bs: int,
    K: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build retrieve_index / retrieve_next_token / retrieve_next_sibling for
    a topk=1 linear chain of K tokens per req.

    Same shapes/dtypes as EAGLE's `build_tree_kernel_efficient` output (the
    int64 / shape=(bs, K) format `verify_tree_greedy` expects).
    """
    # retrieve_index[i, j] = i*K + j → GLOBAL flat index into bs*K candidates
    # array. The kernel uses this to look up the actual draft token value
    # given a (req, local_node) pair.
    retrieve_index = torch.arange(bs * K, dtype=torch.int64, device=device).view(bs, K)

    # retrieve_next_token[i, j] = LOCAL index (per-req, range [0, K-1] or -1)
    # of node j's first child. For linear chain: child of j is j+1, leaf is K-1.
    # NOT a flat global index — kernel walks within per-req tree, then resolves
    # to candidates[retrieve_index[i, that_local]].
    local_arange = torch.arange(K, dtype=torch.int64, device=device).view(1, K)
    next_token_local = local_arange + 1
    next_token_local = next_token_local.expand(bs, K).contiguous()
    next_token_local[:, -1] = -1
    retrieve_next_token = next_token_local

    # retrieve_next_sibling: linear chain has no siblings anywhere → all -1
    retrieve_next_sibling = torch.full((bs, K), -1, dtype=torch.int64, device=device)

    return retrieve_index, retrieve_next_token, retrieve_next_sibling


def build_linear_positions(seq_lens: torch.Tensor, K: int) -> torch.Tensor:
    """Build per-draft-token absolute positions for a linear chain.

    positions[i*K + j] = seq_lens[i] + j

    Shape: (bs*K,) int64. Matches EAGLE's `positions` tensor format consumed
    by the target attention forward.
    """
    bs = seq_lens.numel()
    device = seq_lens.device
    # offsets[j] = j  (broadcast across bs)
    offsets = torch.arange(K, dtype=torch.int64, device=device).view(1, K)
    seq_lens_i64 = seq_lens.to(torch.int64).view(bs, 1)
    positions = (seq_lens_i64 + offsets).view(bs * K)
    return positions


def build_suffix_v2_verify_arrays(
    bonus_tokens: torch.Tensor,
    suffix_draft_tokens: torch.Tensor,
    seq_lens: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """One-shot builder for the SUFFIX V2 → EagleVerifyInput tensor pieces.

    Args:
        bonus_tokens: (bs,) int32 — first token of each chain (the "+1" from
            previous verify, becomes draft position 0).
        suffix_draft_tokens: (bs, K-1) int32 — remaining K-1 spec tokens per
            req, queried from the suffix tree. Pad shorter chains with any
            value; target.argmax will reject them and accept length will cut.
        seq_lens: (bs,) int32/int64 — current sequence lengths (pre-verify).

    Returns:
        draft_token_flat:        (bs*K,) int32
        positions:               (bs*K,) int64
        retrieve_index:          (bs, K) int64
        retrieve_next_token:     (bs, K) int64
        retrieve_next_sibling:   (bs, K) int64

    Caller still needs to attach `custom_mask` and the other EagleVerifyInput
    fields (spec_steps, topk=1, draft_token_num=K, etc.) — see
    `build_suffix_v2_eagle_verify_input`.
    """
    bs = bonus_tokens.numel()
    K = 1 + suffix_draft_tokens.size(1)
    device = bonus_tokens.device
    assert (
        suffix_draft_tokens.size(0) == bs
    ), f"bs mismatch: bonus_tokens={bs}, suffix_draft_tokens={suffix_draft_tokens.shape}"
    assert (
        seq_lens.numel() == bs
    ), f"bs mismatch: seq_lens={seq_lens.numel()} vs bonus_tokens={bs}"

    # Flat draft array: [bonus_0, sfx_0_1..sfx_0_(K-1), bonus_1, sfx_1_1..., ...]
    # Match EAGLE convention: bonus prepended, then spec tokens.
    # Dtype = int64: that's what verify_tree_greedy's sgl_kernel expects
    # (matches EagleVerifyInput.create_idle_input dtype=torch.long).
    draft_token_flat = (
        torch.cat((bonus_tokens.view(bs, 1), suffix_draft_tokens), dim=1)
        .view(bs * K)
        .to(torch.int64)
    )

    positions = build_linear_positions(seq_lens, K)
    retrieve_index, retrieve_next_token, retrieve_next_sibling = (
        build_linear_retrieve_arrays(bs, K, device)
    )

    return (
        draft_token_flat,
        positions,
        retrieve_index,
        retrieve_next_token,
        retrieve_next_sibling,
    )
