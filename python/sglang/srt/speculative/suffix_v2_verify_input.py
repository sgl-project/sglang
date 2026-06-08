"""Build a complete EagleVerifyInput from a SUFFIX linear-chain draft.

M2 of the SUFFIX V2 overlap work. Builds on M1's draft_builder (retrieve_*
arrays + positions) and adds:
  - custom_mask construction for the FULL_MASK layout (per-row prefix=True
    + lower-triangular K-block for linear chain)
  - All remaining EagleVerifyInput dataclass fields
    (spec_steps, topk=1, draft_token_num, capture_hidden_mode,
     seq_lens_sum, seq_lens_cpu)

Once this returns a populated EagleVerifyInput, the EAGLE V2 verify scaffold
(`prepare_for_v2_verify` → target forward(is_verify=True) → `sample`) can be
reused verbatim — no SUFFIX-specific verify path.
"""

from __future__ import annotations

from typing import Optional

import torch

from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sglang.srt.speculative.eagle_info import EagleVerifyInput
from sglang.srt.speculative.suffix_v2_draft_builder import build_suffix_v2_verify_arrays


def build_suffix_v2_custom_mask(
    bs: int,
    K: int,
    seq_lens_cpu: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Build the FULL_MASK custom_mask for a topk=1 linear chain.

    Per-req block layout (K rows × (seq_lens[i] + K) cols, row-major):
      row j: [True] * seq_lens[i] + [True] * (j+1) + [False] * (K-j-1)
      (i.e., prefix all attended, then lower-triangular within K-block)

    Concatenated across reqs in flat layout. Total size:
        sum_i K * (seq_lens[i] + K) = K * seq_lens_sum + K * K * bs

    Args:
        bs: batch size
        K: draft_token_num (chain width per req)
        seq_lens_cpu: per-req seq_lens, shape (bs,) on CPU (int)
        device: target device for the output mask

    Returns:
        custom_mask: (K * seq_lens_sum + K*K*bs,) bool tensor on `device`
    """
    assert (
        seq_lens_cpu.numel() == bs
    ), f"seq_lens_cpu has {seq_lens_cpu.numel()} entries, expected bs={bs}"

    # Build the K-block lower-triangular mask once (same for every req).
    # k_block[j, k] = True iff k <= j  (i.e., position j attends to k in chain)
    k_block = torch.tril(torch.ones((K, K), dtype=torch.bool, device=device))  # (K, K)

    # Concatenate per-req chunks. Each chunk is K rows × (seq_lens[i] + K) cols
    # flattened row-major.
    chunks = []
    seq_lens_list = seq_lens_cpu.tolist()
    for sl in seq_lens_list:
        prefix_block = torch.ones((K, sl), dtype=torch.bool, device=device)
        per_req = torch.cat((prefix_block, k_block), dim=1)  # (K, sl + K)
        chunks.append(per_req.reshape(-1))  # flatten row-major to (K * (sl + K),)
    custom_mask = (
        torch.cat(chunks) if chunks else torch.empty(0, dtype=torch.bool, device=device)
    )
    return custom_mask


def build_suffix_v2_eagle_verify_input(
    bonus_tokens: torch.Tensor,
    suffix_draft_tokens: torch.Tensor,
    seq_lens: torch.Tensor,
    seq_lens_cpu: Optional[torch.Tensor] = None,
    capture_hidden_mode: CaptureHiddenMode = CaptureHiddenMode.NULL,
) -> EagleVerifyInput:
    """Build a complete EagleVerifyInput from SUFFIX linear chain draft.

    Args:
        bonus_tokens: (bs,) int — first token per chain (from prev step's verify)
        suffix_draft_tokens: (bs, K-1) int — suffix-cache-queried spec tokens
        seq_lens: (bs,) int — per-req current seq length (on device)
        seq_lens_cpu: (bs,) int on CPU; computed from seq_lens if None
        capture_hidden_mode: NULL for standalone SUFFIX, FULL if downstream
            needs hidden states (e.g., HYBRID V2 with MTP keep-up).

    Returns:
        EagleVerifyInput ready to be assigned to the batch's spec_info
        and passed through EAGLE V2's verify scaffold (`prepare_for_v2_verify`,
        target forward(is_verify=True), `sample`).
    """
    bs = bonus_tokens.numel()
    K = 1 + suffix_draft_tokens.size(1)
    device = bonus_tokens.device

    if seq_lens_cpu is None:
        seq_lens_cpu = seq_lens.cpu()

    # M1 builder: draft_token_flat + positions + retrieve_*
    (
        draft_token_flat,
        positions,
        retrieve_index,
        retrieve_next_token,
        retrieve_next_sibling,
    ) = build_suffix_v2_verify_arrays(
        bonus_tokens=bonus_tokens,
        suffix_draft_tokens=suffix_draft_tokens,
        seq_lens=seq_lens,
    )

    # M2: custom_mask (FULL_MASK layout, linear chain = lower-triangular K-block)
    custom_mask = build_suffix_v2_custom_mask(bs, K, seq_lens_cpu, device)

    seq_lens_sum = int(seq_lens_cpu.sum().item())

    return EagleVerifyInput(
        draft_token=draft_token_flat,
        custom_mask=custom_mask,
        positions=positions,
        retrieve_index=retrieve_index,
        retrieve_next_token=retrieve_next_token,
        retrieve_next_sibling=retrieve_next_sibling,
        retrieve_cum_len=None,
        spec_steps=K - 1,  # number of draft steps (excluding bonus)
        topk=1,  # linear chain = topk=1
        draft_token_num=K,  # K tokens per req in candidates
        capture_hidden_mode=capture_hidden_mode,
        seq_lens_sum=seq_lens_sum,
        seq_lens_cpu=seq_lens_cpu,
    )
