"""
Triton fused kernel for P-EAGLE parallel draft input construction.

P-EAGLE: all K draft inputs built in ONE fused kernel call.
  pos=0:   h_fused[seq] + embed[last_token[seq]]   (seq-specific)
  pos>0:   h_shared     + embed[mask_token]          (shared, identical across seqs)

Optimization vs naive (torch.stack + view):
  - 3D grid (batch, K, hidden_tiles): no inner loop, high SM occupancy
  - Positions 1..K-1 all hit the same embed row → L2 cache reuse across programs
  - Single kernel launch — no inter-kernel synchronization overhead

Reference: vLLM v0.16.0 p_eagle_proposer.py
SGLang issue: github.com/sgl-project/sglang/issues/23171
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_parallel_draft_input_kernel(
    h_fused_ptr,      # [batch, hidden_dim] fp16
    embed_ptr,        # [vocab_size, hidden_dim] fp16
    last_tokens_ptr,  # [batch] int64
    h_shared_ptr,     # [hidden_dim] fp16
    mask_token_id,    # scalar constexpr
    output_ptr,       # [batch * K, hidden_dim] fp16
    hidden_dim: tl.constexpr,
    K: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """
    3D grid: (batch, K, hidden_dim // BLOCK_H)
    No inner loop — each program handles exactly BLOCK_H hidden elements.
    High SM occupancy: programs = batch × K × (hidden_dim // BLOCK_H).
    """
    seq_id = tl.program_id(0)
    pos_id = tl.program_id(1)
    h_tile = tl.program_id(2)

    h_offs = h_tile * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = h_offs < hidden_dim

    out_row = seq_id * K + pos_id

    if pos_id == 0:
        # Sequence-specific: h_fused[seq] + embed[last_token[seq]]
        token_id = tl.load(last_tokens_ptr + seq_id)
        h = tl.load(h_fused_ptr + seq_id * hidden_dim + h_offs, mask=h_mask, other=0.0)
        e = tl.load(embed_ptr   + token_id * hidden_dim + h_offs, mask=h_mask, other=0.0)
    else:
        # Shared context: h_shared + embed[mask_token]
        # All pos>0 programs access the same embed row → L2 cache hit after first
        h = tl.load(h_shared_ptr + h_offs, mask=h_mask, other=0.0)
        e = tl.load(embed_ptr + mask_token_id * hidden_dim + h_offs, mask=h_mask, other=0.0)

    tl.store(output_ptr + out_row * hidden_dim + h_offs, h + e, mask=h_mask)


def fused_parallel_draft_input(
    h_fused: torch.Tensor,          # [batch, hidden_dim]
    embed_table: torch.Tensor,      # [vocab_size, hidden_dim]
    last_tokens: torch.Tensor,      # [batch] int64
    h_shared: torch.Tensor,         # [hidden_dim]
    mask_token_id: int,
    K: int,
) -> torch.Tensor:
    """
    Construct all K parallel draft inputs.  Returns [batch * K, hidden_dim].

    Grid design (optimised):
      3D grid (batch, K, hidden_tiles) — no inner loop, single kernel launch.
      BLOCK_H = 512 → hidden_tiles = hidden_dim / 512
      Occupancy: batch × K × hidden_tiles programs (e.g. 16 × 4 = 64 for K=4, h=2048)
    """
    batch_size, hidden_dim = h_fused.shape

    output = torch.empty(batch_size * K, hidden_dim, dtype=h_fused.dtype, device=h_fused.device)

    BLOCK_H      = min(triton.next_power_of_2(hidden_dim), 512)
    hidden_tiles = triton.cdiv(hidden_dim, BLOCK_H)

    grid = (batch_size, K, hidden_tiles)
    _fused_parallel_draft_input_kernel[grid](
        h_fused, embed_table, last_tokens, h_shared,
        mask_token_id, output,
        hidden_dim, K, BLOCK_H,
    )

    return output


def fused_parallel_draft_input_torch(
    h_fused: torch.Tensor,
    embed_table: torch.Tensor,
    last_tokens: torch.Tensor,
    h_shared: torch.Tensor,
    mask_token_id: int,
    K: int,
) -> torch.Tensor:
    """PyTorch reference (correctness / fallback). Returns [batch * K, hidden_dim]."""
    batch_size = h_fused.shape[0]
    pos0       = h_fused + embed_table[last_tokens]                      # [B, H]
    pos_shared = h_shared + embed_table[mask_token_id]                   # [H]
    pos_shared = pos_shared.unsqueeze(0).expand(batch_size, -1)          # [B, H]
    all_inputs = torch.stack([pos0] + [pos_shared] * (K - 1), dim=1)    # [B, K, H]
    return all_inputs.reshape(batch_size * K, -1)
