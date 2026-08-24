"""Selective sparse attention: 656B unpack + SFA BSND for selected layers.

This module implements the attention computation for layers whose KV cache
is offloaded to Host DRAM.  The packed 656-byte FP8 record is unpacked to
BF16 and fed to ``npu_sparse_flash_attention`` in BSND layout.

See design document Section 9.4 for the full specification.
"""

from __future__ import annotations

import torch

from sglang.srt.utils.common import is_npu

if is_npu():
    import torch_npu

DSA_KV_QUANT_TILE_SIZE = 128


def selective_sparse_attention(
    *,
    q_nope: torch.Tensor,           # [T, H, 512] BF16
    q_rope: torch.Tensor,           # [T, H, 64]  BF16
    packed_staging: torch.Tensor,   # [T, K, 656] uint8
    valid_mask: torch.Tensor,       # [T, K] bool
    scale: float,
    unpack_k_nope_bf16: torch.Tensor,   # [Tcap, K, 512] BF16 (pre-allocated)
    unpack_k_rope_bf16: torch.Tensor,   # [Tcap, K, 64]  BF16 (pre-allocated)
    sparse_indices_buf: torch.Tensor,   # [Tcap, 1, 1, K] int32 (pre-allocated)
    actual_seq_lens_kv_buf: torch.Tensor,  # [Tcap] int32 (pre-allocated)
    kv_lora_rank: int = 512,
    qk_rope_head_dim: int = 64,
    actual_seq_lens_q_buf: torch.Tensor = None,
    arange_k_buf: torch.Tensor = None,
    fp8_nope_buf: torch.Tensor = None,
    scales_buf: torch.Tensor = None,
    graph_mode: bool = False,
) -> torch.Tensor:
    """Unpack 656B packed FP8 records and run SFA BSND attention.

    Args:
        q_nope: Query non-positional part, ``[T, H, kv_lora_rank]`` BF16.
        q_rope: Query RoPE part, ``[T, H, qk_rope_head_dim]`` BF16.
        packed_staging: Packed FP8 KV records from H2D, ``[T, K, 656]`` uint8.
        valid_mask: Boolean mask for valid KV entries, ``[T, K]``.
        scale: Attention scale factor (``layer.scaling``).
        unpack_k_nope_bf16: Pre-allocated BF16 workspace for unpacked k_nope.
        unpack_k_rope_bf16: Pre-allocated BF16 workspace for unpacked k_rope.
        sparse_indices_buf: Pre-allocated SFA sparse indices buffer.
        actual_seq_lens_kv_buf: Pre-allocated SFA actual seq lengths buffer.
        kv_lora_rank: Latent rank (default 512 for GLM-5.2).
        qk_rope_head_dim: RoPE head dim (default 64 for GLM-5.2).

    Returns:
        Attention output, ``[T, H * kv_lora_rank]`` BF16.
    """
    T = q_nope.shape[0]
    K = packed_staging.shape[1]
    H = q_nope.shape[1]
    device = q_nope.device

    # === Unpack 656B → BF16 (two-step: byte copy, then Cast on contiguous) ===
    staging_flat = packed_staging.reshape(T * K, -1)  # [T*K, 656]

    fp8_bytes = kv_lora_rank  # 512
    rope_bytes = qk_rope_head_dim * 2  # 128 (BF16 = 2 bytes each)
    scale_bytes = (kv_lora_rank // DSA_KV_QUANT_TILE_SIZE) * 4  # 16 (FP32)
    num_tiles = kv_lora_rank // DSA_KV_QUANT_TILE_SIZE  # 4
    scales_start = fp8_bytes + rope_bytes  # 640

    N = T * K

    # Step 1: byte-level copy from non-contiguous staging to contiguous buffers
    if fp8_nope_buf is not None:
        fp8_nope_buf.view(torch.uint8)[:N].copy_(
            staging_flat[:, :fp8_bytes]
        )
        k_nope_fp8 = fp8_nope_buf[:N]
    else:
        k_nope_fp8 = staging_flat[:, :fp8_bytes].view(
            torch.float8_e4m3fn
        ).contiguous()

    if scales_buf is not None:
        scales_buf.view(torch.uint8)[:N].copy_(
            staging_flat[:, scales_start:scales_start + scale_bytes]
        )
        scales = scales_buf[:N]
    else:
        scales = staging_flat[:, scales_start:scales_start + scale_bytes].view(
            torch.float32
        ).reshape(N, num_tiles).contiguous()

    # Step 2: Cast on contiguous memory (safe for NPU Cast kernel)
    k_nope_bf16 = k_nope_fp8.to(torch.bfloat16)  # [N, 512]

    # Dequant: each 128-dim tile multiplied by its scale
    k_nope_bf16 = k_nope_bf16.reshape(N, num_tiles, DSA_KV_QUANT_TILE_SIZE)
    k_nope_bf16 = k_nope_bf16 * scales.unsqueeze(-1)  # broadcast [N, 4, 1]
    k_nope_bf16 = k_nope_bf16.reshape(N, kv_lora_rank)

    # BF16 RoPE: byte copy to contiguous, then view
    rope_start = fp8_bytes  # 512
    rope_end = fp8_bytes + rope_bytes  # 640
    k_rope_bytes = staging_flat[:, rope_start:rope_end]  # [N, 128] stride=(656,1)
    k_rope_bf16 = k_rope_bytes.view(
        torch.bfloat16
    ).reshape(N, qk_rope_head_dim).contiguous()  # [N, 64]

    # Write into pre-allocated workspaces
    unpack_k_nope_bf16[:T].copy_(k_nope_bf16.view(T, K, kv_lora_rank))
    unpack_k_rope_bf16[:T].copy_(k_rope_bf16.view(T, K, qk_rope_head_dim))

    # === Construct SFA inputs (compacted) ===
    # Valid entries must be compacted to the front of sparse_indices because
    # npu_sparse_flash_attention only processes the first actual_seq_lengths_kv
    # entries.  Using K (out-of-range) as sentinel causes OOB reads → NaN/Inf.
    #
    # After compaction:
    #   sparse_indices[t] = [orig_pos_0, orig_pos_1, ..., orig_pos_{N-1}, -1, ...]
    #   actual_seq_lengths_kv[t] = N (count of valid entries)
    # The kernel reads k_nope_sfa[t, sparse_indices[t, p]] for p in [0, N).
    valid = valid_mask[:T]  # [T, K]
    if arange_k_buf is not None:
        arange_k = arange_k_buf
    else:
        arange_k = torch.arange(K, device=device, dtype=torch.int32)

    arange_k_exp = arange_k.unsqueeze(0).expand(T, K)  # [T, K]

    # Sort key: valid entries get 0..K-1 (ascending), invalid get K..2K-1.
    # No ties → deterministic even without stable sort.
    sort_key = torch.where(
        valid,
        arange_k_exp,
        arange_k_exp + K,
    )  # [T, K] int32

    # sorted_indices[t, p] = original K-position of the p-th compacted entry
    _, sorted_indices = torch.sort(sort_key, dim=1)

    n_valid = valid.sum(dim=1)  # [T]

    # Build compacted sparse_indices: first n_valid entries are original
    # positions (in ascending order), remaining entries are -1 sentinel.
    pos_range = arange_k_exp  # [T, K] = 0, 1, ..., K-1
    sparse_indices_compact = torch.where(
        pos_range < n_valid.unsqueeze(1).to(torch.int32),
        sorted_indices.to(torch.int32),
        torch.full((T, K), -1, device=device, dtype=torch.int32),
    )

    # Empty-row guard: rows with zero valid entries would have -1 at position 0
    # but actual_seq_lengths_kv is clamped to min=1, causing the kernel to read
    # sparse_indices[t, 0] = -1 (OOB).  Force to 0 so it reads the zero-init
    # sentinel record instead.
    empty_rows = (n_valid == 0)  # [T] bool
    sparse_indices_compact[:, 0] = torch.where(
        empty_rows,
        torch.zeros(T, dtype=torch.int32, device=device),
        sparse_indices_compact[:, 0],
    )

    sparse_indices_buf[:T, 0, 0, :].copy_(sparse_indices_compact)

    # actual_seq_lengths_kv: count of valid entries per query
    actual_seq_lens_kv_buf[:T].copy_(
        n_valid.clamp(min=1, max=K).to(torch.int32)
    )

    # Query reshape: [T, H, D] → [T, 1, H, D] (BSND)
    q_nope_sfa = q_nope[:T].unsqueeze(1).contiguous()  # [T, 1, H, 512]
    q_rope_sfa = q_rope[:T].unsqueeze(1).contiguous()  # [T, 1, H, 64]

    # KV reshape: [T, K, D] → [T, K, 1, D] (BSND, head=1 for MLA)
    k_nope_sfa = (
        unpack_k_nope_bf16[:T].unsqueeze(2).contiguous()
    )  # [T, K, 1, 512]
    k_rope_sfa = (
        unpack_k_rope_bf16[:T].unsqueeze(2).contiguous()
    )  # [T, K, 1, 64]

    if actual_seq_lens_q_buf is not None:
        actual_seq_lens_q_buf.fill_(1)
        actual_seq_lens_q = actual_seq_lens_q_buf[:T]
    else:
        actual_seq_lens_q = torch.ones(
            T, dtype=torch.int32, device=device
        )

    # === SFA BSND ===
    ret = torch_npu.npu_sparse_flash_attention(
        q_nope_sfa,                  # [T, 1, H, 512]
        k_nope_sfa,                  # [T, K, 1, 512]
        k_nope_sfa,                  # value = key (MLA)
        sparse_indices_buf[:T],      # [T, 1, 1, K]
        scale,
        actual_seq_lengths_query=actual_seq_lens_q,
        actual_seq_lengths_kv=actual_seq_lens_kv_buf[:T],
        query_rope=q_rope_sfa,       # [T, 1, H, 64]
        key_rope=k_rope_sfa,         # [T, K, 1, 64]
        sparse_block_size=1,
        layout_query="BSND",
        layout_kv="BSND",
        sparse_mode=0,
        attention_mode=2,
        return_softmax_lse=False,
    )

    attn_out = ret[0] if isinstance(ret, tuple) else ret

    # Graph bucket padding produces rows with no valid KV.  The SFA API cannot
    # express a zero KV length, so those rows are submitted with a sentinel
    # length of one.  Explicitly zero their outputs: otherwise they can read a
    # stale staging record from the previous replay and affect downstream MoE
    # routing/reductions even though the runner later slices padded logits.
    attn_out = torch.where(
        empty_rows.view(T, 1, 1, 1),
        torch.zeros_like(attn_out),
        attn_out,
    )

    # [T, 1, H, 512] → [T, H * 512]
    return attn_out[:, 0, :, :].reshape(T, H * kv_lora_rank)
