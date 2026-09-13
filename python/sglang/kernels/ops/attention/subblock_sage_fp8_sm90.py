# SPDX-License-Identifier: Apache-2.0
"""SM90 adapter from a SubBlock plan to SpargeAttention's native Sage kernel.

The public backend mode is ``sage_fp8`` across GPU generations. On Hopper,
the fastest available implementation is SageAttention2: it uses 64-token query
blocks and 128-token key blocks, quantizes Q/K to INT8, and quantizes V plus
softmax probabilities to E4M3 online. BF16 model activations and weights remain
unchanged.
"""

from __future__ import annotations

import functools

import torch
import triton
import triton.language as tl

SAGE_FP8_SM90_QUERY_BLOCK_SIZE = 64
SAGE_FP8_SM90_KEY_BLOCK_SIZE = 128

_INSTALL_HELP = (
    "Install SpargeAttention with "
    "`pip install git+https://github.com/thu-ml/SpargeAttn.git "
    "--no-build-isolation` to use SubBlock compute_mode='sage_fp8' on SM90."
)


@functools.lru_cache(maxsize=1)
def _load_sparge_attention_sm90_ops():
    """Load every private SpargeAttention symbol used by the SM90 adapter."""
    try:
        import spas_sage_attn._fused as fused
        import spas_sage_attn._qattn as qattn
        from spas_sage_attn.utils import block_map_lut_triton, get_vanilla_qk_quant

        transpose_pad_permute_cuda = fused.transpose_pad_permute_cuda
        scale_fuse_quant_cuda = fused.scale_fuse_quant_cuda
        kernel = (
            qattn.qk_int8_sv_f8_accum_f32_block_sparse_attn_inst_buf_fuse_v_scale_sm90
        )
    except (ImportError, OSError, AttributeError) as exc:
        raise ImportError(_INSTALL_HELP) from exc
    return (
        get_vanilla_qk_quant,
        block_map_lut_triton,
        transpose_pad_permute_cuda,
        scale_fuse_quant_cuda,
        kernel,
    )


@triton.jit
def _routing_plan_to_block_map_kernel(
    block_index,
    block_counts,
    block_map,
    width: tl.constexpr,
    num_key_blocks: tl.constexpr,
    block: tl.constexpr,
):
    row = tl.program_id(0)
    slots = tl.arange(0, block)
    count = tl.load(block_counts + row)
    active = slots < count
    key_block = tl.load(
        block_index + row * width + slots,
        mask=active,
        other=0,
    )
    tl.store(
        block_map + row * num_key_blocks + key_block,
        1,
        mask=active,
    )


def _routing_plan_to_block_map(
    block_index: torch.Tensor,
    block_counts: torch.Tensor,
    num_key_blocks: int,
) -> torch.Tensor:
    """Convert compact absolute block ids to SpargeAttention's dense bool map."""
    if block_index.ndim != 4:
        raise ValueError(
            "SubBlock Sage FP8 block_index must have shape [B, H, Gq, K], got "
            f"{tuple(block_index.shape)}"
        )
    if block_counts.shape != block_index.shape[:-1]:
        raise ValueError(
            "SubBlock Sage FP8 block_counts must match block_index[:-1], got "
            f"{tuple(block_counts.shape)} and {tuple(block_index.shape)}"
        )

    width = block_index.shape[-1]
    if block_index.device.type == "cuda":
        flat_index = block_index.contiguous().view(-1, width)
        flat_counts = block_counts.contiguous().view(-1)
        block_map = torch.zeros(
            (flat_index.shape[0], num_key_blocks),
            dtype=torch.bool,
            device=block_index.device,
        )
        _routing_plan_to_block_map_kernel[(flat_index.shape[0],)](
            flat_index,
            flat_counts,
            block_map,
            width=width,
            num_key_blocks=num_key_blocks,
            block=triton.next_power_of_2(width),
            num_warps=4 if width >= 128 else 2,
        )
        return block_map.view(*block_index.shape[:-1], num_key_blocks)

    flat_index = block_index.reshape(-1, width).long()
    flat_counts = block_counts.reshape(-1)
    slots = torch.arange(width, device=block_index.device)
    active = slots[None, :] < flat_counts[:, None]

    # Only active entries are written. This matters for heterogeneous plans:
    # ignored suffix values can duplicate an active id, and scatter(False) could
    # otherwise race with scatter(True) for the same destination.
    row = torch.arange(flat_index.shape[0], device=block_index.device)
    row = row[:, None].expand_as(flat_index)
    block_map = torch.zeros(
        (flat_index.shape[0], num_key_blocks),
        dtype=torch.bool,
        device=block_index.device,
    )
    block_map[row[active], flat_index[active]] = True
    return block_map.view(*block_index.shape[:-1], num_key_blocks)


@torch.no_grad()
def subblock_sage_fp8_sm90_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_index: torch.Tensor,
    topk: int,
    softmax_scale: float,
    block_counts: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run a 64x128 SubBlock plan with the native SM90 SageAttention2 kernel.

    Q/K/V use SGLang's normal ``[batch, sequence, heads, head_dim]`` layout.
    Quantization is performed online and the returned tensor has the same BF16
    dtype and layout as Q.
    """
    (
        get_vanilla_qk_quant,
        block_map_lut_triton,
        transpose_pad_permute_cuda,
        scale_fuse_quant_cuda,
        kernel,
    ) = _load_sparge_attention_sm90_ops()

    if q.device.type != "cuda":
        raise ValueError("SubBlock Sage FP8 requires CUDA tensors.")
    if torch.cuda.get_device_capability(q.device) != (9, 0):
        raise ValueError("This SubBlock Sage FP8 implementation requires SM90.")
    if q.dtype != torch.bfloat16 or k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError("SubBlock Sage FP8 requires BF16 Q, K, and V.")
    if k.device != q.device or v.device != q.device:
        raise ValueError("SubBlock Sage FP8 requires Q, K, and V on one device.")
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("SubBlock Sage FP8 expects Q, K, and V in [B, S, H, D].")
    if (
        q.shape[0] != k.shape[0]
        or q.shape[0] != v.shape[0]
        or q.shape[2:] != k.shape[2:]
        or q.shape[2:] != v.shape[2:]
        or k.shape[1] != v.shape[1]
        or q.shape[-1] != 128
    ):
        raise ValueError(
            "SubBlock Sage FP8 requires compatible Q, K, and V with head_dim=128."
        )

    expected_q_blocks = -(-q.shape[1] // SAGE_FP8_SM90_QUERY_BLOCK_SIZE)
    num_key_blocks = -(-k.shape[1] // SAGE_FP8_SM90_KEY_BLOCK_SIZE)
    if block_index.shape[:3] != (q.shape[0], q.shape[2], expected_q_blocks):
        raise ValueError(
            "SubBlock Sage FP8 routing shape does not match Q: expected "
            f"{(q.shape[0], q.shape[2], expected_q_blocks)}, got "
            f"{tuple(block_index.shape[:3])}"
        )
    if block_counts is None:
        block_counts = torch.full(
            block_index.shape[:-1],
            topk,
            dtype=torch.int32,
            device=block_index.device,
        )
    block_map = _routing_plan_to_block_map(block_index, block_counts, num_key_blocks)

    # Keep the external package's online quantizers and native Hopper kernel,
    # but not its second P/V-threshold sparsifier. SubBlock already chose the
    # exact blocks to compute; another pruning rule would alter that plan and
    # add a reduction to every iteration.
    with torch.cuda.device(q.device):
        q_hnd = q.transpose(1, 2).contiguous()
        k_hnd = k.transpose(1, 2).contiguous()
        v_hnd = v.transpose(1, 2).contiguous()
        k_mean = k_hnd.mean(dim=-2, keepdim=True)
        q_int8, q_scale, k_int8, k_scale = get_vanilla_qk_quant(
            q_hnd,
            k_hnd,
            k_mean,
            SAGE_FP8_SM90_QUERY_BLOCK_SIZE,
            SAGE_FP8_SM90_KEY_BLOCK_SIZE,
        )
        lut, valid_block_counts = block_map_lut_triton(block_map)

        padded_kv_len = num_key_blocks * SAGE_FP8_SM90_KEY_BLOCK_SIZE
        v_transposed = torch.empty(
            (q.shape[0], q.shape[2], q.shape[3], padded_kv_len),
            dtype=v.dtype,
            device=v.device,
        )
        transpose_pad_permute_cuda(v_hnd, v_transposed, 1)
        v_fp8 = torch.empty_like(v_transposed, dtype=torch.float8_e4m3fn)
        v_scale = torch.empty(
            (q.shape[0], q.shape[2], q.shape[3]),
            dtype=torch.float32,
            device=v.device,
        )
        scale_fuse_quant_cuda(
            v_transposed,
            v_fp8,
            v_scale,
            k.shape[1],
            2.25,
            1,
        )

        # The extension honors output strides. Let its HND kernel write through
        # an HND view of BSHD storage, avoiding a full output transpose/copy.
        output_bshd = torch.empty_like(q)
        output_hnd = output_bshd.transpose(1, 2)
        kernel(
            q_int8,
            k_int8,
            v_fp8,
            output_hnd,
            lut,
            valid_block_counts,
            q_scale,
            k_scale,
            v_scale,
            1,  # HND tensor layout
            False,
            1,  # per-block Q/K scales
            softmax_scale,
        )
    return output_bshd


__all__ = ["subblock_sage_fp8_sm90_attention"]
