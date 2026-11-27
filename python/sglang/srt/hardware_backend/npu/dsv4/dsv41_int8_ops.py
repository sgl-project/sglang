"""Native CANN bindings for the DeepSeek-V4.1 A3 signed INT8 CSA cache."""

from functools import cache
from importlib import import_module
from math import isclose

import torch


@cache
def int8_csa_ops():
    try:
        import_module("cann_ops_transformer")
        ops = torch.ops.cann_ops_transformer
        return (
            ops.kv_compress_epilog,
            ops.mixed_quant_sparse_flash_mla_metadata,
            ops.mixed_quant_sparse_flash_mla,
        )
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "DeepSeek-V4.1 INT8 KV requires the CANN arch22 INT8 KvCompressEpilog, "
            "MixedQuantSparseFlashMla and MixedQuantSparseFlashMlaMetadata package."
        ) from exc


@cache
def check_int8_csa_support(device):
    import acl

    from sglang.srt.hardware_backend.npu.utils import is_npu_arch22

    soc_name = acl.get_soc_name()
    if not is_npu_arch22() or not soc_name.startswith("Ascend910_93"):
        raise ValueError(f"DeepSeek-V4.1 INT8 KV requires Atlas A3, got {soc_name}.")

    # Dispatch a legal A16C8 call. Older packages expose the same op names
    # but implement a different format (or have no kernel for arch22).
    try:
        win = torch.zeros((2, 16, 1, 608), dtype=torch.int8, device=device)
        cmp = torch.zeros_like(win)
        q = torch.zeros((1, 1, 512), dtype=torch.bfloat16, device=device)
        kv = torch.zeros_like(q).view(1, 512)
        kv[:, -1] = 2
        slot = torch.tensor([16], dtype=torch.int32, device=device)
        lens = torch.ones(1, dtype=torch.int32, device=device)
        common = dict(
            cu_seqlens_q=torch.tensor([0, 1], dtype=torch.int32, device=device),
            seqused_ori_kv=lens,
            seqused_cmp_kv=lens,
            cmp_ratio=1,
            ori_mask_mode=4,
            cmp_mask_mode=3,
            ori_win_left=127,
            ori_win_right=0,
            layout_q="TND",
            layout_kv="PA_BBND",
        )
        write_int8_csa_rows(win, slot, kv)
        write_int8_csa_rows(cmp, slot, kv)
        page_table = lens.view(1, 1)
        metadata = int8_csa_metadata(
            1,
            1,
            512,
            batch_size=1,
            cmp_topk=16,
            has_ori_kv=True,
            has_cmp_kv=True,
            **common,
        )
        indices = torch.full((1, 1, 16), -1, dtype=torch.int32, device=device)
        indices[..., 0] = 0
        out, _ = int8_csa_attention(
            q,
            ori_kv=win,
            cmp_kv=cmp,
            cmp_sparse_indices=indices,
            ori_block_table=page_table,
            cmp_block_table=page_table,
            metadata=metadata,
            sinks=torch.zeros(1, dtype=torch.float32, device=device),
            softmax_scale=512**-0.5,
            **common,
        )
        # Two identical latent rows with zero logits and one zero-logit sink
        # give 4/3 in the last output column.
        if out.dtype != torch.bfloat16 or not isclose(
            out[0, 0, -1].item(), 4 / 3, abs_tol=0.02
        ):
            raise RuntimeError(
                "The native INT8 CSA startup probe returned an invalid result."
            )
    except (ImportError, AttributeError, RuntimeError, ValueError) as exc:
        raise RuntimeError(
            f"DeepSeek-V4.1 INT8 KV failed the native CANN probe on {soc_name}. "
            "Install a matching arch22 INT8 ops-transformer vendor and Python binding."
        ) from exc


def write_int8_csa_rows(cache, loc, kv):
    rows = kv.reshape(-1, 512).contiguous()
    if rows.shape[0] == 0:
        return
    slots = loc.reshape(-1)
    if rows.shape[0] != slots.numel() or rows.dtype != torch.bfloat16:
        raise ValueError("INT8 CSA requires one BF16 latent row per cache slot.")
    page_rows = cache.shape[1]
    live = (slots >= page_rows) & (slots < cache.shape[0] * page_rows)
    # Page zero is reserved in SGLang. CANN treats it as writable.
    slots = torch.where(live, slots, -1).contiguous()
    writer, _, _ = int8_csa_ops()
    writer(
        cache.view(torch.uint8),
        rows,
        slots,
        quant_group_size=64,
        quant_mode="int8_bf16",
        round_scale=False,
        x_scale=1.0,
    )


def int8_csa_metadata(num_heads_q, num_heads_kv, head_dim, **kwargs):
    _, metadata, _ = int8_csa_ops()
    return metadata(num_heads_q, num_heads_kv, head_dim, 1, rope_head_dim=64, **kwargs)


def int8_csa_attention(q, **kwargs):
    _, _, attention = int8_csa_ops()
    return attention(q, 1, rope_head_dim=64, topk_value_mode=1, **kwargs)
