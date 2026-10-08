"""Opt-in Kimi-K3 gfx950 attention-residual specializations."""

from __future__ import annotations

import functools
import os

import torch

from sglang.kernels.ops.attention.kda_whole_layer_gluon_hip import (
    _has_required_gluon_api,
    _rocm_arch,
)
from sglang.srt.runtime_context import get_exec, get_parallel, get_server_args
from sglang.srt.utils import is_hip
from sglang.srt.utils.common import rank0_log

_installed = False


def enabled() -> bool:
    return (
        os.environ.get("SGLANG_ROCM_K3_ATTN_RESIDUAL_FUSED_BACKEND", "").lower()
        == "gluon"
    )


def entrypoint_name(rows: int, valid_rows: int, mode: int) -> str | None:
    """Compactly encode the 177 exact PR17 profile signatures."""
    if rows not in (1, 2, 4, 8, 16, 32, 64, 128, 256):
        return None
    allowed_banks = {1, 4, 5, 6, 7, 8} if rows <= 16 else set(range(1, 9))
    if valid_rows not in allowed_banks or mode not in (4, 5, 6):
        return None
    if valid_rows == 8 and mode == 6:
        return None

    if rows == 1:
        if (valid_rows, mode) == (1, 4):
            return "attention_residual_norm_m1_banks1_modes4"
        if (valid_rows, mode) == (4, 5):
            return "attention_residual_norm_m1_2_banks4_modes5"
        if (valid_rows, mode) == (8, 5):
            return "attention_residual_norm_m1_2_8_16_banks8_modes5"
        return "attention_residual_norm_m1_banks1_4_8_modes4_6"
    if rows == 2:
        if (valid_rows, mode) == (1, 4):
            return "attention_residual_norm_m2_banks1_modes4"
        if (valid_rows, mode) == (4, 5):
            return "attention_residual_norm_m1_2_banks4_modes5"
        if (valid_rows, mode) == (8, 5):
            return "attention_residual_norm_m1_2_8_16_banks8_modes5"
        return "attention_residual_norm_m2_8_banks1_4_8_modes4_6"
    if rows == 4:
        if (valid_rows, mode) in ((1, 4), (4, 5)):
            return "attention_residual_norm_m4_8_16_banks1_4_modes4_5"
        if (valid_rows, mode) == (8, 5):
            return "attention_residual_norm_m4_banks8_modes5"
        return "attention_residual_norm_m4_16_banks1_4_8_modes4_6"
    if rows == 8:
        if (valid_rows, mode) == (1, 4):
            return "attention_residual_norm_m8_banks1_modes4"
        if (valid_rows, mode) == (4, 5):
            return "attention_residual_norm_m4_8_16_banks1_4_modes4_5"
        if (valid_rows, mode) == (8, 5):
            return "attention_residual_norm_m1_2_8_16_banks8_modes5"
        return "attention_residual_norm_m2_8_banks1_4_8_modes4_6"
    if rows == 16:
        if (valid_rows, mode) == (1, 4):
            return "attention_residual_norm_m16_banks1_modes4"
        if (valid_rows, mode) == (4, 5):
            return "attention_residual_norm_m4_8_16_banks1_4_modes4_5"
        if (valid_rows, mode) == (8, 5):
            return "attention_residual_norm_m1_2_8_16_banks8_modes5"
        return "attention_residual_norm_m4_16_banks1_4_8_modes4_6"
    if rows == 32:
        if (valid_rows, mode) == (1, 4):
            return "attention_residual_norm_m32_banks1_modes4"
        if (valid_rows, mode) == (4, 5):
            return "attention_residual_norm_m32_64_banks4_modes5"
        if (valid_rows, mode) == (8, 5):
            return "attention_residual_norm_m32_banks8_modes5"
    if rows == 64:
        if (valid_rows, mode) == (1, 4):
            return "attention_residual_norm_m64_banks1_modes4"
        if (valid_rows, mode) == (4, 5):
            return "attention_residual_norm_m32_64_banks4_modes5"
        if (valid_rows, mode) == (8, 5):
            return "attention_residual_norm_m64_banks8_modes5"
    return "attention_residual_norm_m32_64_128_256_banks1_8_modes4_6"


def qualified_model(model) -> bool:
    if not enabled():
        return False
    server_args = get_server_args()
    parameter = next(model.parameters(), None)
    return (
        is_hip()
        and _has_required_gluon_api()
        and parameter is not None
        and _rocm_arch(parameter.device) == "gfx950"
        and getattr(model.config, "hidden_size", None) == 7168
        and get_parallel().attn_tp_size == 8
        and not getattr(server_args, "enable_lora", False)
        and not getattr(server_args, "speculative_algorithm", None)
        and not get_exec().deterministic.enable_deterministic_inference
    )


def covered(
    prefix, addend, bank, valid_rows, score_proj, score_norm, out_norm, write_bank
):
    rows = (
        prefix.shape[0] if isinstance(prefix, torch.Tensor) and prefix.ndim == 2 else 0
    )
    mode = (
        int(addend is not None)
        | (int(write_bank) << 1)
        | (int(out_norm is not None) << 2)
    )
    name = entrypoint_name(rows, valid_rows, mode)
    return (
        name
        if (
            name is not None
            and tuple(prefix.shape) == (rows, 7168)
            and prefix.dtype == torch.bfloat16
            and prefix.is_contiguous()
            and (
                addend is None
                or (
                    tuple(addend.shape) == tuple(prefix.shape)
                    and addend.dtype == prefix.dtype
                    and addend.device == prefix.device
                    and addend.is_contiguous()
                )
            )
            and bank.ndim == 3
            and tuple(bank.shape[:1]) == (rows,)
            and bank.shape[1] >= valid_rows + int(write_bank)
            and bank.shape[2] == 7168
            and bank.dtype == prefix.dtype
            and bank.device == prefix.device
            and bank.stride(2) == 1
            and tuple(score_proj.weight.shape) == (1, 7168)
            and tuple(score_norm.weight.shape) == (7168,)
            and out_norm is not None
            and tuple(out_norm.weight.shape) == (7168,)
            and score_norm.weight.device == out_norm.weight.device == prefix.device
            and score_norm.variance_epsilon == 1e-5
            and out_norm.variance_epsilon == 1e-5
        )
        else None
    )


def run(
    name, prefix, addend, bank, score_weight, output_weight, *, valid_rows, write_bank
):
    from sglang.kernels.ops.attention.mla_gluon import attention_residual_norm

    return getattr(attention_residual_norm, name)(
        prefix,
        prefix if addend is None else addend,
        bank,
        score_weight,
        output_weight,
        valid_rows=valid_rows,
        has_addend=addend is not None,
        write_bank=write_bank,
        apply_output_norm=True,
        score_eps=1e-5,
        output_eps=1e-5,
    )


def install(model) -> bool:
    """Register once, and only after a qualified Kimi model is loaded."""
    global _installed
    if _installed:
        return True
    if not qualified_model(model):
        return False

    from sglang.srt.layers import attn_residual as native

    original = native._aggregate_hip

    @functools.wraps(original)
    def aggregate(
        prefix,
        addend,
        bank,
        valid_rows,
        score_proj,
        score_norm,
        out_norm,
        write_bank_row=False,
    ):
        name = covered(
            prefix,
            addend,
            bank,
            valid_rows,
            score_proj,
            score_norm,
            out_norm,
            write_bank_row,
        )
        if name is None:
            return original(
                prefix,
                addend,
                bank,
                valid_rows,
                score_proj,
                score_norm,
                out_norm,
                write_bank_row,
            )
        output, current, returned_bank = run(
            name,
            prefix,
            addend,
            bank,
            native.get_cw(score_proj, score_norm),
            out_norm.weight,
            valid_rows=valid_rows,
            write_bank=write_bank_row,
        )
        if (
            tuple(output.shape) != tuple(prefix.shape)
            or output.dtype != prefix.dtype
            or output.device != prefix.device
            or tuple(current.shape) != tuple(prefix.shape)
            or current.dtype != prefix.dtype
            or current.device != prefix.device
            or returned_bank.untyped_storage().data_ptr()
            != bank.untyped_storage().data_ptr()
        ):
            raise RuntimeError("Kimi-K3 attention residual output ABI changed")
        return output, current

    native._aggregate_hip = aggregate
    _installed = True
    rank0_log("K3 Gluon attention residual enabled for 177 profiled signatures.")
    return True
