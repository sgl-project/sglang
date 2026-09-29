# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SGLang Team

import ast
import logging
import types

import torch
import triton
import triton.language as tl

from sglang.srt.utils.custom_op import register_custom_op

from .source import compile_forward, read_forward, self_attribute

logger = logging.getLogger(__name__)


@triton.jit
def _residual_norm_kernel(
    x_ptr,
    residual_ptr,
    weight_ptr,
    output_ptr,
    sum_ptr,
    width: tl.constexpr,
    epsilon: tl.constexpr,
    cast_before_weight: tl.constexpr,
    zero_centered: tl.constexpr,
    block: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.arange(0, block)
    mask = columns < width
    x = tl.load(x_ptr + row * width + columns, mask, 0).to(tl.float32)
    residual = tl.load(residual_ptr + row * width + columns, mask, 0).to(tl.float32)
    # The residual addition rounds before the variance reduction in HF forwards.
    summed = (x + residual).to(sum_ptr.dtype.element_ty).to(tl.float32)
    tl.store(sum_ptr + row * width + columns, summed, mask)
    variance = tl.sum(summed * summed, 0) / width
    normalized = summed * tl.rsqrt(variance + epsilon)
    weight = tl.load(weight_ptr + columns, mask, 0).to(tl.float32)
    if zero_centered:
        weight = weight + 1.0
    if cast_before_weight:
        normalized = normalized.to(output_ptr.dtype.element_ty).to(tl.float32)
    tl.store(output_ptr + row * width + columns, normalized * weight, mask)


def _fake_residual_norm(
    x, residual, weight, epsilon, cast_before_weight, zero_centered
):
    return torch.empty_like(x), torch.empty_like(x)


@register_custom_op(fake_impl=_fake_residual_norm)
def transformers_residual_rms_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
    cast_before_weight: bool,
    zero_centered: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    output, summed = torch.empty_like(x), torch.empty_like(x)
    width = x.shape[-1]
    rows = x.numel() // width
    if rows:
        _residual_norm_kernel[(rows,)](
            x,
            residual,
            weight,
            output,
            summed,
            width,
            epsilon,
            cast_before_weight,
            zero_centered,
            triton.next_power_of_2(width),
            num_warps=4 if width <= 2048 else 8,
            enable_fp_fusion=False,
        )
    return output, summed


def residual_norm(norm, x, residual):
    if x.shape != residual.shape or x.dtype != residual.dtype:
        summed = x + residual
        return norm(summed), summed
    if (
        x.is_cuda
        and x.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and norm.weight.dtype == x.dtype
        and x.shape[-1] <= 16384
    ):
        return transformers_residual_rms_norm(
            x.contiguous(),
            residual.contiguous(),
            norm.weight.contiguous(),
            norm.variance_epsilon,
            getattr(norm, "cast_x_before_out_mul", False),
            norm._hf_zero_centered,
        )
    summed = x + residual
    return norm(summed), summed


def _assignment(statement):
    if (
        isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
    ):
        return statement.targets[0].id, statement.value
    return None, None


def fuse_residual_norm(module):
    from ..layers import HFCompatibleGemmaRMSNorm, HFCompatibleRMSNorm

    if not any(
        isinstance(child, (HFCompatibleRMSNorm, HFCompatibleGemmaRMSNorm))
        for child in module.children()
    ):
        return False
    try:
        function, original = read_forward(module)
    except (OSError, TypeError, SyntaxError):
        return False
    changed = False
    body = []
    index = 0
    while index < len(function.body):
        if index + 2 < len(function.body):
            output, addition = _assignment(function.body[index])
            residual, alias = _assignment(function.body[index + 1])
            norm_output, call = _assignment(function.body[index + 2])
            if (
                output is not None
                and residual is not None
                and output != residual
                and isinstance(addition, ast.BinOp)
                and isinstance(addition.op, ast.Add)
                and isinstance(addition.left, ast.Name)
                and isinstance(addition.right, ast.Name)
                and {addition.left.id, addition.right.id} == {output, residual}
                and isinstance(alias, ast.Name)
                and alias.id == output
                and norm_output == output
                and isinstance(call, ast.Call)
                and len(call.args) == 1
                and not call.keywords
                and isinstance(call.args[0], ast.Name)
                and call.args[0].id == output
            ):
                name = self_attribute(call.func)
                norm = getattr(module, name or "", None)
                if isinstance(norm, (HFCompatibleRMSNorm, HFCompatibleGemmaRMSNorm)):
                    replacement = ast.parse(
                        f"{output}, {residual} = self.{name}.forward_add({output}, {residual})"
                    ).body[0]
                    body.append(replacement)
                    changed = True
                    index += 3
                    continue
        body.append(function.body[index])
        index += 1
    if changed:
        function.body = body
        try:
            forward = compile_forward(function, original)
        except (TypeError, SyntaxError, ValueError) as error:
            logger.debug(
                "Cannot fuse residual norm in %s: %s", type(module).__name__, error
            )
            return False
        module.forward = types.MethodType(forward, module)
    return changed
