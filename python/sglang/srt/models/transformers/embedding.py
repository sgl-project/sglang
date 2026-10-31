# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SGLang Team

import ast

import torch
from torch import nn

from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding

from .fusers.source import read_forward


def scaled_embedding_contract(module):
    if not isinstance(module, nn.Embedding) or module.max_norm is not None:
        return None
    scale = getattr(module, "embed_scale", None)
    if not isinstance(scale, (float, int, torch.Tensor)) or (
        isinstance(scale, torch.Tensor) and scale.numel() != 1
    ):
        return None
    try:
        source = read_forward(module)
    except (OSError, TypeError, ValueError, StopIteration):
        return None
    function, _ = source
    body = [
        node
        for node in function.body
        if not (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        )
    ]
    if len(body) != 1 or not isinstance(body[0], ast.Return):
        return None
    expression = body[0].value
    if not isinstance(expression, ast.BinOp) or not isinstance(expression.op, ast.Mult):
        return None
    arguments = [arg.arg for arg in function.args.args]
    if len(arguments) != 2:
        return None
    if ast.unparse(expression.left) != f"super().forward({arguments[1]})":
        return None
    value = ast.unparse(expression.right)
    if value == "self.embed_scale":
        return False
    if value == "self.embed_scale.to(self.weight.dtype)":
        return True
    return None


class ScaledVocabParallelEmbedding(VocabParallelEmbedding):
    def set_scale(self, value, cast_to_weight):
        self.register_buffer(
            "embed_scale", torch.as_tensor(value).detach().clone(), persistent=False
        )
        self.cast_scale_to_weight = cast_to_weight

    def forward(self, inputs):
        scale = self.embed_scale
        if self.cast_scale_to_weight:
            scale = scale.to(self.weight.dtype)
        return super().forward(inputs) * scale
