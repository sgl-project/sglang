"""Compile boundary for the plain MHA Decode graph, with native length updates."""

from typing import List

import torch
import torch_npu
from torch_npu.npu import NpuGraphOpHandler, register_npu_graph_handler


def _prepare(query, key, value, kwargs):
    kwargs = dict(kwargs)
    v_head_dim = kwargs.pop("v_head_dim")
    kwargs.update(input_layout="BSH", sparse_mode=0)
    workspace = torch_npu._npu_fused_infer_attention_score_get_max_workspace(
        query, key, value, **kwargs
    )
    output = query.new_empty((query.shape[0], 1, kwargs["num_heads"] * v_head_dim))
    lse = query.new_empty(1)
    kwargs.update(workspace=workspace, out=[output, lse])
    return kwargs


@torch.library.custom_op("sglang::npu_graph_mha_attention", mutates_args=())
def npu_graph_mha_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    block_table: torch.Tensor,
    *,
    block_size: int,
    num_heads: int,
    num_key_value_heads: int,
    scale: float,
    actual_seq_lengths_kv: List[int],
    v_head_dim: int,
) -> torch.Tensor:
    kwargs = _prepare(
        query,
        key,
        value,
        dict(
            block_table=block_table,
            block_size=block_size,
            num_heads=num_heads,
            num_key_value_heads=num_key_value_heads,
            scale=scale,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
            v_head_dim=v_head_dim,
        ),
    )
    torch_npu.npu_fused_infer_attention_score.out(query, key, value, **kwargs)
    return kwargs["out"][0]


@npu_graph_mha_attention.register_fake
def _fake(
    query,
    key,
    value,
    block_table,
    *,
    block_size,
    num_heads,
    num_key_value_heads,
    scale,
    actual_seq_lengths_kv,
    v_head_dim,
):
    return query.new_empty((query.shape[0], 1, num_heads * v_head_dim))


@register_npu_graph_handler(["npu_graph_mha_attention.default"])
class _MHAHandler(NpuGraphOpHandler):
    @classmethod
    def prepare_capture(cls, func, args, kwargs):
        # Record the native .out operation, not an opaque replay of its wrapper.
        # Keep lengths as named arguments so NPUGraph.update can rebind them.
        kwargs = dict(kwargs)
        kwargs["block_table"] = args[3]
        kwargs = _prepare(*args[:3], kwargs)
        return torch_npu.npu_fused_infer_attention_score.out, args[:3], kwargs

    @classmethod
    def postprocess_result(cls, result, kwargs):
        return kwargs["out"][0]
