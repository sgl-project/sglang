"""Run real DeepSeek staged MoE operations with identity attention fixtures.

Uses SGLang's split/pad/merge and interleaved executor. Attention and its
collectives are outside this focused MoE fixture.
"""

from types import SimpleNamespace

import torch

from sglang.srt.batch_overlap.operations import execute_overlapped_operations
from sglang.srt.batch_overlap.operations_strategy import (
    OperationsStrategy,
    _compute_moe_deepseek_layer_operations_strategy_tbo,
)
from sglang.srt.batch_overlap.two_batch_overlap import (
    _model_forward_tbo_merge_outputs,
    _model_forward_tbo_split_inputs_raw,
)
from sglang.srt.layers.dp_attention import DpPaddingMode
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context


def _layer(model, residual_scale=None, input_transform=None):
    def begin(state, hidden_states, forward_batch, tbo_subbatch_index, **kwargs):
        if hasattr(model.topk, "set_subbatch"):
            model.topk.set_subbatch(
                tbo_subbatch_index, forward_batch.tbo_parent_token_range
            )
        state.hidden_states_mlp_input = (
            input_transform(hidden_states)
            if input_transform is not None
            else hidden_states
        )
        state.forward_batch = forward_batch
        state.tbo_subbatch_index = tbo_subbatch_index
        if residual_scale is not None:
            state.layer_input = hidden_states

    def finish(state):
        hidden_states = state.pop("hidden_states_mlp_output")
        if residual_scale is not None:
            hidden_states = state.pop("layer_input") + hidden_states * residual_scale
        result = dict(
            hidden_states=hidden_states,
            forward_batch=state.pop("forward_batch"),
            tbo_subbatch_index=state.pop("tbo_subbatch_index"),
            residual=None,
        )
        state.clear(expect_keys=[])
        return result

    return SimpleNamespace(
        is_layer_sparse=True,
        mlp=model,
        op_comm_prepare_attn=begin,
        self_attn=SimpleNamespace(
            op_prepare=lambda state: None, op_core=lambda state: None
        ),
        op_comm_prepare_mlp=lambda state: None,
        op_comm_postprocess_layer=finish,
    )


def forward_tbo(
    models,
    x,
    *,
    split=None,
    padded=None,
    counts=None,
    mode,
    children=None,
    residual_scale=None,
    input_transform=None,
):
    children = (
        children
        if children is not None
        else [
            SimpleNamespace(
                forward_mode=mode,
                num_token_non_padded=counts[index],
                tbo_parent_token_range=bounds,
                tbo_padded_len=rows,
                global_dp_buffer_len=rows,
                global_num_tokens_cpu=[rows],
                dp_padding_mode=DpPaddingMode.MAX_LEN,
            )
            for index, (bounds, rows) in enumerate(
                zip(((0, split), (split, len(x))), padded)
            )
        ]
    )
    inputs = _model_forward_tbo_split_inputs_raw(
        hidden_states=x,
        residual=None,
        positions=torch.arange(len(x), device=x.device),
        forward_batch=SimpleNamespace(tbo_children=children),
        zero_allocator=None,
    )
    strategy = OperationsStrategy.concat(
        [
            _compute_moe_deepseek_layer_operations_strategy_tbo(
                _layer(model, residual_scale, input_transform), mode
            )
            for model in models
        ]
    )
    with forward_context(ForwardContext(attn_backend=None)):
        outputs = execute_overlapped_operations(
            inputs_arr=inputs,
            operations_arr=[strategy.operations] * 2,
            delta_stages=[0, strategy.tbo_delta_stages],
        )
    return _model_forward_tbo_merge_outputs(*outputs, len(x))[0]
