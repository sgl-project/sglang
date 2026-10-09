from __future__ import annotations

from typing import TYPE_CHECKING, Union

import torch

from sglang.srt.layers.cp.utils import cp_gather_after_forward, cp_shard_model_inputs
from sglang.srt.runtime_context import get_parallel

if TYPE_CHECKING:
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput
    from sglang.srt.model_executor.forward_batch_info import (
        ForwardBatch,
        PPProxyTensors,
    )


def runner_owns_cp_boundary(model) -> bool:
    # GLM delegates its CP boundary to the runner via prepare_cp_inputs.
    # Other group-sharing models keep owning the layout in forward().
    return (
        not get_parallel().enable_cp_tp_group_sharing
        or getattr(model, "prepare_cp_inputs", None) is not None
    )


def cp_extend_forward(
    *, model, forward_batch: ForwardBatch, kwargs: dict
) -> Union[LogitsProcessorOutput, PPProxyTensors]:
    """CP extend: shard inputs at the model boundary, run the body on the
    rank-local slice, then gather hidden states before the logits step.
    """
    if prepare_inputs := getattr(model, "prepare_cp_inputs", None):
        input_embeds, positions, model_kwargs = prepare_inputs(forward_batch, **kwargs)
    else:
        input_embeds = kwargs.get("input_embeds")
        if input_embeds is None:
            input_embeds = model.get_input_embeddings()(forward_batch.input_ids)
        positions = forward_batch.positions
        model_kwargs = {}
        if (pp_proxy_tensors := kwargs.get("pp_proxy_tensors")) is not None:
            model_kwargs["pp_proxy_tensors"] = pp_proxy_tensors
    with cp_shard_model_inputs(
        input_embeds,
        positions,
        forward_batch,
        forward_batch.input_ids,
    ) as (sharded_input_embeds, sharded_positions, model_input_ids):
        model_kwargs["input_embeds"] = sharded_input_embeds
        hidden_states = model.model(
            model_input_ids,
            sharded_positions,
            forward_batch,
            **model_kwargs,
        )
    capture_aux_hidden_states = getattr(model, "capture_aux_hidden_states", False)
    aux_hidden_states = None
    if capture_aux_hidden_states:
        hidden_states, aux_hidden_states = hidden_states

    if not model.pp_group.is_last_rank:
        return (
            (hidden_states, aux_hidden_states)
            if capture_aux_hidden_states
            else hidden_states
        )

    stream = torch.cuda.current_stream()
    hidden_states = cp_gather_after_forward(hidden_states, forward_batch, stream)
    # DSpark aux tensors ride the same CP token split; gather them the same way.
    if aux_hidden_states is not None:
        if isinstance(aux_hidden_states, torch.Tensor):
            aux_hidden_states = cp_gather_after_forward(
                aux_hidden_states, forward_batch, stream
            )
        else:
            aux_hidden_states = [
                cp_gather_after_forward(aux, forward_batch, stream)
                for aux in aux_hidden_states
            ]
    logits_kwargs = {}
    # DSV4 returns (hidden_states, hidden_states_before_norm) from its model body.
    if isinstance(hidden_states, tuple):
        hidden_states, hidden_states_before_norm = hidden_states
        # Mirror DeepseekV4ForCausalLM.forward: drop pre_hc_head when
        # DSpark aux capture is on, else it overrides the packed aux.
        if aux_hidden_states is None:
            logits_kwargs["hidden_states_before_norm"] = hidden_states_before_norm
    return model.logits_processor(
        forward_batch.input_ids,
        hidden_states,
        model.lm_head,
        forward_batch,
        aux_hidden_states,
        **logits_kwargs,
    )
