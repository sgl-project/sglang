from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

import torch
from sglang.srt.distributed import get_tensor_model_parallel_world_size, tensor_model_parallel_all_reduce
from sglang.srt.layers.dp_attention import attn_tp_all_gather, dp_gather_partial, dp_scatter, attn_tp_reduce_scatter, \
    get_local_attention_dp_size, get_attention_tp_size, get_attention_tp_rank
from sglang.srt.managers.schedule_batch import global_server_args_dict
from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class ScatterMode(Enum):
    SCATTERED = auto()
    TP_ATTN_FULL = auto()
    FULL = auto()


@dataclass
class _LayerModeComputationContext:
    num_layers: int
    layer_id: int
    is_layer_sparse: bool
    is_previous_layer_sparse: Optional[bool]

    def previous_layer(self):
        assert self.is_previous_layer_sparse is not None
        return _LayerModeComputationContext(
            layer_id=self.layer_id,
            is_layer_sparse=self.is_previous_layer_sparse,
            is_previous_layer_sparse=None,

            # unchanged
            num_layers=self.num_layers,
        )


@dataclass
class LayerScatterModes:
    layer_input_mode: ScatterMode
    attn_mode: ScatterMode
    # Can be further split into e.g. ffn_input_mode and ffn_output_mode if needed
    mlp_mode: ScatterMode
    layer_output_mode: ScatterMode

    @classmethod
    def init_new(cls, **kwargs):
        context = _LayerModeComputationContext(**kwargs)
        return cls(
            layer_input_mode=cls._compute_layer_input_mode(context),
            attn_mode=ScatterMode.TP_ATTN_FULL,
            mlp_mode=cls._compute_mlp_mode(context),
            layer_output_mode=cls._compute_layer_output_mode(context),
        )

    @classmethod
    def _compute_layer_input_mode(cls, context: _LayerModeComputationContext):
        if context.layer_id == 0:
            return ScatterMode.TP_ATTN_FULL
        return cls._compute_layer_output_mode(context.previous_layer())

    @classmethod
    def _compute_mlp_mode(cls, context: _LayerModeComputationContext):
        if context.is_layer_sparse:
            return (
                ScatterMode.SCATTERED
                if global_server_args_dict["enable_deepep_moe"]
                else ScatterMode.FULL
            )
        else:
            return (
                ScatterMode.SCATTERED
                if enable_moe_dense_fully_dp()
                else ScatterMode.FULL
            )

    @classmethod
    def _compute_layer_output_mode(cls, context: _LayerModeComputationContext):
        if context.layer_id == context.num_layers - 1:
            return ScatterMode.TP_ATTN_FULL
        return cls._compute_mlp_mode(context)


def enable_moe_dense_fully_dp():
    return global_server_args_dict["moe_dense_tp_size"] == 1


class LayerCommunicator:
    def __init__(self, layer_scatter_modes: LayerScatterModes):
        self.layer_scatter_modes = layer_scatter_modes
        self.local_dp_size = get_local_attention_dp_size()
        self.attn_tp_size = get_attention_tp_size()
        self.attn_tp_rank = get_attention_tp_rank()
        self.tp_size = get_tensor_model_parallel_world_size()

    def forward_pre_attn(
        self,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ):
        return _communicate_simple(
            hidden_states=hidden_states,
            forward_batch=forward_batch,
            input_mode=self.layer_scatter_modes.layer_input_mode,
            output_mode=self.layer_scatter_modes.attn_mode,
            attn_tp_size=self.attn_tp_size,
        )

    def forward_pre_mlp(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        forward_batch: ForwardBatch,
    ):
        if self.layer_scatter_modes.mlp_mode == ScatterMode.FULL:
            if self.attn_tp_size != 1 and self.layer_scatter_modes.layer_input_mode == ScatterMode.SCATTERED:
                raise AssertionError("moe_layer_freq > 1 is not supported when attn_tp_size > 1")

            if self.tp_size > 1:
                # all gather and all reduce
                if self.local_dp_size != 1:
                    if self.attn_tp_rank == 0:
                        hidden_states += residual
                    hidden_states, local_hidden_states = (
                        forward_batch.gathered_buffer,
                        hidden_states,
                    )
                    dp_gather_partial(hidden_states, local_hidden_states, forward_batch)
                    dp_scatter(residual, hidden_states, forward_batch)
                    hidden_states = self.post_attention_layernorm(hidden_states)
                else:
                    hidden_states = tensor_model_parallel_all_reduce(hidden_states)
                    hidden_states, residual = self.post_attention_layernorm(
                        hidden_states, residual
                    )
            else:
                hidden_states, residual = self.post_attention_layernorm(
                    hidden_states, residual
                )
        elif self.layer_scatter_modes.mlp_mode == ScatterMode.SCATTERED:
            if self.attn_tp_size != 1:
                if self.layer_scatter_modes.layer_input_mode == ScatterMode.SCATTERED:
                    tensor_list = list(hidden_states.tensor_split(self.attn_tp_size))
                    hidden_states = tensor_list[self.attn_tp_rank]
                    attn_tp_reduce_scatter(hidden_states, tensor_list)
                else:
                    tensor_list = list(hidden_states.tensor_split(self.attn_tp_size))
                    hidden_states = tensor_list[self.attn_tp_rank]
                    attn_tp_reduce_scatter(hidden_states, tensor_list)
                    residual = residual.tensor_split(self.attn_tp_size)[self.attn_tp_rank]
            if hidden_states.shape[0] != 0:
                hidden_states, residual = self.post_attention_layernorm(
                    hidden_states, residual
                )
        else:
            raise NotImplementedError
        return hidden_states, residual

    def forward_layer_end(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        forward_batch: ForwardBatch,
    ):
        return _communicate_summable_tensor_pair(
            hidden_states=hidden_states,
            residual=residual,
            forward_batch=forward_batch,
            hidden_states_input_mode=self.layer_scatter_modes.mlp_mode,
            residual_input_mode=TODO,
            output_mode=self.layer_scatter_modes.layer_output_mode,
            local_dp_size=self.local_dp_size,
            attn_tp_size=self.attn_tp_size,
        )


def _communicate_simple(
    hidden_states: torch.Tensor,
    forward_batch: ForwardBatch,
    input_mode: ScatterMode,
    output_mode: ScatterMode,
    attn_tp_size: int,
) -> torch.Tensor:
    if input_mode == output_mode:
        return hidden_states

    if input_mode == ScatterMode.SCATTERED and output_mode == ScatterMode.TP_ATTN_FULL:
        if attn_tp_size != 1:
            hidden_states, local_hidden_states = (
                forward_batch.gathered_buffer[: forward_batch.input_ids.shape[0]],
                hidden_states,
            )
            attn_tp_all_gather(
                list(hidden_states.tensor_split(attn_tp_size)), local_hidden_states
            )
        return hidden_states

    raise NotImplementedError(f"{input_mode=} {output_mode=}")


def _communicate_summable_tensor_pair(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    forward_batch: ForwardBatch,
    hidden_states_input_mode: ScatterMode,
    residual_input_mode: ScatterMode,
    output_mode: ScatterMode,
    local_dp_size: int,
    attn_tp_size: int,
):
    """It is allowed to sum hidden_states and residual if needed."""
    if hidden_states_input_mode == ScatterMode.FULL:
        TODO_add_more_guards
        # TODO(ch-wan): use reduce-scatter in MLP to avoid this scatter
        if local_dp_size != 1:
            # important: forward batch.gathered_buffer is used both after scatter and after gather.
            # be careful about this!
            hidden_states, global_hidden_states = (
                forward_batch.gathered_buffer[: forward_batch.input_ids.shape[0]],
                hidden_states,
            )
            dp_scatter(hidden_states, global_hidden_states, forward_batch)
        return hidden_states, residual

    if hidden_states_input_mode == ScatterMode.SCATTERED:
        TODO_add_more_guards
        if output_mode == ScatterMode.TP_ATTN_FULL and attn_tp_size != 1:
            hidden_states += residual
            residual = None
            hidden_states, local_hidden_states = (
                forward_batch.gathered_buffer[: forward_batch.input_ids.shape[0]],
                hidden_states,
            )
            attn_tp_all_gather(
                list(hidden_states.tensor_split(attn_tp_size)), local_hidden_states
            )
        return hidden_states, residual

    raise NotImplementedError
