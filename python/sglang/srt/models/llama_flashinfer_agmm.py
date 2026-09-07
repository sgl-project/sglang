# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Exact FlashInfer AGMM sequence-parallel route for Llama 3.1 70B.

This module intentionally supports only exact unquantized BF16 Llama 3.1 70B
configurations: TP4 or TP8 on SM103 and an extend batch containing exactly 4096
token rows. Decode and other row counts continue through the native Llama model
path.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    import torch

    from sglang.srt.model_executor.forward_batch_info import (
        ForwardBatch,
        PPProxyTensors,
    )


_EXPECTED_MODEL = {
    "hidden_size": 8192,
    "intermediate_size": 28672,
    "num_hidden_layers": 80,
    "num_attention_heads": 64,
    "num_key_value_heads": 8,
}
_FULL_ROWS = 4096
_MODEL_TOKEN_ATTRIBUTE = "_flashinfer_agmm_true_sp_model_token"


@dataclass(frozen=True)
class _Topology:
    tp_size: int
    local_rows: int
    q_size: int
    kv_size: int
    intermediate_size: int

    @property
    def packed_qkv_n(self) -> int:
        return self.q_size + 2 * self.kv_size

    @property
    def ranks(self) -> tuple[int, ...]:
        return tuple(range(self.tp_size))


_TOPOLOGIES = {
    4: _Topology(
        tp_size=4,
        local_rows=1024,
        q_size=2048,
        kv_size=256,
        intermediate_size=7168,
    ),
    8: _Topology(
        tp_size=8,
        local_rows=512,
        q_size=1024,
        kv_size=128,
        intermediate_size=3584,
    ),
}


def _topology_for_tp_size(tp_size: int) -> _Topology:
    try:
        return _TOPOLOGIES[tp_size]
    except KeyError as error:
        raise RuntimeError(
            "FlashInfer AGMM true-SP requires tensor parallel size 4 or 8"
        ) from error


def _row_partition(total_rows: int, topology: _Topology) -> Optional[int]:
    return topology.local_rows if total_rows == _FULL_ROWS else None


def _forward_mode_reason(forward_batch: Any) -> Optional[str]:
    if forward_batch is None:
        return "missing_forward_batch"
    mode = getattr(forward_batch, "forward_mode", None)
    if mode is None:
        return "missing_forward_mode"
    is_extend = getattr(mode, "is_extend", None)
    if not callable(is_extend) or not is_extend():
        return "not_extend"
    is_target_verify = getattr(mode, "is_target_verify", None)
    if callable(is_target_verify) and is_target_verify():
        return "target_verify"
    if bool(getattr(forward_batch, "can_run_tbo", False)):
        return "two_batch_overlap"
    return None


def _model_contract_reason(
    model: Any, torch_module: Any, topology: _Topology
) -> Optional[str]:
    pp_group = getattr(model, "pp_group", None)
    if (
        pp_group is None
        or int(getattr(pp_group, "world_size", -1)) != 1
        or not bool(getattr(pp_group, "is_first_rank", False))
        or not bool(getattr(pp_group, "is_last_rank", False))
    ):
        return "pipeline_parallel"
    if int(getattr(model, "start_layer", -1)) != 0:
        return "nonzero_start_layer"
    if int(getattr(model, "end_layer", -1)) != 80:
        return "partial_layer_range"
    if list(getattr(model, "layers_to_capture", ())) != []:
        return "aux_layer_capture"
    config = getattr(model, "config", None)
    if config is None:
        return "missing_config"
    for name, expected in _EXPECTED_MODEL.items():
        if int(getattr(config, name, -1)) != expected:
            return f"config_{name}"
    layers = getattr(model, "layers", None)
    if layers is None or len(layers) != 80:
        return "layer_count"
    for layer_index, layer in enumerate(layers):
        if type(layer).__name__ != "LlamaDecoderLayer":
            return f"layer_{layer_index}_type"
        attention = getattr(layer, "self_attn", None)
        mlp = getattr(layer, "mlp", None)
        if attention is None or mlp is None:
            return f"layer_{layer_index}_modules"
        qkv = getattr(attention, "qkv_proj", None)
        o_proj = getattr(attention, "o_proj", None)
        gate_up = getattr(mlp, "gate_up_proj", None)
        down = getattr(mlp, "down_proj", None)
        modules = (qkv, o_proj, gate_up, down)
        if any(module is None for module in modules):
            return f"layer_{layer_index}_linear_modules"
        if (
            type(qkv).__name__ != "QKVParallelLinear"
            or type(o_proj).__name__ != "RowParallelLinear"
            or type(gate_up).__name__ != "MergedColumnParallelLinear"
            or type(down).__name__ != "RowParallelLinear"
        ):
            return f"layer_{layer_index}_linear_types"
        if (
            tuple(qkv.weight.shape) != (topology.packed_qkv_n, 8192)
            or tuple(o_proj.weight.shape) != (8192, topology.q_size)
            or tuple(gate_up.weight.shape) != (2 * topology.intermediate_size, 8192)
            or tuple(down.weight.shape) != (8192, topology.intermediate_size)
        ):
            return f"layer_{layer_index}_weight_shapes"
        if any(module.weight.dtype != torch_module.bfloat16 for module in modules):
            return f"layer_{layer_index}_weight_dtype"
        if any(not module.weight.is_cuda for module in modules):
            return f"layer_{layer_index}_weight_device"
        if any(module.bias is not None for module in modules):
            return f"layer_{layer_index}_bias"
        if any(
            type(module.quant_method).__name__ != "UnquantizedLinearMethod"
            for module in modules
        ):
            return f"layer_{layer_index}_quantization"
        if (
            int(qkv.tp_size) != topology.tp_size
            or int(qkv.q_proj_shard_size) != topology.q_size
            or int(qkv.kv_proj_shard_size) != topology.kv_size
            or int(qkv.v_proj_shard_size) != topology.kv_size
            or bool(qkv.gather_output)
        ):
            return f"layer_{layer_index}_qkv_contract"
        if (
            int(attention.q_size) != topology.q_size
            or int(attention.kv_size) != topology.kv_size
        ):
            return f"layer_{layer_index}_attention_split"
        if (
            int(gate_up.tp_size) != topology.tp_size
            or bool(gate_up.gather_output)
            or int(o_proj.tp_size) != topology.tp_size
            or int(down.tp_size) != topology.tp_size
            or not bool(o_proj.input_is_parallel)
            or not bool(down.input_is_parallel)
            or not bool(o_proj.reduce_results)
            or not bool(down.reduce_results)
        ):
            return f"layer_{layer_index}_parallel_contract"
    if int(getattr(model.norm, "hidden_size", -1)) != 8192:
        return "final_norm"
    return None


def _validate_prepare_signature(prepare: Callable[..., Any]) -> None:
    parameters = inspect.signature(prepare).parameters
    if tuple(parameters) != ("inp", "w", "group", "backend", "verbose"):
        raise RuntimeError("FlashInfer prepared AGMM API has an incompatible signature")
    if parameters["backend"].kind is not inspect.Parameter.KEYWORD_ONLY:
        raise RuntimeError("FlashInfer prepared AGMM backend must be keyword-only")
    if parameters["backend"].default != "auto":
        raise RuntimeError("FlashInfer prepared AGMM backend default changed")
    if parameters["verbose"].kind is not inspect.Parameter.KEYWORD_ONLY:
        raise RuntimeError("FlashInfer prepared AGMM verbose must be keyword-only")
    if parameters["verbose"].default is not False:
        raise RuntimeError("FlashInfer prepared AGMM verbose default changed")


def _bind_model_contract(
    model: Any, torch_module: Any, topology: _Topology
) -> tuple[int, object]:
    reason = _model_contract_reason(model, torch_module, topology)
    if reason is not None:
        raise RuntimeError(
            "--enable-flashinfer-agmm-true-sp requires the exact "
            f"unquantized BF16 Llama 3.1 70B model contract: {reason}"
        )
    if hasattr(model, _MODEL_TOKEN_ATTRIBUTE):
        raise RuntimeError("Llama model already owns an AGMM true-SP token")
    token = object()
    setattr(model, _MODEL_TOKEN_ATTRIBUTE, token)
    return id(model), token


@dataclass(frozen=True)
class _PreparedBinding:
    weight: Any
    group: Any
    local_rows: int
    launcher: Callable[[Any], Any]


class LlamaFlashInferAgmmTrueSP:
    """Own the opt-in true sequence-parallel route for one Llama model."""

    def __init__(self, model: Any) -> None:
        import torch
        from flashinfer.comm import prepare_all_gather_matmul

        self._torch = torch
        self._topology = self._validate_runtime_config()
        if not callable(prepare_all_gather_matmul):
            raise RuntimeError("FlashInfer prepared AGMM API is unavailable")
        _validate_prepare_signature(prepare_all_gather_matmul)
        self._model_id, self._model_token = _bind_model_contract(
            model, torch, self._topology
        )
        self._prepare_all_gather_matmul = prepare_all_gather_matmul
        self._coordinator = None
        self._group = None
        self._rank = None
        self._packed_weights: dict[int, Any] = {}
        self._bindings: dict[tuple[int, int], _PreparedBinding] = {}

    @staticmethod
    def _validate_runtime_config() -> _Topology:
        from sglang.srt.runtime_context import (
            get_exec,
            get_memory,
            get_parallel,
            get_schedule,
        )

        parallel = get_parallel()
        tp_size = int(parallel.tp_size)
        topology = _topology_for_tp_size(tp_size)
        if (
            int(parallel.attn_tp_size) != tp_size
            or int(parallel.attn_dp_size) != 1
            or int(parallel.attn_cp_size) != 1
            or int(parallel.pp_size) != 1
        ):
            raise RuntimeError(
                "--enable-flashinfer-agmm-true-sp requires TP4 or TP8, DP1, CP1, PP1"
            )
        if not bool(get_exec().graph.disable_cuda_graph):
            raise RuntimeError(
                "--enable-flashinfer-agmm-true-sp requires --disable-cuda-graph"
            )
        if not bool(get_schedule().disable_overlap_schedule):
            raise RuntimeError(
                "--enable-flashinfer-agmm-true-sp requires --disable-overlap-schedule"
            )
        if not bool(get_memory().disable_radix_cache):
            raise RuntimeError(
                "--enable-flashinfer-agmm-true-sp requires --disable-radix-cache"
            )
        if int(get_schedule().chunked_prefill_size) != _FULL_ROWS:
            raise RuntimeError(
                "--enable-flashinfer-agmm-true-sp requires "
                f"--chunked-prefill-size {_FULL_ROWS}"
            )
        if int(get_schedule().max_running_requests) != 1:
            raise RuntimeError(
                "--enable-flashinfer-agmm-true-sp requires --max-running-requests 1"
            )
        if get_exec().kernel.attention_backend != "flashinfer":
            raise RuntimeError(
                "--enable-flashinfer-agmm-true-sp requires "
                "--attention-backend flashinfer"
            )
        return topology

    def _input_reason(
        self,
        input_ids: Any,
        positions: Any,
        forward_batch: Any,
        input_embeds: Any,
        pp_proxy_tensors: Any,
    ) -> Optional[str]:
        reason = _forward_mode_reason(forward_batch)
        if reason is not None:
            return reason
        if input_embeds is not None:
            return "input_embeds"
        if pp_proxy_tensors is not None:
            return "pp_proxy_tensors"
        torch = self._torch
        if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 1:
            return "input_ids"
        if not isinstance(positions, torch.Tensor) or positions.ndim != 1:
            return "positions"
        total_rows = int(input_ids.shape[0])
        if int(positions.shape[0]) != total_rows:
            return "position_rows"
        if _row_partition(total_rows, self._topology) is None:
            return "full_rows"
        batch_input_ids = getattr(forward_batch, "input_ids", None)
        if (
            not isinstance(batch_input_ids, torch.Tensor)
            or batch_input_ids.ndim != 1
            or int(batch_input_ids.shape[0]) != total_rows
        ):
            return "forward_batch_input_ids"
        if (
            not input_ids.is_cuda
            or not positions.is_cuda
            or not batch_input_ids.is_cuda
        ):
            return "input_device"
        return None

    def _runtime_group(self, model: Any):
        import torch.distributed as dist
        import torch.distributed._symmetric_memory as symm_mem

        from sglang.srt.distributed import get_tp_group

        coordinator = get_tp_group()
        group = coordinator.device_group
        rank = int(coordinator.rank_in_group)
        topology = self._topology
        if (
            int(coordinator.world_size) != topology.tp_size
            or tuple(coordinator.ranks) != topology.ranks
            or int(dist.get_world_size(group)) != topology.tp_size
            or int(dist.get_rank(group)) != rank
            or str(dist.get_backend(group)).lower() != "nccl"
        ):
            raise RuntimeError("FlashInfer AGMM true-SP found an invalid TP group")
        group_name = getattr(group, "group_name", None)
        if not group_name:
            raise RuntimeError("FlashInfer AGMM true-SP group has no stable name")
        device = model.layers[0].self_attn.qkv_proj.weight.device
        if tuple(self._torch.cuda.get_device_capability(device)) != (10, 3):
            raise RuntimeError("--enable-flashinfer-agmm-true-sp requires an SM103 GPU")
        if str(symm_mem.get_backend(device)).upper() != "NVSHMEM":
            raise RuntimeError(
                "--enable-flashinfer-agmm-true-sp requires the NVSHMEM backend"
            )
        if self._coordinator is None:
            symm_mem.enable_symm_mem_for_group(group_name)
            self._coordinator = coordinator
            self._group = group
            self._rank = rank
        elif (
            coordinator is not self._coordinator
            or group is not self._group
            or rank != self._rank
        ):
            raise RuntimeError("FlashInfer AGMM true-SP TP group changed")
        return coordinator, group, rank

    def _packed_qkv_weight(self, qkv: Any):
        key = id(qkv.weight)
        packed = self._packed_weights.get(key)
        if packed is None:
            packed = qkv.weight.t().contiguous()
            self._packed_weights[key] = packed
        if (
            tuple(packed.shape) != (8192, self._topology.packed_qkv_n)
            or not packed.is_contiguous()
        ):
            raise RuntimeError("FlashInfer AGMM packed-QKV weight changed")
        return packed

    def _prepared_qkv(self, inp: Any, qkv: Any, group: Any):
        weight = self._packed_qkv_weight(qkv)
        local_rows = int(inp.shape[0])
        key = (id(weight), local_rows)
        binding = self._bindings.get(key)
        if binding is None:
            launcher = self._prepare_all_gather_matmul(
                inp,
                weight,
                group,
                backend="auto",
                verbose=False,
            )
            if not callable(launcher):
                raise RuntimeError("FlashInfer AGMM preparation returned no launcher")
            binding = _PreparedBinding(weight, group, local_rows, launcher)
            self._bindings[key] = binding
        elif binding.weight is not weight or binding.group is not group:
            raise RuntimeError("FlashInfer AGMM prepared binding changed")
        return binding.launcher(inp)

    def _all_gather_rows(self, coordinator: Any, local: Any):
        if local.ndim != 2 or not local.is_contiguous():
            raise RuntimeError("AGMM all-gather input must be contiguous and 2D")
        output = local.new_empty(
            local.shape[0] * self._topology.tp_size, local.shape[1]
        )
        coordinator.all_gather_into_tensor(output, local)
        return output

    def _reduce_scatter_rows(self, coordinator: Any, partial: Any):
        if partial.ndim != 2 or not partial.is_contiguous():
            raise RuntimeError("AGMM reduce-scatter input must be contiguous and 2D")
        if partial.shape[0] % self._topology.tp_size:
            raise RuntimeError(
                "AGMM reduce-scatter row count is not divisible by the TP size"
            )
        output = partial.new_empty(
            partial.shape[0] // self._topology.tp_size, partial.shape[1]
        )
        coordinator.reduce_scatter_tensor(output, partial)
        return output

    def _decoder_layer_forward(
        self,
        layer: Any,
        positions: Any,
        hidden_states: Any,
        residual: Any,
        forward_batch: Any,
        coordinator: Any,
        group: Any,
    ):
        topology = self._topology
        if tuple(hidden_states.shape) != (topology.local_rows, 8192):
            raise RuntimeError("AGMM decoder-layer input shape changed")
        if residual is not None and residual.shape != hidden_states.shape:
            raise RuntimeError("AGMM hidden and residual row shards disagree")

        qkv_proj = layer.self_attn.qkv_proj
        if residual is None:
            residual = hidden_states
            normalized = layer.input_layernorm(hidden_states, quant_linear=qkv_proj)
        else:
            normalized, residual = layer.input_layernorm(
                hidden_states, residual, quant_linear=qkv_proj
            )
        if not normalized.is_contiguous():
            raise RuntimeError("AGMM input RMSNorm produced noncontiguous rows")
        qkv = self._prepared_qkv(normalized, qkv_proj, group)
        q, k, v = qkv.split(
            [topology.q_size, topology.kv_size, topology.kv_size], dim=-1
        )
        q, k = layer.self_attn.rotary_emb(positions, q, k)
        attention_output = layer.self_attn.attn(q, k, v, forward_batch)
        o_partial, _ = layer.self_attn.o_proj(
            attention_output,
            skip_all_reduce=True,
            forward_batch=forward_batch,
        )
        local_attention = self._reduce_scatter_rows(coordinator, o_partial)

        mlp_local, residual = layer.post_attention_layernorm(
            local_attention,
            residual,
            quant_linear=layer.mlp.gate_up_proj,
        )
        if not mlp_local.is_contiguous():
            raise RuntimeError(
                "AGMM post-attention RMSNorm produced noncontiguous rows"
            )
        mlp_full = self._all_gather_rows(coordinator, mlp_local)
        gate_up, _ = layer.mlp.gate_up_proj(mlp_full)
        activated = layer.mlp.act_fn(gate_up)
        down_partial, _ = layer.mlp.down_proj(
            activated,
            skip_all_reduce=True,
            forward_batch=forward_batch,
        )
        local_output = self._reduce_scatter_rows(coordinator, down_partial)
        return local_output, residual

    def maybe_forward(
        self,
        model: Any,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> Optional[torch.Tensor]:
        reason = self._input_reason(
            input_ids,
            positions,
            forward_batch,
            input_embeds,
            pp_proxy_tensors,
        )
        if reason is not None:
            return None
        if id(model) != self._model_id:
            raise RuntimeError("FlashInfer AGMM true-SP model owner changed")
        if getattr(model, _MODEL_TOKEN_ATTRIBUTE, None) is not self._model_token:
            raise RuntimeError("FlashInfer AGMM true-SP model ownership changed")
        coordinator, group, rank = self._runtime_group(model)
        replicated = model.embed_tokens(input_ids)
        if (
            tuple(replicated.shape) != (_FULL_ROWS, 8192)
            or replicated.dtype != self._torch.bfloat16
            or not replicated.is_cuda
            or not replicated.is_contiguous()
        ):
            raise RuntimeError("FlashInfer AGMM embedding contract changed")
        hidden_states = replicated.narrow(
            0,
            rank * self._topology.local_rows,
            self._topology.local_rows,
        ).contiguous()
        residual = None
        for layer_index in range(80):
            hidden_states, residual = self._decoder_layer_forward(
                model.layers[layer_index],
                positions,
                hidden_states,
                residual,
                forward_batch,
                coordinator,
                group,
            )
        local_final, _ = model.norm(hidden_states, residual)
        output = self._all_gather_rows(coordinator, local_final)
        if tuple(output.shape) != (_FULL_ROWS, 8192):
            raise RuntimeError("FlashInfer AGMM final output shape changed")
        return output


__all__ = ["LlamaFlashInferAgmmTrueSP"]
