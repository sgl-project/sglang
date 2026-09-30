# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2025 SGLang Team

import inspect
import logging
from collections.abc import Iterable
from typing import List, Optional, Tuple
from weakref import ref

import torch
import torch.nn.functional as F
from torch import nn

from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.eplb.expert_location import ModelConfigForExpertLocation
from sglang.srt.eplb.expert_location_dispatch import ExpertLocationDispatchInfo
from sglang.srt.layers.moe import get_moe_runner_backend, post_experts_all_reduce
from sglang.srt.layers.moe.utils import filter_moe_weight_param_global_expert
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.runtime_context import get_exec, get_parallel
from sglang.srt.utils.common import direct_register_custom_op

from .execution_context import (
    get_execution_layer,
    get_transformers_execution_context,
    register_execution_layer,
)
from .utils import _getattr_first, log_replacement, maybe_prefix

logger = logging.getLogger(__name__)


def native_router_contract(module: nn.Module) -> str | None:
    contracts = {
        (
            "transformers.models.qwen2_moe.modeling_qwen2_moe",
            "Qwen2MoeSparseMoeBlock",
        ): "qwen2",
        (
            "transformers.models.qwen3_moe.modeling_qwen3_moe",
            "Qwen3MoeSparseMoeBlock",
        ): "qwen3",
        (
            "transformers.models.deepseek_v3.modeling_deepseek_v3",
            "DeepseekV3MoE",
        ): "deepseek_v3",
    }
    contract = contracts.get((type(module).__module__, type(module).__name__))
    gate = getattr(module, "gate", None)
    if contract is None or gate is None or not hasattr(module, "experts"):
        return None
    weight = getattr(gate, "weight", None)
    if weight is None or weight.ndim != 2 or getattr(gate, "bias", None) is not None:
        return None
    if not all(
        hasattr(gate, key) for key in ("top_k", "num_experts", "norm_topk_prob")
    ):
        return None
    expected = {"gate", "experts"}
    if contract == "qwen2":
        expected |= {"shared_expert", "shared_expert_gate"}
    elif contract == "deepseek_v3":
        expected |= {"shared_experts"}
        if not all(
            hasattr(gate, key)
            for key in (
                "e_score_correction_bias",
                "num_group",
                "topk_group",
                "routed_scaling_factor",
            )
        ):
            return None
    return contract if set(module._modules) == expected else None


def native_routing_method_type(module: nn.Module, contract: str | None):
    if contract is None:
        return None
    from sglang.srt.layers.moe.utils import RoutingMethodType

    if contract == "deepseek_v3":
        return RoutingMethodType.DeepSeekV3
    if module.gate.norm_topk_prob:
        return RoutingMethodType.Renormalize
    return RoutingMethodType.Default


class TransformersNativeMoE(nn.Module):
    def __init__(self, original: nn.Module, experts, contract: str):
        super().__init__()
        self.gate = original.gate
        self.experts = experts
        self.contract = contract
        for name in ("shared_expert", "shared_experts", "shared_expert_gate"):
            if hasattr(original, name):
                self.add_module(name, getattr(original, name))
        self.experts.configure_router(self.gate, contract)

    @property
    def topk(self):
        return self.experts.native_topk

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        shape = hidden_states.shape
        flat = hidden_states.reshape(-1, shape[-1])
        if self.contract == "deepseek_v3":
            router_logits = F.linear(flat.float(), self.gate.weight.float())
        else:
            router_logits = F.linear(flat, self.gate.weight)
        output = self.experts.forward_router(flat, router_logits)
        if self.contract == "qwen2":
            shared = self.shared_expert(flat)
            shared = torch.sigmoid(self.shared_expert_gate(flat)) * shared
            output = output + shared
        elif self.contract == "deepseek_v3":
            output = output + self.shared_experts(flat)
        return output.reshape(shape)


class TransformersFusedMoE(nn.Module):
    def __init__(
        self,
        *,
        num_experts,
        top_k,
        hidden_size,
        intermediate_size,
        layer_id,
        quant_config,
        prefix,
        activation,
        with_bias,
        expert_mapping,
        routing_method_type=None,
    ):
        super().__init__()
        from sglang.srt.layers.moe.ep_moe.layer import DeepEPMoE, get_moe_impl_class
        from sglang.srt.layers.moe.topk import TopKConfig

        experts_cls = get_moe_impl_class(quant_config)
        self.experts = experts_cls(
            num_experts=num_experts + get_exec().moe.ep_num_redundant_experts,
            top_k=top_k,
            layer_id=layer_id,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            reduce_results=False,
            quant_config=quant_config,
            activation=activation,
            with_bias=with_bias,
            prefix=prefix,
            routing_method_type=routing_method_type,
        )
        self._combine_completes_output = isinstance(self.experts, DeepEPMoE)
        if self._combine_completes_output and self.tp_size != 1:
            raise ValueError("Transformers all-to-all MoE requires moe_tp_size=1")
        self.layer_name = prefix
        self.num_experts = num_experts
        self.top_k = top_k
        self._expert_mapping = expert_mapping
        self._loaded_expert_shards = set()
        self._execution_handle = register_execution_layer(self)
        self.topk_config = TopKConfig(top_k=top_k)
        self.native_topk = None
        self._gate_ref = None

    @property
    def tp_size(self):
        return getattr(self.experts, "moe_tp_size", 1)

    @property
    def ep_size(self):
        return getattr(self.experts, "moe_ep_size", 1)

    def configure_router(self, gate, contract):
        from sglang.srt.layers.moe.topk import TopK

        self._gate_ref = ref(gate)
        grouped = contract == "deepseek_v3"
        self.native_topk = TopK(
            top_k=gate.top_k,
            layer_id=self.experts.layer_id,
            renormalize=gate.norm_topk_prob,
            use_grouped_topk=grouped,
            num_expert_group=gate.num_group if grouped else None,
            topk_group=gate.topk_group if grouped else None,
            scoring_func="sigmoid" if grouped else "softmax",
            correction_bias=gate.e_score_correction_bias if grouped else None,
            routed_scaling_factor=gate.routed_scaling_factor if grouped else None,
        )

    def complete_output(self, output):
        if self._combine_completes_output:
            return output
        return post_experts_all_reduce(output)

    def get_expert_weights(self):
        return getattr(self.experts, "get_expert_weights", lambda: None)()

    def get_moe_weights(self):
        num_local = getattr(self.experts, "num_local_experts", self.num_experts)
        return [
            p.data
            for name, p in self.experts.named_parameters()
            if name != "correction_bias"
            and filter_moe_weight_param_global_expert(name, p, num_local)
        ]

    def forward(self, hidden_states, topk_ids, topk_weights, **kwargs):
        return torch.ops.sglang.transformers_moe_forward(
            hidden_states,
            topk_ids.to(torch.int32),
            topk_weights.to(torch.float32),
            self._execution_handle,
        )

    def forward_router(self, hidden_states, router_logits):
        return torch.ops.sglang.transformers_native_moe_forward(
            hidden_states, router_logits, self._execution_handle
        )

    def _load_expert(self, parameter, weight, name, shard_id, expert_id):
        loader = getattr(parameter, "weight_loader", None)
        if loader is None:
            raise ValueError(f"MoE parameter for {name!r} has no expert weight loader")
        loader(parameter, weight, name, shard_id=shard_id, expert_id=expert_id)

    def _record_expert_shard(self, parameter_name, expert_id, shard_id):
        if parameter_name not in (
            "experts.w13_weight",
            "experts.w2_weight",
            "experts.w13_qweight",
            "experts.w2_qweight",
        ):
            return
        if not hasattr(self, "_loaded_expert_shards"):
            self._loaded_expert_shards = set()
        self._loaded_expert_shards.add((expert_id, shard_id))

    def validate_loaded_weights(self):
        required = {
            (expert_id, shard)
            for expert_id in range(self.num_experts)
            for shard in ("w1", "w2", "w3")
        }
        missing = required - self._loaded_expert_shards
        if missing:
            raise ValueError(
                f"Incomplete expert weights for {self.layer_name}: missing {len(missing)} logical expert shards, including {sorted(missing)[:4]}"
            )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loaded = set()
        parameters = dict(self.named_parameters())
        for name, weight in weights:
            if name in (
                "gate_up_proj",
                "gate_up_proj.weight",
                "down_proj",
                "down_proj.weight",
            ):
                if weight.ndim != 3 or weight.shape[0] != self.num_experts:
                    raise ValueError(
                        f"Unexpected packed expert weight shape for {name}: {tuple(weight.shape)}"
                    )
                packed_gate = name.startswith("gate_up_proj")
                target_name = (
                    "experts.w13_weight" if packed_gate else "experts.w2_weight"
                )
                target = parameters[target_name]
                for expert_id, expert_weight in enumerate(weight.unbind(0)):
                    if packed_gate:
                        for shard_id, shard in zip(
                            ("w1", "w3"), expert_weight.chunk(2, dim=0)
                        ):
                            self._load_expert(target, shard, name, shard_id, expert_id)
                            self._record_expert_shard(target_name, expert_id, shard_id)
                    else:
                        self._load_expert(target, expert_weight, name, "w2", expert_id)
                        self._record_expert_shard(target_name, expert_id, "w2")
                loaded.update((name, target_name))
                continue
            matched = False
            for (
                parameter_name,
                weight_name,
                expert_id,
                shard_id,
            ) in self._expert_mapping:
                if not name.startswith(weight_name):
                    continue
                mapped_name = name.replace(weight_name, parameter_name, 1)
                if mapped_name not in parameters:
                    continue
                self._load_expert(
                    parameters[mapped_name], weight, name, shard_id, expert_id
                )
                self._record_expert_shard(mapped_name, expert_id, shard_id)
                loaded.add(mapped_name)
                matched = True
                break
            if not matched:
                direct_name = name if name in parameters else f"experts.{name}"
                if direct_name not in parameters:
                    raise ValueError(
                        f"Unrecognized MoE weight {self.layer_name}.{name}"
                    )
                parameter = parameters[direct_name]
                getattr(parameter, "weight_loader", default_weight_loader)(
                    parameter, weight
                )
                loaded.add(direct_name)
            loaded.add(name)
        return loaded


def _dispatch_info(layer_id):
    if get_exec().moe.ep_dispatch_algorithm is None:
        return None
    return ExpertLocationDispatchInfo.init_new(layer_id)


def _transformers_moe_forward(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    layer_handle: str,
) -> torch.Tensor:
    from sglang.srt.layers.moe.topk import StandardTopKOutput, _post_process_topk_ids

    layer = get_execution_layer(layer_handle)
    context = get_transformers_execution_context()
    recorder = get_global_expert_distribution_recorder()
    with recorder.with_current_layer(layer.experts.layer_id):
        ids, weights, recorder_ids = _post_process_topk_ids(
            topk_ids=topk_ids.clone(),
            topk_weights=topk_weights.clone(),
            topk_config=layer.topk_config,
            router_logits=hidden_states.new_empty((hidden_states.shape[0], 0)),
            layer_id=layer.experts.layer_id,
            num_token_non_padded=context.num_token_non_padded(),
            expert_location_dispatch_info=_dispatch_info(layer.experts.layer_id),
        )
        if recorder_ids is not None:
            recorder.on_select_experts(topk_ids=recorder_ids)
        topk_output = StandardTopKOutput(weights, ids, None)
        output = layer.experts(hidden_states.clone(), topk_output)
    return layer.complete_output(output)


def _transformers_native_moe_forward(
    hidden_states: torch.Tensor, router_logits: torch.Tensor, layer_handle: str
) -> torch.Tensor:
    layer = get_execution_layer(layer_handle)
    context = get_transformers_execution_context()
    gate = layer._gate_ref()
    if hasattr(gate, "e_score_correction_bias"):
        layer.native_topk.topk_config.correction_bias = gate.e_score_correction_bias
    recorder = get_global_expert_distribution_recorder()
    with recorder.with_current_layer(layer.experts.layer_id):
        if hidden_states.shape[0]:
            topk_output = layer.native_topk(
                hidden_states,
                router_logits,
                num_token_non_padded=context.num_token_non_padded(),
                expert_location_dispatch_info=_dispatch_info(layer.experts.layer_id),
            )
        else:
            topk_output = layer.native_topk.empty_topk_output(
                hidden_states.device, layer_id=layer.experts.layer_id
            )
        output = layer.experts(hidden_states.clone(), topk_output)
    return layer.complete_output(output)


def _transformers_moe_forward_fake(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    layer_handle: str,
) -> torch.Tensor:
    return torch.empty_like(hidden_states)


def _transformers_native_moe_forward_fake(
    hidden_states: torch.Tensor, router_logits: torch.Tensor, layer_handle: str
) -> torch.Tensor:
    return torch.empty_like(hidden_states)


_CPU_MOE_LIBRARY = torch.library.Library("sglang", "IMPL", "CPU")


for name, implementation, fake in (
    (
        "transformers_moe_forward",
        _transformers_moe_forward,
        _transformers_moe_forward_fake,
    ),
    (
        "transformers_native_moe_forward",
        _transformers_native_moe_forward,
        _transformers_native_moe_forward_fake,
    ),
):
    direct_register_custom_op(
        op_name=name, op_func=implementation, mutates_args=[], fake_impl=fake
    )
    if not torch._C._dispatch_has_kernel_for_dispatch_key(f"sglang::{name}", "CPU"):
        _CPU_MOE_LIBRARY.impl(name, implementation)
    from sglang.srt.compilation.compilation_config import SPLIT_OPS

    if f"sglang.{name}" not in SPLIT_OPS:
        SPLIT_OPS.append(f"sglang.{name}")


class MoEMixin:
    def _validate_expert_weights(self):
        if getattr(self, "_weights_loaded", False):
            return
        for layer in self.moe_layers:
            layer.validate_loaded_weights()

    @classmethod
    def get_model_config_for_expert_location(
        cls, config
    ) -> Optional[ModelConfigForExpertLocation]:
        text_config = getattr(config, "text_config", config)
        num_experts = _getattr_first(
            text_config, ("num_local_experts", "num_experts", "n_routed_experts")
        )
        if num_experts is None:
            return None
        num_groups = getattr(text_config, "n_group", None)
        return ModelConfigForExpertLocation(
            num_layers=text_config.num_hidden_layers,
            num_logical_experts=num_experts,
            num_groups=num_groups,
        )

    @property
    def routed_experts_weights_of_layer(self) -> dict[int, list[torch.Tensor]]:
        return {
            fused.experts.layer_id: fused.get_moe_weights() for fused in self.moe_layers
        }

    def _get_expert_mapping(self, num_experts: int) -> List[Tuple[str, str, int, str]]:
        from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE

        ckpt_names = [
            ("gate_proj", "down_proj", "up_proj"),
            ("w1", "w2", "w3"),
            ("linear", "linear_1", "linear_v"),
        ]
        mapping: list = []
        for gate, down, up in ckpt_names:
            mapping.extend(
                FusedMoE.make_expert_params_mapping(
                    ckpt_gate_proj_name=gate,
                    ckpt_down_proj_name=down,
                    ckpt_up_proj_name=up,
                    num_experts=num_experts,
                )
            )
        mapping = [
            (pn, wn.removeprefix("experts."), eid, sid) for pn, wn, eid, sid in mapping
        ]
        return mapping

    def recursive_replace(self):
        config = self.text_config
        num_experts = _getattr_first(
            config, ("num_local_experts", "num_experts", "n_routed_experts")
        )
        top_k = _getattr_first(config, ("num_experts_per_tok", "top_k"))
        intermediate_size = _getattr_first(
            config, ("moe_intermediate_size", "intermediate_size")
        )
        if not num_experts or not top_k or not intermediate_size:
            raise ValueError("Cannot determine MoE dimensions from the model config")
        self.mlp_moe_layers = []
        self.moe_layers = []
        self.num_moe_layers = 0
        self.num_logical_experts = num_experts
        self.num_redundant_experts = get_exec().moe.ep_num_redundant_experts
        self.num_physical_experts = num_experts + self.num_redundant_experts
        self.num_local_physical_experts = (
            self.num_physical_experts // get_parallel().moe_ep_size
        )
        self.num_shared_experts = _getattr_first(
            config, ("n_shared_experts", "moe_num_shared_experts"), 0
        )
        activation = getattr(config, "hidden_act", "silu")
        if activation not in ("silu", "gelu"):
            raise ValueError(
                f"Unsupported Transformers fused-expert activation: {activation}"
            )
        mappings = self._get_expert_mapping(num_experts)

        def replace(parent, prefix):
            for name, child in list(parent.named_children()):
                path = maybe_prefix(prefix, name)
                experts = getattr(child, "experts", None)
                if experts is None:
                    replace(child, path)
                    continue
                contract = native_router_contract(child)
                if contract is None:
                    if isinstance(experts, nn.ModuleList):
                        raise ValueError(
                            f"MoE layer {path} has no supported callable expert contract"
                        )
                    signature = inspect.signature(experts.forward)
                    if len(signature.parameters) < 3:
                        raise ValueError(
                            f"MoE layer {path} has no selected-expert forward contract"
                        )
                    backend = get_moe_runner_backend()
                    if (
                        backend.is_triton_kernels()
                        or backend.is_flashinfer_trtllm()
                        or backend.is_flashinfer_mxfp4()
                    ):
                        raise ValueError(
                            f"MoE layer {path} requires a STANDARD-output expert runner"
                        )
                layer_id = next(
                    (
                        int(part)
                        for part in reversed(path.split("."))
                        if part.isdecimal()
                    ),
                    self.num_moe_layers,
                )
                fused = TransformersFusedMoE(
                    num_experts=num_experts,
                    top_k=top_k,
                    hidden_size=config.hidden_size,
                    intermediate_size=intermediate_size,
                    layer_id=layer_id,
                    quant_config=self.quant_config,
                    prefix=path + ".experts",
                    activation=activation,
                    with_bias=any(
                        "bias" in key for key, _ in experts.named_parameters()
                    ),
                    expert_mapping=mappings,
                    routing_method_type=native_routing_method_type(child, contract),
                )
                child.experts = fused
                replacement = (
                    TransformersNativeMoE(child, fused, contract) if contract else child
                )
                setattr(parent, name, replacement)
                log_replacement(path, child, replacement)
                self.mlp_moe_layers.append(replacement)
                self.moe_layers.append(fused)
                self.num_moe_layers += 1

        replace(self.model, "model")
        if not self.moe_layers:
            raise ValueError(
                "No supported MoE layers were found in the Transformers model"
            )
        super().recursive_replace()
