# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
from typing import Callable, Iterable, List, Optional, Tuple, Type, Union

import numpy as np
import torch
from mindspore import Parameter, Tensor, dtype, from_numpy, mint, nn, ops
from mindspore.ops.auto_generate import (
    FusedAddTopKDiv,
    GroupedMatmulV4,
    MoeDistributeCombine,
    MoeDistributeDispatch,
    MoeInitRoutingV2,
    MoeTokenUnpermute,
)

from sglang.srt.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from sglang.srt.models.mindspore_models.utils import (
    _get_tp_group_name,
    _get_world_group_name,
    split_loaded_weight,
    tensor_torch2ms,
)


def fused_topk(
    hidden_states: Tensor,
    gating_output: Tensor,
    topk: int,
    renormalize: bool,
    indices_type=None,
) -> Tuple[Tensor, Tensor]:
    score = mint.softmax(gating_output, dim=-1)
    topk_weights, topk_ids = mint.topk(score, k=topk, dim=-1)
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    if indices_type is not None:
        topk_ids = topk_ids.to(indices_type)
    return topk_weights, topk_ids


def grouped_topk(
    hidden_states: Tensor,
    gating_output: Tensor,
    topk: int,
    renormalize: bool,
    num_expert_group: int = 0,
    topk_group: int = 0,
    topk_in_group: int = 2,
    routed_scaling_factor: float = 2.5,
    scoring_func: str = "softmax",
    e_score_correction_bias: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    fused_add_topk_div = FusedAddTopKDiv()
    scoring_type = 0 if scoring_func == "sigmoid" else -1
    gating_output = gating_output.to(dtype.float32)
    topk_weights, topk_ids = fused_add_topk_div(
        gating_output,
        e_score_correction_bias,
        num_expert_group,
        topk_group,
        topk_in_group,
        topk,
        scoring_type,
        renormalize,
        routed_scaling_factor,
    )
    return topk_weights, topk_ids


class FusedExperts(nn.Cell):
    def __init__(
        self,
        num_experts: int,
        num_local_experts: int,
        ep_size: int,
        ep_rank: int,
        dp_size: int,
        dp_rank: int,
        tp_size: int,
        tp_rank: int,
        pure_tp: bool,
        tp_ep: bool,
        optim_tp_ep_gating_perf: bool,
        use_all2all_kernels: bool,
    ) -> None:
        super().__init__()

        self.group_matmul_op = GroupedMatmulV4()
        self.moe_init_routing_op = MoeInitRoutingV2()
        self.moe_token_unpermute = MoeTokenUnpermute()

        self.pure_tp = True
        self.pure_ep = False
        self.tp_ep = False

        self.experts_num = num_experts
        self.local_experts_num = num_local_experts
        self.ep_size = ep_size
        self.ep_rank = ep_rank
        self.dp_size = dp_size
        self.dp_rank = dp_rank
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.optim_tp_ep_gating_perf = optim_tp_ep_gating_perf

        if self.ep_size > 1:
            self.experts_num_map = [
                (self.experts_num // self.ep_size) for _ in range(self.ep_size - 1)
            ]
            self.experts_num_map.append(
                self.experts_num
                - ((self.experts_num // self.ep_size) * (self.ep_size - 1))
            )
            self.ep_group = "ep_group"

        if self.ep_size > 1 and self.tp_size == 1:
            self.pure_ep = True

            self.use_all2all_kernels = use_all2all_kernels
        elif self.ep_size == 1 and self.tp_size >= 1:
            self.pure_tp = True
        else:
            self.tp_ep = True
            experts_num_map_np = np.array(self.experts_num_map, dtype=np.int32)
            experts_num_map_cu_np = np.cumsum(experts_num_map_np, dtype=np.int32)
            self.expert_start_index = (
                0 if self.ep_rank == 0 else int(experts_num_map_cu_np[self.ep_rank - 1])
            )

    def construct(
        self,
        hidden_states: Tensor,
        w1: Tensor,
        w2: Tensor,
        topk_weights: Tensor,
        topk_ids: Tensor,
        activation: str = "silu",
        global_num_experts: int = 1,
        apply_router_weight_on_input: bool = False,
    ) -> Tensor:
        if self.pure_tp:
            hidden_states = self.run_tp_moe(
                hidden_states,
                w1,
                w2,
                topk_ids,
                topk_weights,
                activation,
                global_num_experts,
                apply_router_weight_on_input,
            )
        elif self.pure_ep:
            hidden_states = self.run_ep_moe(
                hidden_states,
                w1,
                w2,
                topk_ids,
                topk_weights,
                activation,
                global_num_experts,
                apply_router_weight_on_input,
            )
        else:
            hidden_states = self.run_tp_ep_moe(
                hidden_states,
                w1,
                w2,
                topk_ids,
                topk_weights,
                activation,
                global_num_experts,
                apply_router_weight_on_input,
            )
        return hidden_states

    def _gate_activation(self, gate, activation):
        if activation == "silu":
            return mint.nn.functional.silu(gate)
        elif activation == "gelu":
            return mint.nn.functional.gelu(gate)
        else:
            raise ValueError(f"unsupported activation function: {activation}")

    def _group_matmul(self, hidden_states, weight, group_list):
        return self.group_matmul_op(
            [hidden_states],
            [weight],
            None,
            None,
            None,
            None,
            None,
            None,
            group_list,
            split_item=3,
            group_type=0,
            group_list_type=1,
        )[0]

    def _ffn(self, hidden_states, w1, w2, group_list, activation):
        gate_hidden_out = self._group_matmul(
            hidden_states=hidden_states, weight=w1, group_list=group_list
        )
        gate, hidden = mint.split(
            gate_hidden_out, (w1.shape[2] // 2, w1.shape[2] // 2), -1
        )
        gate = self._gate_activation(gate=gate, activation=activation)
        hidden = mint.mul(hidden, gate)
        expert_output = self._group_matmul(
            hidden_states=hidden, weight=w2, group_list=group_list
        )
        expert_output = mint.nan_to_num(expert_output, 0, 0, 0)
        return expert_output

    def run_tp_moe(
        self,
        hidden_states: Tensor,
        w1: Tensor,
        w2: Tensor,
        topk_ids: Tensor,
        topk_weights: Tensor,
        activation: str = "silu",
        global_num_experts: int = 1,
        apply_router_weight_on_input: bool = False,
    ) -> Tensor:
        topk_weights = topk_weights.astype(hidden_states.dtype)
        topk_ids = topk_ids.astype(dtype.int32)

        sorted_input_tensor, unsort_map, group_list, _ = self.moe_init_routing_op(
            hidden_states,
            topk_ids,
            active_num=0,
            expert_capacity=0,
            expert_num=global_num_experts,
            drop_pad_mode=0,
            expert_tokens_count_or_cumsum_flag=2,
            expert_tokens_before_capacity_flag=True,
        )
        group_list = group_list.astype(dtype.int64)
        expert_output = self._ffn(
            sorted_input_tensor,
            w1=w1,
            w2=w2,
            group_list=group_list,
            activation=activation,
        )

        moe_output = self.moe_token_unpermute(
            permuted_tokens=expert_output,
            sorted_indices=unsort_map,
            probs=topk_weights,
            padded_mode=False,
            restore_shape=None,
        )
        return moe_output

    def run_tp_ep_moe(
        self,
        hidden_states: Tensor,
        w1: Tensor,
        w2: Tensor,
        topk_ids: Tensor,
        topk_weights: Tensor,
        activation: str = "silu",
        global_num_experts: int = 1,
        apply_router_weight_on_input: bool = False,
    ) -> Tensor:
        topk_weights = topk_weights.astype(hidden_states.dtype)
        topk_ids = topk_ids.astype(dtype=dtype.int32)

        if self.dp_size > 1 or not self.optim_tp_ep_gating_perf:
            topk_mask = topk_ids < self.expert_start_index
            local_topk_ids = topk_ids - self.expert_start_index
            local_topk_ids = local_topk_ids.astype(dtype.in32)

            local_topk_ids = ops.masked_fill(
                local_topk_ids, topk_mask, self.experts_num - 1
            )
        else:
            local_topk_ids = topk_ids

        weight_mask = local_topk_ids >= self.local_experts_num
        topk_weights = ops.masked_fill(topk_weights, weight_mask, 0)

        sorted_input_tensor, unsort_map, group_list, _ = self.moe_init_routing_op(
            hidden_states,
            local_topk_ids,
            active_num=0,
            expert_capacity=0,
            expert_num=global_num_experts,
            drop_pad_mode=0,
            expert_tokens_count_or_cumsum_flag=2,
            expert_tokens_before_capacity_flag=True,
        )

        group_list = group_list[: self.local_experts_num]
        group_list = group_list.astype(dtype.int64)
        expert_output = self._ffn(
            sorted_input_tensor,
            w1=w1,
            w2=w2,
            group_list=group_list,
            activation=activation,
        )
        moe_output = self.moe_token_unpermute(
            permuted_tokens=expert_output,
            sorted_indices=unsort_map,
            probs=topk_weights,
            padded_mode=False,
            restore_shape=None,
        )
        return moe_output

    def run_ep_moe(
        self,
        hidden_states: Tensor,
        w1: Tensor,
        w2: Tensor,
        topk_ids: Tensor,
        topk_weights: Tensor,
        activation: str = "silu",
        global_num_experts: int = 1,
        apply_router_weight_on_input: bool = False,
    ) -> Tensor:
        topk_weights = topk_weights.astype(hidden_states.dtype)
        topk_ids = topk_ids.astype(dtype=dtype.int32)

        return

    def _ep_with_dispatch_combine(
        self,
        hidden_states,
        w1,
        w2,
        topk_ids,
        topk_weights,
        activation,
        global_num_experts,
        apply_router_weight_on_input,
    ):
        raise NotImplementedError(
            "ep parallel with dispatch and combine is not implemented."
        )


class FusedMoe(nn.Cell):
    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        *,
        param_dtype: Optional[Type] = None,
        reduce_results: bool = False,
        renormalize: bool = True,
        use_grouped_topk: bool = False,
        num_expert_group: Optional[int] = None,
        topk_group: Optional[int] = None,
        tp_size: Optional[int] = None,
        ep_size: Optional[int] = None,
        dp_size: Optional[int] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        e_score_correction_bias: Optional[Tensor] = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
        num_redundant_experts: int = 0,
        optim_tp_ep_gating_perf: bool = False,
    ):
        super().__init__()

        assert apply_router_weight_on_input == False
        self.param_dtype = param_dtype
        self.global_num_experts = num_experts + num_redundant_experts
        self.local_num_experts, self.expert_map = (self.global_num_experts, None)

        assert ep_size == 1
        self.ep_size = ep_size
        self.ep_rank = 0
        assert dp_size == 1
        self.dp_size = dp_size
        self.dp_rank = 0
        self.tp_size = tp_size
        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_ep = False
        self.pure_tp = True
        self.pure_ep = False
        self.optim_tp_ep_gating_perf = optim_tp_ep_gating_perf and self.tp_ep
        self.expert_start_index = 0
        self.expert_end_index = self.global_num_experts
        self.top_k = top_k
        assert intermediate_size % self.tp_size == 0
        self.hidden_size = hidden_size
        self.intermediate_size_per_partition = intermediate_size // self.tp_size
        self.reduce_results = reduce_results
        self.renormalize = renormalize
        self.use_grouped_topk = use_grouped_topk
        if self.use_grouped_topk:
            assert num_expert_group is not None and topk_group is not None
        self.num_expert_group = num_expert_group
        self.topk_group = topk_group
        self.scoring_func = scoring_func
        self.apply_router_weight_on_input = apply_router_weight_on_input
        self.activation = activation
        assert self.scoring_func == "softmax" or self.use_grouped_topk
        self.tp_group = _get_tp_group_name()
        self.custom_routing_function = custom_routing_function
        self.e_score_correction_bias = e_score_correction_bias

        self.w13_weight = Parameter(
            mint.empty(
                self.global_num_experts,
                self.hidden_size,
                2 * self.intermediate_size_per_partition,
                dtype=self.param_dtype,
            ),
            requires_grad=False,
        )
        setattr(self.w13_weight, "weight_load", self.weight_load)
        setattr(self.w13_weight, "is_transpose", True)

        self.w2_weight = Parameter(
            mint.empty(
                self.global_num_experts,
                self.intermediate_size_per_partition,
                self.hidden_size,
                dtype=self.param_dtype,
            ),
            requires_grad=False,
        )
        setattr(self.w2_weight, "weight_load", self.weight_load)
        setattr(self.w2_weight, "is_transpose", True)

        self.all_reduce_from_tp_group = ops.AllReduce(group=self.tp_group)
        self.fused_experts = FusedExperts(
            num_experts=self.global_num_experts,
            num_local_experts=self.local_num_experts,
            ep_size=self.ep_size,
            ep_rank=self.ep_rank,
            dp_size=self.dp_size,
            dp_rank=self.dp_rank,
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
            pure_tp=self.pure_tp,
            tp_ep=self.tp_ep,
            optim_tp_ep_gating_perf=self.optim_tp_ep_gating_perf,
            use_all2all_kernels=False,
        )

    def _load_w13(
        self,
        param: Parameter,
        shard_dim: int,
        shard_id: str,
        loaded_weight,
        expert_id: int,
        tp_rank: int,
    ):
        is_param_transpose = (
            param.is_transpose if hasattr(param, "is_transpose") else False
        )

        if is_param_transpose:
            shard_size = param.shape[-1] // 2
        else:
            shard_size = param.shape[-2] // 2

        loaded_weight = split_loaded_weight(
            loaded_weight=loaded_weight,
            shard_dim=shard_dim,
            start_idx=shard_size * tp_rank,
            shard_size=shard_size,
        )

        if is_param_transpose:
            loaded_weight = from_numpy(loaded_weight.swapaxes(-1, -2))
        else:
            loaded_weight = from_numpy(loaded_weight)

        if shard_id == "w1":
            if is_param_transpose:
                param[expert_id, :, 0:shard_size] = loaded_weight
            else:
                param[expert_id, 0:shard_size, :] = loaded_weight
        else:
            assert shard_id == "w3"
            if is_param_transpose:
                param[expert_id, :, shard_size : shard_size * 2] = loaded_weight
            else:
                param[expert_id, shard_size : shard_size * 2, :] = loaded_weight

    def _load_w2(
        self,
        param: Parameter,
        shard_dim: int,
        loaded_weight,
        tp_rank: int,
        expert_id: int,
        load_full: bool = False,
    ):
        is_param_transpose = (
            param.is_transpose if hasattr(param, "is_transpose") else False
        )

        if not load_full:
            if is_param_transpose:
                shard_size = param.shape[-2]
            else:
                shard_size = param.shape[-1]

            loaded_weight = split_loaded_weight(
                loaded_weight=loaded_weight,
                shard_dim=shard_dim,
                start_idx=shard_size * tp_rank,
                shard_size=shard_size,
            )

            if is_param_transpose:
                loaded_weight = from_numpy(loaded_weight.swapaxes(-1, -2))
            else:
                loaded_weight = from_numpy(loaded_weight)

            param[expert_id] = loaded_weight
        else:
            if is_param_transpose:
                loaded_weight = from_numpy(loaded_weight.swapaxes(-1, -2))
            else:
                loaded_weight = from_numpy(loaded_weight)

            param.set_data(loaded_weight)

    def _load_model_weight_or_group_weight_scale(
        self,
        shard_dim: int,
        param: Parameter,
        shard_id: str,
        loaded_weight,
        tp_rank: int,
        expert_id: int,
        load_full_w2: bool = False,
    ):
        if shard_id == "w2":
            self._load_w2(
                shard_dim=shard_dim,
                param=param,
                loaded_weight=loaded_weight,
                tp_rank=tp_rank,
                expert_id=expert_id,
                load_full=load_full_w2,
            )
        elif shard_id in ("w1", "w3"):
            self._load_w13(
                shard_id=shard_id,
                shard_dim=shard_dim,
                param=param,
                loaded_weight=loaded_weight,
                expert_id=expert_id,
                tp_rank=tp_rank,
            )

    def _load_single_value(self, param: Parameter, loaded_weight, expert_id: int):
        is_param_transpose = (
            param.is_transpose if hasattr(param, "is_transpose") else False
        )
        loaded_weight = loaded_weight[:]
        if is_param_transpose:
            loaded_weight = from_numpy(loaded_weight.swapaxes(-1, -2))
        else:
            loaded_weight = from_numpy(loaded_weight)
        param[expert_id] = loaded_weight

    def _load_g_idx(
        self,
        shard_id: str,
        param: Parameter,
        shard_dim: int,
        loaded_weight,
        tp_rank: int,
        expert_id: int,
    ):
        if shard_id == "w2":
            self._load_w2(
                shard_dim=shard_dim,
                loaded_weight=loaded_weight,
                param=param,
                expert_id=expert_id,
                tp_rank=tp_rank,
            )
        else:
            assert shard_id in ("w1", "w3")
            is_param_transpose = (
                param.is_transpose if hasattr(param, "is_transpose") else False
            )
            loaded_weight = loaded_weight[:]
            if is_param_transpose:
                loaded_weight = from_numpy(loaded_weight.swapaxes(-1, -2))
            else:
                loaded_weight = from_numpy(loaded_weight)
            param[expert_id] = loaded_weight

    def _map_global_expert_id_to_local_expert_id(self, expert_id: int) -> int:
        if self.expert_map is None:
            return expert_id
        return self.expert_map[expert_id].item()

    def weight_load(
        self,
        param: Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str,
        expert_id: int,
    ) -> None:
        loaded_weight = loaded_weight.contiguous().to(torch.float32).numpy()
        expert_id = self._map_global_expert_id_to_local_expert_id(expert_id=expert_id)
        if expert_id == -1:
            return

        assert shard_id in ("w1", "w2", "w3")

        SHARD_ID_TO_SHARD_DIM = {"w1": 0, "w2": 1, "w3": 0}
        shard_dim = SHARD_ID_TO_SHARD_DIM[shard_id]

        if "g_idx" in weight_name:
            self._load_g_idx(
                shard_dim=0,
                shard_id=shard_id,
                loaded_weight=loaded_weight,
                param=param,
                tp_rank=self.tp_rank,
                expert_id=expert_id,
            )
            return

        if "weight_shape" in weight_name:
            self._load_single_value(
                param=param, loaded_weight=loaded_weight, expert_id=expert_id
            )
            return

        if "weight" in weight_name:
            self._load_model_weight_or_group_weight_scale(
                shard_id=shard_id,
                shard_dim=shard_dim,
                loaded_weight=loaded_weight,
                param=param,
                expert_id=expert_id,
                tp_rank=self.tp_rank,
            )
            return

    @classmethod
    def make_expert_params_mapping(
        cls,
        ckpt_gate_proj_name: str,
        ckpt_down_proj_name: str,
        ckpt_up_proj_name: str,
        num_experts: int,
    ) -> List[Tuple[str, str, int, str]]:
        return [
            (
                (
                    "experts.w13_"
                    if weight_name in [ckpt_gate_proj_name, ckpt_up_proj_name]
                    else "experts.w2_"
                ),
                f"experts.{expert_id}.{weight_name}.",
                expert_id,
                shard_id,
            )
            for expert_id in range(num_experts)
            for shard_id, weight_name in [
                ("w1", ckpt_gate_proj_name),
                ("w2", ckpt_down_proj_name),
                ("w3", ckpt_up_proj_name),
            ]
        ]

    @staticmethod
    def select_experts(
        hidden_states: Tensor,
        router_logits: Tensor,
        top_k: int,
        use_grouped_topk: bool,
        renormalize: bool,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        custom_routing_functions: Optional[Callable] = None,
        scoring_func: str = "softmax",
        e_score_correction_bias: Optional[Tensor] = None,
        indices_type=None,
    ):
        if use_grouped_topk:
            assert topk_group is not None
            assert num_expert_group is not None
            topk_weights, topk_ids = grouped_topk(
                hidden_states=hidden_states,
                gating_output=router_logits,
                topk=top_k,
                renormalize=renormalize,
                num_expert_group=num_expert_group,
                topk_group=topk_group,
                scoring_func=scoring_func,
                e_score_correction_bias=e_score_correction_bias,
            )
            if indices_type is not None:
                topk_ids = topk_ids.to(dtype=indices_type)
        else:
            topk_weights, topk_ids = fused_topk(
                hidden_states=hidden_states,
                gating_output=router_logits,
                topk=top_k,
                renormalize=renormalize,
                indices_type=indices_type,
            )

        return topk_weights, topk_ids

    def must_reduce_shared_expert_outputs(self) -> bool:
        # If dp_size == 1, means routed expert use the same tensor parallel group as shared expert.
        # And meanwhile if ep_size == 1, it means using tensor parallel to compute routed expert.
        # So we can delay the shared expert outputs reduce after the routed expert and
        # the shared expert are added.
        return not (self.pure_tp and self.dp_size == 1)

    def maybe_all_reduce_tensor_model_parallel(self, final_hidden_states: Tensor):
        if self.pure_tp or self.tp_ep:
            return self.all_reduce_from_tp_group(final_hidden_states)
        return final_hidden_states

    def construct(
        self,
        hidden_states: Tensor,
        router_logits: Tensor,
    ) -> Tensor:
        topk_weights, topk_ids = FusedMoe.select_experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
            use_grouped_topk=self.use_grouped_topk,
            top_k=self.top_k,
            renormalize=self.renormalize,
            topk_group=self.topk_group,
            num_expert_group=self.num_expert_group,
            custom_routing_functions=self.custom_routing_function,
            scoring_func=self.scoring_func,
            e_score_correction_bias=self.e_score_correction_bias,
            indices_type=None,
        )

        final_hidden_states = self.fused_experts(
            hidden_states=hidden_states,
            w1=self.w13_weight,
            w2=self.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=self.activation,
            global_num_experts=self.global_num_experts,
            apply_router_weight_on_input=self.apply_router_weight_on_input,
        )

        return final_hidden_states
