# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Optional, Union

import torch
from torch import nn

from sglang.srt.custom_op import CustomOp
from sglang.srt.distributed import (
    divide,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.layers.linear import ColumnParallelLinear, RowParallelLinear
from sglang.srt.layers.quantization import QuantizationConfig
from sglang.srt.mem_cache.memory_pool import MambaPool
from sglang.srt.model_loader.weight_utils import (
    composed_weight_loader,
    sharded_weight_loader,
)
from sglang.srt.utils import set_weight_attrs

from .causal_conv1d_triton import causal_conv1d_fn, causal_conv1d_update
from .layernorm_gated import rms_norm_gated
from .mamba import extra_groups_for_head_shards, mamba_v2_sharded_weight_loader
from .mamba2_metadata import Mamba2Metadata
from .mamba_ssm import selective_state_update
from .ssd_combined import mamba_chunk_scan_combined

# Added by the IBM Team, 2024


# Adapted from transformers.models.mamba2.modeling_mamba2.MambaRMSNormGated
class Mixer2RMSNormGated(CustomOp):

    def __init__(
        self,
        full_hidden_size: int,
        full_n_groups: int,
        use_rms_norm: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.full_hidden_size = full_hidden_size
        self.group_size = full_hidden_size // full_n_groups
        self.per_rank_hidden_size = full_hidden_size // self.tp_size
        self.n_groups = full_hidden_size // self.group_size

        self.variance_epsilon = eps
        self.use_rms_norm = use_rms_norm
        if self.use_rms_norm:
            # Register norm weight only if we're actually applying RMSNorm
            self.weight = nn.Parameter(torch.ones(self.per_rank_hidden_size))
            set_weight_attrs(self.weight, {"weight_loader": sharded_weight_loader(0)})
        else:
            # Avoid checkpoint mismatch by skipping unused parameter
            self.register_parameter("weight", None)
        assert (
            self.full_hidden_size % self.tp_size == 0
        ), "Tensor parallel world size must divide hidden size."

    def forward_native(
        self,
        x: torch.Tensor,
        gate: torch.Tensor,
    ):
        # Three tensor-parallel cases:
        #   1. n_groups is 1
        #      In this case we parallelize along the reduction dim.
        #      Each rank computes a local sum of squares followed by AllReduce
        #   2. tp_size divides n_groups
        #      Each rank only reduces within its local group(s).
        #      No collective ops necessary.
        #   3. The general case can be pretty complicated so we AllGather
        #      the input and then redundantly compute the RMSNorm.
        input_dtype = x.dtype
        x = x * nn.functional.silu(gate.to(torch.float32))
        if not self.use_rms_norm:
            return x.to(input_dtype)

        if self.n_groups == 1:
            if self.tp_size > 1:
                # Compute local sum and then reduce to obtain global sum
                local_sums = x.pow(2).sum(dim=-1, keepdim=True)
                global_sums = tensor_model_parallel_all_reduce(local_sums)
                # Calculate the variance
                count = self.tp_size * x.shape[-1]
                variance = global_sums / count

            else:
                variance = x.pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(variance + self.variance_epsilon)
        else:
            redundant_tp: bool = self.n_groups % self.tp_size != 0
            if redundant_tp:
                # To handle the general case, redundantly apply the variance
                x = tensor_model_parallel_all_gather(x, -1)

            *prefix_dims, hidden_dim = x.shape
            group_count = hidden_dim // self.group_size
            x_grouped = x.view(*prefix_dims, group_count, self.group_size)
            variance = x_grouped.pow(2).mean(-1, keepdim=True)
            x_grouped = x_grouped * torch.rsqrt(variance + self.variance_epsilon)
            x = x_grouped.view(*prefix_dims, hidden_dim)

            if redundant_tp:
                start = self.per_rank_hidden_size * self.tp_rank
                end = start + self.per_rank_hidden_size
                x = x[..., start:end]

        return self.weight * x.to(input_dtype)

    def forward_cuda(
        self,
        x: torch.Tensor,
        gate: torch.Tensor,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        input_dtype = x.dtype
        if not self.use_rms_norm:
            # Keep gate in float32 for numerical stability during silu
            return x * nn.functional.silu(gate.to(torch.float32)).to(input_dtype)

        if ((self.n_groups % self.tp_size) != 0) or self.n_groups != 1:
            return self.forward_native(x, gate)

        return rms_norm_gated(
            x,
            self.weight.data,
            bias=None,
            z=gate,
            eps=self.variance_epsilon,
            norm_before_gate=False,
        )


# Adapted from transformers.models.mamba.modeling_mamba.MambaMixer
class MambaMixer2(CustomOp):
    """
    Compute ∆, A, B, C, and D the state space parameters and compute
    the `contextualized_states`. A, D are input independent
    (see Mamba paper [1] Section 3.5.2 "Interpretation of A"
    for why A isn't selective) ∆, B, C are input-dependent
    (this is a key difference between Mamba and the linear time
    invariant S4, and is why Mamba is called
    **selective** state spaces)
    """

    def __init__(
        self,
        hidden_size: int,
        ssm_state_size: int,
        conv_kernel_size: int,
        intermediate_size: int,
        use_conv_bias: bool,
        use_bias: bool,
        n_groups: int = 1,
        num_heads: int = 128,
        head_dim: int = 64,
        rms_norm_eps: float = 1e-5,
        activation: str = "silu",
        use_rms_norm: bool = True,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()

        # For TP, the sharding plan is as follows:
        # - for the conv modules, since
        #   conv_dim = intermediate_size * 2 * n_groups * ssm_state_size,
        #   we shard intermediate_size and n_groups
        # - since intermediate_size = n_heads * head_dim, sharding on
        #   intermediate_size is achieved by sharding on n_heads.
        # - IF, world_size divides groups, then sharding
        #   (n_groups / world_size, n_heads / world_size)
        #   also maintains the invariant n_heads % n_groups == 0
        # - HOWEVER IF, world_size DOES NOT divide groups, then we need
        #   to allocate extra space in the shard, such that groups
        #   may be replicated to follow the head shard.
        # - NOTE: currently for the world size DOES NOT divide groups
        #   case, we only support the case when n_groups == 1
        self.tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()

        assert (
            num_heads % self.tp_size == 0
        ), "Tensor parallel world size must divide num heads."

        assert (n_groups % self.tp_size) == 0 or n_groups == 1, (
            "If tensor parallel world size does not divide num_heads, "
            "then num_groups must equal 1."
        )

        assert (
            self.tp_size == 1 or quant_config is None
        ), "Tensor parallel currently not supported for quantized models."

        self.ssm_state_size = ssm_state_size
        self.conv_kernel_size = conv_kernel_size
        self.activation = activation

        self.intermediate_size = intermediate_size
        self.head_dim = head_dim
        self.num_heads = num_heads

        self.n_groups = n_groups
        if n_groups % self.tp_size != 0:
            # - for TP we shard conv_dim by sharding on n_groups,
            # - but if n_groups cannot divide tp_size, we need to
            #   extend some extra groups
            groups = extra_groups_for_head_shards(n_groups, self.tp_size)
            self.n_groups = n_groups + groups

        self.conv_dim = intermediate_size + 2 * self.n_groups * ssm_state_size
        self.conv1d = ColumnParallelLinear(
            input_size=conv_kernel_size,
            output_size=self.conv_dim,
            bias=use_conv_bias,
            quant_config=None,
        )
        # unsqueeze to fit conv1d weights shape into the linear weights shape.
        # Can't do this in `weight_loader` since it already exists in
        # `ColumnParallelLinear` and `set_weight_attrs`
        # doesn't allow to override it
        self.conv1d.weight.data = self.conv1d.weight.data.unsqueeze(1)

        self.in_proj = ColumnParallelLinear(
            input_size=hidden_size,
            output_size=intermediate_size + self.conv_dim + self.num_heads,
            bias=use_bias,
            quant_config=quant_config,
        )

        # - because in_proj is a concatenation of 3 weights, we
        #   need to interleave them before sharding
        # - use the custom weight loader mamba_v2_sharded_weight_loader
        #   for conv1d.bias, covn1d.weight and in_proj.weight
        # - need to set these settings, to assign the groups to the head shards
        group_shard_settings = (
            self.n_groups * self.ssm_state_size,  # expected model size
            (self.n_groups - n_groups) * self.ssm_state_size,  # extra dims assigned
            n_groups == 1,  # if there was only one group
        )
        intermediate_settings = (intermediate_size, 0, False)
        head_settings = (self.num_heads, 0, False)

        # - the weight already has a "weight_loader" attribute
        #   which set_weight_attrs will raise if we do not
        #   delete before trying to override it
        # - ditto for the otther two weights below
        delattr(self.conv1d.bias, "weight_loader")
        set_weight_attrs(
            self.conv1d.bias,
            {
                "weight_loader": mamba_v2_sharded_weight_loader(
                    [
                        intermediate_settings,
                        group_shard_settings,
                        group_shard_settings,
                    ],
                    self.tp_size,
                    tp_rank,
                )
            },
        )

        delattr(self.conv1d.weight, "weight_loader")
        set_weight_attrs(
            self.conv1d.weight,
            {
                "weight_loader": mamba_v2_sharded_weight_loader(
                    [
                        intermediate_settings,
                        group_shard_settings,
                        group_shard_settings,
                    ],
                    self.tp_size,
                    tp_rank,
                )
            },
        )

        if quant_config is None:
            # - quant layers do not have a weight loader
            delattr(self.in_proj.weight, "weight_loader")
            set_weight_attrs(
                self.in_proj.weight,
                {
                    "weight_loader": mamba_v2_sharded_weight_loader(
                        [
                            intermediate_settings,  # for gate
                            intermediate_settings,
                            group_shard_settings,
                            group_shard_settings,
                            head_settings,  # for dt
                        ],
                        self.tp_size,
                        tp_rank,
                    )
                },
            )

        # - these are TPed by heads to reduce the size of the
        #   temporal shape
        self.A = nn.Parameter(
            torch.empty(
                divide(num_heads, self.tp_size),
                dtype=torch.float32,
            )
        )
        self.D = nn.Parameter(torch.ones(num_heads // self.tp_size))
        self.dt_bias = nn.Parameter(torch.ones(num_heads // self.tp_size))
        self.use_rms_norm = use_rms_norm

        set_weight_attrs(self.D, {"weight_loader": sharded_weight_loader(0)})
        a_weight_loader = composed_weight_loader(
            sharded_weight_loader(0), lambda x: -torch.exp(x.float())
        )
        set_weight_attrs(self.A, {"weight_loader": a_weight_loader})
        set_weight_attrs(self.dt_bias, {"weight_loader": sharded_weight_loader(0)})

        self.out_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=use_bias,
            input_is_parallel=True,
            quant_config=quant_config,
        )

        self.norm = Mixer2RMSNormGated(
            intermediate_size, n_groups, self.use_rms_norm, eps=rms_norm_eps
        )

        self.prefix = prefix

    def forward_native(
        self,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
        layer_cache: MambaPool.State,
        layer_cache_indices: torch.Tensor,
        metadata: Mamba2Metadata,
    ):
        pass

    def forward(
        self,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
        layer_cache: MambaPool.State,
        layer_cache_indices: torch.Tensor,
        metadata: Mamba2Metadata,
    ):
        CustomOp.forward(
            self, hidden_states, output, layer_cache, layer_cache_indices, metadata
        )

    def forward_cuda(
        self,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
        layer_cache: MambaPool.State,
        layer_cache_indices: torch.Tensor,
        metadata: Mamba2Metadata,
    ):
        # metadata contains metadata necessary for the mamba2 triton
        # kernels to operate in continuous batching and in chunked prefill
        # modes; they are computed at top-level model forward since they
        # stay the same and reused for all mamba layers in the same iteration
        conv_state = layer_cache.conv
        ssm_state = layer_cache.temporal
        state_indices_tensor = layer_cache_indices
        has_initial_states_p = metadata.has_initial_states
        prep_initial_states = metadata.prep_initial_states
        chunk_size = metadata.chunk_size
        seq_idx_p = metadata.seq_idx
        chunk_indices_p = metadata.chunk_indices
        chunk_offsets_p = metadata.chunk_offsets
        seq_lens_cpu_p = metadata.seq_lens_cpu

        groups_time_state_size = self.n_groups * self.ssm_state_size

        # 1. Gated MLP's linear projection
        projected_states, _ = self.in_proj(hidden_states)

        gate, hidden_states_B_C, dt = torch.split(
            projected_states,
            [
                self.intermediate_size // self.tp_size,
                self.conv_dim // self.tp_size,
                self.num_heads // self.tp_size,
            ],
            dim=-1,
        )

        conv_weights = self.conv1d.weight.view(
            self.conv1d.weight.size(0), self.conv1d.weight.size(2)
        )

        # - get hidden_states, B and C after depthwise convolution.
        split_hidden_states_B_C_fn = lambda hidden_states_B_C: torch.split(
            hidden_states_B_C,
            [
                self.intermediate_size // self.tp_size,
                groups_time_state_size // self.tp_size,
                groups_time_state_size // self.tp_size,
            ],
            dim=-1,
        )

        num_prefills = metadata.num_prefills  # request count
        num_decodes = metadata.num_decodes  # token count (=request)
        num_prefill_tokens = metadata.num_prefill_tokens  # token count
        has_prefill = num_prefills > 0
        has_decode = num_decodes > 0
        num_actual_tokens = num_prefill_tokens + num_decodes

        # NOTE: V0 put prefill before decode
        # Separate prefill and decode by splitting varlen input
        # Split along token dimension
        hidden_states_B_C_p, hidden_states_B_C_d = torch.split(
            hidden_states_B_C,
            [num_prefill_tokens, num_decodes],
            dim=0,
        )
        dt_p, dt_d = torch.split(
            dt,
            [num_prefill_tokens, num_decodes],
            dim=0,
        )
        # Split along batch dimension
        state_indices_tensor_p, state_indices_tensor_d = torch.split(
            state_indices_tensor,
            [num_prefills, num_decodes],
            dim=0,
        )
        query_start_loc_p = (
            metadata.query_start_loc[: num_prefills + 1] if has_prefill else None
        )

        # Preallocate output tensor to avoid memcpy cost for merging prefill
        # and decode outputs
        preallocated_ssm_out = torch.empty(
            [
                num_prefill_tokens + num_decodes,
                (self.num_heads // self.tp_size) * self.head_dim,
            ],
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        preallocated_ssm_out_p, preallocated_ssm_out_d = torch.split(
            preallocated_ssm_out,
            [num_prefill_tokens, num_decodes],
            dim=0,
        )

        # Process prefill requests
        if has_prefill:
            # 2. Convolution sequence transformation
            # - "cache_indices" updates the conv_state cache in positions
            #   pointed to by "state_indices_tensor"
            x = hidden_states_B_C_p.transpose(
                0, 1
            )  # this is the form that causal-conv see
            hidden_states_B_C_p = causal_conv1d_fn(
                x,
                conv_weights,
                self.conv1d.bias,
                conv_state,
                query_start_loc_p,
                seq_lens_cpu_p,
                state_indices_tensor_p,
                has_initial_states_p,
                self.activation,
            ).transpose(0, 1)[:num_prefill_tokens]

            hidden_states_p, B_p, C_p = split_hidden_states_B_C_fn(hidden_states_B_C_p)

            # 3. State Space Model sequence transformation
            initial_states = None
            if has_initial_states_p is not None and prep_initial_states:
                # making a copy of the states
                initial_states = torch.where(
                    has_initial_states_p[:num_prefills, None, None, None],
                    ssm_state[state_indices_tensor_p],
                    0,
                )

            # NOTE: final output is an in-place update of out tensor
            varlen_state = mamba_chunk_scan_combined(
                hidden_states_p.view(
                    1, num_prefill_tokens, self.num_heads // self.tp_size, self.head_dim
                ),
                dt_p.unsqueeze(0),
                self.A,
                B_p.view(1, num_prefill_tokens, self.n_groups // self.tp_size, -1),
                C_p.view(1, num_prefill_tokens, self.n_groups // self.tp_size, -1),
                chunk_size=chunk_size,
                D=self.D,
                z=None,
                dt_bias=self.dt_bias,
                seq_idx=seq_idx_p,
                chunk_indices=chunk_indices_p,
                chunk_offsets=chunk_offsets_p,
                cu_seqlens=query_start_loc_p,
                initial_states=initial_states,
                return_varlen_states=True,
                return_final_states=False,
                dt_softplus=True,
                dt_limit=(0.0, float("inf")),
                out=preallocated_ssm_out_p.view(
                    1, num_prefill_tokens, -1, self.head_dim
                ),
                state_dtype=ssm_state.dtype,
            )

            # update ssm states
            # - varlen state is a (num_prefills, nheads, headdim, dstate) tensor
            ssm_state[state_indices_tensor_p] = varlen_state

        # Process decode requests
        if has_decode:
            # 2. Convolution sequence transformation
            hidden_states_B_C_d = causal_conv1d_update(
                hidden_states_B_C_d,
                conv_state,
                conv_weights,
                self.conv1d.bias,
                self.activation,
                conv_state_indices=state_indices_tensor_d,
            )

            hidden_states_d, B_d, C_d = split_hidden_states_B_C_fn(hidden_states_B_C_d)

            # 3. State Space Model sequence transformation
            n_groups = self.n_groups // self.tp_size
            A_d = (
                self.A[:, None, ...][:, :, None]
                .expand(-1, self.head_dim, self.ssm_state_size)
                .to(dtype=torch.float32)
            )
            dt_d = dt_d[:, :, None].expand(-1, -1, self.head_dim)
            dt_bias = self.dt_bias[:, None, ...].expand(-1, self.head_dim)
            D_d = self.D[:, None, ...].expand(-1, self.head_dim)
            B_d = B_d.view(-1, n_groups, B_d.shape[1] // n_groups)
            C_d = C_d.view(-1, n_groups, C_d.shape[1] // n_groups)
            hidden_states_d = hidden_states_d.view(
                -1, self.num_heads // self.tp_size, self.head_dim
            )

            # - the hidden is reshaped into (bs, num_heads, head_dim)
            # - layer_state.ssm_state's slots will be selected
            #   using state_indices_tensor_d
            # NOTE: final output is an in-place update of out tensor
            selective_state_update(
                ssm_state,
                hidden_states_d,
                dt_d,
                A_d,
                B_d,
                C_d,
                D_d,
                z=None,
                dt_bias=dt_bias,
                dt_softplus=True,
                state_batch_indices=state_indices_tensor_d,
                out=preallocated_ssm_out_d.view(num_decodes, -1, self.head_dim),
            )

        # 4. gated MLP
        # GatedRMSNorm internally applying SiLU to the gate
        # SiLU is applied internally before normalization, unlike standard
        # norm usage
        hidden_states = self.norm(preallocated_ssm_out, gate[:num_actual_tokens])

        # 5. Final linear projection
        output[:num_actual_tokens], _ = self.out_proj(hidden_states)
