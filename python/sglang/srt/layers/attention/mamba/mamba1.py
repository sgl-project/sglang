# Copyright 2025 SGLang Team
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
"""Generic Mamba1 mixer layer.

This is the Mamba1 equivalent of MambaMixer2 in mamba.py. It provides a
reusable mixer that can be used by any Mamba1-based hybrid model (e.g., Jamba).

Key differences from MambaMixer2:
- 2D temporal state (intermediate_size/tp, state_size), no groups/heads
- Uses dt_proj to project dt_rank -> intermediate_size
- Uses x_proj to compute dt, B, C from conv output
- Uses selective_scan_fn for prefill, triton kernel for decode
"""

from typing import Optional

import torch
import torch.nn as nn

from sglang.srt.configs.mamba_utils import Mamba1CacheParams
from sglang.srt.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from sglang.srt.layers.attention.mamba.mamba1_metadata import Mamba1Metadata
from sglang.srt.layers.attention.mamba.ops import (
    mamba1_selective_scan,
    mamba1_selective_state_update,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.mem_cache.memory_pool import MambaPool
from sglang.srt.model_loader.weight_utils import sharded_weight_loader
from sglang.srt.utils import is_cuda, set_weight_attrs

if is_cuda():
    from sglang.srt.layers.attention.mamba.causal_conv1d import (
        causal_conv1d_fn,
        causal_conv1d_update,
    )


class MambaMixer1(nn.Module):
    """Generic Mamba1 mixer layer.

    Compute delta, A, B, C, and D the state space parameters and compute
    the contextualized states. A, D are input independent.
    delta, B, C are input-dependent (selective).

    This is the Mamba1 equivalent of MambaMixer2.
    """

    def __init__(
        self,
        cache_params: Mamba1CacheParams,
        hidden_size: int,
        dt_rank: int,
        use_conv_bias: bool = True,
        use_bias: bool = False,
        use_dt_bc_layernorm: bool = False,
        rms_norm_eps: float = 1e-6,
        activation: str = "silu",
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()

        self.hidden_size = hidden_size
        self.intermediate_size = cache_params.shape.intermediate_size
        self.state_size = cache_params.shape.state_size
        self.conv_kernel = cache_params.shape.conv_kernel
        self.dt_rank = dt_rank
        self.use_dt_bc_layernorm = use_dt_bc_layernorm
        self.activation = activation

        # Intermediate size after TP sharding
        self.intermediate_size_tp = self.intermediate_size // self.tp_size

        # Input projection: hidden -> (x, z) where x goes through conv+SSM, z is gate
        self.in_proj = MergedColumnParallelLinear(
            hidden_size,
            [self.intermediate_size, self.intermediate_size],
            bias=use_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.in_proj",
        )

        # Conv1d on intermediate dimension
        self.conv1d = ColumnParallelLinear(
            self.conv_kernel,
            self.intermediate_size,
            bias=use_conv_bias,
            quant_config=None,
            prefix=f"{prefix}.conv1d",
        )
        self.conv1d.weight.data = self.conv1d.weight.data.unsqueeze(1)

        # dt projection: dt_rank -> intermediate_size
        self.dt_proj = ColumnParallelLinear(
            dt_rank,
            self.intermediate_size,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.dt_proj",
        )

        # x_proj: x -> (dt, B, C) projections
        # Input is already sharded across TP, need all-reduce for correct output
        self.x_proj = RowParallelLinear(
            self.intermediate_size,
            dt_rank + 2 * self.state_size,
            bias=False,
            input_is_parallel=True,
            reduce_results=True,
            quant_config=quant_config,
            prefix=f"{prefix}.x_proj",
        )

        # Optional layer norms for dt, B, C (used by Jamba for numerical stability)
        if use_dt_bc_layernorm:
            self.dt_layernorm = RMSNorm(dt_rank, eps=rms_norm_eps)
            self.b_layernorm = RMSNorm(self.state_size, eps=rms_norm_eps)
            self.c_layernorm = RMSNorm(self.state_size, eps=rms_norm_eps)

        # A parameter (stored as log, -exp applied during forward)
        self.A_log = nn.Parameter(
            torch.empty(self.intermediate_size_tp, self.state_size)
        )
        set_weight_attrs(self.A_log, {"weight_loader": sharded_weight_loader(0)})

        # D parameter (skip connection)
        self.D = nn.Parameter(torch.empty(self.intermediate_size_tp))
        set_weight_attrs(self.D, {"weight_loader": sharded_weight_loader(0)})

        # Output projection
        self.out_proj = RowParallelLinear(
            self.intermediate_size,
            hidden_size,
            bias=use_bias,
            input_is_parallel=True,
            quant_config=quant_config,
            prefix=f"{prefix}.out_proj",
        )

    def forward(
        self,
        *,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
        layer_cache: MambaPool.State,
        metadata: Mamba1Metadata,
    ):
        """Forward pass for Mamba1 mixer.

        Args:
            hidden_states: (num_tokens, hidden_size) input tensor
            output: (num_tokens, hidden_size) preallocated output tensor
            layer_cache: MambaPool.State with conv[0] and temporal states
            metadata: Mamba1Metadata with cache indices and batch counts
        """
        conv_state = layer_cache.conv[0]
        ssm_state = layer_cache.temporal
        state_indices = metadata.mamba_cache_indices
        query_start_loc = metadata.query_start_loc
        num_prefills = metadata.num_prefills
        num_decodes = metadata.num_decodes

        num_tokens = hidden_states.shape[0]
        num_prefill_tokens = num_tokens - num_decodes

        # 1. Input projection
        projected, _ = self.in_proj(hidden_states)
        x, z = projected.chunk(2, dim=-1)

        # Split by prefill/decode
        x_p, x_d = x[:num_prefill_tokens], x[num_prefill_tokens:]
        z_p, z_d = z[:num_prefill_tokens], z[num_prefill_tokens:]
        state_indices_p = state_indices[:num_prefills]
        state_indices_d = state_indices[num_prefills:]

        # Prepare intermediate output tensor (intermediate_size_tp dim)
        ssm_output = torch.empty_like(x)

        # 2. Process prefill requests
        if num_prefills > 0 and num_prefill_tokens > 0:
            conv_weight = self.conv1d.weight.view(
                self.conv1d.weight.size(0), self.conv1d.weight.size(2)
            )

            has_initial_states = metadata.has_initial_states
            prep_initial_states = metadata.prep_initial_states

            x_conv = causal_conv1d_fn(
                x_p.transpose(0, 1),
                conv_weight,
                self.conv1d.bias,
                activation=self.activation,
                conv_states=conv_state,
                has_initial_state=has_initial_states,
                cache_indices=state_indices_p,
                query_start_loc=query_start_loc,
            ).transpose(0, 1)[:num_prefill_tokens]

            # x_proj: compute dt, B, C projections
            x_dbc, _ = self.x_proj(x_conv)
            dt, B, C = x_dbc.split(
                [self.dt_rank, self.state_size, self.state_size], dim=-1
            )

            # Apply optional layer norms
            if self.use_dt_bc_layernorm:
                dt = self.dt_layernorm(dt)
                B = self.b_layernorm(B)
                C = self.c_layernorm(C)

            # dt projection
            dt, _ = self.dt_proj(dt)

            # A = -exp(A_log)
            A = -torch.exp(self.A_log.float())

            # Process per-sequence for selective scan
            batch_sizes = (query_start_loc[1:] - query_start_loc[:-1]).tolist()
            x_batched = torch.split(x_conv, batch_sizes, dim=0)
            dt_batched = torch.split(dt, batch_sizes, dim=0)
            B_batched = torch.split(B, batch_sizes, dim=0)
            C_batched = torch.split(C, batch_sizes, dim=0)
            z_batched = torch.split(z_p, batch_sizes, dim=0)

            outputs_p = []
            for i, (x_i, dt_i, B_i, C_i, z_i) in enumerate(
                zip(x_batched, dt_batched, B_batched, C_batched, z_batched)
            ):
                dtype_in = x_i.dtype
                x_i = x_i.unsqueeze(0).float()
                dt_i = dt_i.unsqueeze(0).float()
                B_i = B_i.unsqueeze(0).float()
                C_i = C_i.unsqueeze(0).float()
                z_i = z_i.unsqueeze(0).float()

                # Load initial state for extending sequences (prefix caching)
                initial_state = None
                if has_initial_states is not None and prep_initial_states:
                    if has_initial_states[i]:
                        initial_state = ssm_state[state_indices_p[i]].unsqueeze(0).float()

                y_i, final_state = mamba1_selective_scan(
                    x_i,
                    dt_i,
                    A,
                    B_i,
                    C_i,
                    D=self.D.float(),
                    z=z_i,
                    delta_softplus=True,
                    return_last_state=True,
                    initial_state=initial_state,
                )

                ssm_state[state_indices_p[i]] = final_state.squeeze(0)
                outputs_p.append(y_i.squeeze(0).to(dtype_in))

            ssm_output[:num_prefill_tokens] = torch.cat(outputs_p, dim=0)

        # 3. Process decode requests
        if num_decodes > 0:
            conv_weight = self.conv1d.weight.view(
                self.conv1d.weight.size(0), self.conv1d.weight.size(2)
            )

            x_conv = causal_conv1d_update(
                x_d,
                conv_state,
                conv_weight,
                self.conv1d.bias,
                self.activation,
                conv_state_indices=state_indices_d,
            )

            x_dbc, _ = self.x_proj(x_conv)
            dt, B, C = x_dbc.split(
                [self.dt_rank, self.state_size, self.state_size], dim=-1
            )

            if self.use_dt_bc_layernorm:
                dt = self.dt_layernorm(dt)
                B = self.b_layernorm(B)
                C = self.c_layernorm(C)

            dt, _ = self.dt_proj(dt)
            A = -torch.exp(self.A_log.float())

            y_d = mamba1_selective_state_update(
                ssm_state,
                x_conv,
                dt,
                A,
                B,
                C,
                D=self.D.float(),
                z=z_d,
                dt_softplus=True,
                state_batch_indices=state_indices_d,
            )

            ssm_output[num_prefill_tokens:] = y_d

        # 4. Output projection
        output[:num_tokens], _ = self.out_proj(ssm_output)

    @property
    def mamba_type(self) -> str:
        return "mamba1"
