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
    selective_scan_fn,
    selective_state_update,
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
    from sglang.srt.layers.attention.mamba.causal_conv1d_triton import (
        causal_conv1d_fn as causal_conv1d_fn_triton,
    )
    from sglang.srt.layers.attention.mamba.causal_conv1d_triton import (
        causal_conv1d_update as causal_conv1d_update_triton,
    )


class MambaMixer1(nn.Module):
    """Generic Mamba1 mixer layer.

    Compute delta, A, B, C, and D the state space parameters and compute
    the contextualized states. A, D are input independent.
    delta, B, C are input-dependent (selective).
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

        self.intermediate_size_tp = self.intermediate_size // self.tp_size

        self.in_proj = MergedColumnParallelLinear(
            hidden_size,
            [self.intermediate_size, self.intermediate_size],
            bias=use_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.in_proj",
        )

        self.conv1d = ColumnParallelLinear(
            self.conv_kernel,
            self.intermediate_size,
            bias=use_conv_bias,
            quant_config=None,
            prefix=f"{prefix}.conv1d",
        )
        self.conv1d.weight.data = self.conv1d.weight.data.unsqueeze(1)

        self.dt_proj = ColumnParallelLinear(
            dt_rank,
            self.intermediate_size,
            bias=True,
            skip_bias_add=True,
            quant_config=quant_config,
            prefix=f"{prefix}.dt_proj",
        )

        self.x_proj = RowParallelLinear(
            self.intermediate_size,
            dt_rank + 2 * self.state_size,
            bias=False,
            input_is_parallel=True,
            reduce_results=True,
            quant_config=quant_config,
            prefix=f"{prefix}.x_proj",
        )

        if use_dt_bc_layernorm:
            self.dt_layernorm = RMSNorm(dt_rank, eps=rms_norm_eps)
            self.b_layernorm = RMSNorm(self.state_size, eps=rms_norm_eps)
            self.c_layernorm = RMSNorm(self.state_size, eps=rms_norm_eps)

        self.A_log = nn.Parameter(
            torch.empty(self.intermediate_size_tp, self.state_size)
        )
        set_weight_attrs(self.A_log, {"weight_loader": sharded_weight_loader(0)})

        self.D = nn.Parameter(torch.empty(self.intermediate_size_tp))
        set_weight_attrs(self.D, {"weight_loader": sharded_weight_loader(0)})

        self.out_proj = RowParallelLinear(
            self.intermediate_size,
            hidden_size,
            bias=use_bias,
            input_is_parallel=True,
            quant_config=quant_config,
            prefix=f"{prefix}.out_proj",
        )

    def _ssm_transform(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project through x_proj, split into dt/B/C, apply layernorm and dt_proj.

        Returns:
            (dt, B, C, A) after all transformations.
        """
        x_dbc, _ = self.x_proj(x)
        dt, B, C = x_dbc.split([self.dt_rank, self.state_size, self.state_size], dim=-1)

        if self.use_dt_bc_layernorm:
            dt = self.dt_layernorm(dt)
            B = self.b_layernorm(B)
            C = self.c_layernorm(C)

        dt, _ = self.dt_proj(dt)
        A = -torch.exp(self.A_log.float())
        return dt, B, C, A

    def _time_proj_bias(self) -> Optional[torch.Tensor]:
        if hasattr(self.dt_proj, "bias") and self.dt_proj.bias is not None:
            return self.dt_proj.bias.float()
        return None

    def forward(
        self,
        *,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
        layer_cache: MambaPool.State,
        metadata: Mamba1Metadata,
        use_triton_causal_conv: bool = False,
    ):
        conv_state = layer_cache.conv[0]
        ssm_state = layer_cache.temporal
        state_indices = metadata.mamba_cache_indices
        query_start_loc = metadata.query_start_loc
        num_prefills = metadata.num_prefills
        num_decodes = metadata.num_decodes

        num_prefill_tokens = metadata.num_prefill_tokens
        has_prefill = num_prefills > 0
        has_decode = num_decodes > 0

        # 1. Input projection
        projected, _ = self.in_proj(hidden_states)
        x, z = projected.chunk(2, dim=-1)

        x_p, x_d = x[:num_prefill_tokens], x[num_prefill_tokens:]
        z_p, z_d = z[:num_prefill_tokens], z[num_prefill_tokens:]
        state_indices_p = state_indices[:num_prefills]
        state_indices_d = state_indices[num_prefills:]

        # Prepare intermediate output tensor (intermediate_size_tp dim)
        ssm_output = torch.empty_like(x)
        conv_weight = self.conv1d.weight.view(
            self.conv1d.weight.size(0), self.conv1d.weight.size(2)
        )
        query_start_loc_p = query_start_loc[: num_prefills + 1] if has_prefill else None

        # 2. Process prefill requests
        if has_prefill:
            mixed_metadata = metadata.mixed_metadata
            has_initial_states = mixed_metadata.has_initial_states

            ccfn = (
                causal_conv1d_fn
                if not use_triton_causal_conv
                else causal_conv1d_fn_triton
            )
            ccfn_kwargs = dict(
                conv_states=conv_state,
                has_initial_state=has_initial_states,
                cache_indices=state_indices_p,
                query_start_loc=query_start_loc_p,
                seq_lens_cpu=mixed_metadata.extend_seq_lens_cpu,
            )

            x_conv = ccfn(
                x_p.transpose(0, 1),
                conv_weight,
                self.conv1d.bias,
                activation=self.activation,
                **ccfn_kwargs,
            ).transpose(0, 1)[:num_prefill_tokens]

            dt, B, C, A = self._ssm_transform(x_conv)
            delta_bias = self._time_proj_bias()

            # Process per-sequence for selective scan
            batch_sizes = (query_start_loc_p[1:] - query_start_loc_p[:-1]).tolist()
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

                initial_state = None
                if has_initial_states is not None:
                    initial_state = ssm_state[state_indices_p[i]].unsqueeze(0).float()

                y_i, final_state = selective_scan_fn(
                    x_i,
                    dt_i,
                    A,
                    B_i,
                    C_i,
                    D=self.D.float(),
                    z=z_i,
                    delta_bias=delta_bias,
                    delta_softplus=True,
                    return_last_state=True,
                    initial_state=initial_state,
                )

                ssm_state[state_indices_p[i]] = final_state.squeeze(0)
                outputs_p.append(y_i.squeeze(0).to(dtype_in))

            ssm_output[:num_prefill_tokens] = torch.cat(outputs_p, dim=0)

        # 3. Process decode requests
        if has_decode:
            ccu = (
                causal_conv1d_update
                if not use_triton_causal_conv
                else causal_conv1d_update_triton
            )
            x_conv = ccu(
                x_d,
                conv_state,
                conv_weight,
                self.conv1d.bias,
                self.activation,
                conv_state_indices=state_indices_d,
            )

            dt, B, C, A = self._ssm_transform(x_conv)
            delta_bias = self._time_proj_bias()

            selective_state_update(
                ssm_state,
                x_conv,
                dt,
                A,
                B,
                C,
                D=self.D.float(),
                z=z_d,
                dt_bias=delta_bias,
                dt_softplus=True,
                state_batch_indices=state_indices_d,
                out=ssm_output[num_prefill_tokens:],
            )

        # 4. Output projection
        num_actual_tokens = num_prefill_tokens + num_decodes
        output[:num_actual_tokens], _ = self.out_proj(ssm_output)

    @property
    def mamba_type(self) -> str:
        return "mamba1"
