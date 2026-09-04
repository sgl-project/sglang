# SPDX-License-Identifier: Apache-2.0
# Copyright 2023-2025 SGLang Team
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
"""Mamba-1 (selective-scan) SSM mixer for SGLang, e.g. Falcon-Mamba.

Unlike Mamba-2 (:class:`MambaMixer2`, SSD / chunked-scan, scalar per-head ``A``),
Mamba-1 keeps a **full-rank** ``A`` of shape ``(intermediate_size, state_size)``
and derives the selective parameters ``dt``/``B``/``C`` from ``x_proj`` applied
*after* the causal conv (so the conv is over ``intermediate_size`` only). This
matches HuggingFace ``FalconMambaMixer`` / the original ``MambaMixer``.

Reuse strategy (to ride the existing Mamba2 attention backend, memory pool and
kernels unchanged): the full-rank state is expressed on the Mamba2 head layout
as ``num_heads == intermediate_size`` and ``head_dim == 1`` (see
``Mamba2StateShape.create_full_rank``). Then:

  - the causal conv uses the shared ``causal_conv1d_fn`` / ``causal_conv1d_update``
    (Triton variants on XPU), exactly like Mamba2;
  - single-token **decode** uses the shared ``selective_state_update`` kernel,
    which already supports a full-rank ``A`` of shape ``(nheads, dim, dstate)``
    and applies the ``silu(z)`` output gate;
  - multi-token **prefill** runs a portable pure-torch selective scan (there is
    no Mamba-1 chunked-scan kernel in-tree), which is device-agnostic and works
    on Intel XPU.

Falcon-Mamba adds a weightless RMSNorm to ``B``, ``C`` and ``dt`` (the "Falcon"
stabilization trick), applied here via :func:`rms_normalize` gated on
``use_bc_dt_rms``.
"""

import logging
from typing import Optional, Tuple

import torch
from torch import nn

from sglang.kernels.ops.mamba.triton_ops import selective_state_update
from sglang.srt.distributed import divide
from sglang.srt.layers.attention.mamba.mamba import (
    causal_conv1d_fn,
    causal_conv1d_fn_triton,
    causal_conv1d_update,
    causal_conv1d_update_triton,
)
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_loader.weight_utils import sharded_weight_loader
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import set_weight_attrs

logger = logging.getLogger(__name__)


def rms_normalize(hidden_states: torch.Tensor, eps: float) -> torch.Tensor:
    """Weightless RMSNorm (matches HF ``falcon_mamba.rms_forward``).

    Falcon-Mamba normalizes ``B``, ``C`` and the time step with a *non-learnable*
    RMSNorm (no weight) before discretization; other Mamba-1 models skip this.
    """
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + eps)
    return hidden_states.to(input_dtype)


class MambaMixer1(nn.Module):
    """Mamba-1 selective-scan mixer.

    Weight names match the HF checkpoint (``in_proj``, ``conv1d``, ``x_proj``,
    ``dt_proj``, ``A_log``, ``D``, ``out_proj``) so the model loader maps them
    directly.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        intermediate_size: int,
        state_size: int,
        conv_kernel: int,
        time_step_rank: int,
        use_conv_bias: bool,
        use_bias: bool,
        activation: str = "silu",
        use_bc_dt_rms: bool = False,
        rms_eps: float = 1e-6,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.tp_size = get_parallel().tp_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.ssm_state_size = state_size
        self.conv_kernel_size = conv_kernel
        self.time_step_rank = time_step_rank
        self.activation = activation
        self.use_bc_dt_rms = use_bc_dt_rms
        self.rms_eps = rms_eps

        assert intermediate_size % self.tp_size == 0, (
            f"Mamba-1 intermediate_size ({intermediate_size}) must be divisible "
            f"by tp_size ({self.tp_size})"
        )
        self.intermediate_size_per_tp = divide(intermediate_size, self.tp_size)

        # in_proj -> [x, gate], each of size intermediate_size (column-sharded).
        self.in_proj = MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[intermediate_size, intermediate_size],
            bias=use_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.in_proj",
        )

        # Depthwise causal conv over the intermediate channels only (column-sharded).
        self.conv1d = ColumnParallelLinear(
            input_size=conv_kernel,
            output_size=intermediate_size,
            bias=use_conv_bias,
            quant_config=None,
            prefix=f"{prefix}.conv1d",
        )
        # Checkpoint stores conv1d.weight as (dim, 1, K); ColumnParallelLinear
        # allocates (dim, K). Re-view to (dim, 1, K) so the conv kernel and the
        # default weight loader agree on shape (same trick as MambaMixer2).
        self.conv1d.weight.data = self.conv1d.weight.data.unsqueeze(1)

        # x_proj: intermediate -> [dt_rank, B(state), C(state)]. Input dim is
        # sharded across TP, so this reduces (RowParallel) to full dt/B/C.
        self.x_proj = RowParallelLinear(
            input_size=intermediate_size,
            output_size=time_step_rank + 2 * state_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.x_proj",
        )

        # dt_proj: dt_rank -> intermediate (column-sharded). dt_rank input is
        # replicated (small), so keep the input unsharded.
        self.dt_proj = ColumnParallelLinear(
            input_size=time_step_rank,
            output_size=intermediate_size,
            bias=True,
            gather_output=False,
            quant_config=quant_config,
            prefix=f"{prefix}.dt_proj",
        )

        # Full-rank A (stored as A_log) and D, sharded along the intermediate dim.
        self.A_log = nn.Parameter(
            torch.empty(self.intermediate_size_per_tp, state_size, dtype=torch.float32)
        )
        self.D = nn.Parameter(torch.ones(self.intermediate_size_per_tp))
        set_weight_attrs(self.A_log, {"weight_loader": sharded_weight_loader(0)})
        set_weight_attrs(self.D, {"weight_loader": sharded_weight_loader(0)})

        # The time-step bias is folded into dt_proj.bias (applied before the
        # scan), so the selective_state_update kernel gets a zero dt_bias. Keep
        # it as a registered buffer of shape (nheads=dim, head_dim=1); passing a
        # real tensor also avoids a `dt_bias is None` unpack path in the kernel.
        self.register_buffer(
            "dt_bias_zero",
            torch.zeros(self.intermediate_size_per_tp, 1),
            persistent=False,
        )

        # out_proj: intermediate -> hidden (input sharded, RowParallel reduces).
        self.out_proj = RowParallelLinear(
            input_size=intermediate_size,
            output_size=hidden_size,
            bias=use_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.out_proj",
        )

    def _ssm_params(
        self, conv_out: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """From convolved x (tokens, dim_per_tp) -> (dt_per_tp, B, C).

        ``dt`` is per-TP-channel (intermediate/tp); ``B``/``C`` are the full,
        replicated state selection vectors. Falcon-Mamba RMS-normalizes all three.
        """
        ssm_params, _ = self.x_proj(conv_out)
        time_step, B, C = torch.split(
            ssm_params,
            [self.time_step_rank, self.ssm_state_size, self.ssm_state_size],
            dim=-1,
        )
        if self.use_bc_dt_rms:
            B = rms_normalize(B, self.rms_eps)
            C = rms_normalize(C, self.rms_eps)
            time_step = rms_normalize(time_step, self.rms_eps)
        dt, _ = self.dt_proj(time_step)  # (tokens, intermediate/tp)
        return dt, B, C

    def forward(
        self,
        *,
        hidden_states: torch.Tensor,
        output: Optional[torch.Tensor],
        layer_cache,
        metadata,
        mup_vector: Optional[torch.Tensor] = None,
        use_triton_causal_conv: bool = False,
    ) -> Tuple[torch.Tensor, None]:
        # Matches the call contract of Mamba2AttnBackend.forward. Falcon-Mamba
        # has no speculative-decode / radix-track support yet.
        assert not metadata.is_target_verify, (
            "Mamba-1 (Falcon-Mamba) does not support speculative decoding yet"
        )

        conv_state = layer_cache.conv[0]
        ssm_state = layer_cache.temporal  # (slots, intermediate/tp, 1, state)
        state_indices = metadata.mamba_cache_indices
        query_start_loc = metadata.query_start_loc

        dim = self.intermediate_size_per_tp
        num_prefills = metadata.num_prefills
        num_prefill_tokens = metadata.num_prefill_tokens
        num_decodes = metadata.num_decodes
        num_actual_tokens = num_prefill_tokens + num_decodes

        # Project and split into x (to be convolved+scanned) and gate.
        projected, _ = self.in_proj(hidden_states)
        if mup_vector is not None:
            projected = projected * mup_vector
        x_in, gate = projected.split([dim, dim], dim=-1)
        x_in = x_in[:num_actual_tokens]
        gate = gate[:num_actual_tokens]

        conv_weights = self.conv1d.weight.view(
            self.conv1d.weight.size(0), self.conv1d.weight.size(2)
        )

        # Split varlen tokens into prefill (front) then decode (back).
        x_p, x_d = torch.split(x_in, [num_prefill_tokens, num_decodes], dim=0)
        gate_p, gate_d = torch.split(gate, [num_prefill_tokens, num_decodes], dim=0)
        state_indices_p = state_indices[:num_prefills]
        state_indices_d = state_indices[num_prefills : num_prefills + num_decodes]

        out = torch.empty(
            (num_actual_tokens, dim), dtype=hidden_states.dtype, device=x_in.device
        )
        out_p, out_d = torch.split(out, [num_prefill_tokens, num_decodes], dim=0)

        A = -torch.exp(self.A_log.float())  # (dim, state)

        if num_prefills > 0:
            self._forward_prefill(
                x=x_p,
                gate=gate_p,
                out=out_p,
                A=A,
                conv_state=conv_state,
                ssm_state=ssm_state,
                conv_weights=conv_weights,
                state_indices=state_indices_p,
                query_start_loc=query_start_loc[: num_prefills + 1],
                metadata=metadata,
                use_triton_causal_conv=use_triton_causal_conv,
            )

        if num_decodes > 0:
            self._forward_decode(
                x=x_d,
                gate=gate_d,
                out=out_d,
                A=A,
                conv_state=conv_state,
                ssm_state=ssm_state,
                conv_weights=conv_weights,
                state_indices=state_indices_d,
                use_triton_causal_conv=use_triton_causal_conv,
            )

        mixer_out, _ = self.out_proj(out)
        if output is not None:
            output[:num_actual_tokens].copy_(mixer_out)
        return mixer_out, None

    def _forward_prefill(
        self,
        *,
        x,
        gate,
        out,
        A,
        conv_state,
        ssm_state,
        conv_weights,
        state_indices,
        query_start_loc,
        metadata,
        use_triton_causal_conv,
    ):
        mixed = metadata.mixed_metadata
        has_initial = mixed.has_initial_states if mixed is not None else None
        # Per-sequence prefill token counts; required by the Triton causal-conv
        # varlen kernel (used on XPU). Fall back to deriving from query_start_loc.
        seq_lens_cpu = mixed.extend_seq_lens_cpu if mixed is not None else None
        if seq_lens_cpu is None:
            seq_lens_cpu = (query_start_loc[1:] - query_start_loc[:-1]).cpu().tolist()
        # The causal-conv kernel needs input, weights and the conv-state cache in
        # one dtype. The cache dtype (SGLANG_MAMBA_CONV_DTYPE) is independent of
        # the model dtype, so cast the conv inputs to it and the result back.
        act_dtype = x.dtype
        conv_dtype = conv_state.dtype
        ccfn = causal_conv1d_fn_triton if use_triton_causal_conv else causal_conv1d_fn
        conv_out = (
            ccfn(
                x.transpose(0, 1).to(conv_dtype),  # (dim, tokens)
                conv_weights.to(conv_dtype),
                (
                    self.conv1d.bias.to(conv_dtype)
                    if self.conv1d.bias is not None
                    else None
                ),
                activation=self.activation,
                conv_states=conv_state,
                has_initial_state=has_initial,
                cache_indices=state_indices,
                query_start_loc=query_start_loc,
                seq_lens_cpu=seq_lens_cpu,
            )
            .transpose(0, 1)[: x.shape[0]]
            .to(act_dtype)
        )  # (tokens, dim)

        dt, B, C = self._ssm_params(conv_out)

        # Sequential selective scan per sequence (portable, device-agnostic).
        seq_lens = (query_start_loc[1:] - query_start_loc[:-1]).tolist()
        for i, seqlen in enumerate(seq_lens):
            start = int(query_start_loc[i])
            end = start + seqlen
            slot = int(state_indices[i])
            if has_initial is not None and bool(has_initial[i]):
                h = ssm_state[slot, :, 0, :].float()  # (dim, state)
            else:
                h = torch.zeros(
                    A.shape[0], A.shape[1], dtype=torch.float32, device=x.device
                )
            # softplus matches HF `discrete_time_step = softplus(dt_proj(time_step))`
            # (the decode kernel applies this internally via dt_softplus=True).
            dt_seq = nn.functional.softplus(dt[start:end].float())
            h, y = self._selective_scan(
                x=conv_out[start:end].float(),  # (seqlen, dim)
                dt=dt_seq,  # (seqlen, dim)
                A=A,  # (dim, state)
                B=B[start:end].float(),  # (seqlen, state)
                C=C[start:end].float(),  # (seqlen, state)
                h0=h,
            )
            y = y + conv_out[start:end].float() * self.D.float()[None, :]
            y = y * nn.functional.silu(gate[start:end].float())
            out[start:end].copy_(y.to(out.dtype))
            # Persist the final recurrent state for subsequent decode.
            ssm_state[slot, :, 0, :].copy_(h.to(ssm_state.dtype))

    @staticmethod
    def _selective_scan(*, x, dt, A, B, C, h0):
        """Reference Mamba-1 recurrence for one sequence.

        Shapes: x/dt (T, dim); A (dim, state); B/C (T, state); h0 (dim, state).
        Returns (final_state (dim, state), y (T, dim)).

        Discretization is computed *per timestep* rather than materializing the
        full (T, dim, state) tensors up front: with dim==intermediate_size (8192)
        and prefill chunks up to 2048 tokens, a materialized (T, dim, state) is
        ~1 GB in fp32 and OOMs the XPU under concurrent prefill. The per-step
        form bounds peak activation to O(dim * state).
        """
        h = h0
        ys = []
        for t in range(x.shape[0]):
            # dA = exp(dt * A), dBx = dt * B * x  (all (dim, state) for this step).
            dt_t = dt[t][:, None]  # (dim, 1)
            dA_t = torch.exp(dt_t * A)  # (dim, state)
            dBx_t = (dt_t * B[t][None, :]) * x[t][:, None]  # (dim, state)
            h = dA_t * h + dBx_t
            ys.append((h * C[t][None, :]).sum(-1))  # (dim,)
        y = torch.stack(ys, dim=0)  # (T, dim)
        return h, y

    def _forward_decode(
        self,
        *,
        x,
        gate,
        out,
        A,
        conv_state,
        ssm_state,
        conv_weights,
        state_indices,
        use_triton_causal_conv,
    ):
        # Match the conv-state cache dtype (see _forward_prefill), then cast the
        # result back to the activation dtype before the x_proj matmul.
        act_dtype = x.dtype
        conv_dtype = conv_state.dtype
        ccu = (
            causal_conv1d_update_triton
            if use_triton_causal_conv
            else causal_conv1d_update
        )
        conv_out = ccu(
            x.to(conv_dtype),
            conv_state,
            conv_weights.to(conv_dtype),
            self.conv1d.bias.to(conv_dtype) if self.conv1d.bias is not None else None,
            self.activation,
            conv_state_indices=state_indices,
        ).to(act_dtype)

        dt, B, C = self._ssm_params(conv_out)

        # Map onto the shared selective_state_update kernel with the
        # (nheads=dim, head_dim=1, ngroups=1) full-rank layout. z=gate applies
        # the silu output gate; D is the per-channel skip connection.
        n_decode = x.shape[0]
        dim = self.intermediate_size_per_tp
        A_k = A[:, None, :]  # (dim, 1, state)
        D_k = self.D.float()[:, None]  # (dim, 1)
        x_k = conv_out.view(n_decode, dim, 1)
        dt_k = dt.view(n_decode, dim, 1)
        gate_k = gate.view(n_decode, dim, 1)
        B_k = B.view(n_decode, 1, self.ssm_state_size)
        C_k = C.view(n_decode, 1, self.ssm_state_size)
        out_k = out.view(n_decode, dim, 1)
        selective_state_update(
            ssm_state,
            x_k,
            dt_k,
            A_k,
            B_k,
            C_k,
            D_k,
            z=gate_k,
            dt_bias=self.dt_bias_zero,
            dt_softplus=True,
            state_batch_indices=state_indices,
            out=out_k,
        )

    @property
    def mamba_type(self) -> str:
        return "mamba1"
