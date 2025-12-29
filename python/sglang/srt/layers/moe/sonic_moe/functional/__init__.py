# ********************************************************************************
# Copyright (c) 2025, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
# ********************************************************************************

import os
from typing import Optional

import torch
import torch.nn.functional as F
from quack.gemm_interface import gemm

from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.sonic_moe.count_cumsum import count_cumsum
from sglang.srt.layers.moe.sonic_moe.enums import ActivationType, is_glu
from sglang.srt.layers.moe.sonic_moe.quack_utils import gemm_dgated, gemm_gated

from .backward import (
    _down_projection_backward,
    _softmax_topk_bwd,
    _token_broadcast_backward,
    _up_projection_backward,
)
from .forward import (
    _down_projection_forward,
    _router_forward,
    _softmax_topk_fwd,
    _up_projection_forward,
)
from .utils import enable_quack_gemm, is_using_quack_gemm


def TC_topk_router_metadata(
    topk_router_indices: torch.Tensor, expert_frequency_offset: torch.Tensor, K: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    s_scatter_idx = torch.argsort(topk_router_indices.view(-1)).int()
    expert_frequency_offset = torch.cat(
        [
            torch.zeros(
                1,
                device=expert_frequency_offset.device,
                dtype=expert_frequency_offset.dtype,
            ),
            expert_frequency_offset,
        ]
    )
    # s_reverse_scatter_idx = torch.argsort(s_scatter_idx).int()
    s_reverse_scatter_idx = torch.empty_like(s_scatter_idx)
    s_reverse_scatter_idx[s_scatter_idx] = torch.arange(
        s_scatter_idx.shape[0], device=s_scatter_idx.device, dtype=s_scatter_idx.dtype
    )

    num_activated_expert_per_token_offset = torch.arange(
        0,
        topk_router_indices.shape[0] * K + 1,
        K,
        dtype=torch.int32,
        device=topk_router_indices.device,
    )
    x_gather_idx = s_scatter_idx // K

    return (
        expert_frequency_offset,
        x_gather_idx,
        s_scatter_idx,
        s_reverse_scatter_idx,
        num_activated_expert_per_token_offset,
    )


def general_routing_router_metadata(
    router_scores_selected: torch.Tensor,
    sorted_selected_T: torch.Tensor,
    selected_E: torch.Tensor,
    T: int,
    E: int,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:

    device = router_scores_selected.device

    expert_frequency, expert_frequency_offset = count_cumsum(
        selected_E, E, do_cumsum=True
    )
    expert_frequency_offset = torch.cat(
        [torch.zeros(1, dtype=torch.int32, device=device), expert_frequency_offset]
    )

    s_scatter_idx = selected_E.argsort().int()
    s_reverse_scatter_idx = torch.empty_like(s_scatter_idx)
    s_reverse_scatter_idx[s_scatter_idx] = torch.arange(
        s_scatter_idx.size(0), device=s_scatter_idx.device, dtype=s_scatter_idx.dtype
    )

    x_gather_idx = sorted_selected_T[s_scatter_idx]

    if T % 4 == 0 and T <= 50000:
        _, num_activated_expert_per_token_offset = count_cumsum(
            sorted_selected_T, T, do_cumsum=True
        )
    else:
        num_activated_expert_per_token_offset = (
            torch.bincount(sorted_selected_T, minlength=T).cumsum(0).int()
        )

    num_activated_expert_per_token_offset = torch.cat(
        [
            torch.zeros(1, dtype=torch.int32, device=device),
            num_activated_expert_per_token_offset,
        ]
    )

    return (
        expert_frequency,
        expert_frequency_offset,
        x_gather_idx,
        s_scatter_idx,
        s_reverse_scatter_idx,
        num_activated_expert_per_token_offset,
    )


def sonic_moe_forward_placeholder(
    hidden_states: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    b13: Optional[torch.Tensor],
    b2: Optional[torch.Tensor],
    router_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    activation_type: ActivationType,
    config: MoeRunnerConfig,
) -> torch.Tensor:
    """
    Placeholder for actual SonicMoE kernel.
    Replace this with the real SonicMoE implementation.
    """
    assert hidden_states.size(0) == selected_experts.size(0), "Batch size mismatch"
    assert hidden_states.size(1) == config.hidden_size, "Hidden size mismatch"
    assert w13.size(2) == config.num_experts, "W13 expert count mismatch"
    assert w13.size(0) == (
        config.intermediate_size_per_partition * 2
        if config.is_gated
        else config.intermediate_size_per_partition
    ), "W13 intermediate size mismatch"
    assert w13.size(1) == config.hidden_size, "W13 hidden size mismatch"
    assert w2.size(2) == config.num_experts, "W2 expert count mismatch"
    assert w2.size(0) == config.hidden_size, "W2 hidden size mismatch"
    assert (
        w2.size(1) == config.intermediate_size_per_partition
    ), "W2 intermediate size mismatch"
    assert selected_experts.size(1) == config.top_k, "Input feature size mismatch"

    T = hidden_states.size(0)
    K = config.top_k

    expert_frequency, expert_frequency_offset = count_cumsum(
        selected_experts.view(-1), config.num_experts, do_cumsum=True
    )
    (
        expert_frequency_offset,
        x_gather_idx,
        s_scatter_idx,
        s_reverse_scatter_idx,
        num_activated_expert_per_token_offset,
    ) = TC_topk_router_metadata(selected_experts, expert_frequency_offset, K)

    y1, z = _UpProjection.apply(
        hidden_states,
        w13,
        b13,
        expert_frequency_offset,
        T * K,
        K,
        config.stream_id,
        x_gather_idx,
        s_scatter_idx,
        s_reverse_scatter_idx,
        num_activated_expert_per_token_offset,
        False,  # is_varlen_K
        activation_type,
        True,  # inference_mode_enabled
    )

    hidden_states = _DownProjection.apply(
        y1,
        z,
        w2,
        b2,
        router_weights,
        expert_frequency_offset,
        T,
        K,
        config.stream_id,
        x_gather_idx,
        s_scatter_idx,
        s_reverse_scatter_idx,
        num_activated_expert_per_token_offset,
        False,  # is_varlen_K
        activation_type,
    )
    return hidden_states


class TC_Softmax_Topk_Router_Function(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, router_logits: torch.Tensor, E: int, K: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        T = router_logits.size(0)

        # change this to router_logits.dtype (bfloat16) increase another 5 tflops at fwd at the cost of numerical accuracy
        topk_router_score = torch.empty(
            T, K, dtype=torch.float32, device=router_logits.device
        )
        topk_router_indices = torch.empty(
            T, K, dtype=torch.int32, device=router_logits.device
        )

        _softmax_topk_fwd(router_logits, topk_router_score, topk_router_indices, E, K)

        ctx.save_for_backward(topk_router_score, topk_router_indices)
        ctx.E = E
        ctx.dtype = router_logits.dtype

        return topk_router_score, topk_router_indices

    @staticmethod
    def backward(
        ctx, dtopk_score: torch.Tensor, _: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        T, K = dtopk_score.size()

        topk_router_score, topk_router_indices = ctx.saved_tensors
        dlogits = torch.zeros(
            T, ctx.E, dtype=ctx.dtype, device=topk_router_score.device
        )

        _softmax_topk_bwd(
            dlogits, None, dtopk_score, topk_router_score, topk_router_indices, K
        )

        return dlogits, None, None


class _UpProjection(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        w1: torch.Tensor,
        b1: torch.Tensor | None,
        expert_frequency_offset: torch.Tensor,
        total_expert_freq: int,
        K: int,
        stream_id: int,
        x_gather_idx: torch.Tensor,
        s_scatter_idx: torch.Tensor,
        s_reverse_scatter_idx: torch.Tensor,
        num_activated_expert_per_token_offset: torch.Tensor,
        is_varlen_K: bool,
        activation_type: ActivationType,
        is_inference_mode_enabled: bool,
    ) -> torch.Tensor:
        T, H = x.shape
        I, H, E = w1.shape
        is_glu_activation = is_glu(activation_type)
        if is_glu_activation:
            I //= 2
        TK = total_expert_freq

        if is_using_quack_gemm():
            assert not torch.compiler.is_compiling()
            assert (
                is_glu_activation
            ), "QuACK GEMM does not support non GLU activation yet"
            z, y1 = gemm_gated(
                x,
                w1.permute(2, 1, 0),
                activation="swiglu",
                cu_seqlens_m=expert_frequency_offset,
                A_idx=x_gather_idx,
                dynamic_scheduler=False,
            )
        else:
            z = torch.empty(
                TK, (2 * I if is_glu_activation else I), dtype=x.dtype, device=x.device
            )
            y1 = torch.empty(TK, I, dtype=x.dtype, device=x.device)
            _up_projection_forward(
                x=x,
                w1=w1,
                z=z,
                y1=y1,
                b1=b1,
                expert_frequency_offset=expert_frequency_offset,
                expert_schedule_order=None,
                x_gather_idx=x_gather_idx,
                stream_id=stream_id,
                activation_type=activation_type.value,
                is_glu_activation=is_glu_activation,
                is_inference_mode_enabled=is_inference_mode_enabled,
            )

        ctx.T = T
        ctx.TK = TK
        ctx.E = E
        ctx.K = K
        ctx.H = H
        ctx.I = I
        ctx.is_varlen_K = is_varlen_K
        ctx.is_glu_activation = is_glu_activation
        ctx.stream_id = stream_id

        ctx.save_for_backward(
            x,
            w1,
            b1,
            expert_frequency_offset,
            x_gather_idx,
            s_scatter_idx,
            s_reverse_scatter_idx,
            num_activated_expert_per_token_offset,
        )

        ctx.mark_non_differentiable(y1)
        ctx.set_materialize_grads(False)

        return y1, z

    @staticmethod
    def backward(ctx, _: None, dz: torch.Tensor):
        is_compiling = torch.compiler.is_compiling()

        if not is_compiling:
            assert _ is None

        T = ctx.T
        TK = ctx.TK
        E = ctx.E
        K = ctx.K
        H = ctx.H
        is_glu_activation = ctx.is_glu_activation
        is_varlen_K = ctx.is_varlen_K
        stream_id = ctx.stream_id

        (
            x,
            w1,
            b1,
            expert_frequency_offset,
            x_gather_idx,
            s_scatter_idx,
            s_reverse_scatter_idx,
            num_activated_expert_per_token_offset,
        ) = ctx.saved_tensors

        dw1 = torch.empty_like(w1)
        db1 = None if b1 is None else torch.empty_like(b1)

        if is_using_quack_gemm():
            assert not is_compiling

            gemm(
                x.T,
                dz,
                out=dw1.permute(2, 1, 0),
                cu_seqlens_k=expert_frequency_offset,
                A_idx=x_gather_idx,
                batch_idx_permute=None,
                dynamic_scheduler=False,
            )
            dx_expanded = gemm(
                dz,
                w1.permute(2, 0, 1),
                cu_seqlens_m=expert_frequency_offset,
                dynamic_scheduler=False,
            )
        else:
            dx_expanded = torch.empty(TK, H, dtype=dz.dtype, device=dz.device)
            _up_projection_backward(
                x=x,
                w1=w1,
                dx_expanded=dx_expanded,
                dw1=dw1,
                dz=dz,
                db1=db1,
                expert_frequency_offset=expert_frequency_offset,
                expert_schedule_order=None,
                x_gather_idx=x_gather_idx,
                s_scatter_idx=s_scatter_idx,
                is_glu_activation=is_glu_activation,
                stream_id=stream_id,
            )

        dx_reduced = torch.empty(T, H, dtype=dz.dtype, device=dz.device)

        _token_broadcast_backward(
            dx_reduced=dx_reduced,
            dx_expanded=dx_expanded,
            s_reverse_scatter_idx=s_reverse_scatter_idx,
            num_activated_expert_per_token_offset=num_activated_expert_per_token_offset,
            varlen_K_max=(E if is_varlen_K else K),
            H=H,
            is_varlen_K=is_varlen_K,
        )

        return dx_reduced, dw1, db1, *[None] * 12


class _DownProjection(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        y1: torch.Tensor,
        z: torch.Tensor,
        w2: torch.Tensor,
        b2: torch.Tensor | None,
        topk_scores: torch.Tensor,
        expert_frequency_offset: torch.Tensor,
        T: int,
        K: int,
        stream_id: int,
        x_gather_idx: torch.Tensor,
        s_scatter_idx: torch.Tensor,
        s_reverse_scatter_idx: torch.Tensor,
        num_activated_expert_per_token_offset: torch.Tensor,
        is_varlen_K: bool,
        activation_type: ActivationType,
    ) -> torch.Tensor:
        TK = y1.size(0)
        H, I, E = w2.shape

        if is_using_quack_gemm():
            assert not torch.compiler.is_compiling()

            assert b2 is None
            y2 = gemm(y1, w2.permute(2, 1, 0), cu_seqlens_m=expert_frequency_offset)
        else:
            y2 = torch.empty(TK, H, dtype=y1.dtype, device=y1.device)
            _down_projection_forward(
                w2=w2,
                y1=y1,
                y2=y2,
                b2=b2,
                expert_frequency_offset=expert_frequency_offset,
                expert_schedule_order=None,
                x_gather_idx=x_gather_idx,
                stream_id=stream_id,
            )

        o = torch.empty(T, H, device=z.device, dtype=z.dtype)
        topk_scores = topk_scores.flatten()

        _router_forward(
            y2=y2,
            o=o,
            topk_scores=topk_scores,
            s_reverse_scatter_idx=s_reverse_scatter_idx,
            num_activated_expert_per_token_offset=num_activated_expert_per_token_offset,
            varlen_K_max=(E if is_varlen_K else K),
            H=H,
            is_varlen_K=is_varlen_K,
        )

        ctx.T = T
        ctx.K = K
        ctx.is_varlen_K = is_varlen_K
        ctx.activation_type = activation_type
        ctx.stream_id = stream_id

        ctx.save_for_backward(
            z,
            w2,
            b2,
            topk_scores,
            expert_frequency_offset,
            x_gather_idx,
            s_scatter_idx,
            s_reverse_scatter_idx,
        )

        return o

    @staticmethod
    def backward(ctx, dout: torch.Tensor):
        T = ctx.T
        K = ctx.K
        stream_id = ctx.stream_id
        is_varlen_K = ctx.is_varlen_K
        activation_type = ctx.activation_type

        (
            z,
            w2,
            b2,
            topk_scores,
            expert_frequency_offset,
            x_gather_idx,
            s_scatter_idx,
            s_reverse_scatter_idx,
        ) = ctx.saved_tensors

        dw2 = torch.empty_like(w2)
        db2 = None if b2 is None else torch.empty_like(b2)
        dz = torch.empty_like(z)

        if is_using_quack_gemm():
            assert not torch.compiler.is_compiling()
            assert is_glu(
                activation_type
            ), "QuACK GEMM does not support non GLU activation yet"

            s = topk_scores[s_scatter_idx]
            _, y1s, ds = gemm_dgated(
                dout,
                w2.permute(2, 0, 1),
                PreAct=z,
                activation="swiglu",
                dx_out=dz,
                colvec_scale=s,
                colvec_reduce=True,
                cu_seqlens_m=expert_frequency_offset,
                A_idx=x_gather_idx,
                dynamic_scheduler=False,
            )
            gemm(
                dout.T,
                y1s,
                out=dw2.permute(2, 0, 1),
                cu_seqlens_k=expert_frequency_offset,
                A_idx=x_gather_idx,
                batch_idx_permute=None,
                dynamic_scheduler=False,
            )

            ds = ds[s_reverse_scatter_idx]
        else:
            ds = torch.empty_like(topk_scores)
            _down_projection_backward(
                dout=dout,
                z=z,
                w2=w2,
                dw2=dw2,
                dz=dz,
                ds=ds,
                b2=b2,
                db2=db2,
                topk_scores=topk_scores,
                expert_frequency_offset=expert_frequency_offset,
                expert_schedule_order=None,
                x_gather_idx=x_gather_idx,
                s_scatter_idx=s_scatter_idx,
                stream_id=stream_id,
                is_glu_activation=is_glu(activation_type),
                activation_type=activation_type.value,
            )

        # TC top-K routing
        if not is_varlen_K:
            ds = ds.view(T, K)

        return None, dz, dw2, db2, ds, *[None] * 10


def moe_TC_softmax_topk_layer(
    x: torch.Tensor,
    router_w: torch.Tensor,
    w1: torch.Tensor,
    b1: torch.Tensor | None,
    w2: torch.Tensor,
    b2: torch.Tensor | None,
    K: int,
    stream_id: int,
    activation_type: ActivationType | str = ActivationType.SWIGLU,
    is_inference_mode_enabled: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert ((b1 is None) and (b2 is None)) or (
        (b1 is not None) and (b2 is not None)
    ), "b1 and b2 has to be None or not None at the same time!"
    router_logits = F.linear(x, router_w)
    topk_scores, topk_indices = TC_Softmax_Topk_Router_Function.apply(
        router_logits, router_w.size(0), K
    )
    expert_frequency, expert_frequency_offset = count_cumsum(
        topk_indices.view(-1), router_w.size(0), do_cumsum=True
    )

    (
        expert_frequency_offset,
        x_gather_idx,
        s_scatter_idx,
        s_reverse_scatter_idx,
        num_activated_expert_per_token_offset,
    ) = TC_topk_router_metadata(topk_indices, expert_frequency_offset, K)

    T = x.size(0)

    if type(activation_type) == str:
        activation_type = ActivationType(activation_type)

    y1, z = _UpProjection.apply(
        x,
        w1,
        b1,
        expert_frequency_offset,
        T * K,
        K,
        stream_id,
        x_gather_idx,
        s_scatter_idx,
        s_reverse_scatter_idx,
        num_activated_expert_per_token_offset,
        False,  # is_varlen_K
        activation_type,
        is_inference_mode_enabled,
    )

    o = _DownProjection.apply(
        y1,
        z,
        w2,
        b2,
        topk_scores,
        expert_frequency_offset,
        T,
        K,
        stream_id,
        x_gather_idx,
        s_scatter_idx,
        s_reverse_scatter_idx,
        num_activated_expert_per_token_offset,
        False,  # is_varlen_K
        activation_type,
    )

    return o, router_logits, expert_frequency


# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# We assume token_indices is already SORTED ascendingly !!!
#   and len(token_indices) = len(expert_indices) = len(router_scores)
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
def moe_general_routing_inputs(
    x: torch.Tensor,
    router_scores: torch.Tensor,
    token_indices: torch.Tensor,
    expert_indices: torch.Tensor,
    w1: torch.Tensor,
    b1: torch.Tensor | None,
    w2: torch.Tensor,
    b2: torch.Tensor | None,
    E: int,
    stream_id: int,
    activation_type: ActivationType,
    is_inference_mode_enabled: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert ((b1 is None) and (b2 is None)) or (
        (b1 is not None) and (b2 is not None)
    ), "b1 and b2 has to be None or not None at the same time!"

    T = x.size(0)
    TK = router_scores.size(0)
    E = w2.size(-1)
    (
        expert_frequency,
        expert_frequency_offset,
        x_gather_idx,
        s_scatter_idx,
        s_reverse_scatter_idx,
        num_activated_expert_per_token_offset,
    ) = general_routing_router_metadata(
        router_scores, token_indices, expert_indices, T, E
    )

    y1, z = _UpProjection.apply(
        x,
        w1,
        b1,
        expert_frequency_offset,
        TK,
        None,  # K, not needed
        stream_id,
        x_gather_idx,
        s_scatter_idx,
        s_reverse_scatter_idx,
        num_activated_expert_per_token_offset,
        True,  # is_varlen_K
        activation_type,
        is_inference_mode_enabled,
    )

    o = _DownProjection.apply(
        y1,
        z,
        w2,
        b2,
        router_scores,
        expert_frequency_offset,
        T,
        None,  # K, not needed
        stream_id,
        x_gather_idx,
        s_scatter_idx,
        s_reverse_scatter_idx,
        num_activated_expert_per_token_offset,
        True,  # is_varlen_K
        activation_type,
    )

    return o, expert_frequency
