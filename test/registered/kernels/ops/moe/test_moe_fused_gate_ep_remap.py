"""GPU equivalence, strided-view, fallback, and CUDA-graph tests for fused EP router remap."""

from __future__ import annotations

import contextlib
import unittest.mock
from types import SimpleNamespace

import pytest
import torch

from sglang.kernels.ops.moe import moe_fused_gate as moe_fused_gate_mod
from sglang.kernels.ops.moe.moe_fused_gate import moe_fused_gate
from sglang.srt.configs.shared_experts import SharedExpertsConfig
from sglang.srt.layers.moe import topk as topk_mod
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.token_dispatcher import standard as standard_mod
from sglang.srt.layers.moe.token_dispatcher.standard import (
    StandardDispatcher,
    _map_topk_ids_to_local_experts,
)
from sglang.srt.layers.moe.topk import (
    StandardTopKOutputLocalMapped,
    TopKConfig,
    select_experts,
)
from sglang.srt.layers.moe.utils import MoeA2ABackend, MoeRunnerBackend
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-small")


def _reference_unfused_gate(
    scores: torch.Tensor,
    bias: torch.Tensor | None,
    topk: int,
    *,
    scoring_func: str,
    renormalize: bool = True,
    num_fused_shared_experts: int = 0,
    routed_scaling_factor: float = 1.0,
    apply_routed_scaling_factor_on_output: bool = False,
    num_expert_group: int = 1,
    topk_group: int = 1,
    num_token_non_padded: torch.Tensor | None = None,
    ep_size: int = 2,
    ep_rank: int = 0,
    interleaved: bool = False,
):
    with unittest.mock.patch.object(
        moe_fused_gate_mod.moe_route_warp, "covered", return_value=False
    ):
        w_ref, ids_ref = moe_fused_gate(
            scores,
            bias,
            topk,
            scoring_func=scoring_func,
            num_fused_shared_experts=num_fused_shared_experts,
            renormalize=renormalize,
            routed_scaling_factor=routed_scaling_factor,
            apply_routed_scaling_factor_on_output=apply_routed_scaling_factor_on_output,
            num_expert_group=num_expert_group,
            topk_group=topk_group,
            num_token_non_padded=None,
        )
    m_rows, num_experts = scores.shape
    if num_token_non_padded is not None and m_rows > 0:
        rows = torch.arange(m_rows, device=scores.device)[:, None]
        pad_mask = rows >= num_token_non_padded
        w_ref = torch.where(pad_mask, torch.zeros_like(w_ref), w_ref)
        ids_ref = torch.where(pad_mask, -1, ids_ref)

    num_local_routed = num_experts // ep_size
    mapping = torch.full(
        (num_experts + num_fused_shared_experts,),
        -1,
        dtype=torch.int32,
        device=scores.device,
    )
    if interleaved:
        mapping[ep_rank:num_experts:ep_size] = torch.arange(
            num_local_routed, dtype=torch.int32, device=scores.device
        )
    else:
        mapping[ep_rank * num_local_routed : (ep_rank + 1) * num_local_routed] = (
            torch.arange(num_local_routed, dtype=torch.int32, device=scores.device)
        )
    if num_fused_shared_experts > 0:
        mapping[num_experts:] = torch.arange(
            num_local_routed,
            num_local_routed + num_fused_shared_experts,
            dtype=torch.int32,
            device=scores.device,
        )
    ids_mapped = torch.where(ids_ref < 0, -1, mapping[ids_ref.clamp_min(0)])
    return w_ref, ids_mapped


@pytest.mark.parametrize("m_rows", [0, 1, 7, 48, 64, 1025])
@pytest.mark.parametrize("ep_size", [2, 4])
@pytest.mark.parametrize("interleaved", [False, True])
def test_boundary_tokens_and_padding(m_rows: int, ep_size: int, interleaved: bool):
    torch.manual_seed(42 + m_rows + ep_size)
    device = torch.device("cuda")
    num_experts, topk = 128, 8
    scores = torch.randn((m_rows, num_experts), dtype=torch.bfloat16, device=device)

    non_padded_options = [None, 0, 1]
    if m_rows >= 2:
        non_padded_options.append(m_rows // 2)
    if m_rows >= 1:
        non_padded_options.append(m_rows)

    for non_padded_val in non_padded_options:
        num_non_padded = (
            None
            if non_padded_val is None
            else torch.tensor([non_padded_val], dtype=torch.int32, device=device)
        )
        for ep_rank in range(ep_size):
            w_ref, ids_ref = _reference_unfused_gate(
                scores,
                None,
                topk,
                scoring_func="softmax",
                num_token_non_padded=num_non_padded,
                ep_size=ep_size,
                ep_rank=ep_rank,
                interleaved=interleaved,
            )
            kwargs = (
                {"ep_interleaved_size": ep_size, "ep_rank": ep_rank}
                if interleaved
                else {
                    "ep_start": ep_rank * (num_experts // ep_size),
                    "ep_end": (ep_rank + 1) * (num_experts // ep_size),
                }
            )
            w_fused, ids_fused = moe_fused_gate(
                scores,
                None,
                topk,
                scoring_func="softmax",
                renormalize=True,
                num_token_non_padded=num_non_padded,
                **kwargs,
            )
            assert torch.equal(ids_ref, ids_fused)
            assert torch.allclose(w_ref, w_fused, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize(
    "num_experts,topk,shared,scoring,bias_dtype,n_group,topk_group,apply_scale,ep_size",
    [
        (128, 8, 0, "softmax", None, 1, 1, False, 2),
        (128, 8, 0, "softmax", None, 1, 1, True, 4),
        (384, 8, 0, "sigmoid", torch.bfloat16, 1, 1, False, 2),
        (384, 8, 0, "sigmoid", torch.float32, 1, 1, True, 4),
        (256, 9, 1, "sqrtsoftplus", torch.bfloat16, 1, 1, False, 2),
        (256, 9, 1, "softmax", None, 1, 1, True, 4),
        (256, 8, 0, "sigmoid", torch.float32, 8, 4, True, 4),
    ],
)
@pytest.mark.parametrize("interleaved", [False, True])
def test_topologies_and_scoring_funcs(
    num_experts: int,
    topk: int,
    shared: int,
    scoring: str,
    bias_dtype: torch.dtype | None,
    n_group: int,
    topk_group: int,
    apply_scale: bool,
    ep_size: int,
    interleaved: bool,
):
    torch.manual_seed(123)
    device = torch.device("cuda")
    m_rows = 37
    scores = torch.randn((m_rows, num_experts), dtype=torch.bfloat16, device=device)
    bias = (
        None
        if bias_dtype is None
        else torch.randn((num_experts,), dtype=bias_dtype, device=device) * 0.1
    )
    num_non_padded = torch.tensor([23], dtype=torch.int32, device=device)
    scale = 2.5

    for ep_rank in (0, ep_size - 1):
        w_ref, ids_ref = _reference_unfused_gate(
            scores,
            bias,
            topk,
            scoring_func=scoring,
            renormalize=True,
            num_fused_shared_experts=shared,
            routed_scaling_factor=scale,
            apply_routed_scaling_factor_on_output=apply_scale,
            num_expert_group=n_group,
            topk_group=topk_group,
            num_token_non_padded=num_non_padded,
            ep_size=ep_size,
            ep_rank=ep_rank,
            interleaved=interleaved,
        )
        kwargs = (
            {"ep_interleaved_size": ep_size, "ep_rank": ep_rank}
            if interleaved
            else {
                "ep_start": ep_rank * (num_experts // ep_size),
                "ep_end": (ep_rank + 1) * (num_experts // ep_size),
            }
        )
        w_fused, ids_fused = moe_fused_gate(
            scores,
            bias,
            topk,
            scoring_func=scoring,
            num_fused_shared_experts=shared,
            renormalize=True,
            routed_scaling_factor=scale,
            apply_routed_scaling_factor_on_output=apply_scale,
            num_expert_group=n_group,
            topk_group=topk_group,
            num_token_non_padded=num_non_padded,
            **kwargs,
        )
        assert torch.equal(ids_ref, ids_fused)
        assert torch.allclose(w_ref, w_fused, atol=1e-6, rtol=1e-5)


def test_offset_and_strided_views():
    torch.manual_seed(7)
    device = torch.device("cuda")
    buf = torch.randn((160, 256), dtype=torch.bfloat16, device=device)
    offset_scores = buf[16:80, :128]
    strided_scores = buf[16:80, ::2]
    num_non_padded = torch.tensor([41], dtype=torch.int32, device=device)

    for view in (offset_scores, strided_scores):
        for ep_rank in (0, 1):
            w_ref, ids_ref = _reference_unfused_gate(
                view,
                None,
                8,
                scoring_func="softmax",
                num_token_non_padded=num_non_padded,
                ep_size=2,
                ep_rank=ep_rank,
            )
            w_view, ids_view = moe_fused_gate(
                view,
                None,
                8,
                scoring_func="softmax",
                num_token_non_padded=num_non_padded,
                ep_start=ep_rank * 64,
                ep_end=(ep_rank + 1) * 64,
            )
            assert torch.equal(ids_ref, ids_view)
            assert torch.allclose(w_ref, w_view, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_standalone_map_topk_ids(dtype: torch.dtype, monkeypatch):
    device = torch.device("cuda")
    num_experts = 128
    mapping = torch.full((num_experts,), -1, dtype=torch.int32, device=device)
    mapping[64:128] = torch.arange(64, dtype=torch.int32, device=device)

    for m_rows in (0, 1, 64):
        raw = torch.randint(-1, num_experts, (m_rows, 8), dtype=dtype, device=device)
        if m_rows > 0:
            raw[0, 0] = -1
            raw[0, 1] = 0
            raw[0, 2] = 63
            raw[0, 3] = 64
            raw[0, 4] = num_experts - 1
        expected = torch.where(
            raw < 0,
            torch.tensor(-1, dtype=dtype, device=device),
            mapping[raw.clamp(0, num_experts - 1)].to(dtype),
        )

        monkeypatch.delenv("SGLANG_DISABLE_FUSED_EP_REMAP", raising=False)
        out_triton = _map_topk_ids_to_local_experts(raw, mapping)
        assert out_triton.dtype == dtype
        assert torch.equal(out_triton, expected)

        monkeypatch.setenv("SGLANG_DISABLE_FUSED_EP_REMAP", "1")
        out_eager = _map_topk_ids_to_local_experts(raw, mapping)
        assert torch.equal(out_eager, expected)

        if m_rows > 0:
            oob = raw.clone()
            oob[0, 0] = num_experts
            oob[0, 1] = num_experts + 5
            monkeypatch.delenv("SGLANG_DISABLE_FUSED_EP_REMAP", raising=False)
            out_oob = _map_topk_ids_to_local_experts(oob, mapping)
            assert int(out_oob[0, 0].item()) == -1
            assert int(out_oob[0, 1].item()) == -1

    monkeypatch.delenv("SGLANG_DISABLE_FUSED_EP_REMAP", raising=False)
    buf = torch.randint(0, num_experts, (16, 8), dtype=dtype, device=device)
    non_contig = buf[::2, :]
    expected_nc = torch.where(
        non_contig < 0, -1, mapping[non_contig.clamp_min(0)].to(dtype)
    )
    out_nc = _map_topk_ids_to_local_experts(non_contig, mapping)
    assert torch.equal(out_nc, expected_nc)


def _mock_ep_runtime(monkeypatch, *, ep_size: int = 2, ep_rank: int = 1):
    from sglang.srt.layers import dp_attention as dp_attn_mod
    from sglang.srt.layers.moe import utils as moe_utils_mod

    parallel = SimpleNamespace(
        moe_ep_size=ep_size,
        moe_ep_rank=ep_rank,
        moe_tp_size=1,
        moe_tp_rank=0,
        dwdp_size=1,
        tp_group=None,
    )
    exec_cfg = SimpleNamespace(
        moe=SimpleNamespace(ep_join_mode=None),
        comm=SimpleNamespace(enable_quant_communications=False),
    )
    monkeypatch.setattr(standard_mod, "get_parallel", lambda: parallel)
    monkeypatch.setattr(
        standard_mod,
        "get_lora",
        lambda: SimpleNamespace(enable_lora=False, lora_paths=None),
    )
    monkeypatch.setattr(standard_mod, "get_device", lambda: torch.device("cuda"))
    monkeypatch.setattr(standard_mod, "is_allocation_symmetric", lambda: False)
    monkeypatch.setattr(
        standard_mod, "should_use_flashinfer_cutlass_moe_fp4_allgather", lambda: False
    )
    monkeypatch.setattr(
        standard_mod, "use_symmetric_memory", lambda *a, **kw: contextlib.nullcontext()
    )
    monkeypatch.setattr(standard_mod, "is_dp_max_padding", lambda: False)
    monkeypatch.setattr(
        standard_mod, "get_moe_runner_backend", lambda: MoeRunnerBackend.TRITON
    )

    monkeypatch.setattr(
        moe_utils_mod, "get_moe_runner_backend", lambda: MoeRunnerBackend.TRITON
    )
    monkeypatch.setattr(
        moe_utils_mod, "get_moe_a2a_backend", lambda: MoeA2ABackend.NONE
    )
    monkeypatch.setattr(
        moe_utils_mod, "should_use_flashinfer_cutlass_moe_fp4_allgather", lambda: False
    )
    monkeypatch.setattr(dp_attn_mod, "is_dp_max_padding", lambda: False)
    monkeypatch.setattr(dp_attn_mod, "is_allocation_symmetric", lambda: False)

    monkeypatch.setattr(topk_mod, "get_parallel", lambda: parallel)
    monkeypatch.setattr(topk_mod, "get_exec", lambda: exec_cfg)
    monkeypatch.setattr(topk_mod, "get_moe_a2a_backend", lambda: MoeA2ABackend.NONE)
    monkeypatch.setattr(
        topk_mod, "get_moe_runner_backend", lambda: MoeRunnerBackend.TRITON
    )
    monkeypatch.setattr(
        topk_mod, "should_use_flashinfer_cutlass_moe_fp4_allgather", lambda: False
    )
    monkeypatch.setattr(topk_mod, "is_dp_max_padding", lambda: False)
    monkeypatch.setattr(topk_mod, "is_expert_remapping_enabled", lambda: False)
    monkeypatch.setattr(topk_mod, "get_global_experts_capturer", lambda: None)
    monkeypatch.setattr(
        topk_mod,
        "get_global_expert_distribution_recorder",
        lambda: SimpleNamespace(
            recording=False,
            get_current_pass_count_buffer=lambda: None,
            on_select_experts=lambda **kw: None,
        ),
    )
    monkeypatch.setattr(topk_mod, "is_allocation_symmetric", lambda: False)
    monkeypatch.setattr(
        topk_mod, "use_symmetric_memory", lambda *a, **kw: contextlib.nullcontext()
    )
    return parallel, exec_cfg


def test_select_experts_and_cuda_graph_replay(monkeypatch):
    _mock_ep_runtime(monkeypatch, ep_size=2, ep_rank=1)
    monkeypatch.delenv("SGLANG_DISABLE_FUSED_EP_REMAP", raising=False)
    monkeypatch.delenv("SGLANG_EP_INTERLEAVED_EXPERTS", raising=False)

    device = torch.device("cuda")
    # Verify EP remap bypasses moe_route_warp cleanly even on warp-covered shapes
    warp_scores = torch.randn((8, 128), dtype=torch.bfloat16, device=device)
    w_warp_bypass, ids_warp_bypass = moe_fused_gate(
        warp_scores,
        None,
        8,
        scoring_func="softmax",
        renormalize=True,
        ep_start=64,
        ep_end=128,
    )
    w_warp_ref, ids_warp_ref = _reference_unfused_gate(
        warp_scores,
        None,
        8,
        scoring_func="softmax",
        ep_size=2,
        ep_rank=1,
    )
    assert torch.equal(ids_warp_bypass, ids_warp_ref)
    assert torch.allclose(w_warp_bypass, w_warp_ref, atol=1e-6, rtol=1e-5)

    monkeypatch.setattr(
        moe_fused_gate_mod.moe_route_warp, "covered", lambda *a, **kw: False
    )
    for scoring, shared in [
        ("softmax", 0),
        ("softmax", 1),
        ("sigmoid", 0),
        ("sqrtsoftplus", 1),
    ]:
        num_experts = 128
        m_rows = 8
        total_topk = 8 + shared
        hidden = torch.randn((m_rows, 64), dtype=torch.bfloat16, device=device)
        logits = torch.randn((m_rows, num_experts), dtype=torch.bfloat16, device=device)
        bias = (
            None
            if scoring == "softmax"
            else torch.randn((num_experts,), dtype=torch.bfloat16, device=device) * 0.1
        )
        non_pad = torch.tensor([5], dtype=torch.int32, device=device)

        cfg = TopKConfig(
            top_k=total_topk,
            renormalize=True,
            scoring_func=scoring,
            correction_bias=bias,
            shared_experts=SharedExpertsConfig(num_fused_shared_experts=shared),
            runner_backend=MoeRunnerBackend.TRITON,
            layer_id=0,
        )

        monkeypatch.setenv("SGLANG_DISABLE_FUSED_EP_REMAP", "1")
        ref_topk = select_experts(hidden, logits, cfg, num_token_non_padded=non_pad)
        d_ctrl = StandardDispatcher(
            MoeRunnerConfig(
                num_experts=num_experts + shared,
                num_local_experts=(num_experts // 2) + shared,
                top_k=total_topk,
                num_fused_shared_experts=shared,
                runner_backend=MoeRunnerBackend.TRITON,
            )
        )
        ref_disp = d_ctrl.dispatch(hidden, ref_topk)

        monkeypatch.delenv("SGLANG_DISABLE_FUSED_EP_REMAP", raising=False)
        treat_topk = select_experts(hidden, logits, cfg, num_token_non_padded=non_pad)
        assert isinstance(treat_topk, StandardTopKOutputLocalMapped)
        d_treat = StandardDispatcher(
            MoeRunnerConfig(
                num_experts=num_experts + shared,
                num_local_experts=(num_experts // 2) + shared,
                top_k=total_topk,
                num_fused_shared_experts=shared,
                runner_backend=MoeRunnerBackend.TRITON,
            )
        )
        treat_disp = d_treat.dispatch(hidden, treat_topk)
        treat_disp2 = d_treat.dispatch(hidden, treat_topk)
        assert torch.equal(
            treat_disp.topk_output.topk_ids[:5], ref_disp.topk_output.topk_ids[:5]
        )
        assert torch.equal(
            treat_disp2.topk_output.topk_ids[:5], ref_disp.topk_output.topk_ids[:5]
        )
        assert torch.allclose(
            treat_disp.topk_output.topk_weights[:5],
            ref_disp.topk_output.topk_weights[:5],
            atol=1e-6,
            rtol=1e-5,
        )
        assert torch.all(treat_disp.topk_output.topk_ids[5:] == -1)
        assert torch.all(treat_disp.topk_output.topk_weights[5:] == 0.0)

    # Full CUDA graph capture + dynamic num_token_non_padded replay
    m_rows, num_experts, topk = 64, 128, 8
    hidden = torch.randn((m_rows, 32), dtype=torch.bfloat16, device=device)
    logits = torch.randn((m_rows, num_experts), dtype=torch.bfloat16, device=device)
    non_pad = torch.tensor([32], dtype=torch.int32, device=device)
    cfg = TopKConfig(
        top_k=topk,
        renormalize=True,
        scoring_func="softmax",
        runner_backend=MoeRunnerBackend.TRITON,
        layer_id=0,
    )
    disp = StandardDispatcher(
        MoeRunnerConfig(
            num_experts=num_experts,
            num_local_experts=num_experts // 2,
            top_k=topk,
            num_fused_shared_experts=0,
            runner_backend=MoeRunnerBackend.TRITON,
        )
    )
    out_ids = torch.empty((m_rows, topk), dtype=torch.int32, device=device)
    out_w = torch.empty((m_rows, topk), dtype=torch.float32, device=device)

    def step():
        tk = select_experts(hidden, logits, cfg, num_token_non_padded=non_pad)
        d = disp.dispatch(hidden, tk)
        out_ids.copy_(d.topk_output.topk_ids)
        out_w.copy_(d.topk_output.topk_weights)

    s = torch.cuda.Stream(device=device)
    s.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(s):
        for _ in range(3):
            step()
    torch.cuda.current_stream(device).wait_stream(s)
    torch.cuda.synchronize(device)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        step()
    torch.cuda.synchronize(device)

    for v in (48, 1, 64, 0):
        non_pad.fill_(v)
        g.replay()
        torch.cuda.synchronize(device)
        w_ref, ids_ref = _reference_unfused_gate(
            logits,
            None,
            topk,
            scoring_func="softmax",
            num_token_non_padded=non_pad,
            ep_size=2,
            ep_rank=1,
        )
        assert torch.equal(out_ids, ids_ref)
        assert torch.allclose(out_w, w_ref, atol=1e-6, rtol=1e-5)
