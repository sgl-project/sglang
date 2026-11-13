import itertools
import unittest
from types import SimpleNamespace
from typing import Callable, Optional

import torch
from sgl_kernel.scalar_type import scalar_types

from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import fused_marlin_moe
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_marlin_utils import awq_marlin_quantize, marlin_quantize
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=242, stage="base-b", runner_config="1-gpu-small")

set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))


def stack_and_dev(tensors: list[torch.Tensor]):
    dev = tensors[0].device
    return torch.stack(tensors, dim=0).to(dev)


def torch_experts(
    a: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weight: torch.Tensor,
    topk_ids: torch.Tensor,
    global_num_experts: int = -1,
    expert_map: Optional[torch.Tensor] = None,
    quant_dtype: Optional[torch.dtype] = None,
    apply_router_weights_on_input: bool = False,
    activation: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> torch.Tensor:
    assert (
        global_num_experts == -1
        or (global_num_experts == w1.shape[0] and expert_map is None)
        or (expert_map is not None and global_num_experts == expert_map.shape[0])
    ), "Invalid expert configuration"

    M, K = a.shape
    topk = topk_ids.shape[1]

    if apply_router_weights_on_input:
        assert topk == 1, "apply_router_weights_on_input only works with topk=1"
        a = a * topk_weight.to(a.dtype)

    a = a.view(M, -1, K).repeat(1, topk, 1).reshape(-1, K)

    out = torch.zeros(M * topk, w2.shape[1], dtype=a.dtype, device=a.device)

    num_experts = w1.shape[0]

    topk_ids = topk_ids.view(-1)
    if expert_map is not None:
        topk_ids = expert_map[topk_ids]

    f32 = torch.float32

    for i in range(num_experts):
        mask = topk_ids == i
        if mask.sum():
            if quant_dtype is None:
                tmp1 = a[mask] @ w1[i].transpose(0, 1)
                tmp2 = (activation or SiluAndMul())(tmp1)
                out[mask] = tmp2 @ w2[i].transpose(0, 1)

    if apply_router_weights_on_input:
        return out
    else:
        return (
            (out.view(M, -1, w2.shape[1]).to(f32) * topk_weight.view(M, -1, 1))
            .sum(dim=1)
            .to(out.dtype)
        )


def torch_moe(
    a: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    score: torch.Tensor,
    topk: int,
    global_num_experts: int = -1,
    expert_map: Optional[torch.Tensor] = None,
    activation: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> torch.Tensor:
    score = torch.softmax(score, dim=-1, dtype=torch.float32)
    topk_weight, topk_ids = torch.topk(score, topk)
    return torch_experts(
        a,
        w1,
        w2,
        topk_weight,
        topk_ids,
        global_num_experts,
        expert_map,
        activation=activation,
    )


def marlin_moe_generate_valid_test_cases():
    m_list = [1, 123, 666]
    n_list = [128, 1024]
    k_list = [256, 2048]
    e_list = [4, 12]
    topk_list = [2, 3]
    dtype_list = [torch.half, torch.bfloat16]
    group_size_list = [64, 128]
    act_order_list = [True, False]
    quant_type_list = [
        scalar_types.uint4,
        scalar_types.uint4b8,
    ]
    is_k_full_list = [True, False]

    all_combinations = itertools.product(
        m_list,
        n_list,
        k_list,
        e_list,
        topk_list,
        dtype_list,
        group_size_list,
        act_order_list,
        quant_type_list,
        is_k_full_list,
    )

    def is_valid(m, n, k, e, topk, dtype, group_size, act_order, quant_type, is_k_full):
        if group_size > 0 and k % group_size != 0:
            return False

        if act_order:
            if group_size in (-1, k, n):
                return False
            if quant_type not in [scalar_types.uint4b8]:
                return False
        else:
            if not is_k_full:
                return False

        return True

    cases = []
    for case in all_combinations:
        if is_valid(*case):
            cases.append(case)
    return cases


class TestFusedMarlinMoe(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("This test requires a CUDA device.")
        torch.set_default_device("cuda")

    def test_fused_marlin_moe(self):
        test_cases = marlin_moe_generate_valid_test_cases()

        for (
            m,
            n,
            k,
            e,
            topk,
            dtype,
            group_size,
            act_order,
            quant_type,
            is_k_full,
        ) in test_cases:
            with self.subTest(
                m=m,
                n=n,
                k=k,
                e=e,
                topk=topk,
                dtype=dtype,
                group_size=group_size,
                act_order=act_order,
                quant_type=quant_type,
                is_k_full=is_k_full,
            ):
                torch.manual_seed(0)

                has_zp = quant_type in [scalar_types.uint4, scalar_types.uint8]

                if act_order:
                    if group_size == -1:
                        continue
                    if group_size in (k, n):
                        continue
                    if has_zp:
                        continue
                else:
                    if not is_k_full:
                        continue

                a = torch.randn((m, k), device="cuda", dtype=dtype) / 10
                w1 = torch.randn((e, 2 * n, k), device="cuda", dtype=dtype) / 20
                w2 = torch.randn((e, k, n), device="cuda", dtype=dtype) / 20

                e_map = None

                w_ref1_l = []
                qweight1_l = []
                scales1_l = []
                zeros1_l = []
                g_idx1_l = []
                sort_indices1_l = []

                for i in range(w1.shape[0]):
                    if has_zp:
                        w_ref1, qweight1, scales1, zeros1 = awq_marlin_quantize(
                            w1[i].transpose(1, 0), quant_type, group_size
                        )

                        w_ref1_l.append(w_ref1.T)
                        qweight1_l.append(qweight1)
                        scales1_l.append(scales1)
                        zeros1_l.append(zeros1)
                    else:
                        test_perm = torch.randperm(k)
                        w_ref1, qweight1, scales1, g_idx1, sort_indices1, _ = (
                            marlin_quantize(
                                w1[i].transpose(1, 0),
                                quant_type,
                                group_size,
                                act_order,
                                test_perm,
                            )
                        )

                        w_ref1_l.append(w_ref1.T)
                        qweight1_l.append(qweight1)
                        scales1_l.append(scales1)
                        g_idx1_l.append(g_idx1)
                        sort_indices1_l.append(sort_indices1)

                w_ref1 = stack_and_dev(w_ref1_l)
                qweight1 = stack_and_dev(qweight1_l).contiguous()
                scales1 = stack_and_dev(scales1_l)
                g_idx1 = stack_and_dev(g_idx1_l) if g_idx1_l else None
                zeros1 = stack_and_dev(zeros1_l) if zeros1_l else None
                sort_indices1 = (
                    stack_and_dev(sort_indices1_l) if sort_indices1_l else None
                )

                w_ref2_l = []
                qweight2_l = []
                scales2_l = []
                zeros2_l = []
                g_idx2_l = []
                sort_indices2_l = []

                for i in range(w2.shape[0]):
                    if has_zp:
                        w_ref2, qweight2, scales2, zeros2 = awq_marlin_quantize(
                            w2[i].transpose(1, 0), quant_type, group_size
                        )

                        w_ref2_l.append(w_ref2.T)
                        qweight2_l.append(qweight2)
                        scales2_l.append(scales2)
                        zeros2_l.append(zeros2)
                    else:
                        test_perm = torch.randperm(n)
                        w_ref2, qweight2, scales2, g_idx2, sort_indices2, _ = (
                            marlin_quantize(
                                w2[i].transpose(1, 0),
                                quant_type,
                                group_size,
                                act_order,
                                test_perm,
                            )
                        )

                        w_ref2_l.append(w_ref2.T)
                        qweight2_l.append(qweight2)
                        scales2_l.append(scales2)
                        g_idx2_l.append(g_idx2)
                        sort_indices2_l.append(sort_indices2)

                w_ref2 = stack_and_dev(w_ref2_l)
                qweight2 = stack_and_dev(qweight2_l).contiguous()
                scales2 = stack_and_dev(scales2_l)
                g_idx2 = stack_and_dev(g_idx2_l) if g_idx2_l else None
                zeros2 = stack_and_dev(zeros2_l) if zeros2_l else None
                sort_indices2 = (
                    stack_and_dev(sort_indices2_l) if sort_indices2_l else None
                )

                score = torch.randn((m, e), device="cuda", dtype=dtype)
                from sglang.srt.layers.moe.topk import fused_topk_torch_native

                topk_weights, topk_ids = fused_topk_torch_native(a, score, topk, False)

                torch_output = torch_moe(
                    a,
                    w_ref1,
                    w_ref2,
                    score,
                    topk,
                    global_num_experts=e,
                    expert_map=e_map,
                )

                marlin_output = fused_marlin_moe(
                    a,
                    qweight1,
                    qweight2,
                    scales1,
                    scales2,
                    score,
                    topk_weights,
                    topk_ids,
                    global_num_experts=e,
                    expert_map=e_map,
                    g_idx1=g_idx1,
                    g_idx2=g_idx2,
                    sort_indices1=sort_indices1,
                    sort_indices2=sort_indices2,
                    w1_zeros=zeros1,
                    w2_zeros=zeros2,
                    num_bits=4,
                    is_k_full=is_k_full,
                )

                torch.testing.assert_close(
                    marlin_output, torch_output, atol=5e-2, rtol=0
                )

    def test_fused_marlin_moe_gpt_oss_activation(self):
        """With gemm1_alpha and clamp_limit, the first GEMM's [gate | up] halves go
        through gate * sigmoid(alpha * gate) * (up + 1), gate clamped above and up
        clamped both ways, into the buffer the second GEMM reads."""
        alpha, limit = 1.702, 7.0
        e, topk, n, k, group_size = 4, 2, 128, 256, 128
        quant_type = scalar_types.uint4b8

        def gpt_oss(x: torch.Tensor) -> torch.Tensor:
            gate = x[:, :n].float().clamp(max=limit)
            up = x[:, n:].float().clamp(min=-limit, max=limit)
            return (gate * torch.sigmoid(alpha * gate) * (up + 1)).to(x.dtype)

        def quantize(w):
            refs, qweights, scales = [], [], []
            for i in range(w.shape[0]):
                ref, qweight, scale, _, _, _ = marlin_quantize(
                    w[i].transpose(1, 0),
                    quant_type,
                    group_size,
                    False,
                    torch.randperm(w.shape[2]),
                )
                refs.append(ref.T)
                qweights.append(qweight)
                scales.append(scale)
            return (
                stack_and_dev(refs),
                stack_and_dev(qweights).contiguous(),
                stack_and_dev(scales),
            )

        for m, dtype in itertools.product((1, 123), (torch.half, torch.bfloat16)):
            with self.subTest(m=m, dtype=dtype):
                torch.manual_seed(0)
                a = torch.randn((m, k), device="cuda", dtype=dtype) / 10
                w1 = torch.randn((e, 2 * n, k), device="cuda", dtype=dtype) / 20
                w2 = torch.randn((e, k, n), device="cuda", dtype=dtype) / 20
                w_ref1, qweight1, scales1 = quantize(w1)
                w_ref2, qweight2, scales2 = quantize(w2)
                score = torch.randn((m, e), device="cuda", dtype=dtype)
                from sglang.srt.layers.moe.topk import fused_topk_torch_native

                topk_weights, topk_ids = fused_topk_torch_native(a, score, topk, False)
                torch_output = torch_moe(
                    a,
                    w_ref1,
                    w_ref2,
                    score,
                    topk,
                    global_num_experts=e,
                    activation=gpt_oss,
                )
                marlin_output = fused_marlin_moe(
                    a,
                    qweight1,
                    qweight2,
                    scales1,
                    scales2,
                    score,
                    topk_weights,
                    topk_ids,
                    global_num_experts=e,
                    num_bits=4,
                    clamp_limit=limit,
                    gemm1_alpha=alpha,
                )
                torch.testing.assert_close(
                    marlin_output, torch_output, atol=5e-2, rtol=0
                )

    def test_fused_marlin_moe_expert_parallelism(self):
        m_list = [1, 16, 128]
        e_list = [8, 16]

        for m in m_list:
            for e in e_list:
                with self.subTest(m=m, e=e):
                    torch.manual_seed(100)

                    n, k = 256, 256
                    topk = 2
                    ep_size = 2
                    group_size = 128
                    dtype = torch.bfloat16
                    quant_type = scalar_types.uint4b8

                    local_e = e // ep_size
                    e_ids = torch.arange(local_e, device="cuda", dtype=torch.int32)
                    e_map = torch.full((e,), -1, device="cuda", dtype=torch.int32)
                    e_map[e_ids] = torch.arange(
                        local_e, device="cuda", dtype=torch.int32
                    )

                    a = torch.randn((m, k), device="cuda", dtype=dtype) / 10
                    w1_full = (
                        torch.randn((e, 2 * n, k), device="cuda", dtype=dtype) / 20
                    )
                    w2_full = torch.randn((e, k, n), device="cuda", dtype=dtype) / 20

                    score = torch.randn((m, e), device="cuda", dtype=dtype)
                    score[:, e_ids] += 10.0

                    w1 = w1_full[e_ids]
                    w2 = w2_full[e_ids]

                    w_ref1_l, qweight1_l, scales1_l = [], [], []
                    for i in range(local_e):
                        test_perm = torch.randperm(k)
                        w_ref1, qweight1, scales1, _, _, _ = marlin_quantize(
                            w1[i].transpose(1, 0),
                            quant_type,
                            group_size,
                            False,
                            test_perm,
                        )
                        w_ref1_l.append(w_ref1.T)
                        qweight1_l.append(qweight1)
                        scales1_l.append(scales1)

                    w_ref2_l, qweight2_l, scales2_l = [], [], []
                    for i in range(local_e):
                        test_perm = torch.randperm(n)
                        w_ref2, qweight2, scales2, _, _, _ = marlin_quantize(
                            w2[i].transpose(1, 0),
                            quant_type,
                            group_size,
                            False,
                            test_perm,
                        )
                        w_ref2_l.append(w_ref2.T)
                        qweight2_l.append(qweight2)
                        scales2_l.append(scales2)

                    w_ref1 = stack_and_dev(w_ref1_l)
                    qweight1 = stack_and_dev(qweight1_l).contiguous()
                    scales1 = stack_and_dev(scales1_l)

                    w_ref2 = stack_and_dev(w_ref2_l)
                    qweight2 = stack_and_dev(qweight2_l).contiguous()
                    scales2 = stack_and_dev(scales2_l)

                    from sglang.srt.layers.moe.topk import fused_topk_torch_native

                    topk_weights, topk_ids = fused_topk_torch_native(
                        a, score, topk, False
                    )

                    w_ref1_full = w1_full.clone()
                    w_ref2_full = w2_full.clone()
                    w_ref1_full[e_ids] = w_ref1
                    w_ref2_full[e_ids] = w_ref2

                    torch_output = torch_moe(
                        a,
                        w_ref1_full,
                        w_ref2_full,
                        score,
                        topk,
                        global_num_experts=e,
                        expert_map=e_map,
                    )

                    marlin_output = fused_marlin_moe(
                        a,
                        qweight1,
                        qweight2,
                        scales1,
                        scales2,
                        score,
                        topk_weights,
                        topk_ids,
                        global_num_experts=e,
                        expert_map=e_map,
                        num_bits=4,
                        is_k_full=True,
                    )

                    torch.testing.assert_close(
                        marlin_output, torch_output, atol=5e-2, rtol=0
                    )


_FP8_MAX = torch.finfo(torch.float8_e4m3fn).max


def _quantize_fp8_experts(w: torch.Tensor, layout: str):
    """Quantize [E, N, K] weights in one FP8 checkpoint layout.

    Returns the e4m3 weights, the Marlin [E, groups, N] FP32 group scales the
    serving methods derive from that layout, and the dequantized weights.
    """
    from sglang.srt.layers.quantization.marlin_utils_fp8 import (
        marlin_moe_fp8_block_scales,
        marlin_moe_fp8_channel_scales,
        marlin_moe_mxfp8_scales,
    )

    e, n, k = w.shape
    w = w.float()
    if layout == "channel":
        scale = w.abs().amax(dim=2, keepdim=True).clamp(min=1e-6) / _FP8_MAX
        q = (w / scale).to(torch.float8_e4m3fn)
        return q, marlin_moe_fp8_channel_scales(scale), q.float() * scale
    if layout == "block128":
        nb, kb = n // 128, k // 128
        blocks = w.view(e, nb, 128, kb, 128)
        scale = blocks.abs().amax(dim=(2, 4)).clamp(min=1e-6) / _FP8_MAX
        q = (blocks / scale[:, :, None, :, None]).to(torch.float8_e4m3fn)
        deq = q.float() * scale[:, :, None, :, None]
        return (
            q.view(e, n, k),
            marlin_moe_fp8_block_scales(scale, size_n=n, block_n=128),
            deq.view(e, n, k),
        )
    assert layout == "mxfp8"
    groups = w.view(e, n, k // 32, 32)
    # The smallest power of two that brings the group's maximum into E4M3.
    exponent = torch.ceil(
        torch.log2(groups.abs().amax(dim=-1).clamp(min=2.0**-120) / _FP8_MAX)
    )
    scale = torch.exp2(exponent)
    q = (groups / scale[..., None]).to(torch.float8_e4m3fn)
    deq = q.float() * scale[..., None]
    e8m0 = (exponent + 127).to(torch.uint8)
    return q.view(e, n, k), marlin_moe_mxfp8_scales(e8m0), deq.view(e, n, k)


def _marlin_fp8_layer(w1: torch.Tensor, w2: torch.Tensor, layout: str, biases=None):
    """A FusedMoE stand-in holding the Marlin W8A16 tiles of these experts,
    with FP32 logical-order expert biases when ``biases`` gives them."""
    from sglang.srt.layers.quantization.marlin_utils_fp8 import (
        prepare_moe_fp8_layer_for_marlin,
    )

    q1, scales1, ref1 = _quantize_fp8_experts(w1, layout)
    q2, scales2, ref2 = _quantize_fp8_experts(w2, layout)
    layer = SimpleNamespace(
        w13_weight=torch.nn.Parameter(q1, requires_grad=False),
        w2_weight=torch.nn.Parameter(q2, requires_grad=False),
        hidden_size=w1.shape[2],
        intermediate_size_per_partition=w2.shape[2],
        moe_runner_config=SimpleNamespace(
            is_gated=True, activation="silu", apply_router_weight_on_input=False
        ),
    )
    if biases is not None:
        layer.w13_weight_bias = torch.nn.Parameter(biases[0], requires_grad=False)
        layer.w2_weight_bias = torch.nn.Parameter(biases[1], requires_grad=False)
    prepare_moe_fp8_layer_for_marlin(
        layer, w13_scales=scales1, w2_scales=scales2, params_dtype=w1.dtype
    )
    return layer, ref1.to(w1.dtype), ref2.to(w2.dtype)


class TestFusedMarlinMoeFp8(CustomTestCase):
    """E4M3 experts served W8A16 track the dequantized reference for every FP8
    scale layout the repack admits: channel-wise, 128-wide blocks and MXFP8's
    32-wide E8M0 groups (the one kernel instantiation added for it)."""

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("This test requires a CUDA device.")
        torch.set_default_device("cuda")

    def test_fused_marlin_moe_fp8_layouts(self):
        from sglang.srt.layers.moe.topk import fused_topk_torch_native

        e, n, k, topk = 4, 128, 256, 2
        for layout, m, dtype in itertools.product(
            ("channel", "block128", "mxfp8"), (1, 37), (torch.bfloat16, torch.float16)
        ):
            with self.subTest(layout=layout, m=m, dtype=dtype):
                torch.manual_seed(0)
                a = torch.randn((m, k), device="cuda", dtype=dtype) / 10
                w1 = torch.randn((e, 2 * n, k), device="cuda", dtype=dtype) / 20
                w2 = torch.randn((e, k, n), device="cuda", dtype=dtype) / 20
                layer, w_ref1, w_ref2 = _marlin_fp8_layer(w1, w2, layout)
                score = torch.randn((m, e), device="cuda", dtype=dtype)
                topk_weights, topk_ids = fused_topk_torch_native(a, score, topk, False)
                torch_output = torch_moe(
                    a, w_ref1, w_ref2, score, topk, global_num_experts=e
                )
                marlin_output = fused_marlin_moe(
                    a,
                    layer.w13_weight,
                    layer.w2_weight,
                    layer.w13_weight_scale,
                    layer.w2_weight_scale,
                    score,
                    topk_weights,
                    topk_ids,
                    global_num_experts=e,
                    num_bits=8,
                    is_fp8=True,
                )
                # W8A16 keeps the activations exact; the scales round to the
                # activation dtype once, so the gap stays well under 1%.
                gap = (marlin_output.float() - torch_output.float()).norm()
                rel = (gap / torch_output.float().norm()).item()
                self.assertLess(rel, 1e-2, (layout, m, dtype, rel))

    def test_fused_marlin_moe_fp8_expert_biases(self):
        """Nonzero, nonuniform expert biases ride the Marlin epilogue: the gate/up
        bias lands before the activation and the down bias before the routing
        weight, in the kernel's tile order and the activation dtype, for every
        FP8 scale layout; the independent reference applies them in FP32."""
        e, n, k, topk = 4, 128, 256, 2
        for layout, dtype in itertools.product(
            ("channel", "block128", "mxfp8"), (torch.bfloat16, torch.float16)
        ):
            with self.subTest(layout=layout, dtype=dtype):
                torch.manual_seed(1)
                m = 37
                a = torch.randn((m, k), device="cuda", dtype=dtype) / 10
                w1 = torch.randn((e, 2 * n, k), device="cuda", dtype=dtype) / 20
                w2 = torch.randn((e, k, n), device="cuda", dtype=dtype) / 20
                b1 = torch.randn((e, 2 * n), device="cuda") * 0.2
                b2 = torch.randn((e, k), device="cuda") * 0.2
                layer, w_ref1, w_ref2 = _marlin_fp8_layer(
                    w1, w2, layout, biases=(b1.clone(), b2.clone())
                )
                score = torch.randn((m, e), device="cuda", dtype=dtype)
                from sglang.srt.layers.moe.topk import fused_topk_torch_native

                topk_weights, topk_ids = fused_topk_torch_native(a, score, topk, False)
                out = torch.zeros((m, topk, k), device="cuda", dtype=torch.float32)
                for t in range(m):
                    for j in range(topk):
                        ex = int(topk_ids[t, j])
                        gate_up = a[t].float() @ w_ref1[ex].float().T + b1[ex]
                        act = torch.nn.functional.silu(gate_up[:n]) * gate_up[n:]
                        out[t, j] = (
                            act @ w_ref2[ex].float().T + b2[ex]
                        ) * topk_weights[t, j]
                reference = out.sum(1)
                marlin_output = fused_marlin_moe(
                    a,
                    layer.w13_weight,
                    layer.w2_weight,
                    layer.w13_weight_scale,
                    layer.w2_weight_scale,
                    score,
                    topk_weights,
                    topk_ids,
                    global_num_experts=e,
                    w1_bias=layer.w13_weight_bias,
                    w2_bias=layer.w2_weight_bias,
                    num_bits=8,
                    is_fp8=True,
                )
                rel = (
                    (marlin_output.float() - reference).norm() / reference.norm()
                ).item()
                self.assertLess(rel, 1e-2, (layout, dtype, rel))
                # Biases must matter: without them the reference moves by far more.
                no_bias = (
                    fused_marlin_moe(
                        a,
                        layer.w13_weight,
                        layer.w2_weight,
                        layer.w13_weight_scale,
                        layer.w2_weight_scale,
                        score,
                        topk_weights,
                        topk_ids,
                        global_num_experts=e,
                        num_bits=8,
                        is_fp8=True,
                    )
                    .float()
                    .sub(reference)
                    .norm()
                    / reference.norm()
                )
                self.assertGreater(no_bias.item(), 0.1)

    def test_online_mxfp8_experts_match_a_serialized_reload(self):
        """A BF16 checkpoint served as MXFP8 quantizes its experts into the
        canonical layout (E4M3 weights, one UE8M0 byte per row and 32 K values,
        the scale that brings the group maximum to at most 448: the rule the
        dense online path uses) and then repacks exactly as a serialized MXFP8
        checkpoint of those bytes does: the Marlin tiles and scales are bitwise
        equal, so the weights are quantized once, never twice."""
        from sglang.srt.layers.moe.utils import MoeRunnerBackend
        from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8MoEMethod
        from sglang.srt.layers.quantization.fp8_utils import mxfp8_quantize_moe_experts
        from sglang.srt.runtime_context import get_flags

        e, n, k = 4, 128, 256
        torch.manual_seed(2)
        w13 = (torch.randn((e, 2 * n, k), device="cuda") * 0.05).to(torch.bfloat16)
        w2 = (torch.randn((e, k, n), device="cuda") * 0.05).to(torch.bfloat16)
        w13[0, 0] = 0  # a zero row: scale byte 0, every code 0

        def reference(w):
            groups = w.float().view(*w.shape[:-1], -1, 32)
            amax = groups.abs().amax(dim=-1)
            exponent = torch.where(
                amax > 0,
                torch.ceil(torch.log2(amax / _FP8_MAX)),
                torch.full_like(amax, -127.0),
            ).clamp(min=-127, max=127)
            q = (groups / torch.exp2(exponent)[..., None]).to(torch.float8_e4m3fn)
            return q.view_as(w), (exponent + 127).to(torch.uint8)

        for w in (w13, w2):
            q, scale = mxfp8_quantize_moe_experts(w)
            q_ref, scale_ref = reference(w)
            self.assertEqual(q.dtype, torch.float8_e4m3fn)
            self.assertEqual(scale.dtype, torch.uint8)
            self.assertTrue(torch.equal(scale, scale_ref))
            self.assertTrue(torch.equal(q.view(torch.uint8), q_ref.view(torch.uint8)))

        def layer_for(w13_param, w2_param, w13_scale, w2_scale):
            return SimpleNamespace(
                w13_weight=torch.nn.Parameter(w13_param.clone(), requires_grad=False),
                w2_weight=torch.nn.Parameter(w2_param.clone(), requires_grad=False),
                w13_weight_scale_inv=torch.nn.Parameter(
                    w13_scale.clone(), requires_grad=False
                ),
                w2_weight_scale_inv=torch.nn.Parameter(
                    w2_scale.clone(), requires_grad=False
                ),
                params_dtype=torch.bfloat16,
                hidden_size=k,
                intermediate_size_per_partition=n,
                moe_runner_config=SimpleNamespace(
                    is_gated=True, activation="silu", apply_router_weight_on_input=False
                ),
            )

        q13, s13 = mxfp8_quantize_moe_experts(w13)
        q2, s2 = mxfp8_quantize_moe_experts(w2)
        with get_flags().moe.override(runner_backend=MoeRunnerBackend.MARLIN):
            online = Fp8MoEMethod(
                Fp8Config(is_checkpoint_fp8_serialized=False, use_mxfp8=True)
            )
            serialized = Fp8MoEMethod(
                Fp8Config(is_checkpoint_fp8_serialized=True, use_mxfp8=True)
            )
        self.assertTrue(online.use_marlin and serialized.use_marlin)
        placeholder13 = torch.zeros(
            (e, 2 * n, k // 32), dtype=torch.uint8, device="cuda"
        )
        placeholder2 = torch.zeros((e, k, n // 32), dtype=torch.uint8, device="cuda")
        layer_online = layer_for(w13, w2, placeholder13, placeholder2)
        layer_serialized = layer_for(q13, q2, s13, s2)
        online._process_weights_for_marlin(layer_online)
        serialized._process_weights_for_marlin(layer_serialized)
        for name in ("w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale"):
            self.assertTrue(
                torch.equal(
                    getattr(layer_online, name), getattr(layer_serialized, name)
                ),
                name,
            )
        self.assertEqual(layer_online.w13_weight.dtype, torch.int32)

    def test_compressed_tensors_per_tensor_experts_keep_separate_scales(self):
        """compressed-tensors per-tensor FP8 experts under Marlin: the loader's
        separate gate and up scales each cover their own rows (no requantization
        to a shared scale), static or dynamic activation metadata is ignored
        (W8A16 reads the floating activations), and the fused runner input the
        scheme builds matches the FP32 reference on the original dequantized
        weights."""
        from compressed_tensors.quantization import QuantizationArgs

        from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
        from sglang.srt.layers.moe.topk import fused_topk_torch_native
        from sglang.srt.layers.moe.utils import MoeRunnerBackend
        from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_fp8_moe import (
            CompressedTensorsW8A8Fp8MoE,
        )
        from sglang.srt.runtime_context import get_flags

        e, n, k, topk, m = 4, 128, 256, 2, 37
        torch.manual_seed(3)
        for dynamic in (True, False):
            with (
                self.subTest(dynamic_activations=dynamic),
                get_flags().moe.override(runner_backend=MoeRunnerBackend.MARLIN),
            ):
                scheme = CompressedTensorsW8A8Fp8MoE(
                    QuantizationArgs(num_bits=8, type="float", strategy="tensor"),
                    QuantizationArgs(
                        num_bits=8, type="float", strategy="tensor", dynamic=dynamic
                    ),
                )
                self.assertTrue(scheme.use_marlin)
                layer = torch.nn.Module()
                layer.moe_tp_size = 1
                layer.num_local_experts = e
                layer.hidden_size = k
                layer.intermediate_size_per_partition = n
                layer.params_dtype = torch.bfloat16
                layer.dispatcher = SimpleNamespace(
                    local_expert_mapping=None, num_experts=e
                )
                layer.moe_runner_config = MoeRunnerConfig(
                    num_experts=e,
                    num_local_experts=e,
                    hidden_size=k,
                    intermediate_size_per_partition=n,
                    layer_id=0,
                    top_k=topk,
                    num_fused_shared_experts=0,
                    params_dtype=torch.bfloat16,
                    activation="silu",
                    inplace=False,
                )
                scheme.create_weights(
                    layer=layer,
                    num_experts=e,
                    hidden_size=k,
                    intermediate_size_per_partition=n,
                    params_dtype=torch.bfloat16,
                )
                layer.cuda()
                # Unequal gate and up scales per expert, one down scale per expert.
                gate_scale = torch.rand(e, device="cuda") * 0.02 + 0.005
                up_scale = gate_scale * torch.tensor(
                    [3.0, 0.5, 1.7, 7.0], device="cuda"
                )
                down_scale = torch.rand(e, device="cuda") * 0.02 + 0.005
                w1 = torch.randn((e, 2 * n, k), device="cuda") / 20
                w2 = torch.randn((e, k, n), device="cuda") / 20
                q1 = torch.empty_like(w1, dtype=torch.float8_e4m3fn)
                q1[:, :n] = (w1[:, :n] / gate_scale[:, None, None]).to(
                    torch.float8_e4m3fn
                )
                q1[:, n:] = (w1[:, n:] / up_scale[:, None, None]).to(
                    torch.float8_e4m3fn
                )
                q2 = (w2 / down_scale[:, None, None]).to(torch.float8_e4m3fn)
                layer.w13_weight.data.copy_(q1)
                layer.w2_weight.data.copy_(q2)
                layer.w13_weight_scale.data.copy_(
                    torch.stack([gate_scale, up_scale], 1)
                )
                layer.w2_weight_scale.data.copy_(down_scale)
                if not dynamic:
                    # Static activation scales that would wreck the output if applied.
                    layer.w13_input_scale.data.fill_(1000.0)
                    layer.w2_input_scale.data.fill_(1000.0)
                scheme.process_weights_after_loading(layer)
                self.assertEqual(layer.w13_weight.dtype, torch.int32)
                info = scheme.get_marlin_quant_info(layer)
                self.assertTrue(info.is_fp8 and info.weight_bits == 8)

                ref1 = torch.cat(
                    [
                        q1[:, :n].float() * gate_scale[:, None, None],
                        q1[:, n:].float() * up_scale[:, None, None],
                    ],
                    dim=1,
                ).to(torch.bfloat16)
                ref2 = (q2.float() * down_scale[:, None, None]).to(torch.bfloat16)
                a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16) / 10
                score = torch.randn((m, e), device="cuda", dtype=torch.bfloat16)
                topk_weights, topk_ids = fused_topk_torch_native(a, score, topk, False)
                reference = torch_moe(a, ref1, ref2, score, topk, global_num_experts=e)
                marlin_output = fused_marlin_moe(
                    a,
                    info.w13_qweight,
                    info.w2_qweight,
                    info.w13_scales,
                    info.w2_scales,
                    score,
                    topk_weights,
                    topk_ids,
                    global_num_experts=e,
                    num_bits=info.weight_bits,
                    is_fp8=info.is_fp8,
                )
                rel = (
                    (marlin_output.float() - reference.float()).norm()
                    / reference.float().norm()
                ).item()
                self.assertLess(rel, 1e-2, (dynamic, rel))
                # A shared scale would have requantized the smaller-scaled half.
                folded = torch.maximum(gate_scale, up_scale)[:, None, None]
                shared = (
                    torch.cat([q1[:, :n].float(), q1[:, n:].float()], dim=1) * folded
                )
                self.assertGreater(
                    (
                        (shared.to(torch.bfloat16).float() - ref1.float()).norm()
                        / ref1.float().norm()
                    ).item(),
                    0.1,
                )

    def test_fp8_channel_scales_are_applied_before_accumulation(self):
        """Channel-wise FP8 experts are repacked with their scale repeated over
        K groups (128-wide, or 32-wide when a stage's K would hold one 128-wide
        group): the kernel's single-group path scales after the K accumulation,
        where small products have already underflowed FP32. Weights of 1 and
        BF16 inputs of 2**-60 through the whole fused MoE must match the FP64
        reference on independently scaled weights and never be 0."""
        from sglang.srt.layers.moe.topk import fused_topk_torch_native

        for hidden, inter, group in ((512, 256, 128), (256, 128, 32)):
            with self.subTest(hidden=hidden, intermediate=inter):
                e, topk, m = 4, 2, 8
                w1 = torch.ones(
                    (e, 2 * inter, hidden), device="cuda", dtype=torch.bfloat16
                )
                w2 = torch.ones((e, hidden, inter), device="cuda", dtype=torch.bfloat16)
                layer, w_ref1, w_ref2 = _marlin_fp8_layer(w1, w2, "channel")
                self.assertEqual(layer.w13_weight_scale.shape[1], hidden // group)
                self.assertEqual(layer.w2_weight_scale.shape[1], inter // group)
                a = torch.full(
                    (m, hidden), 2.0**-60, dtype=torch.bfloat16, device="cuda"
                )
                score = torch.randn((m, e), device="cuda", dtype=torch.bfloat16)
                topk_weights, topk_ids = fused_topk_torch_native(a, score, topk, True)
                reference = torch.zeros((m, hidden), device="cuda", dtype=torch.float64)
                for t in range(m):
                    for j in range(topk):
                        ex = int(topk_ids[t, j])
                        gate_up = a[t].double() @ w_ref1[ex].double().T
                        act = (
                            torch.nn.functional.silu(gate_up[:inter]) * gate_up[inter:]
                        )
                        reference[t] += (act @ w_ref2[ex].double().T) * topk_weights[
                            t, j
                        ].double()
                out = fused_marlin_moe(
                    a,
                    layer.w13_weight,
                    layer.w2_weight,
                    layer.w13_weight_scale,
                    layer.w2_weight_scale,
                    score,
                    topk_weights,
                    topk_ids,
                    global_num_experts=e,
                    num_bits=8,
                    is_fp8=True,
                )
                self.assertTrue(bool((out != 0).all()))
                torch.testing.assert_close(out.double(), reference, rtol=2e-2, atol=0)

    def test_fp8_channel_layers_keep_64_aligned_intermediates(self):
        """Channel-wise FP8 experts with intermediate sizes that are multiples
        of 64 but not of 128 (320, 448) served through the channel-wise kernel
        before; they now take 32-wide groups and must still repack and run,
        matching the dequantized reference like the other layouts."""
        from sglang.srt.layers.moe.topk import fused_topk_torch_native

        e, hidden, topk, m = 4, 512, 2, 37
        for inter in (320, 448):
            with self.subTest(intermediate=inter):
                torch.manual_seed(4)
                a = torch.randn((m, hidden), device="cuda", dtype=torch.bfloat16) / 10
                w1 = (
                    torch.randn(
                        (e, 2 * inter, hidden), device="cuda", dtype=torch.bfloat16
                    )
                    / 20
                )
                w2 = (
                    torch.randn((e, hidden, inter), device="cuda", dtype=torch.bfloat16)
                    / 20
                )
                layer, w_ref1, w_ref2 = _marlin_fp8_layer(w1, w2, "channel")
                self.assertEqual(layer.w13_weight_scale.shape[1], hidden // 32)
                self.assertEqual(layer.w2_weight_scale.shape[1], inter // 32)
                score = torch.randn((m, e), device="cuda", dtype=torch.bfloat16)
                topk_weights, topk_ids = fused_topk_torch_native(a, score, topk, False)
                reference = torch_moe(
                    a, w_ref1, w_ref2, score, topk, global_num_experts=e
                )
                out = fused_marlin_moe(
                    a,
                    layer.w13_weight,
                    layer.w2_weight,
                    layer.w13_weight_scale,
                    layer.w2_weight_scale,
                    score,
                    topk_weights,
                    topk_ids,
                    global_num_experts=e,
                    num_bits=8,
                    is_fp8=True,
                )
                rel = (
                    (out.float() - reference.float()).norm() / reference.float().norm()
                ).item()
                self.assertLess(rel, 1e-2, (inter, rel))

    def test_fp16_dequantized_weights_fit_or_are_refused(self):
        """With FP16 activations the grouped kernel forms code * scale in FP16
        before the MMA, with the scale as the kernel holds it: the compensated
        scale rounded to FP16. Codes of 448 under a scale of 255 on the gate
        columns (dequantized weight 114240, FP16 infinity, while the gate GEMM
        result over K = 256 with inputs 2**-10 is a finite 28560) are refused
        at repack, and so is 65504 / 448 although 448 * 65504 / 448 fits FP16:
        its compensated scale rounds up to 37440 and 1.75 * 37440 is infinity.
        146.125 is the largest effective scale the kernel can hold for codes of
        448 (compensated 37408, product 65464); every input scale that rounds to
        it is admitted (146.18 rounds to 146.125 too), and the layer runs and
        matches the FP64 reference. BF16 activations take the scale of 255 and
        match the reference."""
        from sglang.srt.layers.moe.topk import fused_topk_torch_native
        from sglang.srt.layers.quantization.marlin_utils_fp8 import (
            marlin_moe_fp8_channel_scales,
            prepare_moe_fp8_layer_for_marlin,
        )

        e, inter, hidden, topk, m = 4, 128, 256, 2, 8

        def layer_for(dtype, scale):
            layer = SimpleNamespace(
                w13_weight=torch.nn.Parameter(
                    torch.full((e, 2 * inter, hidden), 448.0, device="cuda").to(
                        torch.float8_e4m3fn
                    ),
                    requires_grad=False,
                ),
                w2_weight=torch.nn.Parameter(
                    torch.ones((e, hidden, inter), device="cuda").to(
                        torch.float8_e4m3fn
                    ),
                    requires_grad=False,
                ),
                params_dtype=dtype,
                hidden_size=hidden,
                intermediate_size_per_partition=inter,
                moe_runner_config=SimpleNamespace(
                    is_gated=True, activation="silu", apply_router_weight_on_input=False
                ),
            )
            # The gate half carries the scale under test; the up half stays small
            # so the gated activation and the down GEMM fit FP16 themselves.
            w13_scales = torch.cat(
                [
                    torch.full((e, inter, 1), scale, device="cuda"),
                    torch.full((e, inter, 1), 1e-4, device="cuda"),
                ],
                dim=1,
            )
            prepare_moe_fp8_layer_for_marlin(
                layer,
                w13_scales=marlin_moe_fp8_channel_scales(w13_scales),
                w2_scales=marlin_moe_fp8_channel_scales(
                    torch.ones((e, hidden, 1), device="cuda")
                ),
                params_dtype=dtype,
            )
            return layer

        for scale in (255.0, 65504.0 / 448.0):
            # 65504 / 448 is refused too: the kernel holds the compensated scale
            # as an FP16 number (37440 after rounding), and 1.75 * 37440 = 65520
            # is already infinity; 146.125 (compensated 37408) is the largest
            # effective scale the kernel can hold for codes of 448, and input
            # scales that round to it (146.18) are admitted.
            with (
                self.subTest(refused=scale),
                self.assertRaisesRegex(ValueError, "beyond the FP16 range"),
            ):
                layer_for(torch.float16, scale)
        for dtype, scale in ((torch.float16, 146.125), (torch.bfloat16, 255.0)):
            with self.subTest(dtype=dtype, scale=scale):
                layer = layer_for(dtype, scale)
                a = torch.full((m, hidden), 2.0**-10, dtype=dtype, device="cuda")
                score = torch.randn((m, e), device="cuda", dtype=dtype)
                topk_weights, topk_ids = fused_topk_torch_native(a, score, topk, True)
                gate = torch.tensor(
                    hidden * 2.0**-10 * 448.0 * scale, dtype=torch.float64
                )
                up = torch.tensor(hidden * 2.0**-10 * 448.0 * 1e-4, dtype=torch.float64)
                expected = float(torch.nn.functional.silu(gate) * up * inter)
                out = fused_marlin_moe(
                    a,
                    layer.w13_weight,
                    layer.w2_weight,
                    layer.w13_weight_scale,
                    layer.w2_weight_scale,
                    score,
                    topk_weights,
                    topk_ids,
                    global_num_experts=e,
                    num_bits=8,
                    is_fp8=True,
                )
                self.assertTrue(bool(torch.isfinite(out).all()))
                torch.testing.assert_close(
                    out.double(),
                    torch.full_like(out.double(), expected),
                    rtol=2e-2,
                    atol=0,
                )

    def test_fused_marlin_moe_fp8_rejects_unsupported_groups(self):
        """A 64-wide K group has no FP8 kernel; the repack refuses it instead
        of handing Marlin tiles it cannot run."""
        from sglang.srt.layers.quantization.marlin_utils_fp8 import (
            prepare_moe_fp8_layer_for_marlin,
        )

        e, n, k = 2, 128, 256
        w = torch.randn((e, 2 * n, k), dtype=torch.bfloat16) / 20
        layer = SimpleNamespace(
            w13_weight=torch.nn.Parameter(
                w.to(torch.float8_e4m3fn), requires_grad=False
            ),
            w2_weight=torch.nn.Parameter(
                torch.zeros((e, k, n), dtype=torch.float8_e4m3fn), requires_grad=False
            ),
            hidden_size=k,
            intermediate_size_per_partition=n,
            moe_runner_config=SimpleNamespace(
                is_gated=True, activation="silu", apply_router_weight_on_input=False
            ),
        )
        with self.assertRaisesRegex(NotImplementedError, "32- or 128-wide"):
            prepare_moe_fp8_layer_for_marlin(
                layer,
                w13_scales=torch.ones((e, k // 64, 2 * n)),
                w2_scales=torch.ones((e, n // 64, k)),
                params_dtype=torch.bfloat16,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
