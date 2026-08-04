"""Numerics for the NVFP4 FusedMoE runner backends (--moe-runner-backend).

Runs the real FusedMoE layer path (construct -> fill NVFP4 checkpoint-format
weights -> process_weights_after_loading -> forward) per backend against a
dequantized torch MoE reference, covering the per-backend weight preparation
(TRTLLM shuffle / CUTLASS swizzle / CuteDSL v2 interleave + MMA blockscales)
and the MoE runner dispatch. Single GPU, tp=ep=1.
"""

import os
import unittest

import torch

from sglang.srt.layers.moe.utils import MoeRunnerBackend
from sglang.srt.layers.quantization.modelopt_quant import ModelOptFp4Config
from sglang.srt.runtime_context import get_context, get_flags, get_parallel
from sglang.srt.utils import get_device_sm
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=120, stage="base-b", runner_config="4-gpu-b200")

E, H, I, TOPK, M = 8, 1024, 1024, 2, 32
FLOAT8_E4M3_MAX = 448.0
FLOAT4_E2M1_MAX = 6.0

kE2M1ToFloat = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32
)


def _init_single_process_dist():
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29631")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    from sglang.srt.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
        model_parallel_is_initialized,
    )

    if not torch.distributed.is_initialized():
        init_distributed_environment(world_size=1, rank=0, local_rank=0, backend="gloo")
    if not model_parallel_is_initialized():
        initialize_model_parallel(
            tensor_model_parallel_size=1,
            expert_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            backend="gloo",
        )


def convert_swizzled_to_linear(a_sf_swizzled, m, k, block_size=16):
    m_tiles = (m + 128 - 1) // 128
    f = block_size * 4
    k_tiles = (k + f - 1) // f
    tmp = torch.reshape(a_sf_swizzled, (1, m_tiles, k_tiles, 32, 4, 4))
    tmp = torch.permute(tmp, (0, 1, 4, 3, 2, 5))
    out = tmp.reshape(m_tiles * 128, k_tiles * f // block_size)
    return out[0:m, 0 : k // block_size]


def break_fp4_bytes(a):
    m, n = a.shape
    a_flat = a.flatten()
    high = (a_flat & 0xF0) >> 4
    low = a_flat & 0x0F
    combined = torch.stack((low, high), dim=1).flatten()
    signs = (combined & 0x08).to(torch.bool)
    abs_vals = (combined & 0x07).to(torch.long)
    kE2M1 = kE2M1ToFloat.to(device=a.device)
    values = kE2M1[abs_vals] * torch.where(signs, -1.0, 1.0)
    return values.reshape(m, n * 2).to(dtype=torch.float32)


def dequant_nvfp4(w_q, sf_swizzled, gs, n, k):
    w_f32 = break_fp4_bytes(w_q).reshape(n, k // 16, 16)
    sf = convert_swizzled_to_linear(sf_swizzled.view(torch.float8_e4m3fn), n, k)
    return (w_f32 * (sf.float() / gs).unsqueeze(-1)).reshape(n, k)


@unittest.skipIf(get_device_sm() < 100, "NVFP4 MoE backends require SM100+")
class TestNvFp4MoeBackends(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        _init_single_process_dist()
        torch.set_default_device("cuda")

    def _run_backend(self, backend: str):
        from flashinfer import fp4_quantize

        from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
        from sglang.srt.layers.moe.topk import StandardTopKOutput

        torch.manual_seed(7)
        quant_config = ModelOptFp4Config(
            is_checkpoint_nvfp4_serialized=True, group_size=16
        )
        with get_context().override_server_args(
            model_path="dummy"
        ), get_flags().moe.override(
            runner_backend=MoeRunnerBackend(backend)
        ), get_parallel().override(
            moe_ep_size=1,
            moe_ep_rank=0,
            moe_tp_size=1,
            moe_tp_rank=0,
            tp_size=1,
            tp_rank=0,
        ):
            layer = FusedMoE(
                num_experts=E,
                hidden_size=H,
                intermediate_size=I,
                layer_id=0,
                top_k=TOPK,
                params_dtype=torch.bfloat16,
                quant_config=quant_config,
                gate_up_interleaved=False,
            ).cuda()

            w13_ref = torch.zeros(E, 2 * I, H, dtype=torch.float32, device="cuda")
            w2_ref = torch.zeros(E, H, I, dtype=torch.float32, device="cuda")
            for e in range(E):
                w13 = torch.randn(2 * I, H, dtype=torch.bfloat16, device="cuda") / 10
                w2 = torch.randn(H, I, dtype=torch.bfloat16, device="cuda") / 10
                w13_gs = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / w13.abs().max().float()
                w2_gs = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / w2.abs().max().float()
                w13_q, w13_sf = fp4_quantize(w13, w13_gs)
                w2_q, w2_sf = fp4_quantize(w2, w2_gs)
                layer.w13_weight.data[e].copy_(w13_q)
                layer.w2_weight.data[e].copy_(w2_q)
                layer.w13_weight_scale.data[e].copy_(
                    convert_swizzled_to_linear(
                        w13_sf.view(torch.float8_e4m3fn), 2 * I, H
                    )
                )
                layer.w2_weight_scale.data[e].copy_(
                    convert_swizzled_to_linear(w2_sf.view(torch.float8_e4m3fn), H, I)
                )
                layer.w13_weight_scale_2.data[e].fill_(1.0 / w13_gs)
                layer.w2_weight_scale_2.data[e].fill_(1.0 / w2_gs)
                w13_ref[e] = dequant_nvfp4(w13_q, w13_sf, w13_gs, 2 * I, H)
                w2_ref[e] = dequant_nvfp4(w2_q, w2_sf, w2_gs, H, I)
            act_scale = 1.0 / (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX)
            layer.w13_input_scale.data.fill_(act_scale)
            layer.w2_input_scale.data.fill_(act_scale)

            layer.quant_method.process_weights_after_loading(layer)

            x = torch.randn(M, H, dtype=torch.bfloat16, device="cuda") / 10
            router_logits = torch.randn(M, E, dtype=torch.float32, device="cuda")
            weights = torch.softmax(router_logits, dim=-1)
            topk_weights, topk_ids = torch.topk(weights, TOPK, dim=-1)
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

            out = layer.forward(
                x,
                StandardTopKOutput(
                    topk_weights=topk_weights,
                    topk_ids=topk_ids.to(torch.int32),
                    router_logits=router_logits,
                ),
            )
            if not isinstance(out, torch.Tensor):
                out = out[0] if isinstance(out, tuple) else out.hidden_states

            ref = self._torch_moe_reference(
                layer, x, topk_weights, topk_ids, w13_ref, w2_ref
            )

            self.assertEqual(out.shape, (M, H))
            cos = torch.nn.functional.cosine_similarity(
                out.float().flatten(), ref.flatten(), dim=0
            ).item()
            self.assertGreater(cos, 0.99)

    @staticmethod
    def _torch_moe_reference(layer, x, topk_weights, topk_ids, w13_ref, w2_ref):
        from flashinfer import fp4_quantize

        def quant_roundtrip(t2d, gs):
            q, sf = fp4_quantize(t2d.to(torch.bfloat16), gs)
            return dequant_nvfp4(q, sf, gs, t2d.shape[0], t2d.shape[1])

        # The kernels quantize the input and the GEMM1->GEMM2 intermediate to
        # NVFP4; mirror both round trips or the comparison carries ~7% noise.
        act_gs = torch.tensor(
            FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX, dtype=torch.float32, device="cuda"
        )
        # TRTLLM consumes w13 as [up; gate] (GEMM1 scales are applied on that
        # assumption); CUTLASS / CuteDSL-v2 load up first as well via
        # load_up_proj_weight_first.
        up_first = (
            layer.quant_method.load_up_proj_weight_first
            or layer.quant_method.enable_flashinfer_trtllm_moe
        )
        x_dq = quant_roundtrip(x.float(), act_gs)
        m, h = x.shape
        ref = torch.zeros(m, h, dtype=torch.float32, device="cuda")
        for t in range(m):
            for j in range(topk_ids.shape[1]):
                e = int(topk_ids[t, j])
                gu = x_dq[t] @ w13_ref[e].T
                if up_first:
                    up, gate = gu[:I], gu[I:]
                else:
                    gate, up = gu[:I], gu[I:]
                act = torch.nn.functional.silu(gate) * up
                act_dq = quant_roundtrip(act.unsqueeze(0), act_gs)[0]
                ref[t] += float(topk_weights[t, j]) * (act_dq @ w2_ref[e].T)
        return ref

    def test_flashinfer_cutlass(self):
        self._run_backend("flashinfer_cutlass")

    def test_flashinfer_trtllm(self):
        self._run_backend("flashinfer_trtllm")

    def test_flashinfer_cutedsl(self):
        self._run_backend("flashinfer_cutedsl")


if __name__ == "__main__":
    unittest.main()
