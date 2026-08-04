"""Numerics for the NVFP4 FusedMoE runner backends (--moe-runner-backend).

Runs the real FusedMoE layer path (construct -> load NVFP4 checkpoint-format
shards through the real weight_loader -> process_weights_after_loading ->
forward) per backend against a dequantized torch MoE reference, covering
the per-backend weight preparation
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
        from sglang.srt.layers.moe.topk import TopKConfig, select_experts

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

            # Feed checkpoint-format shards through the real weight_loader so
            # gate/up placement and scale gathering stay the loader's job.
            refs = {
                "w1": torch.zeros(E, I, H, dtype=torch.float32, device="cuda"),
                "w3": torch.zeros(E, I, H, dtype=torch.float32, device="cuda"),
                "w2": torch.zeros(E, H, I, dtype=torch.float32, device="cuda"),
            }
            act_scale = 1.0 / (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX)
            for e in range(E):
                for shard_id in ("w1", "w3", "w2"):
                    rows, cols = (I, H) if shard_id in ("w1", "w3") else (H, I)
                    w = (
                        torch.randn(rows, cols, dtype=torch.bfloat16, device="cuda")
                        / 10
                    )
                    gs = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / w.abs().max().float()
                    w_q, w_sf = fp4_quantize(w, gs)
                    sf_linear = convert_swizzled_to_linear(
                        w_sf.view(torch.float8_e4m3fn), rows, cols
                    )
                    prefix = "w13" if shard_id in ("w1", "w3") else "w2"
                    for suffix, loaded in (
                        ("weight", w_q),
                        ("weight_scale", sf_linear),
                        ("weight_scale_2", (1.0 / gs).clone()),
                        ("input_scale", torch.tensor(act_scale, device="cuda")),
                    ):
                        name = f"{prefix}_{suffix}"
                        param = getattr(layer, name)
                        layer.weight_loader(
                            param, loaded, name, shard_id=shard_id, expert_id=e
                        )
                    refs[shard_id][e] = dequant_nvfp4(w_q, w_sf, gs, rows, cols)

            layer.quant_method.process_weights_after_loading(layer)

            x = torch.randn(M, H, dtype=torch.bfloat16, device="cuda") / 10
            router_logits = torch.randn(M, E, dtype=torch.float32, device="cuda")
            # Route through the real topk compute path, not a hand-rolled one.
            topk_output = select_experts(
                hidden_states=x,
                router_logits=router_logits,
                topk_config=TopKConfig(top_k=TOPK, renormalize=True),
            )
            topk_weights, topk_ids = topk_output.topk_weights, topk_output.topk_ids

            out = layer.forward(x, topk_output)
            if not isinstance(out, torch.Tensor):
                out = out[0] if isinstance(out, tuple) else out.hidden_states

            ref = self._torch_moe_reference(
                x, topk_weights, topk_ids, refs["w1"], refs["w3"], refs["w2"]
            )

            self.assertEqual(out.shape, (M, H))
            cos = torch.nn.functional.cosine_similarity(
                out.float().flatten(), ref.flatten(), dim=0
            ).item()
            self.assertGreater(cos, 0.99)

    @staticmethod
    def _torch_moe_reference(x, topk_weights, topk_ids, w1_ref, w3_ref, w2_ref):
        from flashinfer import fp4_quantize

        def quant_roundtrip(t2d, gs):
            q, sf = fp4_quantize(t2d.to(torch.bfloat16), gs)
            return dequant_nvfp4(q, sf, gs, t2d.shape[0], t2d.shape[1])

        # The kernels quantize the input and the GEMM1->GEMM2 intermediate to
        # NVFP4; mirror both round trips or the comparison carries ~7% noise.
        # Shards were fed through the real weight_loader, so the reference
        # stays in checkpoint semantics (w1=gate, w3=up); physical gate/up
        # placement is the loader's job.
        act_gs = torch.tensor(
            FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX, dtype=torch.float32, device="cuda"
        )
        x_dq = quant_roundtrip(x.float(), act_gs)
        m, h = x.shape
        ref = torch.zeros(m, h, dtype=torch.float32, device="cuda")
        for t in range(m):
            for j in range(topk_ids.shape[1]):
                e = int(topk_ids[t, j])
                gate = x_dq[t] @ w1_ref[e].T
                up = x_dq[t] @ w3_ref[e].T
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
