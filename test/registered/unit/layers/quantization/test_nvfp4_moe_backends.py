"""Numerics for the NVFP4 FusedMoE runner backends (--moe-runner-backend).

Real FusedMoE path (NVFP4 checkpoint shards through the real weight_loader
-> weight processing -> forward) per backend vs a dequantized torch MoE
reference. Single GPU, tp=ep=1.

W4A16 raw weight reloads also reuse the original destination buffers and a
captured CUDA graph, comparing each update with a freshly loaded layer.
"""

import unittest

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.moe.utils import MoeRunnerBackend
from sglang.srt.layers.quantization.modelopt_quant import ModelOptFp4Config
from sglang.srt.runtime_context import get_context, get_flags, get_parallel
from sglang.srt.utils import get_device_sm
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.layer_ut_utils import assert_output_close, init_single_process_dist
from sglang.test.quant_ref_utils import (
    FLOAT4_E2M1_MAX,
    FLOAT8_E4M3_MAX,
    dequantize_nvfp4_to_dtype,
    quantize_nvfp4_shard,
)
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=45, stage="base-b", runner_config="4-gpu-b200")

E, H, I, TOPK, M = 8, 1024, 1024, 2, 32


@unittest.skipIf(get_device_sm() < 100, "NVFP4 MoE backends require SM100+")
class TestNvFp4MoeBackends(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        init_single_process_dist(master_port=29631)
        torch.set_default_device("cuda")

    def _run_backend(self, backend: str):
        from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
        from sglang.srt.layers.moe.topk import TopKConfig, select_experts

        torch.manual_seed(7)
        quant_config = ModelOptFp4Config(
            is_checkpoint_nvfp4_serialized=True, group_size=16
        )
        with (
            get_context().override_server_args(model_path="dummy"),
            get_flags().moe.override(runner_backend=MoeRunnerBackend(backend)),
            get_parallel().override(
                moe_ep_size=1,
                moe_ep_rank=0,
                moe_tp_size=1,
                moe_tp_rank=0,
                tp_size=1,
                tp_rank=0,
            ),
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

            # Checkpoint-format shards through the real weight_loader;
            # gate/up placement stays the loader's job.
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
                    w_q, sf_linear, gs, w_dequant = quantize_nvfp4_shard(w)
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
                    refs[shard_id][e] = w_dequant
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
            assert_output_close(self, out, ref)

    @staticmethod
    def _torch_moe_reference(x, topk_weights, topk_ids, w1_ref, w3_ref, w2_ref):
        from flashinfer import fp4_quantize

        def quant_roundtrip(t2d, gs):
            q, sf = fp4_quantize(t2d.to(torch.bfloat16), gs)
            return dequantize_nvfp4_to_dtype(q, sf, gs, torch.float32)

        # Mirror the kernels' two fp4 activation round trips (input and
        # GEMM1->GEMM2) or the comparison carries ~7% noise. The reference
        # stays in checkpoint semantics (w1=gate, w3=up).
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

    def test_flashinfer_cutedsl_w4a16_weight_reload(self):
        from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
        from sglang.srt.layers.moe.topk import TopKConfig, select_experts

        def make_layer(device):
            with torch.device(device):
                return FusedMoE(
                    num_experts=E,
                    hidden_size=H,
                    intermediate_size=I,
                    layer_id=0,
                    top_k=TOPK,
                    params_dtype=torch.bfloat16,
                    quant_config=ModelOptFp4Config(
                        is_checkpoint_nvfp4_serialized=True, group_size=16
                    ),
                    gate_up_interleaved=False,
                )

        def checkpoint(seed):
            torch.manual_seed(seed)
            shards = []
            for expert in range(E):
                w13 = torch.randn(2, I, H, dtype=torch.bfloat16) / 10
                gs13 = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / w13.abs().max().float()
                for shard_id, weight in (
                    ("w1", w13[0]),
                    ("w3", w13[1]),
                    ("w2", torch.randn(H, I, dtype=torch.bfloat16) / 10),
                ):
                    q, sf, gs, _ = quantize_nvfp4_shard(
                        weight, gs=gs13 if shard_id != "w2" else None
                    )
                    prefix = "w2" if shard_id == "w2" else "w13"
                    for suffix, value in (
                        ("weight", q),
                        ("weight_scale", sf),
                        ("weight_scale_2", 1.0 / gs),
                        ("input_scale", torch.tensor(1.0)),
                    ):
                        shards.append(
                            (f"{prefix}_{suffix}", shard_id, expert, value.cpu())
                        )
            return shards

        def load(layer, shards):
            for name, shard_id, expert, value in shards:
                param = getattr(layer, name)
                layer.weight_loader(
                    param, value.to(param.device), name, shard_id, expert
                )

        def forward(layer):
            out = layer.forward(x, topk)
            if isinstance(out, torch.Tensor):
                return out
            return out[0] if isinstance(out, tuple) else out.hidden_states

        def pointers(layer):
            tensors = [*layer.parameters(), *layer._cutedsl_scales]
            tensors.append(layer._cutedsl_input_scale)
            return [(t.data_ptr(), t.shape, t.dtype, t.device) for t in tensors]

        with (
            get_context().override_server_args(
                model_path="dummy", chunked_prefill_size=M
            ),
            get_flags().moe.override(
                runner_backend=MoeRunnerBackend.FLASHINFER_CUTEDSL
            ),
            get_parallel().override(
                moe_ep_size=1,
                moe_ep_rank=0,
                moe_tp_size=1,
                moe_tp_rank=0,
                tp_size=1,
                tp_rank=0,
            ),
            envs.SGLANG_FLASHINFER_CUTEDSL_NVFP4_W4A16.override(True),
            envs.SGLANG_FLASHINFER_MOE_FUSED_FINALIZE.override(False),
            torch.no_grad(),
        ):
            a, b = checkpoint(31), checkpoint(57)
            cpu = make_layer("cpu")
            # Match a dummy CPU replica followed by pinned staging allocation.
            for param in cpu.parameters():
                param.fill_(1)
            cpu.quant_method.process_weights_after_loading(cpu)
            for param in cpu.parameters():
                param.data = param.data.pin_memory()
            raw = {
                name: param
                for name, param in cpu.named_parameters()
                if hasattr(param, "weight_loader")
            }
            gpu = make_layer("cuda")
            load(gpu, a)
            gpu.quant_method.process_weights_after_loading(gpu)
            # Retain the initial registered buffers across every update.
            registered = {name: getattr(gpu, name).detach() for name in raw}
            x = torch.randn(M, H, dtype=torch.bfloat16) / 10
            topk = select_experts(
                hidden_states=x,
                router_logits=torch.randn(M, E, dtype=torch.float32),
                topk_config=TopKConfig(top_k=TOPK, renormalize=True),
            )
            baseline = forward(gpu).clone()
            self.assertEqual(gpu._cutedsl_wrapper.quant_mode, "w4a16")
            initial_pointers = pointers(gpu)
            warmup = torch.cuda.Stream()
            warmup.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(warmup):
                for _ in range(3):
                    forward(gpu)
            torch.cuda.current_stream().wait_stream(warmup)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                graph_output = forward(gpu)

            for state, shards in (("A1", a), ("B", b), ("A2", a)):
                with self.subTest(state=state):
                    load(cpu, shards)
                    for name, source in raw.items():
                        target = registered[name]
                        self.assertEqual(
                            (source.shape, source.dtype), (target.shape, target.dtype)
                        )
                        self.assertTrue(
                            source.is_contiguous() and target.is_contiguous()
                        )
                        target.view(torch.uint8).view(-1).copy_(
                            source.view(torch.uint8).view(-1)
                        )
                    gpu.quant_method.process_weights_after_loading(gpu)
                    self.assertEqual(pointers(gpu), initial_pointers)
                    graph_output.fill_(float("nan"))
                    graph.replay()
                    torch.cuda.synchronize()
                    replay = graph_output.clone()
                    eager = forward(gpu).clone()
                    fresh = make_layer("cuda")
                    load(fresh, shards)
                    fresh.quant_method.process_weights_after_loading(fresh)
                    expected = forward(fresh)
                    torch.testing.assert_close(eager, expected, rtol=0, atol=0)
                    torch.testing.assert_close(replay, expected, rtol=0, atol=0)
                    if state == "B":
                        self.assertFalse(torch.equal(eager, baseline))
                    else:
                        torch.testing.assert_close(eager, baseline, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
