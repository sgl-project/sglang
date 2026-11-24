"""Numerics for the NVFP4 FusedMoE runner backends (--moe-runner-backend).

Real FusedMoE path (NVFP4 checkpoint shards through the real weight_loader
-> weight processing -> forward) per backend vs a dequantized torch MoE
reference. Single GPU, tp=ep=1.
"""

import unittest
from unittest import mock

import torch

from sglang.srt.layers.moe.utils import MoeRunnerBackend
from sglang.srt.layers.quantization import modelopt_quant
from sglang.srt.layers.quantization.modelopt_quant import ModelOptFp4Config
from sglang.srt.runtime_context import get_context, get_flags, get_parallel
from sglang.srt.utils import get_device_sm, round_up
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.layer_ut_utils import assert_output_close, init_single_process_dist
from sglang.test.quant_ref_utils import (
    FLOAT4_E2M1_MAX,
    FLOAT8_E4M3_MAX,
    dequantize_nvfp4_to_dtype,
    quantize_nvfp4_shard,
)
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, stage="base-b", runner_config="4-gpu-b200")

E, H, I, TOPK, M = 8, 1024, 1024, 2, 32


@unittest.skipIf(get_device_sm() < 100, "NVFP4 MoE backends require SM100+")
class TestNvFp4MoeBackends(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        init_single_process_dist(master_port=29631)
        torch.set_default_device("cuda")

    def _run_backend(
        self,
        backend: str,
        intermediate_size: int = I,
        swiglu_limit: float | None = None,
        *,
        top_k: int = TOPK,
        is_w4a16: bool = False,
        apply_router_weight_on_input: bool = False,
        check_magnitude: bool = False,
    ):
        from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
        from sglang.srt.layers.moe.topk import TopKConfig, select_experts

        torch.manual_seed(7)
        quant_config = ModelOptFp4Config(
            is_checkpoint_nvfp4_serialized=True, group_size=16
        )
        quant_config.is_w4a16 = is_w4a16
        with (
            get_context().override_server_args(model_path="dummy"),
            # As a launch records it: a backend other than auto was passed.
            get_flags().moe.override(
                runner_backend=MoeRunnerBackend(backend),
                runner_backend_explicit=backend != "auto",
            ),
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
                intermediate_size=intermediate_size,
                layer_id=0,
                top_k=top_k,
                params_dtype=torch.bfloat16,
                quant_config=quant_config,
                swiglu_limit=swiglu_limit,
                gate_up_interleaved=False,
                apply_router_weight_on_input=apply_router_weight_on_input,
            ).cuda()

            # Checkpoint-format shards through the real weight_loader;
            # gate/up placement stays the loader's job.
            refs = {
                "w1": torch.zeros(
                    E, intermediate_size, H, dtype=torch.float32, device="cuda"
                ),
                "w3": torch.zeros(
                    E, intermediate_size, H, dtype=torch.float32, device="cuda"
                ),
                "w2": torch.zeros(
                    E, H, intermediate_size, dtype=torch.float32, device="cuda"
                ),
            }
            act_scale = 1.0 / (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX)
            for e in range(E):
                ws = {}
                for shard_id in ("w1", "w3", "w2"):
                    rows, cols = (
                        (intermediate_size, H)
                        if shard_id in ("w1", "w3")
                        else (H, intermediate_size)
                    )
                    ws[shard_id] = (
                        torch.randn(rows, cols, dtype=torch.bfloat16, device="cuda")
                        / 10
                    )
                # auto runs flashinfer_cutlass on SM110/SM120, which refuses
                # differing w1 and w3 global scales unless it was passed, so there
                # the auto cases quantize both with one scale that fits either.
                gs13 = None
                if backend == "auto" and get_device_sm() // 10 != 10:
                    amax = torch.maximum(ws["w1"].abs().max(), ws["w3"].abs().max())
                    gs13 = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / amax.to(torch.float32)
                for shard_id, w in ws.items():
                    w_q, sf_linear, gs, w_dequant = quantize_nvfp4_shard(
                        w, gs=gs13 if shard_id in ("w1", "w3") else None
                    )
                    prefix = "w13" if shard_id in ("w1", "w3") else "w2"
                    shards = [
                        ("weight", w_q),
                        ("weight_scale", sf_linear),
                        ("weight_scale_2", (1.0 / gs).clone()),
                    ]
                    if not is_w4a16:
                        # W4A16 checkpoints carry no activation scales.
                        shards.append(
                            ("input_scale", torch.tensor(act_scale, device="cuda"))
                        )
                    for suffix, loaded in shards:
                        name = f"{prefix}_{suffix}"
                        param = getattr(layer, name)
                        layer.weight_loader(
                            param, loaded, name, shard_id=shard_id, expert_id=e
                        )
                    refs[shard_id][e] = w_dequant
            if backend == "flashinfer_cutedsl":
                canonical_w13 = layer.w13_weight.clone()
                swap_ab = get_device_sm() in (100, 103)
                group_size = 16 if swap_ab else 64
                expected_w13 = torch.cat(
                    [
                        canonical_w13[:, offset + start : offset + start + group_size]
                        for start in range(0, intermediate_size, group_size)
                        for offset in (0, intermediate_size)
                    ],
                    dim=1,
                )
            layer.quant_method.process_weights_after_loading(layer)
            if backend == "flashinfer_cutedsl":
                torch.testing.assert_close(layer.w13_weight, expected_w13)
            if backend == "flashinfer_trtllm":
                expected_intermediate = round_up(intermediate_size, 64)
                self.assertEqual(
                    layer.intermediate_size_per_partition, expected_intermediate
                )
                self.assertEqual(
                    layer.moe_runner_config.intermediate_size_per_partition,
                    expected_intermediate,
                )
                self.assertEqual(
                    layer.w13_weight.shape, (E, 2 * expected_intermediate, H // 2)
                )
                self.assertEqual(
                    layer.w2_weight.shape, (E, H, expected_intermediate // 2)
                )

            x = torch.randn(M, H, dtype=torch.bfloat16, device="cuda") / 10
            router_logits = torch.randn(M, E, dtype=torch.float32, device="cuda")
            # Route through the real topk compute path, not a hand-rolled one.
            topk_output = select_experts(
                hidden_states=x,
                router_logits=router_logits,
                # Renormalized top-1 weights are all 1, which would hide where
                # a router-on-input layer applies them.
                topk_config=TopKConfig(
                    top_k=top_k, renormalize=not apply_router_weight_on_input
                ),
            )
            if apply_router_weight_on_input:
                # Llama 4's router returns its weights in the activation dtype.
                topk_output = topk_output._replace(
                    topk_weights=topk_output.topk_weights.to(torch.bfloat16)
                )
            topk_weights, topk_ids = topk_output.topk_weights, topk_output.topk_ids

            out = layer.forward(x, topk_output)
            if backend == "flashinfer_cutedsl":
                self.assertEqual(layer._cutedsl_wrapper.swap_ab, swap_ab)
            if not isinstance(out, torch.Tensor):
                out = out[0] if isinstance(out, tuple) else out.hidden_states

            ref = self._torch_moe_reference(
                x,
                topk_weights,
                topk_ids,
                refs["w1"],
                refs["w3"],
                refs["w2"],
                swiglu_limit,
                # A W4A16 layer's activations are quantized with the default
                # activation scale of 1.
                act_scale=1.0 if is_w4a16 else act_scale,
                apply_router_weight_on_input=apply_router_weight_on_input,
            )
            assert_output_close(self, out, ref)
            if check_magnitude:
                # The cosine check cannot see a scale error, such as a router
                # weight applied twice or not at all.
                norm_ratio = (out.float().norm() / ref.norm()).item()
                self.assertLess(abs(norm_ratio - 1), 0.1)
            return layer

    @staticmethod
    def _torch_moe_reference(
        x,
        topk_weights,
        topk_ids,
        w1_ref,
        w3_ref,
        w2_ref,
        swiglu_limit,
        act_scale=1.0 / (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX),
        apply_router_weight_on_input=False,
    ):
        from flashinfer import fp4_quantize

        def quant_roundtrip(t2d, gs):
            q, sf = fp4_quantize(t2d.to(torch.bfloat16), gs)
            return dequantize_nvfp4_to_dtype(q, sf, gs, torch.float32)

        # Mirror the kernels' two fp4 activation round trips (input and
        # GEMM1->GEMM2) or the comparison carries ~7% noise. The reference
        # stays in checkpoint semantics (w1=gate, w3=up).
        act_gs = torch.tensor(1.0 / act_scale, dtype=torch.float32, device="cuda")
        m, h = x.shape
        ref = torch.zeros(m, h, dtype=torch.float32, device="cuda")
        for t in range(m):
            for j in range(topk_ids.shape[1]):
                e = int(topk_ids[t, j])
                weight = float(topk_weights[t, j])
                x_t = x[t : t + 1].float()
                if apply_router_weight_on_input:
                    # The runner scales the BF16 input before quantizing it.
                    x_t = (x_t * weight).to(torch.bfloat16).float()
                    weight = 1.0
                x_dq = quant_roundtrip(x_t, act_gs)[0]
                gate = x_dq @ w1_ref[e].T
                up = x_dq @ w3_ref[e].T
                if swiglu_limit is not None:
                    gate = gate.clamp(max=swiglu_limit)
                    up = up.clamp(-swiglu_limit, swiglu_limit)
                act = torch.nn.functional.silu(gate) * up
                act_dq = quant_roundtrip(act.unsqueeze(0), act_gs)[0]
                ref[t] += weight * (act_dq @ w2_ref[e].T)
        return ref

    def test_flashinfer_cutlass(self):
        self._run_backend("flashinfer_cutlass")

    def test_auto(self):
        """auto must prepare the weights for the runner it resolves to (TRT-LLM
        on SM100, FlashInfer CUTLASS on SM110/SM120); a layer prepared for one
        runner and dispatched to another fails in its first forward."""
        layer = self._run_backend("auto")
        expected = (
            MoeRunnerBackend.FLASHINFER_TRTLLM
            if get_device_sm() // 10 == 10
            else MoeRunnerBackend.FLASHINFER_CUTLASS
        )
        self.assertIs(layer.quant_method.get_moe_runner_backend(layer), expected)

    def test_flashinfer_cutlass_router_weight_on_input(self):
        """A layer that applies its top-1 router weight to the input (Llama 4)
        dispatches BF16, since flashinfer_cutlass scales that input before
        quantizing it, while the launch's other layers still dispatch FP4."""
        with mock.patch.object(
            modelopt_quant, "_use_nvfp4_dispatch", return_value=True
        ):
            layer = self._run_backend(
                "flashinfer_cutlass",
                top_k=1,
                apply_router_weight_on_input=True,
                check_magnitude=True,
            )
            self.assertIsNone(layer.dispatcher.quant_config["input_global_scale"])
            layer = self._run_backend("flashinfer_cutlass")
            self.assertIsNotNone(layer.dispatcher.quant_config["input_global_scale"])

    def test_auto_w4a16(self):
        """W4A16 layers resolve under auto like W4A4 ones: TRT-LLM on SM100,
        FlashInfer CUTLASS on SM110/SM120, both quantizing their activations
        to FP4 with the default activation scale."""
        layer = self._run_backend("auto", is_w4a16=True, check_magnitude=True)
        expected = (
            MoeRunnerBackend.FLASHINFER_TRTLLM
            if get_device_sm() // 10 == 10
            else MoeRunnerBackend.FLASHINFER_CUTLASS
        )
        self.assertIs(layer.quant_method.get_moe_runner_backend(layer), expected)

    def test_flashinfer_trtllm(self):
        self._run_backend("flashinfer_trtllm")

    def test_flashinfer_trtllm_intermediate_alignment(self):
        """Gated NVFP4 keeps 64-aligned expert widths without padding to 128."""
        for intermediate_size in (160, 192, 256):
            with self.subTest(intermediate_size=intermediate_size):
                self._run_backend("flashinfer_trtllm", intermediate_size)

    def test_flashinfer_cutedsl(self):
        self._run_backend("flashinfer_cutedsl")

    def test_flashinfer_cutedsl_swiglu_limit(self):
        """The limit sits inside the GEMM1 outputs' spread (std ~0.3), so the clamp binds."""
        self._run_backend("flashinfer_cutedsl", swiglu_limit=0.1)


if __name__ == "__main__":
    unittest.main()
