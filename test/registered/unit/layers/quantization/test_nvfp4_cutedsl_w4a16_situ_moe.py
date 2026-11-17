"""CuTe DSL NVFP4 W4A16 MoE with Kimi-K3's SiTU activation.

Real FusedMoE path (NVFP4 checkpoint shards through the real weight_loader ->
weight processing -> forward) on --moe-runner-backend flashinfer_cutedsl with
SGLANG_FLASHINFER_CUTEDSL_NVFP4_W4A16=1, against a dequantized torch reference
of SiTU: (beta*tanh(g/beta)*sigmoid(g)) * (linear_beta*tanh(u/linear_beta)).
Single GPU, tp=ep=1.
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
    quantize_nvfp4_shard,
)
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=120, stage="base-b", runner_config="4-gpu-b200")

# Kimi-K3 MoE activation parameters (config.activation_situ_beta / _linear_beta).
SITU_BETA, SITU_LINEAR_BETA = 4.0, 25.0
E, H, INTER, TOPK, M = 8, 1024, 1024, 4, 32
# Large enough that gate/up reach SiTU's tanh soft caps (|g| >> beta,
# |u| ~ linear_beta), where SiTU and SiLU differ.
X_STD, W_STD = 2.0, 0.25


def _situ(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    gate = SITU_BETA * torch.tanh(gate / SITU_BETA) * torch.sigmoid(gate)
    return gate * (SITU_LINEAR_BETA * torch.tanh(up / SITU_LINEAR_BETA))


@unittest.skipIf(get_device_sm() < 100, "CuTe DSL NVFP4 MoE requires SM100+")
class TestNvFp4CuteDslW4A16SituMoe(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        init_single_process_dist(master_port=29633)
        torch.set_default_device("cuda")

    def _build_layer(self):
        """FusedMoE loaded with random NVFP4 experts; returns (layer, refs)."""
        from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE

        layer = FusedMoE(
            num_experts=E,
            hidden_size=H,
            intermediate_size=INTER,
            layer_id=0,
            top_k=TOPK,
            params_dtype=torch.bfloat16,
            quant_config=ModelOptFp4Config(
                is_checkpoint_nvfp4_serialized=True, group_size=16
            ),
            activation="situ",
            gemm1_alpha=SITU_BETA,
            gemm1_clamp_limit=SITU_LINEAR_BETA,
            gate_up_interleaved=False,
        ).cuda()
        refs = {
            "w1": torch.zeros(E, INTER, H, dtype=torch.float32),
            "w3": torch.zeros(E, INTER, H, dtype=torch.float32),
            "w2": torch.zeros(E, H, INTER, dtype=torch.float32),
        }
        act_scale = torch.tensor(1.0 / (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX))
        for e in range(E):
            ws = {
                "w1": torch.randn(INTER, H, dtype=torch.bfloat16) * W_STD,
                "w3": torch.randn(INTER, H, dtype=torch.bfloat16) * W_STD,
                "w2": torch.randn(H, INTER, dtype=torch.bfloat16) * W_STD,
            }
            # CuTe DSL applies one GEMM1 alpha to gate and up before SiTU's
            # tanh, so gate and up share one global scale (as checkpoints do).
            gate_up_amax = torch.maximum(ws["w1"].abs().max(), ws["w3"].abs().max())
            gate_up_gs = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / gate_up_amax.float()
            for shard_id, w in ws.items():
                gs = None if shard_id == "w2" else gate_up_gs
                w_q, sf_linear, gs, w_dequant = quantize_nvfp4_shard(w, gs)
                prefix = "w2" if shard_id == "w2" else "w13"
                for suffix, loaded in (
                    ("weight", w_q),
                    ("weight_scale", sf_linear),
                    ("weight_scale_2", (1.0 / gs).clone()),
                    ("input_scale", act_scale),
                ):
                    name = f"{prefix}_{suffix}"
                    layer.weight_loader(
                        getattr(layer, name),
                        loaded,
                        name,
                        shard_id=shard_id,
                        expert_id=e,
                    )
                refs[shard_id][e] = w_dequant
        return layer, refs

    def test_cutedsl_w4a16_situ_matches_dequantized_reference(self):
        from sglang.srt.layers.moe.topk import TopKConfig, select_experts

        torch.manual_seed(11)
        with (
            envs.SGLANG_FLASHINFER_CUTEDSL_NVFP4_W4A16.override(True),
            get_context().override_server_args(model_path="dummy"),
            get_flags().moe.override(
                runner_backend=MoeRunnerBackend("flashinfer_cutedsl")
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
            layer, refs = self._build_layer()
            layer.quant_method.process_weights_after_loading(layer)

            x = torch.randn(M, H, dtype=torch.bfloat16) * X_STD
            topk_output = select_experts(
                hidden_states=x,
                router_logits=torch.randn(M, E, dtype=torch.float32),
                topk_config=TopKConfig(top_k=TOPK, renormalize=True),
            )
            out = layer.forward(x, topk_output)
            if not isinstance(out, torch.Tensor):
                out = out[0] if isinstance(out, tuple) else out.hidden_states

        ref, silu_ref = self._torch_moe_reference(
            x, topk_output.topk_weights, topk_output.topk_ids, refs
        )
        # The inputs must reach SiTU's soft caps, or a SiLU kernel would pass.
        silu_cos = torch.nn.functional.cosine_similarity(
            silu_ref.flatten(), ref.flatten(), dim=0
        ).item()
        self.assertLess(silu_cos, 0.99)
        # W4A16 keeps activations in BF16, so only BF16 rounding separates the
        # kernel from the dequantized-weight reference.
        assert_output_close(
            self,
            out,
            ref,
            cos_threshold=0.9995,
            rtol=3e-2,
            atol=3e-2 * ref.abs().max().item(),
        )

    @staticmethod
    def _torch_moe_reference(x, topk_weights, topk_ids, refs):
        """(SiTU reference, same reference with SiLU) in fp32, with the
        GEMM1 output rounded to BF16 as the W4A16 kernel stores it."""
        xf = x.float()
        out = torch.zeros(x.shape[0], H, dtype=torch.float32)
        out_silu = torch.zeros_like(out)
        for t in range(x.shape[0]):
            for j in range(topk_ids.shape[1]):
                e = int(topk_ids[t, j])
                weight = float(topk_weights[t, j])
                gate = xf[t] @ refs["w1"][e].T
                up = xf[t] @ refs["w3"][e].T
                act = _situ(gate, up).to(torch.bfloat16).float()
                act_silu = (torch.nn.functional.silu(gate) * up).bfloat16().float()
                out[t] += weight * (act @ refs["w2"][e].T)
                out_silu[t] += weight * (act_silu @ refs["w2"][e].T)
        return out, out_silu


if __name__ == "__main__":
    unittest.main()
