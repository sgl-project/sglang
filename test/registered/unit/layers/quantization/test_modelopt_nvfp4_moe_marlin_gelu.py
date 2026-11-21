"""A ModelOpt NVFP4 MoE with a GELU expert activation (Gemma 4) must run on the Marlin
runner, the default for NVFP4 MoE on SM80-SM90, and match a dequantized reference."""

import unittest

import torch
import torch.nn.functional as F

from sglang.srt.layers.moe.utils import MoeRunnerBackend
from sglang.srt.layers.quantization.modelopt_quant import (
    ModelOptFp4Config,
    ModelOptNvFp4FusedMoEMethod,
)
from sglang.srt.model_loader.loader import DefaultModelLoader
from sglang.srt.runtime_context import get_context, get_flags, get_parallel
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.layer_ut_utils import assert_output_close, init_single_process_dist
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-small")

E, H, N_INTER, TOPK, M = 8, 1024, 512, 2, 32
GROUP_SIZE = 16
DEVICE = torch.device("cuda")
QUANT_CONFIG = {
    "quant_method": "modelopt",
    "quant_algo": "NVFP4",
    "group_size": GROUP_SIZE,
    "exclude_modules": ["lm_head"],
}
E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
# ModelOpt NVFP4 MoE accepts only these activation names (_SUPPORTED_ACT_STRS).
ACTIVATIONS = {"silu": F.silu, "gelu": F.gelu}


def _random_shard(rows, cols, global_scale, gen):
    """e2m1 codes packed two per byte (low nibble = even input column), e4m3 block scales."""
    codes = torch.randint(0, 16, (rows, cols), device=DEVICE, generator=gen)
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).to(torch.uint8)
    lut = torch.tensor(E2M1, device=DEVICE)
    values = lut[codes & 7] * torch.where(codes >= 8, -1.0, 1.0)
    block_scale = (
        torch.rand(rows, cols // GROUP_SIZE, device=DEVICE, generator=gen) + 0.5
    ).to(torch.float8_e4m3fn)
    dequant = (
        values * block_scale.float().repeat_interleave(GROUP_SIZE, dim=1) * global_scale
    )
    tensors = {
        "weight": packed,
        "weight_scale": block_scale,
        "weight_scale_2": torch.tensor(global_scale, device=DEVICE),
        "input_scale": torch.tensor(1.0, device=DEVICE),
    }
    return tensors, dequant


def _random_checkpoint(seed):
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    shards, w13, w2 = {}, [], []
    for expert_id in range(E):
        # ModelOpt exports one global scale shared by the fused gate and up projections.
        w13_scale = 2e-3 * (1 + expert_id / E)
        dense = {}
        for shard_id, (rows, cols), global_scale in (
            ("w1", (N_INTER, H), w13_scale),
            ("w3", (N_INTER, H), w13_scale),
            ("w2", (H, N_INTER), 3e-3),
        ):
            shards[expert_id, shard_id], dense[shard_id] = _random_shard(
                rows, cols, global_scale, gen
            )
        w13.append(torch.cat([dense["w1"], dense["w3"]]))
        w2.append(dense["w2"])
    return shards, torch.stack(w13), torch.stack(w2)


def _reference(x, w13, w2, topk_weights, topk_ids, act):
    out = torch.zeros(x.shape[0], H, device=DEVICE)
    for t in range(x.shape[0]):
        for k in range(TOPK):
            e = int(topk_ids[t, k])
            gate_up = w13[e] @ x[t].float()
            h = act(gate_up[:N_INTER]) * gate_up[N_INTER:]
            out[t] += topk_weights[t, k].float() * (w2[e] @ h)
    return out


class TestModelOptNvfp4MoEMarlinGelu(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        init_single_process_dist(master_port=29635)
        torch.set_default_device("cuda")

    def _run(self, activation):
        from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
        from sglang.srt.layers.moe.topk import TopKConfig, select_experts

        shards, w13, w2 = _random_checkpoint(0)
        with (
            get_context().override_server_args(model_path="dummy"),
            get_flags().moe.override(runner_backend=MoeRunnerBackend.MARLIN),
            get_parallel().override(
                moe_ep_size=1,
                moe_ep_rank=0,
                moe_tp_size=1,
                moe_tp_rank=0,
                tp_size=1,
                tp_rank=0,
            ),
        ):
            model = torch.nn.Module()
            model.experts = FusedMoE(
                num_experts=E,
                hidden_size=H,
                intermediate_size=N_INTER,
                layer_id=0,
                top_k=TOPK,
                params_dtype=torch.bfloat16,
                quant_config=ModelOptFp4Config.from_config(QUANT_CONFIG),
                prefix="model.layers.0.mlp.experts",
                activation=activation,
            ).cuda()
            self.assertIs(type(model.experts.quant_method), ModelOptNvFp4FusedMoEMethod)
            for (expert_id, shard_id), tensors in shards.items():
                prefix = "w2" if shard_id == "w2" else "w13"
                for suffix, tensor in tensors.items():
                    name = f"{prefix}_{suffix}"
                    model.experts.weight_loader(
                        getattr(model.experts, name),
                        tensor,
                        name,
                        shard_id=shard_id,
                        expert_id=expert_id,
                    )
            DefaultModelLoader.postprocess_weights(model, DEVICE)

            torch.manual_seed(0)
            x = torch.randn(M, H, dtype=torch.bfloat16)
            topk_output = select_experts(
                hidden_states=x,
                router_logits=torch.randn(M, E, dtype=torch.float32),
                topk_config=TopKConfig(top_k=TOPK, renormalize=True),
            )
            out = model.experts.forward(x.clone(), topk_output)
            if not isinstance(out, torch.Tensor):
                out = out[0] if isinstance(out, tuple) else out.hidden_states
        ref = _reference(
            x,
            w13,
            w2,
            topk_output.topk_weights,
            topk_output.topk_ids,
            ACTIVATIONS[activation],
        )
        assert_output_close(self, out, ref, cos_threshold=0.999)

    def test_activations_on_marlin(self):
        for activation in ACTIVATIONS:
            with self.subTest(activation=activation):
                self._run(activation)


if __name__ == "__main__":
    unittest.main()
