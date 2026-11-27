"""A W4A16 compressed-tensors MoE with a GELU expert activation (Gemma 4) must run on
the Marlin and Triton runners and match a dequantized reference."""

import copy
import unittest
from unittest import mock

import torch
import torch.nn.functional as F

from sglang.srt.layers.moe.utils import MoeRunnerBackend
from sglang.srt.layers.quantization.compressed_tensors import (
    compressed_tensors as ct_module,
)
from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsWNA16MoE,
    CompressedTensorsWNA16TritonMoE,
)
from sglang.srt.model_loader.loader import DefaultModelLoader
from sglang.srt.runtime_context import get_context, get_flags, get_parallel
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.layer_ut_utils import assert_output_close, init_single_process_dist
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-small")

E, H, N_INTER, TOPK, M = 8, 1024, 512, 2, 32
GROUP_SIZE, PACK_FACTOR = 32, 8
DEVICE = torch.device("cuda")
QUANT_CONFIG = {
    "quant_method": "compressed-tensors",
    "format": "pack-quantized",
    "config_groups": {
        "group_0": {
            "targets": ["re:.*mlp\\.experts.*"],
            "weights": {
                "num_bits": 4,
                "type": "int",
                "symmetric": True,
                "strategy": "group",
                "group_size": GROUP_SIZE,
            },
            "input_activations": None,
        }
    },
    "ignore": ["lm_head"],
}
ACTIVATIONS = {
    "silu": F.silu,
    "gelu": F.gelu,
    "gelu_tanh": lambda x: F.gelu(x, approximate="tanh"),
}
SCHEMES = {
    MoeRunnerBackend.MARLIN: CompressedTensorsWNA16MoE,
    MoeRunnerBackend.TRITON: CompressedTensorsWNA16TritonMoE,
}


def _random_shard(rows, cols, gen):
    """Signed int4 values stored with a +8 offset, eight per int32 along the input dim."""
    q = torch.randint(-8, 8, (rows, cols), device=DEVICE, generator=gen)
    scale = (
        torch.rand(rows, cols // GROUP_SIZE, device=DEVICE, generator=gen) / 50 + 1e-3
    ).to(torch.bfloat16)
    nibbles = (q + 8).to(torch.int64).view(rows, cols // PACK_FACTOR, PACK_FACTOR)
    shifts = 4 * torch.arange(PACK_FACTOR, device=DEVICE)
    packed = (nibbles << shifts).sum(dim=-1)
    packed = torch.where(packed >= 2**31, packed - 2**32, packed).to(torch.int32)
    dequant = q.float() * scale.float().repeat_interleave(GROUP_SIZE, dim=1)
    tensors = {
        "weight_packed": packed,
        "weight_scale": scale,
        "weight_shape": torch.tensor([rows, cols], dtype=torch.float32, device=DEVICE),
    }
    return tensors, dequant


def _random_checkpoint(seed):
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    shards, w13, w2 = {}, [], []
    for expert_id in range(E):
        dense = {}
        for shard_id, (rows, cols) in (
            ("w1", (N_INTER, H)),
            ("w3", (N_INTER, H)),
            ("w2", (H, N_INTER)),
        ):
            shards[expert_id, shard_id], dense[shard_id] = _random_shard(
                rows, cols, gen
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


class TestWNA16MoEGelu(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        init_single_process_dist(master_port=29634)
        torch.set_default_device("cuda")

    def _run(self, backend, activation):
        from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
        from sglang.srt.layers.moe.topk import TopKConfig, select_experts

        shards, w13, w2 = _random_checkpoint(0)
        with (
            get_context().override_server_args(model_path="dummy"),
            get_flags().moe.override(runner_backend=backend),
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
                quant_config=CompressedTensorsConfig.from_config(QUANT_CONFIG),
                prefix="model.layers.0.mlp.experts",
                activation=activation,
            ).cuda()
            self.assertIs(type(model.experts.scheme), SCHEMES[backend])
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
            # The Triton runner writes its output into the input buffer.
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

    def test_activations_on_each_runner(self):
        for backend in SCHEMES:
            for activation in ACTIVATIONS:
                with self.subTest(backend=backend.value, activation=activation):
                    self._run(backend, activation)


class TestRocmWNA16MoESchemeSelection(unittest.TestCase):
    """The ROCm Triton runner passes no zero points, so it must refuse asymmetric checkpoints."""

    def _scheme(self, symmetric):
        cfg = copy.deepcopy(QUANT_CONFIG)
        cfg["config_groups"]["group_0"]["weights"]["symmetric"] = symmetric
        config = CompressedTensorsConfig.from_config(cfg)
        with mock.patch.object(ct_module, "_is_hip", True):
            return config.get_moe_scheme(
                torch.nn.Module(), "model.layers.0.mlp.experts"
            )

    def test_rejects_asymmetric_checkpoint(self):
        with self.assertRaisesRegex(ValueError, "zero points"):
            self._scheme(symmetric=False)

    def test_keeps_symmetric_checkpoint_on_triton(self):
        self.assertIs(
            type(self._scheme(symmetric=True)), CompressedTensorsWNA16TritonMoE
        )


if __name__ == "__main__":
    unittest.main()
