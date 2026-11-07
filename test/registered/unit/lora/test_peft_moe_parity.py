"""Compare a real saved PEFT ParamWrapper adapter with SGLang on CPU.

The same fixture exercises historical/transposed factors with PEFT 0.18.0 and
native factors with PEFT 0.21.2. Both versions adapt the same [experts, out, in]
base parameters; the checkpoint is always produced by save_pretrained(), never
by manually transposing factors. No pretrained model or download is required.
"""

import copy
import tempfile
import unittest
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from peft.tuners.lora.layer import ParamWrapper
from safetensors.torch import load_file
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.lora.lora import LoRAAdapter
from sglang.srt.lora.lora_config import LoRAConfig
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_EXPERTS, _RANK, _HIDDEN, _INTERMEDIATE = 3, 2, 5, 7
_PREFIX = "base_model.model.model.layers.0.mlp.experts"
_TARGETS = ["mlp.experts.gate_up_proj", "mlp.experts.down_proj"]


class _Experts(nn.Module):
    def __init__(self, down_first):
        super().__init__()
        parameters = [
            ("gate_up_proj", (_EXPERTS, 2 * _INTERMEDIATE, _HIDDEN)),
            ("down_proj", (_EXPERTS, _HIDDEN, _INTERMEDIATE)),
        ]
        if down_first:
            parameters.reverse()
        for name, shape in parameters:
            value = torch.arange(torch.Size(shape).numel(), dtype=torch.float64)
            self.register_parameter(name, nn.Parameter(value.reshape(shape) / 400))

    def forward(self, x, expert_ids):
        gate_up = torch.einsum("th,toh->to", x, self.gate_up_proj[expert_ids])
        gate, up = gate_up.chunk(2, dim=-1)
        return torch.einsum(
            "ti,thi->th", nn.functional.silu(gate) * up, self.down_proj[expert_ids]
        )


class _SyntheticMoE(nn.Module):
    def __init__(self, down_first):
        super().__init__()
        self.config = PretrainedConfig(
            num_hidden_layers=1,
            num_experts=_EXPERTS,
            hidden_size=_HIDDEN,
            moe_intermediate_size=_INTERMEDIATE,
        )
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Module()])
        self.model.layers[0].mlp = nn.Module()
        self.model.layers[0].mlp.experts = _Experts(down_first)

    def forward(self, x, expert_ids):
        return self.model.layers[0].mlp.experts(x, expert_ids)


class TestPEFTMoEParity(CustomTestCase):
    @torch.no_grad()
    def test_saved_adapter_matches_peft_delta_and_forward(self):
        for down_first in (False, True):
            with self.subTest(down_first=down_first):
                self._check_saved_adapter(down_first)

    def _check_saved_adapter(self, down_first):
        base = _SyntheticMoE(down_first)
        restored = copy.deepcopy(base)
        with torch.random.fork_rng(devices=[]):
            peft_model = get_peft_model(
                base,
                LoraConfig(
                    r=_RANK,
                    lora_alpha=3,
                    target_modules=[],
                    target_parameters=_TARGETS,
                ),
            ).eval()

        # Nonzero, nonconstant factors expose expert/rank permutations and
        # double scaling. PEFT's default zero B would make parity vacuous.
        for index, (name, parameter) in enumerate(peft_model.named_parameters()):
            if "lora_" in name:
                values = torch.arange(parameter.numel(), dtype=parameter.dtype)
                parameter.copy_(
                    values.reshape(parameter.shape) / 100 + (index + 1) / 20
                )

        expected_deltas = {
            module.parameter_name: module.get_delta_weight("default").clone()
            for module in peft_model.modules()
            if isinstance(module, ParamWrapper)
        }
        self.assertEqual(set(expected_deltas), {"gate_up_proj", "down_proj"})
        x = torch.arange(5 * _HIDDEN, dtype=torch.float64).reshape(5, _HIDDEN) / 100
        expert_ids = torch.tensor([2, 0, 1, 2, 1])
        base_output = restored(x, expert_ids)
        expected_output = peft_model(x, expert_ids)
        self.assertFalse(torch.equal(base_output, expected_output))

        with tempfile.TemporaryDirectory() as directory:
            peft_model.save_pretrained(directory, save_embedding_layers=False)
            config = LoRAConfig(path=directory)
            self.assertEqual(config.hf_config["target_modules"], [])
            self.assertCountEqual(config.hf_config["target_parameters"], _TARGETS)
            self.assertCountEqual(config.target_modules, ["gate_up_proj", "down_proj"])
            saved = load_file(str(Path(directory) / "adapter_model.safetensors"))
            self.assertEqual(len(saved), 4)
            self.assertTrue(all(tensor.ndim == 2 for tensor in saved.values()))
            adapter = LoRAAdapter("saved-peft-moe", config, restored.config, None, None)
            adapter.initialize_weights_from_tensors(saved)

        weights = adapter.layers[0].weights
        self.assertEqual(len(weights), 4)
        for projection, expected_delta in expected_deltas.items():
            a = weights[f"{_PREFIX}.{projection}.lora_A.weight"]
            b = weights[f"{_PREFIX}.{projection}.lora_B.weight"]
            if projection == "gate_up_proj":
                # The complete pipeline must stack A exactly once, producing
                # one rank block for gate and one for up, with B unstacked.
                self.assertEqual(tuple(a.shape), (_EXPERTS, 2 * _RANK, _HIDDEN))
                self.assertEqual(tuple(b.shape), (_EXPERTS, 2 * _INTERMEDIATE, _RANK))
                actual_delta = torch.cat(
                    (
                        torch.bmm(b[:, :_INTERMEDIATE], a[:, :_RANK]),
                        torch.bmm(b[:, _INTERMEDIATE:], a[:, _RANK:]),
                    ),
                    dim=1,
                )
            else:
                self.assertEqual(tuple(a.shape), (_EXPERTS, _RANK, _INTERMEDIATE))
                self.assertEqual(tuple(b.shape), (_EXPERTS, _HIDDEN, _RANK))
                actual_delta = torch.bmm(b, a)
            actual_delta *= adapter.scaling
            torch.testing.assert_close(
                actual_delta, expected_delta, rtol=1e-12, atol=1e-12
            )
            getattr(restored.model.layers[0].mlp.experts, projection).add_(actual_delta)

        torch.testing.assert_close(
            restored(x, expert_ids), expected_output, rtol=1e-12, atol=1e-12
        )


if __name__ == "__main__":
    unittest.main()
