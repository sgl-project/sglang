import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.engram import EngramEmbedding, engram_gate
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, stage="base-a", runner_config="cpu")


class TestModelSlimEngram(CustomTestCase):
    def test_lookup_signed_shard_with_e8m0_scale(self):
        embedding = EngramEmbedding.__new__(EngramEmbedding)
        torch.nn.Module.__init__(embedding)
        embedding.row_start, embedding.rows, embedding.dim = 3, 2, 64
        embedding.int8_sym = True
        embedding.host_table = None
        embedding.weight = torch.nn.Parameter(
            torch.empty(2, 64, dtype=torch.int8), requires_grad=False
        )
        embedding.scale = torch.nn.Parameter(
            torch.empty(2, 2, dtype=torch.uint8), requires_grad=False
        )
        weight = torch.arange(5 * 64, dtype=torch.int16).reshape(5, 64).to(torch.int8)
        exponent = torch.tensor(
            [[127, 127], [127, 127], [127, 127], [125, 129], [128, 126]],
            dtype=torch.uint8,
        )
        embedding._load_rows(embedding.weight, weight)
        embedding._load_rows(embedding.scale, exponent.view(torch.float8_e8m0fnu))
        out = embedding._owned_rows(torch.tensor([4, 2, 3, 6]))
        scale = 2.0 ** (exponent[3:5].float() - 127)
        ref = (
            (weight[3:5].float().view(2, 2, 32) * scale.unsqueeze(-1))
            .flatten(1)
            .bfloat16()
        )
        torch.testing.assert_close(out[[0, 2]], ref[[1, 0]], rtol=0, atol=0)
        self.assertEqual(out[[1, 3]].count_nonzero().item(), 0)

    def test_rotation_sidecar_in_a_cached_checkpoint(self):
        from safetensors.torch import save_file

        from sglang.srt.models.deepseek_v4 import DeepseekV4ForCausalLM

        rotation = torch.tensor([[0.0, 1.0], [-1.0, 0.0]])
        gates = [SimpleNamespace(q_weight=torch.zeros(1, 2)) for _ in range(2)]
        rel = "optional/quarot.safetensors"
        description = {
            "optional": {"quarot": {"rotation_map": {"global_rotation": rel}}}
        }
        model = SimpleNamespace(
            config=SimpleNamespace(hidden_size=2),
            quant_config=SimpleNamespace(quant_description=description),
            model=SimpleNamespace(
                layers=[SimpleNamespace(engram=gate) for gate in gates]
            ),
        )
        declaration = {
            "value_projection_rotated": True,
            "value_basis": "quarot_global",
            "key_and_gate_basis": "original",
            "runtime_delta_rotation": False,
        }
        with tempfile.TemporaryDirectory() as temp:
            cache = Path(temp)
            snapshot = cache / "snapshot"
            (snapshot / "optional").mkdir(parents=True)
            (snapshot / "config.json").write_text(
                json.dumps({"engram_rotation_config": declaration})
            )
            save_file({"global_rotation": rotation}, str(cache / "blob"))
            (snapshot / rel).symlink_to(cache / "blob")
            with patch(
                "sglang.srt.runtime_context.get_model",
                return_value=SimpleNamespace(model_path=str(snapshot), revision=None),
            ):
                DeepseekV4ForCausalLM._load_engram_gate_basis(model)
        torch.testing.assert_close(gates[0].gate_basis, rotation)
        self.assertIs(gates[0].gate_basis, gates[1].gate_basis)

    def test_rotated_gate_preserves_the_residual_and_value_basis(self):
        h = torch.tensor([[[1.0, 2.0]]])
        kv = torch.tensor([[-2.0, 3.0, 4.0, -1.0]])
        norm = torch.ones(1, 2)
        rotation = torch.tensor([[0.0, 1.0], [-1.0, 0.0]])
        actual = engram_gate(h, kv, norm, norm, 1e-6, 1e-6, gate_basis=rotation)
        unrotated = h @ rotation.T
        update = engram_gate(unrotated, kv, norm, norm, 1e-6, 1e-6) - unrotated
        torch.testing.assert_close(actual, h + update)


if __name__ == "__main__":
    unittest.main()
