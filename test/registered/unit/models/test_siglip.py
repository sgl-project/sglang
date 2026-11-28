import unittest

import torch
from transformers import SiglipVisionConfig
from transformers import SiglipVisionModel as HFSiglipVisionModel

from sglang.srt.layers.activation import QuickGELU
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models import siglip
from sglang.srt.runtime_context import reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.parallel_groups import publish
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

VISION = dict(
    hidden_size=32,
    intermediate_size=64,
    num_hidden_layers=2,
    num_attention_heads=4,
    image_size=28,
    patch_size=14,
    vision_use_head=False,
)
QKV_SHARDS = {".q_proj.": "q", ".k_proj.": "k", ".v_proj.": "v"}


def _load_hf_weights(tower, hf_state):
    params = dict(tower.named_parameters())
    for name, weight in hf_state.items():
        name = f"vision_model.{name}".replace(
            ".self_attn.out_proj.", ".self_attn.proj."
        )
        shard = next((s for key, s in QKV_SHARDS.items() if key in name), None)
        if shard is None:
            param = params[name]
            getattr(param, "weight_loader", default_weight_loader)(param, weight)
            continue
        for key in QKV_SHARDS:
            name = name.replace(key, ".qkv_proj.")
        param = params[name]
        param.weight_loader(param, weight, shard)


class TestSiglipVisionTowerAgainstHF(CustomTestCase):
    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)
        publish(
            ServerArgs(
                model_path="dummy", device="cpu", tp_size=1, mm_attention_backend="sdpa"
            ),
            role="test",
        )
        torch.manual_seed(0)
        self.pixels = torch.randn(1, 3, VISION["image_size"], VISION["image_size"])

    def _outputs(self, config):
        hf = HFSiglipVisionModel(config).eval()
        tower = siglip.SiglipVisionModel(config).eval()
        _load_hf_weights(tower, hf.state_dict())
        with torch.no_grad():
            return tower(self.pixels), hf(pixel_values=self.pixels).last_hidden_state

    def test_tower_uses_the_checkpoint_hidden_act(self):
        for hidden_act in ("gelu_pytorch_tanh", "quick_gelu", "gelu"):
            with self.subTest(hidden_act=hidden_act):
                config = SiglipVisionConfig(**VISION, hidden_act=hidden_act)
                ours, ref = self._outputs(config)
                torch.testing.assert_close(ours, ref)

    def test_explicit_act_layer_overrides_the_config(self):
        config = SiglipVisionConfig(**VISION, hidden_act="gelu_pytorch_tanh")
        tower = siglip.SiglipVisionModel(config, act_layer=QuickGELU)
        acts = {type(layer.mlp.act) for layer in tower.vision_model.encoder.layers}
        self.assertEqual(acts, {QuickGELU})


if __name__ == "__main__":
    unittest.main()
