"""CLIP's vision tower and embedding head against the Hugging Face reference.

HF applies post_layernorm only to the pooled CLS token: VLM adaptors read the
tower's per-token states un-normed, while CLIPModel norms just the CLS vector.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from transformers import CLIPConfig
from transformers import CLIPModel as HFCLIPModel
from transformers import CLIPVisionConfig
from transformers import CLIPVisionModel as HFCLIPVisionModel
from transformers.models.auto.modeling_auto import AutoModel

from sglang.srt.model_loader.loader import DefaultModelLoader
from sglang.srt.models import clip
from sglang.srt.runtime_context import reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.parallel_groups import publish
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

VISION = dict(
    hidden_size=32,
    intermediate_size=64,
    num_hidden_layers=2,
    num_attention_heads=4,
    image_size=28,
    patch_size=14,
)
TEXT = dict(
    hidden_size=32,
    intermediate_size=64,
    num_hidden_layers=2,
    num_attention_heads=4,
    vocab_size=99,
    max_position_embeddings=16,
)


def _vision_names(state_dict):
    # transformers 5.x dropped the `vision_model.` wrapper on CLIPVisionModel.
    return [
        (name if name.startswith("vision_model.") else f"vision_model.{name}", w)
        for name, w in state_dict.items()
    ]


class TestClipAgainstHF(CustomTestCase):
    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)
        publish(ServerArgs(model_path="dummy", device="cpu", tp_size=1), role="test")
        torch.manual_seed(0)
        self.pixels = torch.randn(2, 3, VISION["image_size"], VISION["image_size"])

    def test_tower_returns_the_raw_encoder_states_llava_reads(self):
        """The tower's plain forward must equal HF's un-normed last_hidden_state;
        LLaVA/Wan I2V reads it as the final layer, so post-norming it corrupted every token."""
        config = CLIPVisionConfig(**VISION)
        hf = HFCLIPVisionModel(config).eval()
        tower = clip.CLIPVisionModel(config).eval()
        tower.load_weights(_vision_names(hf.state_dict()))

        with torch.no_grad():
            ref = hf(pixel_values=self.pixels, output_hidden_states=True)
            last = tower(self.pixels)
            every_layer = tower(self.pixels, output_hidden_states=True)

        torch.testing.assert_close(last, ref.last_hidden_state)
        torch.testing.assert_close(every_layer.last_hidden_state, last)
        # llava-1.5 reads vision_feature_layer=-2.
        torch.testing.assert_close(every_layer.hidden_states[-2], ref.hidden_states[-2])

    def test_image_embedding_norms_only_the_pooled_cls_token(self):
        config = CLIPConfig(text_config=TEXT, vision_config=VISION, projection_dim=16)
        hf = HFCLIPModel(config).eval()
        # CLIPModel.__init__ patches the weight loader for open_clip checkpoints.
        with patch.object(
            DefaultModelLoader, "_prepare_weights", DefaultModelLoader._prepare_weights
        ):
            model = clip.CLIPModel(config).eval()
        model.load_weights(hf.state_dict().items())
        forward_batch = SimpleNamespace(
            mm_inputs=[SimpleNamespace(mm_items=[SimpleNamespace(feature=self.pixels)])]
        )

        with torch.no_grad():
            embeddings = model(None, None, forward_batch).embeddings
            pooled = hf.vision_model(pixel_values=self.pixels).pooler_output
            ref = torch.nn.functional.normalize(hf.visual_projection(pooled), dim=-1)

        torch.testing.assert_close(embeddings, ref)

    def test_tower_is_registered_under_hf_architecture_name(self):
        """LLaVA resolves its tower by the HF architecture name for its config; a
        missing entry silently falls back to the text-only Transformers backend."""
        arch = AutoModel._model_mapping[CLIPVisionConfig].__name__
        entries = clip.EntryClass
        entries = entries if isinstance(entries, list) else [entries]
        registered = {cls.__name__: cls for cls in entries}
        self.assertIs(registered.get(arch), clip.CLIPVisionModel)


if __name__ == "__main__":
    unittest.main()
