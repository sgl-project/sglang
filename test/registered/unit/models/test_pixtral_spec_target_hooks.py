import unittest

from torch import nn

from sglang.srt.model_executor.model_runner_components.attention_backend_setup import (
    configure_aux_hidden_state_capture,
)
from sglang.srt.models.pixtral import PixtralForConditionalGeneration
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _LanguageModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(8, 4)
        self.lm_head = nn.Linear(4, 8, bias=False)
        self.captured_layer_ids = None

    def get_input_embeddings(self) -> nn.Module:
        return self.embed_tokens

    def set_dflash_layers_to_capture(self, layer_ids: list[int]) -> None:
        self.captured_layer_ids = layer_ids


class TestPixtralSpecTargetHooks(CustomTestCase):
    def test_dspark_target_hooks_reach_language_model(self):
        wrapper = PixtralForConditionalGeneration.__new__(
            PixtralForConditionalGeneration
        )
        nn.Module.__init__(wrapper)
        wrapper.language_model = _LanguageModel()

        configure_aux_hidden_state_capture(
            model=wrapper,
            eagle_use_aux_hidden_state=False,
            eagle_aux_hidden_state_layer_ids=None,
            dflash_use_aux_hidden_state=True,
            dflash_target_layer_ids=[1, 36, 70],
            is_dspark=True,
        )

        self.assertEqual(wrapper.language_model.captured_layer_ids, [1, 36, 70])
        self.assertIs(getattr(wrapper, "lm_head", None), wrapper.language_model.lm_head)
        self.assertIs(
            wrapper.get_input_embeddings(), wrapper.language_model.embed_tokens
        )


if __name__ == "__main__":
    unittest.main()
