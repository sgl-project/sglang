import unittest
from types import SimpleNamespace
from unittest.mock import patch

from torch import nn

from sglang.srt.configs.model_config import is_deepseek_dsa
from sglang.srt.models.mistral_large_3_eagle import MistralLarge3EagleModel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _DummyModule(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()


class _FakePPGroup:
    world_size = 1
    is_first_rank = True
    is_last_rank = True


def _make_eagle_config_with_index_topk():
    return SimpleNamespace(
        architectures=["MistralLarge3ForCausalLMEagle"],
        index_topk=1,
        vocab_size=32000,
        hidden_size=16,
        num_hidden_layers=3,
        rms_norm_eps=1e-6,
    )


class TestMistralLarge3EagleModel(CustomTestCase):
    def test_initializes_non_dsa_state_for_inherited_forward_path(self):
        with (
            patch(
                "sglang.srt.models.mistral_large_3_eagle.get_pp_group",
                return_value=_FakePPGroup(),
            ),
            patch(
                "sglang.srt.models.mistral_large_3_eagle.is_dsa_enable_prefill_cp",
                return_value=False,
            ),
            patch(
                "sglang.srt.models.mistral_large_3_eagle."
                "is_prefill_context_parallel_enabled",
                return_value=False,
            ),
            patch(
                "sglang.srt.models.mistral_large_3_eagle.VocabParallelEmbedding",
                _DummyModule,
            ),
            patch(
                "sglang.srt.models.mistral_large_3_eagle.RowParallelLinear",
                _DummyModule,
            ),
            patch(
                "sglang.srt.models.mistral_large_3_eagle.DeepseekV2DecoderLayer",
                _DummyModule,
            ),
            patch("sglang.srt.models.mistral_large_3_eagle.RMSNorm", _DummyModule),
            patch(
                "sglang.srt.models.deepseek_v2.get_attn_backend",
                return_value=SimpleNamespace(use_mha=False),
            ),
        ):
            config = _make_eagle_config_with_index_topk()
            model = MistralLarge3EagleModel(config)

            self.assertFalse(is_deepseek_dsa(config))
            self.assertFalse(model.use_dsa)
            self.assertFalse(model._dsa_forward_uses_topk())
            self.assertEqual(model.next_full_attention_layer_id, {0: 1, 1: 2})


if __name__ == "__main__":
    unittest.main()
