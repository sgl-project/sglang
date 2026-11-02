# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Which config the EAGLE aux-hidden-state resolver sizes the draft KV from,
and the stage count it records for a DSpark draft.

python -m pytest test/registered/unit/model_executor/test_spec_aux_hidden_state_resolver.py -v
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import sglang.srt.model_executor.model_runner_components.spec_aux_hidden_state as _mod
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, stage="weekly", runner_config="cpu")


def _config(*, num_nextn_predict_layers, num_hidden_layers=1):
    return SimpleNamespace(
        num_nextn_predict_layers=num_nextn_predict_layers,
        num_hidden_layers=num_hidden_layers,
        num_attention_layers=num_hidden_layers,
        is_hybrid_swa=False,
        is_deepseek_v4_arch=False,
    )


class TestEagleResolverDraftConfig(unittest.TestCase):
    def _resolve_with(self, *, draft_path, target, draft):
        """Run the EAGLE resolver with the config factory stubbed: `draft`
        answers the draft-mode read, `target` is the runner's own config."""
        config = _mod.SpecAuxHiddenStateConfig()
        spec = SimpleNamespace(
            speculative_draft_model_path=draft_path,
            speculative_draft_model_revision=None,
        )
        with (
            patch.object(_mod, "get_spec", return_value=spec),
            patch.object(
                _mod.ModelConfig, "from_server_args", return_value=draft
            ) as factory,
        ):
            _mod._resolve_eagle_aux_hidden_state(
                config=config,
                server_args=None,
                model_config=target,
                spec_algorithm=SimpleNamespace(
                    is_eagle=lambda: True,
                    is_standalone=lambda: False,
                    is_eagle3=lambda: False,
                ),
                is_draft_worker=False,
            )
        self.assertTrue(factory.call_args.kwargs["is_draft_model"])
        return config

    def test_path_less_nextn_fuses_without_resizing_the_private_pool(self):
        """A path-less MTP head sizes its fused draft KV from the draft-mode
        config (the only one that names its NEXTN layers), while the private
        draft pool keeps reading the target config, so its token budget is
        unchanged."""
        target = _config(num_nextn_predict_layers=None)
        draft = _config(num_nextn_predict_layers=1)
        config = self._resolve_with(draft_path=None, target=target, draft=draft)
        self.assertIs(config.draft_model_config, draft)
        self.assertEqual(config.draft_kv_num_layers, 1)
        self.assertIsNone(config.eagle_draft_num_layers)

    def test_draft_path_sizes_both_from_the_draft_checkpoint(self):
        target = _config(num_nextn_predict_layers=None)
        draft = _config(num_nextn_predict_layers=None, num_hidden_layers=2)
        config = self._resolve_with(draft_path="/draft", target=target, draft=draft)
        self.assertIs(config.draft_model_config, draft)
        self.assertEqual(config.draft_kv_num_layers, 2)
        self.assertEqual(config.eagle_draft_num_layers, 2)


class TestDSparkResolverStages(unittest.TestCase):
    def _resolve(self, architecture):
        """Run the DFLASH-family resolver for a DSpark draft that inherits a
        40-layer config but captures three target layers."""
        config = _mod.SpecAuxHiddenStateConfig()
        draft = SimpleNamespace(
            hf_config=SimpleNamespace(
                architectures=[architecture],
                num_hidden_layers=40,
                dspark_block_size=5,
                dspark_markov_rank=1,
                dspark_target_layer_ids=[1, 20, 39],
            )
        )
        target = SimpleNamespace(
            hf_config=SimpleNamespace(architectures=["DeepseekV4ForCausalLM"]),
            hf_text_config=SimpleNamespace(num_hidden_layers=43),
        )
        spec = SimpleNamespace(
            speculative_draft_model_path="/draft",
            speculative_draft_model_revision=None,
        )
        with (
            patch.object(_mod, "get_spec", return_value=spec),
            patch.object(_mod.ModelConfig, "from_server_args", return_value=draft),
        ):
            _mod._resolve_dflash_aux_hidden_state(
                config=config,
                server_args=None,
                model_config=target,
                spec_algorithm=SimpleNamespace(
                    is_dflash_family=lambda: True, is_dspark=lambda: True
                ),
                is_draft_worker=False,
            )
        self.assertEqual(config.dflash_draft_num_layers, 40)
        self.assertEqual(config.dflash_target_layer_ids, [1, 20, 39])
        return config

    def test_deepseek_v4_draft_counts_target_layer_stages(self):
        config = self._resolve("DeepseekV4ForCausalLMDSpark")
        self.assertEqual(config.dspark_num_stages, 3)

    def test_dense_draft_leaves_stages_unset(self):
        config = self._resolve("Qwen3DSparkModel")
        self.assertIsNone(config.dspark_num_stages)


if __name__ == "__main__":
    unittest.main()
