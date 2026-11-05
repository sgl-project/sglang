# Copyright 2026 Baidu ERNIE Team and The SGLang Team. All rights reserved.
#
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
import unittest
from types import SimpleNamespace

from sglang.srt.arg_groups.model_overrides.ernie55 import _ernie55_overrides
from sglang.srt.configs.ernie55 import (
    _REQUIRED_CHECKPOINT_FIELDS,
    _REQUIRED_CHECKPOINT_VALUES,
    ERNIE5_5_DSPARK_ARCH,
    ERNIE5_5_NEXTN_ARCH,
    Ernie5_5_MoeConfig,
    to_dspark_draft_config,
    to_nextn_draft_config,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_MAIN = [128, 128, -2, 128]
# Stand-ins for the checkpoint fields the model reads; only their presence is
# validated, except for the few that must carry one specific value.
_CHECKPOINT_FIELDS = {
    **{name: 1 for name in _REQUIRED_CHECKPOINT_FIELDS},
    **_REQUIRED_CHECKPOINT_VALUES,
}


def _config(**kwargs):
    fields = {**_CHECKPOINT_FIELDS, **kwargs}
    return Ernie5_5_MoeConfig(
        num_hidden_layers=len(_MAIN),
        **fields,
    )


class TestErnie5_5_MoeConfigCompressRatios(CustomTestCase):
    def test_accepts_main_stack_only(self):
        config = _config(compress_ratios=_MAIN)
        self.assertEqual(config.mla_layer_ids, (2,))

    def test_accepts_one_entry_per_nextn_layer(self):
        for num_nextn in (1, 2):
            with self.subTest(num_nextn=num_nextn):
                config = _config(
                    compress_ratios=_MAIN + [-2] * num_nextn,
                    num_nextn_predict_layers=num_nextn,
                )
                # The trailing NextN entries never count as main-stack MLA layers.
                self.assertEqual(config.mla_layer_ids, (2,))

    def test_rejects_length_not_matching_nextn(self):
        with self.assertRaisesRegex(ValueError, "num_hidden_layers \\+ num_nextn"):
            _config(compress_ratios=_MAIN + [-2, -2], num_nextn_predict_layers=1)

    def test_rejects_unknown_ratio(self):
        with self.assertRaisesRegex(ValueError, "supports compress_ratios"):
            _config(compress_ratios=[128, 4, -2, 128])


class TestErnie5_5_MoeConfigRequiredFields(CustomTestCase):
    def test_accepts_a_complete_checkpoint(self):
        _config(compress_ratios=_MAIN).validate_checkpoint_fields()

    def test_rejects_checkpoint_missing_a_required_field(self):
        dropped = "hybrid_mla_kv_lora_rank"
        fields = {k: v for k, v in _CHECKPOINT_FIELDS.items() if k != dropped}
        config = Ernie5_5_MoeConfig(
            num_hidden_layers=len(_MAIN),
            compress_ratios=_MAIN,
            **fields,
        )
        with self.assertRaisesRegex(ValueError, dropped):
            config.validate_checkpoint_fields()

    def test_names_every_missing_field(self):
        config = Ernie5_5_MoeConfig(num_hidden_layers=len(_MAIN), compress_ratios=_MAIN)
        with self.assertRaises(ValueError) as caught:
            config.validate_checkpoint_fields()
        message = str(caught.exception)
        for name in _REQUIRED_CHECKPOINT_FIELDS:
            self.assertIn(name, message)

    def test_rejects_a_field_the_model_does_not_implement(self):
        config = _config(compress_ratios=_MAIN, n_group=2, hidden_act="silu")
        with self.assertRaisesRegex(ValueError, "'hidden_act': 'silu'.*'n_group': 2"):
            config.validate_checkpoint_fields()


class TestErnie5_5DraftConfigs(CustomTestCase):
    """The draft configs the speculative runners build from the target's."""

    def test_nextn_draft_is_the_checkpoint_mtp_layer_as_one_mla_layer(self):
        config = _config(compress_ratios=_MAIN + [-2], num_nextn_predict_layers=1)
        to_nextn_draft_config(config)
        self.assertEqual(config.architectures, [ERNIE5_5_NEXTN_ARCH])
        # The KV pool gives the draft a latent sub-pool only for these layers.
        self.assertEqual(config.mla_layer_ids, (0,))
        # The draft loader reads the checkpoint layer stored after the main stack.
        self.assertEqual(config.mtp_checkpoint_layer_id, len(_MAIN))

    def test_nextn_draft_needs_an_mtp_layer(self):
        config = _config(compress_ratios=_MAIN, num_nextn_predict_layers=0)
        with self.assertRaisesRegex(ValueError, "no MTP layer"):
            to_nextn_draft_config(config)

    def test_dspark_draft_has_one_moe_stage_per_target_layer_and_no_mla(self):
        config = _config(
            compress_ratios=_MAIN,
            first_k_dense_replace=1,
            dspark_block_size=7,
            dspark_markov_rank=256,
            dspark_target_layer_ids=[1, 2, 3],
        )
        to_dspark_draft_config(config)
        self.assertEqual(config.architectures, [ERNIE5_5_DSPARK_ARCH])
        self.assertEqual(config.num_hidden_layers, 3)
        # Every stage is uncompressed: no C128 state, no latent sub-pool.
        self.assertEqual(config.compress_ratios, [0, 0, 0])
        self.assertEqual(config.mla_layer_ids, ())
        # The checkpoint's draft stages are all MoE.
        self.assertEqual(config.first_k_dense_replace, 0)

    def test_dspark_draft_needs_bundled_dspark_keys(self):
        with self.assertRaisesRegex(ValueError, "bundles no DSpark draft"):
            to_dspark_draft_config(_config(compress_ratios=_MAIN))


class TestErnie5_5ServerArgOverrides(CustomTestCase):
    """Server features the model cannot serve are rejected at startup by name."""

    _SUPPORTED = dict(
        kv_cache_dtype="fp8_e4m3",
        moe_a2a_backend="deepep",
        enable_eplb=False,
        ep_num_redundant_experts=0,
        init_expert_location="trivial",
        enable_unified_cache_external_linker=False,
        enable_hisparse=False,
        enable_encoder_swa_bounded_replay=False,
        enable_decoder_swa_bounded_replay=False,
        speculative_algorithm=None,
        speculative_num_steps=None,
        speculative_eagle_topk=None,
        speculative_num_draft_tokens=None,
        speculative_adaptive=False,
        swa_full_tokens_ratio=None,
        dsa_prefill_backend=None,
        dsa_decode_backend=None,
        enable_two_batch_overlap=False,
        enable_single_batch_overlap=False,
        dcp_size=1,
        disaggregation_mode="null",
        pp_size=1,
    )

    def test_supported_args_pass(self):
        overrides = _ernie55_overrides(SimpleNamespace(**self._SUPPORTED), None)
        self.assertEqual(overrides["page_size"], 256)

    def test_each_unsupported_feature_is_rejected(self):
        for field, value, message in (
            ("moe_a2a_backend", "megamoe", "megamoe"),
            ("enable_eplb", True, "EPLB"),
            ("ep_num_redundant_experts", 8, "EPLB"),
            ("enable_unified_cache_external_linker", True, "external-linker"),
            ("enable_hisparse", True, "hisparse"),
            ("speculative_algorithm", "NGRAM", "DSPARK and EAGLE"),
            ("enable_encoder_swa_bounded_replay", True, "encoder-swa-bounded-replay"),
            ("enable_decoder_swa_bounded_replay", True, "decoder-swa-bounded-replay"),
            ("dsa_prefill_backend", "flashmla_kv", "flashmla_sparse only"),
            ("dsa_decode_backend", "trtllm", "flashmla_sparse only"),
            ("enable_two_batch_overlap", True, "batch overlap"),
            ("enable_single_batch_overlap", True, "batch overlap"),
            ("dcp_size", 2, "decode context parallelism"),
        ):
            with self.subTest(field=field):
                args = SimpleNamespace(**{**self._SUPPORTED, field: value})
                with self.assertRaisesRegex(ValueError, message):
                    _ernie55_overrides(args, None)

    def test_pd_rejects_pipeline_parallelism(self):
        """PD with a split pipeline would corrupt the MLA layers' transferred KV;
        either feature alone still starts."""
        for mode, pp, rejected in (
            ("prefill", 2, True),
            ("null", 2, False),
            ("decode", 1, False),
        ):
            with self.subTest(mode=mode, pp=pp):
                args = SimpleNamespace(
                    **{**self._SUPPORTED, "disaggregation_mode": mode, "pp_size": pp}
                )
                if rejected:
                    with self.assertRaisesRegex(ValueError, "PD disaggregation"):
                        _ernie55_overrides(args, None)
                else:
                    _ernie55_overrides(args, None)

    def test_mtp_accepts_any_chain_but_no_tree(self):
        """ECHO runs as a chain of any length; a tree (topk > 1) is rejected at
        startup, and the EAGLE handler, not the override, fixes the draft-token count."""
        eagle = {**self._SUPPORTED, "speculative_algorithm": "EAGLE"}
        _ernie55_overrides(SimpleNamespace(**eagle), None)
        for steps in (1, 2, 3):
            chain = dict(
                speculative_num_steps=steps,
                speculative_eagle_topk=1,
                speculative_num_draft_tokens=steps + 1,
            )
            _ernie55_overrides(SimpleNamespace(**{**eagle, **chain}), None)
        mismatched = dict(
            speculative_num_steps=3,
            speculative_eagle_topk=1,
            speculative_num_draft_tokens=6,
        )
        _ernie55_overrides(SimpleNamespace(**{**eagle, **mismatched}), None)
        tree = dict(
            speculative_num_steps=3,
            speculative_eagle_topk=2,
            speculative_num_draft_tokens=4,
        )
        with self.assertRaisesRegex(ValueError, "eagle-topk 1 only"):
            _ernie55_overrides(SimpleNamespace(**{**eagle, **tree}), None)

    def test_raw_cli_spellings_are_accepted(self):
        """Overrides see the CLI value before NEXTN -> EAGLE and upper-casing, so
        NEXTN and lowercase eagle must pass and still get the MTP topk check."""
        for algo in ("NEXTN", "eagle", "dspark"):
            with self.subTest(algo=algo):
                args = {**self._SUPPORTED, "speculative_algorithm": algo}
                _ernie55_overrides(SimpleNamespace(**args), None)
        nextn_tree = {
            **self._SUPPORTED,
            "speculative_algorithm": "NEXTN",
            "speculative_num_steps": 3,
            "speculative_eagle_topk": 4,
            "speculative_num_draft_tokens": 4,
        }
        with self.assertRaisesRegex(ValueError, "eagle-topk 1 only"):
            _ernie55_overrides(SimpleNamespace(**nextn_tree), None)


if __name__ == "__main__":
    unittest.main()
