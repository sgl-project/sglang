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
    def test_accepts_main_stack_plus_one_entry_per_nextn_layer(self):
        for num_nextn in (0, 1, 2):
            with self.subTest(num_nextn=num_nextn):
                config = _config(
                    compress_ratios=_MAIN + [-2] * num_nextn,
                    num_nextn_predict_layers=num_nextn,
                )
                self.assertEqual(config.mla_layer_ids, (2,))

    def test_rejects_length_not_matching_nextn(self):
        with self.assertRaisesRegex(ValueError, "num_hidden_layers \\+ num_nextn"):
            _config(compress_ratios=_MAIN + [-2, -2], num_nextn_predict_layers=1)

    def test_rejects_unknown_ratio(self):
        with self.assertRaisesRegex(ValueError, "supports compress_ratios"):
            _config(compress_ratios=[128, 4, -2, 128])


class TestErnie5_5_MoeConfigRequiredFields(CustomTestCase):
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
    def test_nextn_draft_is_the_checkpoint_mtp_layer_as_one_mla_layer(self):
        config = _config(compress_ratios=_MAIN + [-2], num_nextn_predict_layers=1)
        to_nextn_draft_config(config)
        self.assertEqual(config.architectures, [ERNIE5_5_NEXTN_ARCH])
        self.assertEqual(config.mla_layer_ids, (0,))
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
        self.assertEqual(config.compress_ratios, [0, 0, 0])
        self.assertEqual(config.mla_layer_ids, ())
        self.assertEqual(config.first_k_dense_replace, 0)

    def test_dspark_draft_needs_bundled_dspark_keys(self):
        with self.assertRaisesRegex(ValueError, "bundles no DSpark draft"):
            to_dspark_draft_config(_config(compress_ratios=_MAIN))


class TestErnie5_5ServerArgOverrides(CustomTestCase):
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
            ("dcp_size", 2, "decode context parallelism"),
        ):
            with self.subTest(field=field):
                args = SimpleNamespace(**{**self._SUPPORTED, field: value})
                with self.assertRaisesRegex(ValueError, message):
                    _ernie55_overrides(args, None)

    def test_pd_rejects_pipeline_parallelism(self):
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
        for algo, steps, draft in (
            ("EAGLE", None, None),
            ("EAGLE", 1, 2),
            ("EAGLE", 3, 4),
            ("EAGLE", 3, 6),
            ("NEXTN", None, None),
            ("eagle", None, None),
            ("dspark", None, None),
        ):
            with self.subTest(algo=algo, steps=steps, draft=draft):
                args = {
                    **self._SUPPORTED,
                    "speculative_algorithm": algo,
                    "speculative_num_steps": steps,
                    "speculative_eagle_topk": None if steps is None else 1,
                    "speculative_num_draft_tokens": draft,
                }
                overrides = _ernie55_overrides(SimpleNamespace(**args), None)
                self.assertEqual(overrides["attention_backend"], "ernie55")
                self.assertEqual(overrides["page_size"], 256)
        for algo in ("EAGLE", "NEXTN"):
            with self.subTest(algo=algo, tree=True):
                tree = {
                    **self._SUPPORTED,
                    "speculative_algorithm": algo,
                    "speculative_num_steps": 3,
                    "speculative_eagle_topk": 4,
                    "speculative_num_draft_tokens": 4,
                }
                with self.assertRaisesRegex(ValueError, "eagle-topk 1 only"):
                    _ernie55_overrides(SimpleNamespace(**tree), None)


if __name__ == "__main__":
    unittest.main()
