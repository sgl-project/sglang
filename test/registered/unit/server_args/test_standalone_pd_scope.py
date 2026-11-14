"""Configuration admission for the initial request-local STANDALONE PD handoff.

Call the real public PD/speculative resolution hooks with real ServerArgs and a
local config-only checkpoint. No worker, model weights, network, or GPU starts.
The unsafe default overlap and rank-local abort path must not be admitted just
because the broader monolithic STANDALONE worker supports those configurations.
"""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import msgspec
from transformers import AutoConfig

from sglang.srt.arg_groups.overrides import resolution_result
from sglang.srt.arg_groups.pd_disaggregation_hook import handle_pd_disaggregation
from sglang.srt.arg_groups.speculative_hook import handle_speculative_decoding
from sglang.srt.environ import envs
from sglang.srt.runtime_context import override_platform
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestStandalonePdScope(CustomTestCase):
    def setUp(self):
        self.enterContext(override_platform(is_cuda=True, is_hip=False))
        self.enterContext(envs.SGLANG_DISAGG_STAGING_BUFFER.override(False))
        self.enterContext(
            envs.SGLANG_ENABLE_DP_SPEC_PREFILL_COORDINATION.override(False)
        )
        directory = self.enterContext(tempfile.TemporaryDirectory())
        self.model_path = directory
        (Path(directory) / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "qwen3",
                    "architectures": ["Qwen3ForCausalLM"],
                    "hidden_size": 64,
                    "intermediate_size": 128,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 4,
                    "num_key_value_heads": 2,
                    "vocab_size": 128,
                }
            )
        )
        self.hf_config = AutoConfig.from_pretrained(directory, local_files_only=True)

    def args(self, role="decode", algorithm="STANDALONE", **overrides):
        # The dummy constructor avoids the full hardware/model pipeline. The
        # production handlers below still resolve the actual algorithm and its
        # auto top-k defaults, including loading the real local draft config.
        args = ServerArgs(model_path="dummy", device="cuda")
        fields = {
            "model_path": self.model_path,
            "speculative_draft_model_path": self.model_path,
            "speculative_algorithm": algorithm,
            "disaggregation_mode": role,
            "disable_overlap_schedule": True,
            "attention_backend": "fa3",
            # The normal pipeline resolves this before the speculative hook;
            # the dummy constructor deliberately skips that earlier stage.
            "page_size": 1,
        }
        fields.update(overrides)
        for name, value in fields.items():
            msgspec.structs.force_setattr(args, name, value)
        args._model_config = SimpleNamespace(hf_config=self.hf_config)
        return args

    def resolve(self, args):
        handle_pd_disaggregation(args)
        handle_speculative_decoding(args)

    def test_auto_defaults_admit_both_synchronous_roles(self):
        for role in ("prefill", "decode"):
            with self.subTest(role=role):
                args = self.args(role)
                self.resolve(args)
                self.assertEqual(resolution_result(args, "speculative_eagle_topk"), 1)
                self.assertEqual(resolution_result(args, "speculative_num_steps"), 3)
                self.assertEqual(
                    resolution_result(args, "speculative_num_draft_tokens"), 4
                )
                # Omitted backup stays omitted: do not reject the default by
                # interpreting every decode node as explicit host offload.
                self.assertIsNone(
                    resolution_result(args, "disaggregation_decode_retraction_backup")
                )

    def test_default_overlap_is_rejected_on_both_roles(self):
        for role in ("prefill", "decode"):
            with self.subTest(role=role):
                args = self.args(
                    role,
                    disable_overlap_schedule=ServerArgs(
                        model_path="dummy"
                    ).disable_overlap_schedule,
                )
                with self.assertRaisesRegex(ValueError, "disable-overlap-schedule"):
                    self.resolve(args)

    def test_rank_local_abort_cannot_enter_parallel_serving(self):
        for role in ("prefill", "decode"):
            for field in ("tp_size", "pp_size", "dp_size", "attn_cp_size", "dcp_size"):
                with self.subTest(role=role, field=field):
                    with self.assertRaisesRegex(ValueError, field.replace("_", "-")):
                        self.resolve(self.args(role, **{field: 2}))
            with self.subTest(role=role, prefill_cp=True):
                with self.assertRaisesRegex(ValueError, "enable-prefill-cp"):
                    self.resolve(self.args(role, enable_prefill_cp=True))

    def test_explicit_tree_seed_shape_is_outside_initial_scope(self):
        with self.assertRaisesRegex(ValueError, "speculative-eagle-topk"):
            self.resolve(
                self.args(
                    speculative_num_steps=3,
                    speculative_eagle_topk=4,
                    speculative_num_draft_tokens=8,
                )
            )

    def test_new_prefill_complete_policy_is_not_implicitly_combined(self):
        for role in ("prefill", "decode"):
            with self.subTest(role=role):
                with self.assertRaisesRegex(
                    ValueError, "allocation-policy must be early"
                ):
                    self.resolve(
                        self.args(
                            role,
                            disaggregation_decode_allocation_policy="prefill_complete",
                        )
                    )

    def test_explicit_host_cache_and_staging_paths_are_rejected(self):
        for field in (
            "enable_hierarchical_cache",
            "enable_unified_cache_external_linker",
            "enable_lmcache",
            "enable_flexkv",
            "enable_hisparse",
            "disaggregation_decode_enable_offload_kvcache",
        ):
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, "STANDALONE PD"):
                    self.resolve(self.args(**{field: True}))
        for overrides in (
            {"hicache_storage_backend": "file"},
            {"radix_cache_backend": "flexkv"},
            {"disaggregation_decode_host_receive_threshold": 0.5},
            {"disaggregation_decode_retraction_backup": "host_pool"},
            {"disaggregation_decode_retraction_backup": "cpu_tensor"},
        ):
            with self.subTest(overrides=overrides):
                with self.assertRaisesRegex(ValueError, "STANDALONE PD"):
                    self.resolve(self.args(**overrides))
        with envs.SGLANG_DISAGG_STAGING_BUFFER.override(True):
            with self.assertRaisesRegex(ValueError, "SGLANG_DISAGG_STAGING_BUFFER"):
                self.resolve(self.args())

    def test_disabled_host_features_and_explicit_no_backup_are_allowed(self):
        args = self.args(disaggregation_decode_retraction_backup="none")
        self.resolve(args)
        self.assertEqual(
            resolution_result(args, "disaggregation_decode_retraction_backup"), "none"
        )

    def test_existing_attention_dp_and_decode_radix_errors_are_preserved(self):
        for overrides, message in (
            ({"attn_dp_size": 2}, "does not support dp attention"),
            (
                {"disaggregation_decode_enable_radix_cache": True},
                "incompatible with speculative decoding",
            ),
        ):
            with self.subTest(overrides=overrides):
                with self.assertRaisesRegex(ValueError, message):
                    self.resolve(self.args(**overrides))

    def test_other_serving_modes_do_not_inherit_the_pd_restriction(self):
        for algorithm, role in (
            ("STANDALONE", "null"),
            (None, "prefill"),
            (None, "decode"),
            ("EAGLE", "decode"),
            ("EAGLE3", "prefill"),
        ):
            with self.subTest(algorithm=algorithm, role=role):
                args = self.args(
                    role,
                    algorithm,
                    tp_size=2,
                    disable_overlap_schedule=False,
                    enable_hierarchical_cache=True,
                    speculative_num_steps=3,
                    speculative_eagle_topk=4,
                    speculative_num_draft_tokens=8,
                )
                self.resolve(args)
                self.assertFalse(resolution_result(args, "disable_overlap_schedule"))


if __name__ == "__main__":
    unittest.main()
