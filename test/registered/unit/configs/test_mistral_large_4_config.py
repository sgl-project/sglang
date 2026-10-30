import copy
import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.srt.arg_groups.model_hook import handle_model_specific_adjustments
from sglang.srt.arg_groups.model_override_base import (
    attention_backends_of,
    resolved_view,
)
from sglang.srt.arg_groups.model_overrides.deepseek_v2 import (
    _deepseek_family_overrides,
)
from sglang.srt.arg_groups.overrides import (
    _attention_backend_default,
    declare_resolution,
    run_post_process_pass,
)
from sglang.srt.configs.model_config import (
    AttentionArch,
    ModelConfig,
    dsa_layer_skips_topk,
    dsa_supports_dense_attention,
    get_dsa_index_topk,
    is_deepseek_dsa,
)
from sglang.srt.runtime_context import override_platform
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.dspark_components.dspark_config import (
    checkpoint_bundles_dspark_draft,
    parse_dspark_draft_config,
)
from sglang.srt.utils.hf_transformers import common
from sglang.srt.utils.hf_transformers.config import get_config
from sglang.srt.utils.hf_transformers.mistral_utils import (
    adapt_config_dict,
    load_mistral_config,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=12, suite="base-a-test-cpu")

_KV_SOURCES = list(range(72))
for _consumer in range(39, 72, 4):
    _KV_SOURCES[_consumer] = 35

_INDEX_SHARING = (
    ["full"] * 17
    + ["shared", "shared", "full", "full"] * 13
    + ["shared", "shared", "full"]
)

# params.json shipped with the checkpoint.
_PARAMS = {
    "dim": 6144,
    "n_layers": 72,
    "head_dim": 128,
    "hidden_dim": 24576,
    "n_heads": 64,
    "n_kv_heads": 64,
    "rope_theta": 10000.0,
    "norm_eps": 1e-05,
    "vocab_size": 204800,
    "tied_embeddings": False,
    "max_position_embeddings": 262144,
    "ragged_attention": None,
    "nope": "rope,rope,rope,nope",
    "softmax_tanh_gating": 30.0,
    "llama_4_scaling": None,
    "q_lora_rank": None,
    "qk_rope_head_dim": 64,
    "qk_nope_head_dim": 64,
    "kv_lora_rank": 512,
    "v_head_dim": 128,
    "ssmax_scaling": True,
    "lerope": True,
    "quantization": {
        "qscheme_act": "BLOCK",
        "qformat_weight": "fp8_e4m3",
        "qdynamic_act": True,
        "qscheme_weight": "BLOCK",
        "use_ue8m0_for_qscales": True,
        "block_size_out": 128,
        "block_size_in": 128,
        "quantize_kv_a": False,
    },
    "yarn": {
        "original_max_position_embeddings": 512,
        "factor": 512,
        "apply_scale": False,
        "beta": 32,
        "alpha": 1,
    },
    "kv_sharing": {"source_layers_str": ",".join(map(str, _KV_SOURCES))},
    "moe": {
        "expert_parallel": 1,
        "expert_model_parallel": 1,
        "route_every_n": 1,
        "first_k_dense_replace": 1,
        "num_experts": 128,
        "num_experts_per_tok": 4,
        "num_expert_groups": 1,
        "num_expert_groups_per_tok": 1,
        "routed_scale": 1.0,
        "expert_hidden_dim": 6144,
        "num_shared_experts": 1,
    },
    "vision_encoder": {
        "image_token_id": 10,
        "image_break_token_id": 12,
        "image_end_token_id": 13,
        "intermediate_size": 6144,
        "num_hidden_layers": 30,
        "num_attention_heads": 16,
        "mm_projector_id": "patch_merge",
        "spatial_merge_size": 2,
        "hidden_size": 2048,
        "num_channels": 3,
        "image_size": 1540,
        "max_image_size": 1540,
        "patch_size": 14,
        "rope_theta": 10000.0,
        "add_pre_mm_projector_layer_norm": True,
        "adapter_bias": False,
    },
    "dsa": {
        "index_topk": 2048,
        "index_n_heads": 32,
        "index_head_dim": 128,
        "index_rope_head_dim": 64,
        "indexer_match_nope": True,
        "indexer_ssmax_scaling": True,
        "indexer_norm_eps": 1e-06,
        "index_sharing": ",".join(_INDEX_SHARING),
        "fp8_indexer": True,
        "fp8_kv_cache": True,
    },
}


# params.json mtp block of the checkpoint revision that bundles the DSpark draft.
_MTP = {
    "n_layers": 3,
    "depth": 3,
    "ragged_attention": "2048,2048,2048",
    "target_activation_layers": "1,18,36,54,71",
    "norm_strategy": "eagle",
    "dspark": {
        "markov_rank": 512,
        "head_type": "rnn",
        "draft_input": "position_specific",
    },
}


class TestMistralLarge4Config(CustomTestCase):
    def test_params_only_checkpoint_resolves_full_config(self):
        with tempfile.TemporaryDirectory() as model_dir:
            with open(os.path.join(model_dir, "params.json"), "w") as f:
                json.dump(_PARAMS, f)
            config = get_config(model_dir, trust_remote_code=False)

        self.assertEqual(config.architectures, ["PixtralForConditionalGeneration"])
        self.assertEqual(config.quantization_config["weight_block_size"], [128, 128])
        self.assertEqual(config.quantization_config["activation_scheme"], "dynamic")
        self.assertIn(
            "model.layers.71.self_attn.kv_b_proj",
            config.quantization_config["ignored_layers"],
        )

        text = config.text_config
        self.assertEqual(text.model_type, "mistral_large_4")
        self.assertEqual(text.architectures, ["MistralLarge4ForCausalLM"])
        self.assertEqual(
            (text.q_lora_rank, text.kv_lora_rank, text.qk_nope_head_dim),
            (None, 512, 64),
        )
        self.assertEqual((text.qk_rope_head_dim, text.v_head_dim), (64, 128))
        self.assertEqual(
            (text.n_routed_experts, text.num_experts_per_tok, text.n_shared_experts),
            (128, 4, 1),
        )
        self.assertEqual(text.first_k_dense_replace, 1)
        self.assertEqual(text.max_seq_len, 262144)
        self.assertEqual(text.softmax_tanh_gating, 30.0)

        expected_nope = [layer % 4 == 3 for layer in range(72)]
        self.assertEqual(text.nope_layer_flags, expected_nope)
        self.assertEqual(text.indexer_nope_layer_flags, expected_nope)
        self.assertEqual(
            text.index_topk_pattern,
            ["S" if value == "shared" else "F" for value in _INDEX_SHARING],
        )
        self.assertEqual(
            (text.index_topk, text.index_n_heads, text.index_head_dim),
            (2048, 32, 128),
        )
        self.assertEqual(text.index_kv_dtype, "fp8")
        self.assertTrue(text.indexer_ssmax_scaling)
        self.assertEqual(
            [int(s) for s in text.kv_sharing_config["source_layers_str"].split(",")],
            _KV_SOURCES,
        )

        vision = config.vision_config
        self.assertEqual(
            (vision.hidden_size, vision.num_hidden_layers, vision.patch_size),
            (2048, 30, 14),
        )

    def test_wrapped_config_exposes_dsa_layout(self):
        # Server-arg hooks and the DSA backend receive the outer Pixtral config.
        with tempfile.TemporaryDirectory() as model_dir:
            with open(os.path.join(model_dir, "params.json"), "w") as f:
                json.dump(_PARAMS, f)
            config = get_config(model_dir, trust_remote_code=False)

        self.assertTrue(is_deepseek_dsa(config))
        self.assertFalse(dsa_supports_dense_attention(config))
        self.assertEqual(get_dsa_index_topk(config), 2048)
        self.assertEqual(
            [dsa_layer_skips_topk(config, layer) for layer in range(72)],
            [value == "shared" for value in _INDEX_SHARING],
        )

    def test_dsa_capability_follows_the_wrapped_text_model(self):
        # A wrapper used to lose sparse attention unless its own name was listed.
        with tempfile.TemporaryDirectory() as model_dir:
            with open(os.path.join(model_dir, "params.json"), "w") as f:
                json.dump(_PARAMS, f)
            model_config = ModelConfig(model_path=model_dir)
        config = model_config.hf_config
        config.architectures = ["FutureWrapperForConditionalGeneration"]
        self.assertTrue(is_deepseek_dsa(config))

        server_args = ServerArgs(model_path="dummy")
        server_args._model_config = model_config
        with self.assertRaisesRegex(ValueError, "is not registered"):
            handle_model_specific_adjustments(server_args)

        dense_text = SimpleNamespace(
            architectures=["MistralForCausalLM"], index_topk=2048
        )
        self.assertFalse(
            is_deepseek_dsa(
                SimpleNamespace(
                    architectures=["PixtralForConditionalGeneration"],
                    text_config=dense_text,
                )
            )
        )

    def test_dense_attention_support_follows_attention_features(self):
        base = {"model_type": "mistral_large_4", "index_topk": 2048}
        self.assertTrue(dsa_supports_dense_attention(SimpleNamespace(**base)))
        self.assertTrue(
            dsa_supports_dense_attention(
                SimpleNamespace(**base, nope_layer_flags=[False, False])
            )
        )
        for feature in (
            {"nope_layer_flags": [False, True]},
            {"ssmax_scaling": True},
            {"kv_sharing_config": {"source_layers_str": "0,0"}},
        ):
            with self.subTest(feature=feature):
                config = SimpleNamespace(
                    **{**base, "model_type": "deepseek_v32", **feature}
                )
                self.assertFalse(dsa_supports_dense_attention(config))

    @staticmethod
    def _resolve_attention_backends(**backends):
        with tempfile.TemporaryDirectory() as model_dir:
            with open(os.path.join(model_dir, "params.json"), "w") as f:
                json.dump(_PARAMS, f)
            model_config = ModelConfig(model_path=model_dir)
        server_args = ServerArgs(model_path="dummy", **backends)
        server_args._model_config = model_config
        # SM100 is where the generic MLA default is the dense flashinfer backend.
        with override_platform(
            is_sm100=True, is_hopper_with_cuda_12_3=False, is_hip=False
        ):
            declare_resolution(
                server_args,
                "deepseek_family",
                **_deepseek_family_overrides(server_args, model_config.hf_config),
            )
            run_post_process_pass(server_args, _attention_backend_default)
        view = resolved_view(server_args)
        return (view.attention_backend, *attention_backends_of(view))

    def test_explicit_dense_attention_backend_is_rejected(self):
        for backends in (
            {"attention_backend": "flashinfer"},
            {"attention_backend": "dsa", "prefill_attention_backend": "fa3"},
            {"decode_attention_backend": "flashmla"},
            {
                "prefill_attention_backend": "dsa",
                "decode_attention_backend": "flashinfer",
            },
        ):
            with self.subTest(**backends):
                with self.assertRaisesRegex(ValueError, "dsa attention backend"):
                    self._resolve_attention_backends(**backends)

    def test_attention_backends_resolve_to_dsa_on_both_phases(self):
        for backends, expected in (
            ({}, ("dsa", "dsa", "dsa")),
            ({"prefill_attention_backend": "dsa"}, ("dsa", "dsa", "dsa")),
            ({"decode_attention_backend": "dsa"}, ("dsa", "dsa", "dsa")),
            ({"prefill_attention_backend": "nsa"}, ("dsa", "nsa", "dsa")),
        ):
            with self.subTest(**backends):
                self.assertEqual(self._resolve_attention_backends(**backends), expected)

    def test_model_config_resolves_mla_with_and_without_vision(self):
        # The DSA backend reads the MLA shapes at startup, so both entry archs need them.
        text_only = {k: v for k, v in _PARAMS.items() if k != "vision_encoder"}
        for params, arch in (
            (_PARAMS, "PixtralForConditionalGeneration"),
            (text_only, "MistralLarge4ForCausalLM"),
        ):
            with self.subTest(arch=arch), tempfile.TemporaryDirectory() as model_dir:
                with open(os.path.join(model_dir, "params.json"), "w") as f:
                    json.dump(params, f)
                config = ModelConfig(model_path=model_dir)

                self.assertEqual(config.hf_config.architectures, [arch])
                self.assertEqual(config.attention_arch, AttentionArch.MLA)
                self.assertEqual(
                    (
                        config.kv_lora_rank,
                        config.qk_nope_head_dim,
                        config.qk_rope_head_dim,
                        config.v_head_dim,
                        config.index_head_dim,
                    ),
                    (512, 64, 64, 128, 128),
                )

    def test_mtp_block_resolves_the_bundled_dspark_draft(self):
        with tempfile.TemporaryDirectory() as model_dir:
            with open(os.path.join(model_dir, "params.json"), "w") as f:
                json.dump({**_PARAMS, "mtp": _MTP}, f)
            target = get_config(model_dir, trust_remote_code=False)
            draft = ModelConfig(
                model_path=model_dir,
                is_draft_model=True,
                speculative_algorithm="DSPARK",
            )

        self.assertTrue(checkpoint_bundles_dspark_draft(target))
        self.assertEqual(
            parse_dspark_draft_config(draft_hf_config=target).resolve_gamma(
                default=None
            ),
            3,
        )
        config = draft.hf_config
        self.assertEqual(config.architectures, ["MistralLarge4DSparkModel"])
        self.assertIsNone(draft.quantization)
        self.assertIsNone(config.quantization_config)
        self.assertEqual(config.num_hidden_layers, 3)
        self.assertEqual(config.layer_types, ["sliding_attention"] * 3)
        self.assertEqual((config.sliding_window, config.is_causal), (2048, False))
        self.assertEqual((config.markov_rank, config.markov_head_type), (512, "rnn"))
        self.assertEqual(config.dspark_draft_input, "position_specific")
        self.assertEqual(config.dflash_config["target_layer_ids"], [1, 18, 36, 54, 71])
        self.assertEqual(config.dflash_config["block_size"], 3)
        # Dense MLA draft layers with the target's geometry and plain YaRN.
        self.assertEqual(
            (
                config.hidden_size,
                config.num_attention_heads,
                config.intermediate_size,
                config.kv_lora_rank,
                config.qk_nope_head_dim,
                config.qk_rope_head_dim,
                config.v_head_dim,
            ),
            (6144, 64, 24576, 512, 64, 64, 128),
        )
        self.assertFalse(hasattr(config, "nope_layer_flags"))
        self.assertFalse(hasattr(config, "kv_sharing_config"))
        self.assertEqual(config.rope_scaling["rope_type"], "deepseek_yarn")
        self.assertEqual((draft.head_dim, draft.v_head_dim), (128, 128))

    def test_mtp_block_without_dspark_bundles_no_draft(self):
        mtp = {key: value for key, value in _MTP.items() if key != "dspark"}
        for params in (_PARAMS, {**_PARAMS, "mtp": mtp}):
            config_dict, _ = adapt_config_dict(copy.deepcopy(params), "mistral-large-4")
            self.assertFalse(checkpoint_bundles_dspark_draft(config_dict))
            self.assertFalse(
                checkpoint_bundles_dspark_draft(config_dict["text_config"])
            )

    def test_mtp_block_rejects_unsupported_draft_variants(self):
        for override, message in (
            ({"ragged_attention": "2048,1024,2048"}, "share one window"),
            ({"norm_strategy": "pre_norm"}, "norm_strategy"),
            ({"dspark": {**_MTP["dspark"], "head_type": "gated"}}, "head_type"),
            ({"dspark": {**_MTP["dspark"], "rank": 512}}, "mtp.dspark"),
            ({"target_activation_layers": "1,71,18"}, "ascending"),
            ({"target_activation_layers": "1,72"}, "ascending"),
        ):
            with self.subTest(override=override):
                params = {**copy.deepcopy(_PARAMS), "mtp": {**_MTP, **override}}
                with self.assertRaisesRegex((ValueError, NotImplementedError), message):
                    adapt_config_dict(params, "mistral-large-4")

    def test_unprefixed_dsa_keys_resolve_like_prefixed_keys(self):
        # The trainer's names, as written by newer checkpoint converters.
        unprefixed = {
            "index_topk": "topk",
            "index_n_heads": "n_heads",
            "index_head_dim": "head_dim",
            "index_rope_head_dim": "rope_head_dim",
            "indexer_match_nope": "match_nope",
            "indexer_ssmax_scaling": "ssmax_scaling",
            "indexer_norm_eps": "norm_eps",
            "index_sharing": "sharing",
        }
        params = copy.deepcopy(_PARAMS)
        params["dsa"] = {unprefixed.get(k, k): v for k, v in _PARAMS["dsa"].items()}

        resolved = [
            adapt_config_dict(copy.deepcopy(p), "mistral-large-4")[0]
            for p in (params, _PARAMS)
        ]
        self.assertEqual(resolved[0], resolved[1])
        self.assertEqual(resolved[0]["text_config"]["index_topk"], 2048)

    def test_dsa_block_rejects_mixed_spellings_and_unknown_keys(self):
        mixed = copy.deepcopy(_PARAMS)
        mixed["dsa"]["topk"] = mixed["dsa"].pop("index_topk")
        unknown = copy.deepcopy(_PARAMS)
        unknown["dsa"]["index_top_k"] = 2048
        # Converter-output names were accepted, then dropped: BF16 requests
        # silently resolved to an FP8 indexer and an unset KV precision.
        output_names = copy.deepcopy(_PARAMS)
        for key in ("fp8_indexer", "fp8_kv_cache"):
            del output_names["dsa"][key]
            output_names["dsa"]["dsa_" + key] = False
        conflicting = copy.deepcopy(_PARAMS)
        conflicting["dsa"]["dsa_fp8_kv_cache"] = False

        for params, message in (
            (mixed, "mixes prefixed keys"),
            (unknown, "Unknown keys"),
            (output_names, "checkpoint spelling is 'fp8_indexer'"),
            (conflicting, "checkpoint spelling is 'fp8_kv_cache'"),
        ):
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    adapt_config_dict(params, "mistral-large-4")

    def test_routing_renormalization_is_explicit(self):
        # Set by the params.json remap rather than a transformers config default.
        config_dict, _ = adapt_config_dict(copy.deepcopy(_PARAMS), "mistral-large-4")
        self.assertIs(config_dict["text_config"]["norm_topk_prob"], True)

    def test_kv_sharing_rejects_nope_mismatch_between_source_and_consumer(self):
        params = copy.deepcopy(_PARAMS)
        sources = list(_KV_SOURCES)
        sources[37] = 35  # Layer 37 uses RoPE while layer 35 is a NoPE layer.
        params["kv_sharing"]["source_layers_str"] = ",".join(map(str, sources))

        with self.assertRaisesRegex(ValueError, "same NoPE/RoPE setting"):
            adapt_config_dict(params, "mistral-large-4")

    def test_hisparse_prefetch_requires_shared_kv_provider_outside_index_groups(self):
        from sglang.srt.managers.hisparse_coordinator import (
            resolve_shared_index_layers,
        )

        with tempfile.TemporaryDirectory() as model_dir:
            with open(os.path.join(model_dir, "params.json"), "w") as f:
                json.dump(_PARAMS, f)
            text = get_config(model_dir, trust_remote_code=False).text_config

        with mock.patch(
            "sglang.srt.managers.hisparse_coordinator.get_parallel",
            return_value=SimpleNamespace(pp_size=1),
        ):
            self.assertEqual(
                resolve_shared_index_layers(hf_text_config=text),
                [value == "shared" for value in _INDEX_SHARING],
            )
            # Layer 36 anchors the shared-index group 37-38; layer 37 belongs to it.
            for provider in (36, 37):
                sources = list(_KV_SOURCES)
                sources[39] = provider
                text.kv_sharing_config = {
                    "source_layers_str": ",".join(map(str, sources))
                }
                with self.subTest(provider=provider):
                    self.assertIsNone(resolve_shared_index_layers(hf_text_config=text))

    def test_pinned_revision_reads_params_from_that_revision(self):
        default_params = copy.deepcopy(_PARAMS)
        default_params["dim"] = 4096
        with tempfile.TemporaryDirectory() as root:
            snapshots = {}
            for revision, params in (("main", default_params), ("pinned", _PARAMS)):
                snapshots[revision] = os.path.join(root, revision)
                os.makedirs(snapshots[revision])
                with open(os.path.join(snapshots[revision], "params.json"), "w") as f:
                    json.dump(params, f)

            def fake_snapshot_download(repo_id, allow_patterns=None, revision=None):
                return snapshots[revision or "main"]

            load_mistral_config.cache_clear()
            with mock.patch.object(
                common, "snapshot_download", side_effect=fake_snapshot_download
            ):
                config = load_mistral_config("org/native-model", revision="pinned")
            load_mistral_config.cache_clear()

        self.assertEqual(config.text_config.hidden_size, 6144)


if __name__ == "__main__":
    unittest.main()
