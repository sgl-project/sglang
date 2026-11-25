import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.arg_groups.boundary_reduction import (
    model_boundary_reduction,
    resolve_boundary_reduction,
)
from sglang.srt.arg_groups.model_override_base import resolved_view
from sglang.srt.arg_groups.overrides import resolution_result
from sglang.srt.arg_groups.speculative_hook import (
    _handle_dspark,
    _resolve_dspark_lm_head_dtype,
    _resolve_speculative_algorithm_alias,
    _target_checkpoint_bundles_dspark_draft,
    _target_checkpoint_dspark_subfolder,
)
from sglang.srt.configs.ernie55 import ERNIE5_5_MOE_ARCH
from sglang.srt.environ import envs
from sglang.srt.runtime_context import override_platform
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.dspark_components.dspark_config import (
    load_dspark_hf_config,
    resolve_namespaced_dspark_draft_config,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")

_HOOK = "sglang.srt.arg_groups.speculative_hook"
_BUNDLED_MODEL_PATH = "deepseek-ai/DeepSeek-V4-Flash-DSpark"
_PLAIN_MODEL_PATH = "deepseek-ai/DeepSeek-V4-Flash"


def _bundled_hf_config() -> SimpleNamespace:
    return SimpleNamespace(
        architectures=["DeepseekV4ForCausalLM"],
        dspark_block_size=5,
        dspark_markov_rank=256,
        dspark_target_layer_ids=[40, 41, 42],
        dspark_noise_token_id=128799,
    )


def _plain_hf_config() -> SimpleNamespace:
    return SimpleNamespace(architectures=["DeepseekV4ForCausalLM"])


def _make_dspark_server_args(
    *, model_path: str, hf_config: SimpleNamespace
) -> ServerArgs:
    server_args = ServerArgs(model_path="dummy")
    server_args.model_path = model_path
    server_args.device = "cuda"
    server_args.speculative_algorithm = "DSPARK"
    server_args.speculative_draft_model_path = None
    server_args.speculative_dspark_block_size = 5
    server_args._model_config = SimpleNamespace(hf_config=hf_config)
    return server_args


class TestTargetCheckpointBundlesDsparkDraft(CustomTestCase):
    def test_bundled_dsv4_config_is_detected(self):
        server_args = _make_dspark_server_args(
            model_path=_BUNDLED_MODEL_PATH, hf_config=_bundled_hf_config()
        )
        self.assertTrue(_target_checkpoint_bundles_dspark_draft(server_args))

    def test_plain_target_config_is_not_detected(self):
        server_args = _make_dspark_server_args(
            model_path=_PLAIN_MODEL_PATH, hf_config=_plain_hf_config()
        )
        self.assertFalse(_target_checkpoint_bundles_dspark_draft(server_args))


class TestDsparkLmHeadDtype(CustomTestCase):
    def _resolve(self, dtype, *, is_gfx95=False, is_blackwell=False):
        server_args = _make_dspark_server_args(
            model_path=_BUNDLED_MODEL_PATH, hf_config=_bundled_hf_config()
        )
        server_args.speculative_dspark_lm_head_dtype = dtype
        facts = SimpleNamespace(is_gfx95=is_gfx95, is_blackwell=is_blackwell)
        with patch(f"{_HOOK}.get_platform", return_value=facts):
            _resolve_dspark_lm_head_dtype(server_args)
        return resolution_result(server_args, "speculative_dspark_lm_head_dtype")

    def test_unsupported_platform_raises(self):
        for dtype in ("fp8", "mxfp8"):
            with self.assertRaises(ValueError):
                self._resolve(dtype)

    def test_supported_platform_keeps_dtype(self):
        self.assertEqual(self._resolve("mxfp8", is_blackwell=True), "mxfp8")
        self.assertEqual(self._resolve("fp8", is_gfx95=True), "fp8")

    def test_unset_is_a_no_op(self):
        self.assertIsNone(self._resolve(None))


class TestDsparkDraftPathDefaulting(CustomTestCase):
    def test_bundled_checkpoint_defaults_draft_path_to_model_path(self):
        server_args = _make_dspark_server_args(
            model_path=_BUNDLED_MODEL_PATH, hf_config=_bundled_hf_config()
        )
        _handle_dspark(server_args)
        self.assertEqual(
            resolution_result(server_args, "speculative_draft_model_path"),
            _BUNDLED_MODEL_PATH,
        )
        self.assertEqual(
            resolution_result(server_args, "speculative_num_draft_tokens"), 6
        )

    def test_plain_target_without_draft_path_raises(self):
        server_args = _make_dspark_server_args(
            model_path=_PLAIN_MODEL_PATH, hf_config=_plain_hf_config()
        )
        with (
            patch(f"{_HOOK}._target_checkpoint_dspark_subfolder", return_value=None),
            self.assertRaises(ValueError),
        ):
            _handle_dspark(server_args)

    def test_local_dspark_subfolder_becomes_the_draft_path(self):
        with tempfile.TemporaryDirectory() as ckpt:
            draft_dir = os.path.join(ckpt, "dspark")
            os.makedirs(draft_dir)
            with open(os.path.join(draft_dir, "config.json"), "w") as f:
                f.write("{}")
            server_args = _make_dspark_server_args(
                model_path=ckpt, hf_config=_plain_hf_config()
            )
            _handle_dspark(server_args)
            self.assertEqual(
                resolution_result(server_args, "speculative_draft_model_path"),
                draft_dir,
            )

    def test_local_checkpoint_without_dspark_config_raises(self):
        with tempfile.TemporaryDirectory() as ckpt:
            # A bare folder is not a draft: the draft needs its own config.json.
            os.makedirs(os.path.join(ckpt, "dspark"))
            server_args = _make_dspark_server_args(
                model_path=ckpt, hf_config=_plain_hf_config()
            )
            with self.assertRaises(ValueError):
                _handle_dspark(server_args)

    def test_explicit_draft_path_is_not_overwritten(self):
        server_args = _make_dspark_server_args(
            model_path=_BUNDLED_MODEL_PATH, hf_config=_bundled_hf_config()
        )
        server_args.speculative_draft_model_path = "deepseek-ai/some-other-dspark-draft"
        _handle_dspark(server_args)
        self.assertEqual(
            resolution_result(server_args, "speculative_draft_model_path"),
            "deepseek-ai/some-other-dspark-draft",
        )


class TestTargetCheckpointDsparkSubfolderOnHub(CustomTestCase):
    """A Hub repo id resolves to the snapshot's dspark/ folder, fetching only it."""

    def test_repo_with_dspark_subfolder_downloads_only_that_folder(self):
        with (
            patch("huggingface_hub.hf_hub_download") as probe,
            patch(
                "huggingface_hub.snapshot_download", return_value="/hf/snap/abc"
            ) as snapshot,
        ):
            draft_dir = _target_checkpoint_dspark_subfolder(
                "org/model", revision="main", cache_dir="/dl"
            )
        self.assertEqual(draft_dir, "/hf/snap/abc/dspark")
        probe.assert_called_once_with(
            "org/model", "dspark/config.json", revision="main", cache_dir="/dl"
        )
        snapshot.assert_called_once_with(
            "org/model", revision="main", cache_dir="/dl", allow_patterns=["dspark/*"]
        )

    def test_repo_without_dspark_subfolder_returns_none(self):
        from huggingface_hub.errors import EntryNotFoundError

        with (
            patch(
                "huggingface_hub.hf_hub_download",
                side_effect=EntryNotFoundError("missing"),
            ),
            patch("huggingface_hub.snapshot_download") as snapshot,
        ):
            self.assertIsNone(
                _target_checkpoint_dspark_subfolder(
                    "org/model", revision=None, cache_dir=None
                )
            )
        snapshot.assert_not_called()


class TestNamespacedDsparkConfig(CustomTestCase):
    def test_resolution_isolates_target_and_maps_mla_attention(self):
        config = SimpleNamespace(
            architectures=["BailingMoeV3VLForConditionalGeneration"],
            text_config=SimpleNamespace(
                architectures=["BailingMoeV3ForCausalLM"],
                num_attention_heads=64,
                quantization_config=None,
            ),
            quantization_config={"quant_method": "fp8"},
            dspark_bundle_format="namespaced_safetensors",
            dspark_bundle_version=1,
            dspark_config={
                "architectures": ["LingV3ForCausalLMDSparkMLA"],
                "attention_type": "mla",
            },
        )

        draft = resolve_namespaced_dspark_draft_config(config)

        self.assertIsNot(draft, config.text_config)
        self.assertEqual(draft.attention_type, 1)
        self.assertEqual(draft.full_attention_type, "mla")
        self.assertEqual(draft.architectures, ["LingV3ForCausalLMDSparkMLA"])
        self.assertEqual(draft.quantization_config, {"quant_method": "fp8"})
        self.assertFalse(hasattr(config.text_config, "attention_type"))
        self.assertIsNone(config.text_config.quantization_config)

    def test_separate_sidecar_uses_an_isolated_target_config(self):
        from sglang.srt.hf_transformers import get_config

        with tempfile.TemporaryDirectory() as root:
            target_dir = os.path.join(root, "target")
            draft_dir = os.path.join(root, "draft")
            os.makedirs(target_dir)
            os.makedirs(draft_dir)
            with open(os.path.join(target_dir, "config.json"), "w") as output:
                json.dump(
                    {
                        "model_type": "llama",
                        "architectures": ["LlamaForCausalLM"],
                        "hidden_size": 4096,
                        "num_hidden_layers": 72,
                        "num_attention_heads": 64,
                        "quantization_config": {"quant_method": "fp8"},
                    },
                    output,
                )
            with open(os.path.join(draft_dir, "dspark_config.json"), "w") as output:
                json.dump(
                    {
                        "dspark_bundle_format": "namespaced_safetensors",
                        "dspark_bundle_version": 1,
                        "dspark_config": {
                            "architectures": ["LingV3ForCausalLMDSparkMLA"],
                            "attention_type": "mla",
                            "num_hidden_layers": 3,
                        },
                    },
                    output,
                )
            target = get_config(target_dir, trust_remote_code=False)

            config = load_dspark_hf_config(
                draft_dir,
                trust_remote_code=False,
                revision=None,
                model_override_args={},
                target_model_path=target_dir,
            )
            draft = resolve_namespaced_dspark_draft_config(config)

            self.assertEqual(draft.hidden_size, 4096)
            self.assertEqual(draft.num_hidden_layers, 3)
            self.assertEqual(draft.architectures, ["LingV3ForCausalLMDSparkMLA"])
            self.assertEqual(draft.quantization_config, {"quant_method": "fp8"})
            self.assertFalse(hasattr(target, "dspark_config"))
            self.assertEqual(
                _resolve_speculative_algorithm_alias(
                    "DSPARK",
                    draft_dir,
                    trust_remote_code=False,
                    kwargs={},
                    target_model_path=target_dir,
                ),
                "DSPARK",
            )
            server_args = ServerArgs(model_path=target_dir)
            server_args.speculative_algorithm = "DSPARK"
            server_args.speculative_draft_model_path = draft_dir
            server_args._model_config = SimpleNamespace(hf_config=config)
            boundary = resolve_boundary_reduction(resolved_view(server_args))
            self.assertEqual(
                boundary["speculative_boundary_reduction"],
                model_boundary_reduction(config),
            )


class TestDsparkDpAttentionMoeA2aGate(CustomTestCase):
    """Gate contract for DSpark + dp attention + MoE a2a backends."""

    def _dp_server_args(
        self, *, moe_a2a_backend: str, arch: str = "DeepseekV4ForCausalLM"
    ) -> ServerArgs:
        hf_config = _bundled_hf_config()
        hf_config.architectures = [arch]
        server_args = _make_dspark_server_args(
            model_path=_BUNDLED_MODEL_PATH, hf_config=hf_config
        )
        server_args.enable_dp_lm_head = True
        server_args.attn_dp_size = 2
        server_args.tp_size = 2
        server_args.moe_a2a_backend = moe_a2a_backend
        return server_args

    def test_supported_moe_backends_are_admitted(self):
        """Preserve megamoe and admit Hopper DeepEP; reject unsupported backends."""
        with (
            envs.SGLANG_RAGGED_VERIFY_MODE.override("static"),
            envs.SGLANG_ENABLE_DP_SPEC_PREFILL_COORDINATION.override(False),
            override_platform(is_sm90=True),
        ):
            for backend in ("megamoe", "deepep"):
                with self.subTest(backend=backend):
                    _handle_dspark(self._dp_server_args(moe_a2a_backend=backend))
            with self.assertRaisesRegex(ValueError, "pplx"):
                _handle_dspark(self._dp_server_args(moe_a2a_backend="pplx"))

    def test_ernie55_admits_deepep_and_rejects_megamoe(self):
        with (
            envs.SGLANG_RAGGED_VERIFY_MODE.override("static"),
            override_platform(is_sm90=True),
        ):
            _handle_dspark(
                self._dp_server_args(moe_a2a_backend="deepep", arch=ERNIE5_5_MOE_ARCH)
            )
            with self.assertRaisesRegex(ValueError, "megamoe"):
                _handle_dspark(
                    self._dp_server_args(
                        moe_a2a_backend="megamoe", arch=ERNIE5_5_MOE_ARCH
                    )
                )

    def test_bailing_admits_deepep_and_preserves_default_backends(self):
        with (
            envs.SGLANG_RAGGED_VERIFY_MODE.override("static"),
            override_platform(is_sm90=True),
        ):
            arch = "BailingMoeV3VLForConditionalGeneration"
            for backend in ("none", "megamoe", "mori", "deepep"):
                with self.subTest(backend=backend):
                    _handle_dspark(
                        self._dp_server_args(moe_a2a_backend=backend, arch=arch)
                    )

    def test_deepep_outside_hopper_is_rejected(self):
        with (
            envs.SGLANG_RAGGED_VERIFY_MODE.override("static"),
            envs.SGLANG_ENABLE_DP_SPEC_PREFILL_COORDINATION.override(False),
            override_platform(is_sm90=False),
        ):
            _handle_dspark(self._dp_server_args(moe_a2a_backend="megamoe"))
            with self.assertRaisesRegex(ValueError, "Hopper"):
                _handle_dspark(self._dp_server_args(moe_a2a_backend="deepep"))

    def test_a2a_backend_with_compact_verify_mode_raises(self):
        server_args = self._dp_server_args(moe_a2a_backend="megamoe")
        with envs.SGLANG_RAGGED_VERIFY_MODE.override("compact"):
            with self.assertRaisesRegex(ValueError, "static"):
                _handle_dspark(server_args)


if __name__ == "__main__":
    unittest.main()
