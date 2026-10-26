import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.arg_groups.overrides import resolution_result
from sglang.srt.arg_groups.speculative_hook import (
    _handle_dspark,
    _target_checkpoint_bundles_dspark_draft,
    _target_checkpoint_dspark_subfolder,
)
from sglang.srt.environ import envs
from sglang.srt.server_args import ServerArgs
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


class TestDsparkDpAttentionMoeA2aGate(CustomTestCase):
    """Gate contract for DSpark + dp attention + MoE a2a backends."""

    def _dp_server_args(self, *, moe_a2a_backend: str) -> ServerArgs:
        server_args = _make_dspark_server_args(
            model_path=_BUNDLED_MODEL_PATH, hf_config=_bundled_hf_config()
        )
        server_args.enable_dp_lm_head = True
        server_args.attn_dp_size = 2
        server_args.tp_size = 2
        server_args.moe_a2a_backend = moe_a2a_backend
        return server_args

    def test_only_megamoe_is_admitted(self):
        """Both sides of the allowlist: megamoe passes, others raise by name."""
        with envs.SGLANG_RAGGED_VERIFY_MODE.override("static"):
            _handle_dspark(self._dp_server_args(moe_a2a_backend="megamoe"))
            for backend in ("deepep", "pplx"):
                with self.assertRaisesRegex(ValueError, backend):
                    _handle_dspark(self._dp_server_args(moe_a2a_backend=backend))

    def test_a2a_backend_with_compact_verify_mode_raises(self):
        server_args = self._dp_server_args(moe_a2a_backend="megamoe")
        with envs.SGLANG_RAGGED_VERIFY_MODE.override("compact"):
            with self.assertRaisesRegex(ValueError, "static"):
                _handle_dspark(server_args)


if __name__ == "__main__":
    unittest.main()
