import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.arg_groups.overrides import resolution_result
from sglang.srt.arg_groups.speculative_hook import (
    _publish_dspark_mla_draft_backend,
    _resolve_dflash_draft_attention_backend,
    _resolve_dflash_draft_kv_on_swa,
)
from sglang.srt.configs.model_config import AttentionArch
from sglang.srt.platforms.interface import SRTPlatform
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")

HOOK_MODULE = "sglang.srt.arg_groups.speculative_hook"


def _platform(
    default_backend: str = "custom_backend",
    supported_backends: set[str] | None = None,
) -> Mock:
    supported_backends = supported_backends or {"custom_backend"}
    platform = Mock(spec=SRTPlatform)
    platform.is_out_of_tree.return_value = True
    platform.supports_speculative_algorithm.return_value = True
    platform.supports_speculative_draft_attention_backend.side_effect = (
        lambda algorithm, backend: (
            algorithm == "DFLASH" and backend in supported_backends
        )
    )
    platform.get_default_speculative_draft_attention_backend.return_value = (
        default_backend
    )
    return platform


def _make_dflash_args(draft_backend: str | None) -> ServerArgs:
    return ServerArgs(
        model_path="dummy",
        device="custom",
        speculative_algorithm="DFLASH",
        speculative_draft_model_path="draft",
        speculative_num_draft_tokens=4,
        speculative_draft_attention_backend=draft_backend,
    )


class TestOOTDFlashHooks(CustomTestCase):
    def _resolve_backend(self, draft_backend: str | None, platform: Mock) -> str | None:
        args = _make_dflash_args(draft_backend)
        with (
            patch(f"{HOOK_MODULE}.current_platform", platform),
            patch(f"{HOOK_MODULE}.attention_backends_of", return_value=(None, None)),
        ):
            # Call the resolver directly: the full hook also resolves the
            # algorithm alias, which loads the draft HF config over the network.
            _resolve_dflash_draft_attention_backend(args)
        # Resolvers declare into the stash; they never write the field.
        return resolution_result(args, "speculative_draft_attention_backend")

    def test_explicit_backends_follow_platform_capabilities(self):
        cases = (
            ("custom_backend", {"custom_backend"}, "custom_backend"),
            ("flashinfer", {"custom_backend"}, "custom_backend"),
            ("flashinfer", {"flashinfer"}, "flashinfer"),
            ("trtllm_mha", {"custom_backend"}, "custom_backend"),
        )
        for draft_backend, supported_backends, expected in cases:
            with self.subTest(
                draft_backend=draft_backend, supported_backends=supported_backends
            ):
                self.assertEqual(
                    self._resolve_backend(
                        draft_backend,
                        _platform(supported_backends=supported_backends),
                    ),
                    expected,
                )

    def test_unknown_backend_warns_and_falls_back(self):
        with self.assertLogs(
            "sglang.srt.arg_groups.speculative_hook", "WARNING"
        ) as logs:
            resolved = self._resolve_backend("typo", _platform())

        self.assertEqual(resolved, "custom_backend")
        self.assertIn("attention_backend 'typo'", "\n".join(logs.output))

    def test_invalid_platform_defaults_raise_actionable_errors(self):
        missing_default = _platform()
        missing_default.get_default_speculative_draft_attention_backend.side_effect = (
            NotImplementedError
        )
        for platform, error in (
            (_platform(default_backend="flashinfer"), "returned unsupported"),
            (
                missing_default,
                "get_default_speculative_draft_attention_backend",
            ),
        ):
            with (
                self.subTest(error=error),
                self.assertRaisesRegex(ValueError, error),
            ):
                self._resolve_backend(None, platform)


class TestDFlashDraftBackendResolution(CustomTestCase):
    """MLA drafts fall back to the default MLA draft backend."""

    def _resolve(
        self,
        draft_backend,
        *,
        mla,
        target_backend="fa3",
        resolver=_resolve_dflash_draft_attention_backend,
    ):
        args = _make_dflash_args(draft_backend)
        platform = Mock(spec=SRTPlatform)
        platform.is_out_of_tree.return_value = False
        config = SimpleNamespace(
            num_hidden_layers=1,
            is_causal=True,
            dflash_config={"attention_mode": "mla" if mla else "gqa"},
        )
        with (
            patch(f"{HOOK_MODULE}.current_platform", platform),
            patch(
                f"{HOOK_MODULE}.get_platform",
                return_value=SimpleNamespace(is_xpu=False, is_hip=False, is_sm100=True),
            ),
            patch(
                f"{HOOK_MODULE}.attention_backends_of",
                return_value=(target_backend, target_backend),
            ),
            patch(
                "sglang.srt.hf_transformers.get_config",
                return_value=config,
            ),
        ):
            resolver(args)
        return resolution_result(args, "speculative_draft_attention_backend")

    def test_mla_drafts_take_the_default_mla_backend(self):
        # On SM100 a non-MLA draft leaving trtllm_mha goes to fa4 instead.
        self.assertEqual(self._resolve("trtllm_mha", mla=True), "flashinfer")
        # The trtllm-gen MLA decode kernel has no bidirectional mask.
        self.assertEqual(
            self._resolve(None, mla=True, target_backend="trtllm_mla"), "flashinfer"
        )
        self.assertEqual(self._resolve("trtllm_mha", mla=False), "trtllm_mha")

    def test_mla_dspark_draft_publishes_its_fallback(self):
        """The KV planners read the published draft backend, so an MLA DSpark
        draft that would inherit trtllm_mha must publish flashinfer."""
        for mla, expected in ((True, "flashinfer"), (False, None)):
            self.assertEqual(
                self._resolve(
                    None,
                    mla=mla,
                    target_backend="trtllm_mha",
                    resolver=lambda args, mla=mla: _publish_dspark_mla_draft_backend(
                        args, SimpleNamespace(is_mla=mla)
                    ),
                ),
                expected,
            )


class TestDFlashDraftKvOnSwaResolution(CustomTestCase):
    def _resolve(self, **overrides):
        kwargs = dict(
            model_path="dummy",
            device="custom",
            speculative_algorithm="DFLASH",
            speculative_draft_model_path="draft",
            speculative_num_draft_tokens=8,
            speculative_draft_attention_backend="fa4",
            disaggregation_mode="prefill",
            disaggregation_transfer_backend="mooncake",
        )
        kwargs.update(overrides)
        args = ServerArgs(**kwargs)
        platform = Mock(spec=SRTPlatform)
        platform.is_out_of_tree.return_value = False
        draft = SimpleNamespace(
            num_hidden_layers=2,
            is_causal=True,
            dflash_config={"attention_mode": "gqa"},
            layer_types=["sliding_attention"] * 2,
            sliding_window=2048,
        )
        target = SimpleNamespace(
            attention_arch=AttentionArch.MHA,
            is_hybrid_swa=True,
            is_hybrid_swa_compress=False,
            sliding_window_size=2048,
            hf_config=SimpleNamespace(
                model_type="muse_spark", architectures=["MuseSparkForCausalLM"]
            ),
        )
        with (
            patch(f"{HOOK_MODULE}.current_platform", platform),
            patch(
                f"{HOOK_MODULE}.get_platform",
                return_value=SimpleNamespace(is_cuda=True),
            ),
            patch(f"{HOOK_MODULE}.model_config_of", return_value=target),
            patch("sglang.srt.hf_transformers.get_config", return_value=draft),
        ):
            _resolve_dflash_draft_kv_on_swa(args)
        return (
            resolution_result(args, "speculative_dflash_draft_kv_on_swa"),
            resolution_result(args, "speculative_dflash_draft_window_transfer"),
        )

    def test_launch_gates(self):
        cases = (
            ({}, (True, 2048)),
            ({"disaggregation_transfer_backend": "nixl"}, (False, None)),
            # A real launch rejects PP > 1 earlier; this pins the resolver's own guard.
            ({"pp_size": 2}, (False, None)),
            ({"attn_cp_size": 2}, (False, None)),
            ({"dcp_size": 2}, (False, None)),
            # The window transfer does not depend on the draft backend.
            ({"speculative_draft_attention_backend": "flashinfer"}, (False, 2048)),
        )
        for overrides, expected in cases:
            with self.subTest(**overrides):
                self.assertEqual(self._resolve(**overrides), expected)


if __name__ == "__main__":
    unittest.main()
