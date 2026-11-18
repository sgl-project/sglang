"""Resolution refuses an EAGLE-family MLA draft under DCP when the draft runs on aiter.

Such a draft (DeepSeek-V2-Lite STANDALONE was observed) used to fail during startup,
after the weights loaded, with an opaque ``AttributeError`` inside the aiter extend:
the MLA model code sends the draft extend through its DCP decode branch without the
step's keys. The check is inert off ROCm, at DCP 1, for drafts on another backend,
for non-MLA drafts, and for ROCm DSA drafts that shard their KV like the target.
"""

import contextlib
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.arg_groups.speculative_hook import (
    _reject_aiter_mla_draft_under_dcp,
    handle_speculative_decoding,
)
from sglang.srt.configs import model_config as model_config_module
from sglang.srt.configs.model_config import AttentionArch, ModelConfig
from sglang.srt.runtime_context import override_platform
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_NEXTN_ARCH = "DeepseekV3ForCausalLMNextN"


def _args(**overrides) -> ServerArgs:
    # model_path="dummy" short-circuits ServerArgs.__post_init__.
    args = ServerArgs(model_path="dummy")
    args.speculative_algorithm = "STANDALONE"
    args.speculative_draft_model_path = "draft"
    args.speculative_num_steps = 3
    args.speculative_eagle_topk = 1
    args.speculative_num_draft_tokens = 4
    args.device = "cuda"
    args.attention_backend = "aiter"
    args.dcp_size = 2
    args._model_config = SimpleNamespace(
        hf_config=SimpleNamespace(
            architectures=["DeepseekV2ForCausalLM"],
            get_text_config=lambda: SimpleNamespace(),
        )
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def _draft_config(*, arch=AttentionArch.MLA, **hf_fields):
    return SimpleNamespace(
        attention_arch=arch,
        hf_config=SimpleNamespace(architectures=[_NEXTN_ARCH], **hf_fields),
    )


@contextlib.contextmanager
def _rocm(*, is_hip=True, draft_config=None):
    """Patch the platform and the draft's ModelConfig; yield the config builder."""
    with (
        override_platform(is_hip=is_hip, is_xpu=False),
        patch.object(model_config_module, "is_hip", return_value=is_hip),
        patch.object(
            ModelConfig,
            "from_server_args",
            return_value=draft_config or _draft_config(),
        ) as build,
    ):
        yield build


class TestAiterMlaDraftDcpGuard(CustomTestCase):
    def test_raises_for_an_mla_draft_on_aiter_under_dcp(self):
        with _rocm() as build, self.assertRaisesRegex(ValueError, "--dcp-size 2"):
            _reject_aiter_mla_draft_under_dcp(_args())
        # Reads only the draft's config, at resolution time.
        build.assert_called_once()
        self.assertEqual(build.call_args.kwargs["model_path"], "draft")
        self.assertTrue(build.call_args.kwargs["is_draft_model"])

    def test_an_explicit_aiter_draft_backend_is_checked(self):
        args = _args(
            attention_backend="triton", speculative_draft_attention_backend="aiter"
        )
        with _rocm(), self.assertRaises(ValueError):
            _reject_aiter_mla_draft_under_dcp(args)

    def test_inert_without_rocm_dcp_or_an_aiter_draft(self):
        for args, is_hip in (
            (_args(), False),
            (_args(dcp_size=1), True),
            (_args(attention_backend="triton"), True),
            (_args(speculative_draft_attention_backend="triton"), True),
        ):
            with (
                self.subTest(dcp=args.dcp_size, is_hip=is_hip),
                _rocm(is_hip=is_hip) as build,
            ):
                _reject_aiter_mla_draft_under_dcp(args)
                # The draft config is only read when the rest applies.
                build.assert_not_called()

    def test_inert_for_a_non_mla_draft(self):
        with _rocm(draft_config=_draft_config(arch=AttentionArch.MHA)):
            _reject_aiter_mla_draft_under_dcp(_args())

    def test_inert_for_a_rocm_dsa_draft_that_shards_like_the_target(self):
        with _rocm(draft_config=_draft_config(index_topk=2048)):
            _reject_aiter_mla_draft_under_dcp(_args())
        # The same NextN arch without an indexer is a plain MLA draft.
        with _rocm(), self.assertRaises(ValueError):
            _reject_aiter_mla_draft_under_dcp(_args())

    def test_the_eagle_family_hook_runs_the_check(self):
        # DeepSeek-V3 NEXTN: the hook points the draft at the target checkpoint first.
        args = _args(speculative_algorithm="NEXTN", speculative_draft_model_path=None)
        args._model_config.hf_config.architectures = ["DeepseekV3ForCausalLM"]
        with (
            _rocm() as build,
            self.assertRaisesRegex(ValueError, "MLA draft model"),
        ):
            handle_speculative_decoding(args)
        self.assertEqual(build.call_args.kwargs["model_path"], "dummy")


if __name__ == "__main__":
    unittest.main()
