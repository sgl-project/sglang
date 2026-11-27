import unittest
from types import SimpleNamespace

from sglang.srt.arg_groups.overrides import declare_resolution, resolution_result
from sglang.srt.arg_groups.speculative_hook import _handle_eagle_family
from sglang.srt.environ import envs
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_MODEL_PATH = "deepseek-ai/DeepSeek-V3"


def _make_nextn_server_args(*, draft_revision) -> ServerArgs:
    server_args = ServerArgs(model_path="dummy")
    server_args.model_path = _MODEL_PATH
    server_args.revision = "target"
    server_args.device = "cuda"
    server_args.speculative_algorithm = "EAGLE"
    server_args.speculative_draft_model_path = None
    server_args.speculative_draft_model_revision = draft_revision
    server_args.speculative_num_steps = 3
    server_args.speculative_eagle_topk = 1
    server_args.speculative_num_draft_tokens = 4
    server_args._model_config = SimpleNamespace(
        hf_config=SimpleNamespace(architectures=["DeepseekV3ForCausalLM"])
    )
    return server_args


class TestEagleBundledDraftPathDefaulting(CustomTestCase):
    def test_bundled_draft_keeps_an_explicit_draft_revision(self):
        for draft_revision, expected in ((None, "target"), ("draft", "draft")):
            with self.subTest(draft_revision=draft_revision):
                server_args = _make_nextn_server_args(draft_revision=draft_revision)
                _handle_eagle_family(server_args)
                self.assertEqual(
                    resolution_result(server_args, "speculative_draft_model_path"),
                    _MODEL_PATH,
                )
                self.assertEqual(
                    resolution_result(server_args, "speculative_draft_model_revision"),
                    expected,
                )

    def test_modelscope_rejects_another_revision_for_a_bundled_draft(self):
        server_args = _make_nextn_server_args(draft_revision="draft")
        declare_resolution(
            server_args, "_handle_modelscope_paths", model_path="/ms/snapshot"
        )
        with (
            envs.SGLANG_USE_MODELSCOPE.override(True),
            self.assertRaisesRegex(ValueError, "ModelScope"),
        ):
            _handle_eagle_family(server_args)

    def test_modelscope_check_ignores_a_later_model_path_change(self):
        server_args = _make_nextn_server_args(draft_revision="draft")
        declare_resolution(
            server_args, "_handle_modelscope_paths", model_path=_MODEL_PATH
        )
        declare_resolution(server_args, "handle_expert_pack", model_path="/meta")
        with envs.SGLANG_USE_MODELSCOPE.override(True):
            _handle_eagle_family(server_args)


if __name__ == "__main__":
    unittest.main()
