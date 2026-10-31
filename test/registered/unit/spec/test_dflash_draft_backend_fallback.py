import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.arg_groups.overrides import resolution_result
from sglang.srt.arg_groups.speculative_hook import (
    _resolve_dflash_draft_attention_backend,
)
from sglang.srt.platforms.interface import SRTPlatform
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

HOOK_MODULE = "sglang.srt.arg_groups.speculative_hook"


class TestDFlashDraftBackendFallback(CustomTestCase):
    def _resolve(self, *, sm100: bool = False, hip: bool = False, xpu: bool = False):
        args = ServerArgs(
            model_path="dummy",
            device="custom",
            speculative_algorithm="DFLASH",
            speculative_draft_model_path="draft",
            speculative_num_draft_tokens=4,
        )
        platform = Mock(spec=SRTPlatform)
        platform.is_out_of_tree.return_value = False
        device = SimpleNamespace(is_xpu=xpu, is_hip=hip, is_sm100_or_sm110=sm100)
        with (
            patch(f"{HOOK_MODULE}.current_platform", platform),
            patch(f"{HOOK_MODULE}.get_platform", return_value=device),
            # A target backend the DFLASH draft worker does not support.
            patch(f"{HOOK_MODULE}.attention_backends_of", return_value=("dsa", None)),
            self.assertLogs(HOOK_MODULE, "WARNING"),
        ):
            _resolve_dflash_draft_attention_backend(args)
        return resolution_result(args, "speculative_draft_attention_backend")

    def test_unsupported_target_backend_falls_back_to_fa4_on_sm100(self):
        self.assertEqual(self._resolve(sm100=True), "fa4")

    def test_other_cuda_gpus_keep_flashinfer(self):
        self.assertEqual(self._resolve(), "flashinfer")

    def test_rocm_and_xpu_keep_triton(self):
        self.assertEqual(self._resolve(hip=True), "triton")
        self.assertEqual(self._resolve(xpu=True), "triton")


if __name__ == "__main__":
    unittest.main()
