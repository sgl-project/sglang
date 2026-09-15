"""Model kernel warmup must cover prefill without a speculative dummy batch."""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.model_executor.runner import flashinfer_autotune as autotune
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, stage="base-a", runner_config="cpu")


class TestModelPrefillAutotune(CustomTestCase):
    def setUp(self):
        self.hook = Mock(return_value=1)
        self.mr = SimpleNamespace(
            model=SimpleNamespace(autotune_prefill_kernels=self.hook),
            is_generation=True,
            is_draft_worker=False,
            dtype=torch.bfloat16,
        )
        self.runner = SimpleNamespace(model_runner=self.mr)
        # Deliberately no dummy-buffer or attention APIs: this path must not
        # construct a TARGET_VERIFY batch or mutate request/KV state.
        for target, kwargs in (
            ("max_prefill_buffer_tokens", {"return_value": 65536}),
            (
                "flashinfer_autotune_context",
                {"side_effect": lambda *a, **k: nullcontext()},
            ),
        ):
            p = patch.object(autotune, target, **kwargs)
            setattr(self, target, p.start())
            self.addCleanup(p.stop)
        p = patch.object(
            autotune.envs.SGLANG_FLASHINFER_AUTOTUNE_EXTEND, "get", return_value=False
        )
        p.start()
        self.addCleanup(p.stop)

    def test_prefill_kernel_hook_uses_large_m_and_runner_dtype(self):
        autotune.maybe_flashinfer_autotune_extend(self.runner, decode_num_tokens=384)
        self.hook.assert_called_once_with(65536, dtype=torch.bfloat16)
        self.flashinfer_autotune_context.assert_called_once_with(
            self.mr, run_lm_head=False
        )

    def test_draft_worker_keeps_its_own_warmup(self):
        self.mr.is_draft_worker = True
        autotune.maybe_flashinfer_autotune_extend(self.runner, decode_num_tokens=384)
        self.hook.assert_not_called()
        self.flashinfer_autotune_context.assert_not_called()

    def test_no_extra_pass_when_decode_already_covers_prefill(self):
        autotune.maybe_flashinfer_autotune_extend(self.runner, decode_num_tokens=65536)
        self.hook.assert_not_called()

    def test_other_models_keep_extend_opt_in(self):
        del self.mr.model.autotune_prefill_kernels
        autotune.maybe_flashinfer_autotune_extend(self.runner, decode_num_tokens=384)
        self.flashinfer_autotune_context.assert_not_called()


if __name__ == "__main__":
    unittest.main()
