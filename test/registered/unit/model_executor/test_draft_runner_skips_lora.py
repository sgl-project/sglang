"""The draft runner must not build a LoRA manager.

Adapters apply to the target model only. Every worker is handed the same
published ServerArgs, so `enable_lora` is True for the draft too -- the
decision is the runner's own, keyed on is_draft_worker. Without it the draft
tries to load the target's adapters into the draft model, whose layer count
differs, and startup fails inside LoRAAdapter weight loading.

The LoRA paths downstream then key on `lora_manager is not None` rather than
the config, so a draft runner skips them by construction.
"""

import unittest
from unittest.mock import patch

from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class TestDraftRunnerSkipsLoRA(CustomTestCase):
    def _init_lora_called(self, *, is_draft_worker: bool, enable_lora: bool) -> bool:
        runner = ModelRunner.__new__(ModelRunner)
        runner.is_draft_worker = is_draft_worker
        runner.lora_manager = None
        with patch.object(ModelRunner, "init_lora_manager") as init_lora:
            with patch("sglang.srt.model_executor.model_runner.get_lora") as get_lora:
                get_lora.return_value.enable_lora = enable_lora
                runner.maybe_init_lora_manager()
        return init_lora.called

    def test_only_the_target_runner_builds_a_lora_manager(self):
        cases = [
            (False, True, True),  # target + LoRA -> builds one
            (True, True, False),  # draft + LoRA -> must not
            (False, False, False),  # LoRA off -> nobody builds one
        ]
        for is_draft_worker, enable_lora, expected in cases:
            with self.subTest(draft=is_draft_worker, enable_lora=enable_lora):
                self.assertEqual(
                    self._init_lora_called(
                        is_draft_worker=is_draft_worker, enable_lora=enable_lora
                    ),
                    expected,
                )


if __name__ == "__main__":
    unittest.main()
