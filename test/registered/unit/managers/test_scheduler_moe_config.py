"""CPU unit tests for when the scheduler seeds the MoE runtime flags."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import json
import os
import tempfile
import unittest
from contextlib import ExitStack
from unittest import mock

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.layers.moe.utils import (
    MoeRunnerBackend,
    draft_model_build_scope,
    get_moe_runner_backend,
    is_moe_runner_backend_explicit,
    speculative_moe_backend_context,
)
from sglang.srt.managers import scheduler as scheduler_module
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.test_utils import CustomTestCase

_DENSE = {
    "model_type": "qwen3",
    "architectures": ["Qwen3ForCausalLM"],
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 1,
    "num_attention_heads": 2,
    "num_key_value_heads": 1,
    "head_dim": 32,
    "vocab_size": 128,
}
_MOE = {
    **_DENSE,
    "model_type": "qwen3_moe",
    "architectures": ["Qwen3MoeForCausalLM"],
    "moe_intermediate_size": 32,
    "num_experts": 4,
    "num_experts_per_tok": 2,
}


class TestSchedulerMoeConfig(CustomTestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.addCleanup(reset_context)
        self.paths = {}
        for name, config in (("dense", _DENSE), ("moe", _MOE)):
            self.paths[name] = os.path.join(tmp.name, name)
            os.makedirs(self.paths[name])
            with open(os.path.join(self.paths[name], "config.json"), "w") as f:
                json.dump(config, f)

    def _init_dense_target_with_draft(self, *, draft):
        reset_context()
        server_args = ServerArgs(
            model_path=self.paths["dense"],
            device="cpu",
            moe_runner_backend="flashinfer_cutlass",
            speculative_algorithm="STANDALONE",
            speculative_draft_model_path=self.paths[draft],
        )
        publish(server_args, role="test")
        scheduler = object.__new__(Scheduler)
        scheduler.server_args = server_args
        scheduler.model_config = ModelConfig.from_server_args(server_args)
        with ExitStack() as stack:
            for name in (
                "initialize_fp8_gemm_config",
                "initialize_fp4_gemm_config",
                "initialize_bf16_gemm_config",
                "require_mlp_sync",
            ):
                stack.enter_context(mock.patch.object(scheduler_module, name))
            scheduler.init_moe_gemm_config()

    def test_moe_draft_of_a_dense_target_runs_the_users_runner(self):
        """With a dense target the MoE flags were never seeded, so an MoE draft
        built its layers with auto, as not passed explicitly, even under
        --moe-runner-backend flashinfer_cutlass. A STANDALONE draft is built with
        the target's runner; an EAGLE draft inside the speculative context."""
        self._init_dense_target_with_draft(draft="moe")
        for worker, contexts in (
            ("standalone", (draft_model_build_scope,)),
            ("eagle", (speculative_moe_backend_context, draft_model_build_scope)),
        ):
            with self.subTest(worker=worker), ExitStack() as built:
                for context in contexts:
                    built.enter_context(context())
                self.assertEqual(
                    get_moe_runner_backend(), MoeRunnerBackend.FLASHINFER_CUTLASS
                )
                self.assertTrue(
                    is_moe_runner_backend_explicit(
                        backend=MoeRunnerBackend.FLASHINFER_CUTLASS
                    )
                )

    def test_dense_draft_of_a_dense_target_leaves_the_moe_flags_unset(self):
        self._init_dense_target_with_draft(draft="dense")
        self.assertEqual(get_moe_runner_backend(), MoeRunnerBackend.AUTO)
        self.assertFalse(is_moe_runner_backend_explicit(backend=MoeRunnerBackend.AUTO))


if __name__ == "__main__":
    unittest.main()
