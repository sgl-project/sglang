import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.layers.layernorm import RMSNorm  # noqa: E402
from sglang.srt.models.dflash import DFlashDraftModel  # noqa: E402
from sglang.srt.model_executor.pool_configurator import MemoryPoolConfig  # noqa: E402
from sglang.srt.speculative.dspark_components.dspark_worker_v2 import (  # noqa: E402
    DSparkWorkerV2,
)

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestDSparkPPContext(CustomTestCase):
    def test_partial_projection_sum_matches_full_projection(self):
        """PP partial projections must preserve the full pre-norm FC result."""
        torch.manual_seed(0)
        hidden_size = 4
        model = DFlashDraftModel.__new__(DFlashDraftModel)
        torch.nn.Module.__init__(model)
        model.config = SimpleNamespace(hidden_size=hidden_size)
        model.num_context_features = 3
        model.fc = torch.nn.Linear(3 * hidden_size, hidden_size, bias=False)
        model.hidden_norm = RMSNorm(hidden_size, eps=1e-6)

        feature_hidden = [
            torch.randn(5, hidden_size, dtype=torch.float32) for _ in range(3)
        ]
        full_hidden = torch.cat(feature_hidden, dim=-1)
        full_projected = model.project_target_hidden(full_hidden)

        stage_0 = model.project_target_hidden_partial(
            torch.cat([feature_hidden[0], feature_hidden[2]], dim=-1),
            [0, 2],
        )
        stage_1 = model.project_target_hidden_partial(feature_hidden[1], [1])
        pp_projected = model.hidden_norm(stage_0 + stage_1)

        torch.testing.assert_close(pp_projected, full_projected)

    def test_non_last_pp_prefill_uses_minimal_draft_kv_pool(self):
        """A context-only PP rank must not reserve the full draft KV capacity."""
        worker = DSparkWorkerV2.__new__(DSparkWorkerV2)
        worker._draft_worker = Mock()
        worker._is_pd_prefill = True
        worker._draft_is_moe = False
        worker.ps = SimpleNamespace(pp_rank=0, pp_size=2)
        worker.page_size = 64
        full_config = MemoryPoolConfig(
            max_total_num_tokens=4096,
            max_running_requests=32,
        )

        worker.alloc_memory_pool(memory_pool_config=full_config)

        passed_config = worker._draft_worker.alloc_memory_pool.call_args.kwargs[
            "memory_pool_config"
        ]
        self.assertEqual(passed_config.max_total_num_tokens, 64)
        self.assertEqual(passed_config.max_running_requests, 32)
        self.assertEqual(full_config.max_total_num_tokens, 4096)

    def test_last_pp_prefill_keeps_full_draft_kv_pool(self):
        worker = DSparkWorkerV2.__new__(DSparkWorkerV2)
        worker._draft_worker = Mock()
        worker._is_pd_prefill = True
        worker._draft_is_moe = False
        worker.ps = SimpleNamespace(pp_rank=1, pp_size=2)
        worker.page_size = 64
        full_config = MemoryPoolConfig(
            max_total_num_tokens=4096,
            max_running_requests=32,
        )

        worker.alloc_memory_pool(memory_pool_config=full_config)

        passed_config = worker._draft_worker.alloc_memory_pool.call_args.kwargs[
            "memory_pool_config"
        ]
        self.assertIs(passed_config, full_config)


if __name__ == "__main__":
    unittest.main()
