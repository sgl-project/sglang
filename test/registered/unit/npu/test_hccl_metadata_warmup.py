import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from sglang.srt.hardware_backend.npu.graph_runner.hccl_warmup import (
    prepare_hccl_before_metadata,
)
from sglang.srt.layers.moe.token_dispatcher.deepep import DeepEPDispatcher
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestHcclWarmup(unittest.TestCase):
    def test_aliased_groups_and_largest_expert_set(self):
        log = []

        def dispatcher(n):
            d = Mock(spec=DeepEPDispatcher)
            d.prebuild_pending = True
            d._low_latency_dispatcher = SimpleNamespace(num_experts=n)
            d.prebuild.side_effect = lambda: log.append(n)
            return d

        small, big = dispatcher(8), dispatcher(32)
        model = Mock()
        model.modules.return_value = [
            SimpleNamespace(dispatcher=SimpleNamespace(_inners=[small, big]))
        ]
        a, b = object(), object()

        def group(g):
            return SimpleNamespace(device_group=g)

        parallel = SimpleNamespace(
            tp_group=group(a), attn_tp_group=group(a), moe_ep_group=group(b)
        )
        with (
            patch.object(torch, "zeros"),
            patch.object(torch.distributed, "get_world_size", return_value=2),
            patch.object(
                torch.distributed,
                "all_reduce",
                side_effect=lambda _, group: log.append(group),
            ),
            patch.object(torch, "npu", create=True) as npu,
        ):
            npu.synchronize.side_effect = lambda: log.append("sync")
            prepare_hccl_before_metadata(model, parallel)
        self.assertEqual(log, [a, b, "sync", 32, "sync"])


if __name__ == "__main__":
    unittest.main()
