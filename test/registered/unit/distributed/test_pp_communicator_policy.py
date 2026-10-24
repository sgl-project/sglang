"""Regression: default PP preserves backend-specific communicator policy."""

import unittest
from contextlib import ExitStack
from unittest.mock import Mock, patch

from sglang.srt.distributed import parallel_state
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, publish_build_topology

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestPPCommunicatorPolicy(CustomTestCase):
    def test_default_transport_preserves_platform_policy(self):
        """Opting out of Gloo staging must not enable NCCL on unsupported platforms."""
        for platform, backend, expected in [
            ("cuda", "nccl", True),
            ("npu", "hccl", False),
            ("xpu", "xccl", False),
            ("cuda", "mooncake", False),
        ]:
            with self.subTest(platform=platform, backend=backend):
                self._check_policy(platform, backend, "default", expected)
        self._check_policy("cuda", "nccl", "gloo", False)

    def _check_policy(self, platform, backend, transport, expected):
        from sglang.srt.runtime_context import reset_context

        groups = {}

        def coordinator(*, group_name, group_ranks, use_pynccl, **kwargs):
            group = Mock()
            group.world_size = len(group_ranks[0])
            group.use_pynccl = use_pynccl
            group.backend = kwargs["torch_distributed_backend"]
            groups[group_name] = group
            return group

        with ExitStack() as stack:
            stack.callback(reset_context)
            for name in (
                "_WORLD",
                "_TP",
                "_ATTN_CP",
                "_ATTN_TP",
                "_PP",
                "_SELF_PP",
                "_MOE_EP",
                "_MOE_TP",
                "_MOE_DP",
                "_DCP",
                "_SHARED_EXPERTS_TP",
            ):
                stack.enter_context(patch.object(parallel_state, name, None))
            stack.enter_context(
                patch.object(parallel_state, "_is_npu", platform == "npu")
            )
            stack.enter_context(
                patch.object(parallel_state, "_is_xpu", platform == "xpu")
            )
            for method, value in [
                ("is_initialized", True),
                ("get_world_size", 2),
                ("get_rank", 0),
                ("get_backend", backend),
            ]:
                stack.enter_context(
                    patch("torch.distributed." + method, return_value=value)
                )
            world = Mock(local_rank=0)
            stack.enter_context(
                patch.object(parallel_state, "get_world_group", return_value=world)
            )
            stack.enter_context(
                patch.object(
                    parallel_state, "GroupCoordinator", side_effect=coordinator
                )
            )
            publish_build_topology(
                tp_size=1, pp_size=2, device="cuda", pp_transport=transport
            )
            parallel_state.initialize_model_parallel()
            self.assertIs(groups["pp"].use_pynccl, expected)
            self.assertEqual(
                groups["pp"].backend, "gloo" if transport == "gloo" else backend
            )
            self.assertEqual(groups["tp"].backend, backend)


if __name__ == "__main__":
    unittest.main()
