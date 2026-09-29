from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.arg_groups.overrides import resolved_view
from sglang.srt.arg_groups.parallel_hook import handle_elastic_ep
from sglang.srt.elastic_ep.runtime_topology import (
    ElasticEPRecoveryRequiredError,
    RuntimeTopology,
)
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    CudaGraphConfig,
    PhaseConfig,
)
from sglang.srt.server_args import ServerArgs

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _args(allocation_index: int, **kwargs) -> ServerArgs:
    values = {
        "model_path": "dummy",
        "elastic_ep_backend": "mooncake",
        "ep_join_mode": "auto",
        "elastic_ep_allocation_index": allocation_index,
        "elastic_ep_allocation_id": f"pod-uid-{allocation_index}",
        "elastic_ep_initial_size": 8,
        "max_ep_size": 16,
        "dist_init_addr": "10.0.0.1:20000",
        "enable_dp_lm_head": True,
        "moe_a2a_backend": "nixl",
        "cuda_graph_config": CudaGraphConfig(
            decode=PhaseConfig(backend=Backend.DISABLED),
            prefill=PhaseConfig(backend=Backend.DISABLED),
        ),
        "load_balance_method": "round_robin",
    }
    values.update(kwargs)
    return ServerArgs(**values)


def _topology(
    *, initial: int = 8, effective: int = 8, maximum: int = 8
) -> RuntimeTopology:
    return RuntimeTopology(
        runtime_instance_id="runtime-1",
        initial_ep_size=initial,
        allocation_width=4,
        effective_ep_size=effective,
        max_committed_ep_size=maximum,
    )


@patch(
    "sglang.srt.arg_groups.parallel_hook._elastic_ep_visible_device_count",
    return_value=4,
)
class TestElasticEPAutoBootstrap(CustomTestCase):
    def test_initial_allocations_form_one_world(self, _device_count):
        for index in (0, 1):
            with (
                self.subTest(index=index),
                patch(
                    "sglang.srt.elastic_ep.runtime_topology.probe_runtime_topology",
                    return_value=None,
                ),
            ):
                server_args = _args(index)

                handle_elastic_ep(server_args)

                cfg = resolved_view(server_args)
                self.assertIsNone(cfg.ep_join_mode)
                self.assertEqual(cfg.ep_join_rank_offset, 0)
                self.assertEqual(cfg.tp_size, 8)
                self.assertEqual(cfg.dp_size, 1)
                self.assertEqual(cfg.attn_dp_size, 8)
                self.assertEqual(cfg.ep_size, 8)
                self.assertEqual(cfg.node_rank, index)
                self.assertEqual(cfg.nnodes, 2)
                self.assertEqual(cfg.elastic_ep_allocation_width, 4)
                self.assertEqual(
                    cfg.elastic_ep_runtime_instance_id is not None,
                    index == 0,
                )

    def test_later_allocation_resolves_to_append_joiner(self, _device_count):
        with patch(
            "sglang.srt.elastic_ep.runtime_topology.probe_runtime_topology",
            return_value=_topology(),
        ):
            server_args = _args(2)

            handle_elastic_ep(server_args)

        cfg = resolved_view(server_args)
        self.assertEqual(cfg.ep_join_mode, "scale")
        self.assertEqual(cfg.elastic_ep_allocation_index, 2)
        self.assertEqual(cfg.elastic_ep_allocation_id, "pod-uid-2")
        self.assertEqual(cfg.tp_size, 4)
        self.assertEqual(cfg.dp_size, 1)
        self.assertEqual(cfg.attn_dp_size, 4)
        self.assertEqual(cfg.ep_size, 4)
        self.assertEqual(cfg.ep_join_rank_offset, 8)
        self.assertEqual(cfg.node_rank, 1)
        self.assertEqual(cfg.nnodes, 2)
        self.assertEqual(cfg.elastic_ep_runtime_instance_id, "runtime-1")

    def test_single_allocation_initial_world_can_append(self, _device_count):
        with patch(
            "sglang.srt.elastic_ep.runtime_topology.probe_runtime_topology",
            return_value=_topology(initial=4, effective=4, maximum=4),
        ):
            server_args = _args(1, elastic_ep_initial_size=4)

            handle_elastic_ep(server_args)

        cfg = resolved_view(server_args)
        self.assertEqual(cfg.ep_join_mode, "scale")
        self.assertEqual(cfg.ep_join_rank_offset, 4)
        self.assertEqual(cfg.tp_size, 4)
        self.assertEqual(cfg.dp_size, 1)
        self.assertEqual(cfg.attn_dp_size, 4)
        self.assertEqual(cfg.ep_size, 4)

    def test_initial_slot_restart_requires_recovery(self, _device_count):
        with (
            patch(
                "sglang.srt.elastic_ep.runtime_topology.probe_runtime_topology",
                return_value=_topology(),
            ),
            self.assertRaisesRegex(
                ElasticEPRecoveryRequiredError,
                "already formed runtime",
            ),
        ):
            handle_elastic_ep(_args(1))

    def test_previously_occupied_append_slot_requires_recovery(self, _device_count):
        with (
            patch(
                "sglang.srt.elastic_ep.runtime_topology.probe_runtime_topology",
                return_value=_topology(effective=8, maximum=12),
            ),
            self.assertRaisesRegex(
                ElasticEPRecoveryRequiredError,
                "previously occupied",
            ),
        ):
            handle_elastic_ep(_args(2))

    def test_initial_size_must_be_divisible_by_allocation_width(self, _device_count):
        with self.assertRaisesRegex(AssertionError, "must be divisible"):
            handle_elastic_ep(_args(0, elastic_ep_initial_size=6))

    def test_automatic_mode_rejects_explicit_geometry(self, _device_count):
        with self.assertRaisesRegex(ValueError, "--tp-size=4"):
            handle_elastic_ep(_args(0, tp_size=4))

    def test_automatic_mode_rejects_device_selection(self, _device_count):
        with self.assertRaisesRegex(ValueError, "--base-gpu-id=1"):
            handle_elastic_ep(_args(0, base_gpu_id=1))

    def test_automatic_mode_derives_attention_dp(self, _device_count):
        server_args = _args(0)

        handle_elastic_ep(server_args)

        cfg = resolved_view(server_args)
        self.assertEqual(cfg.dp_size, 1)
        self.assertEqual(cfg.attn_dp_size, 8)
        self.assertFalse(cfg.enable_dp_attention)

    def test_automatic_mode_rejects_explicit_attention_dp_geometry(self, _device_count):
        with self.assertRaisesRegex(ValueError, "--attn-dp-size=4"):
            handle_elastic_ep(_args(0, attn_dp_size=4))

    def test_automatic_mode_rejects_attention_context_parallelism(self, _device_count):
        with self.assertRaisesRegex(AssertionError, "--attn-cp-size 1"):
            handle_elastic_ep(_args(0, attn_cp_size=2))


if __name__ == "__main__":
    import unittest

    unittest.main()
