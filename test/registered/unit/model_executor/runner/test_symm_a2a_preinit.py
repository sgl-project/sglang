"""Unit tests for symmetric-memory DCP A2A runner integration."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.model_executor.runner.base_runner import BaseRunner  # noqa: E402
from sglang.srt.layers.dcp import estimate_symm_a2a_workspace_nbytes  # noqa: E402
from sglang.srt.runtime_context import (  # noqa: E402
    get_context,
    get_parallel,
    reset_context,
)


register_cpu_ci(est_time=2, suite="base-a-test-cpu")


_CUSTOM_AR_V2 = (
    "sglang.srt.distributed.device_communicators.custom_all_reduce_v2."
    "can_use_custom_all_reduce_v2"
)
_SAME_NODE = "sglang.srt.distributed.parallel_state.in_the_same_node_as"
_INIT_WORKSPACE = "sglang.srt.layers.dcp.init_symm_a2a_workspace"
_ESTIMATE_WORKSPACE = "sglang.srt.layers.dcp.estimate_symm_a2a_workspace_nbytes"


class _FakeModelConfig:
    def __init__(self, kv_lora_rank):
        self.kv_lora_rank = kv_lora_rank
        self.attn_tp_size = None

    def get_num_attention_heads(self, attn_tp_size):
        self.attn_tp_size = attn_tp_size
        return 16


class TestSymmA2APreinit(CustomTestCase):
    def setUp(self):
        super().setUp()
        reset_context()
        self.addCleanup(reset_context)

    def _run_preinit(
        self,
        *,
        dcp_size=2,
        backend="symm_a2a",
        can_use=True,
        same_node=(True, True, True, True),
        eager_max_bs=40,
        kv_lora_rank=512,
        capability_probe=None,
        same_node_probe=None,
    ):
        cp_group = SimpleNamespace(cpu_group=object(), world_size=4, rank_in_group=0)
        model_config = _FakeModelConfig(kv_lora_rank)
        init_workspace = MagicMock()
        capability_probe = capability_probe or MagicMock(return_value=can_use)
        same_node_probe = same_node_probe or MagicMock(return_value=list(same_node))

        with (
            get_context().override_server_args(
                dcp_size=dcp_size, dcp_comm_backend=backend
            ) as server_args,
            get_parallel().override(dcp_group=cp_group, attn_tp_size=2),
            patch(_CUSTOM_AR_V2, capability_probe),
            patch(_SAME_NODE, same_node_probe),
            patch(_INIT_WORKSPACE, init_workspace),
        ):
            model_runner = SimpleNamespace(
                server_args=server_args,
                device="cpu",
                dtype=torch.bfloat16,
                model_config=model_config,
                max_decode_logits_rows=lambda: 96,
            )
            runner = SimpleNamespace(
                model_runner=model_runner,
                _eager_max_bs=eager_max_bs,
                _eager_num_tokens_per_req=3,
            )
            BaseRunner._pre_initialize_symm_a2a_workspace(runner)

        return SimpleNamespace(
            cp_group=cp_group,
            model_runner=model_runner,
            model_config=model_config,
            init_workspace=init_workspace,
            capability_probe=capability_probe,
            same_node_probe=same_node_probe,
        )

    def test_gate_is_noop_unless_dcp_symm_a2a_is_configured(self):
        for dcp_size, backend in ((1, "symm_a2a"), (2, "a2a")):
            with self.subTest(dcp_size=dcp_size, backend=backend):
                result = self._run_preinit(dcp_size=dcp_size, backend=backend)
                result.init_workspace.assert_not_called()
                result.capability_probe.assert_not_called()

    def test_initializes_workspace_with_runner_geometry_and_max_decode_tokens(self):
        result = self._run_preinit()

        self.assertEqual(result.model_config.attn_tp_size, 2)
        result.init_workspace.assert_called_once_with(
            result.cp_group,
            device=torch.device("cpu"),
            max_num_tokens=120,
            heads_per_rank=16,
            head_dim=512,
            dtype=torch.bfloat16,
            num_ubatches=1,
        )

    def test_rejects_unsupported_or_multi_node_topology_before_allocation(self):
        for can_use, same_node in (
            (False, (True, True, True, True)),
            (True, (True, False, True, True)),
        ):
            with self.subTest(can_use=can_use, same_node=same_node):
                with self.assertRaisesRegex(RuntimeError, "a2a.*ag_rs|ag_rs.*a2a"):
                    self._run_preinit(can_use=can_use, same_node=same_node)

    def test_capability_failure_still_runs_same_node_collective(self):
        capability_probe = MagicMock(return_value=False)
        same_node_probe = MagicMock(return_value=[True] * 4)

        with self.assertRaises(RuntimeError):
            self._run_preinit(
                capability_probe=capability_probe,
                same_node_probe=same_node_probe,
            )

        capability_probe.assert_called_once()
        same_node_probe.assert_called_once()

    def test_missing_kv_lora_rank_fails_with_mla_specific_message(self):
        with self.assertRaisesRegex(RuntimeError, "MLA.*kv_lora_rank"):
            self._run_preinit(kv_lora_rank=None)

    def test_warns_when_workspace_estimate_exceeds_512_mib(self):
        with (
            patch(_ESTIMATE_WORKSPACE, return_value=512 * 1024**2 + 1) as estimate,
            self.assertLogs(
                "sglang.srt.model_executor.runner.base_runner", level="WARNING"
            ) as logs,
        ):
            result = self._run_preinit()

        estimate.assert_called_once_with(
            world_size=4,
            max_num_tokens=120,
            heads_per_rank=16,
            head_dim=512,
            dtype=torch.bfloat16,
            num_ubatches=1,
        )
        result.init_workspace.assert_called_once()
        self.assertEqual(len(logs.output), 1)
        self.assertIn("512 MiB", logs.output[0])

    def test_runner_uses_shared_workspace_estimator(self):
        with patch(
            _ESTIMATE_WORKSPACE, wraps=estimate_symm_a2a_workspace_nbytes
        ) as estimate:
            self._run_preinit()

        estimate.assert_called_once()

if __name__ == "__main__":
    unittest.main()
