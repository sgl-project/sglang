import types
import unittest
from unittest.mock import patch

import torch

from sglang.srt.layers import flashinfer_comm_fusion as fusion
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, stage="unit-test", runner_config="1-gpu-small")


class _FakeWorkspace:
    backend = "fake"

    def is_buffer_size_sufficient(self, **_kwargs):
        return True


class _FakeFlashInferComm:
    class AllReduceFusionPattern:
        kARResidualRMSNorm = object()

    def __init__(self):
        self.calls = []

    def create_allreduce_fusion_workspace(self, **kwargs):
        self.calls.append(kwargs)
        workspace = _FakeWorkspace()
        workspace.backend = kwargs["backend"]
        return workspace


class TestFlashInferCommFusion(unittest.TestCase):
    def test_auto_backend_resolves_by_arch(self):
        args = types.SimpleNamespace(flashinfer_allreduce_fusion_backend="auto")

        with patch.object(fusion, "is_sm100_supported", return_value=True):
            self.assertEqual(
                fusion.resolve_flashinfer_allreduce_fusion_backend(args), "mnnvl"
            )

        with patch.object(fusion, "is_sm100_supported", return_value=False):
            self.assertEqual(
                fusion.resolve_flashinfer_allreduce_fusion_backend(args), "trtllm"
            )

    def test_workspace_creation_for_supported_backends(self):
        fake_comm = _FakeFlashInferComm()
        original_comm = fusion._flashinfer_comm
        original_create = fusion._create_allreduce_fusion_workspace
        original_group_support = fusion._flashinfer_create_workspace_supports_group
        original_comm_backend_support = (
            fusion._flashinfer_create_workspace_supports_comm_backend
        )
        try:
            fusion._flashinfer_comm = fake_comm
            fusion._create_allreduce_fusion_workspace = (
                fake_comm.create_allreduce_fusion_workspace
            )
            fusion._flashinfer_create_workspace_supports_group = True
            fusion._flashinfer_create_workspace_supports_comm_backend = True

            for backend in ("trtllm", "mnnvl"):
                manager = fusion.FlashInferWorkspaceManager()
                with (
                    patch.object(fusion, "_preflight_check_workspace_memory", return_value=True),
                    patch.object(fusion, "in_the_same_node_as", return_value=[True] * 4),
                ):
                    manager.initialize(
                        world_size=4,
                        rank=0,
                        max_token_num=8,
                        hidden_dim=16,
                        backend=backend,
                        dtype=torch.float16,
                    )
                self.assertTrue(manager.initialized)
                self.assertEqual(fake_comm.calls[-1]["backend"], backend)
                self.assertEqual(fake_comm.calls[-1]["gpus_per_node"], 4)
        finally:
            fusion._flashinfer_comm = original_comm
            fusion._create_allreduce_fusion_workspace = original_create
            fusion._flashinfer_create_workspace_supports_group = original_group_support
            fusion._flashinfer_create_workspace_supports_comm_backend = (
                original_comm_backend_support
            )


if __name__ == "__main__":
    unittest.main()
