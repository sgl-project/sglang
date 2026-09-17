import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.runner import eager_runner
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestEagerDCPVerify(CustomTestCase):
    def _forward(self, mode):
        batch = SimpleNamespace(
            forward_mode=mode,
            input_ids=torch.tensor([2, 3]),
            positions=torch.tensor([4, 5]),
            seq_lens=torch.tensor([6]),
            extend_prefix_lens=(None if mode.is_target_verify() else torch.tensor([4])),
            extend_prefix_lens_cpu=None if mode.is_target_verify() else [4],
            extend_seq_lens=None if mode.is_target_verify() else torch.tensor([2]),
            req_pool_indices=torch.tensor([0]),
            seq_lens_sum=6,
            attn_dcp_metadata=None,
            needs_forward_metadata_init=lambda: not mode.is_target_verify(),
        )

        def prepare_prefix(_seq_lens, prefix_lens, *_args):
            return torch.cumsum(prefix_lens, dim=0)

        def init_metadata(batch):
            batch.planned_positions = batch.positions.clone()

        def forward(input_ids, positions, batch):
            return input_ids + batch.planned_positions

        runner = eager_runner.EagerRunner.__new__(eager_runner.EagerRunner)
        runner.enable_pdmux = False
        runner.load_batch = lambda batch, *_args: batch
        runner.model_runner = SimpleNamespace(
            ps=SimpleNamespace(attn_dcp_size=8),
            model=SimpleNamespace(
                prepare_context_parallel_metadata_for_dcp=prepare_prefix,
                forward=forward,
            ),
            attn_backend=SimpleNamespace(
                init_forward_metadata=init_metadata,
                prepare_prefill_shared_read_snapshot=lambda *args, **kwargs: None,
            ),
            _extend_forward_kwargs=lambda *args: {},
            device="cpu",
            kv_cache_dtype=torch.bfloat16,
            device_timer=None,
            prefill_cuda_graph_runner=None,
        )
        with (
            patch.object(eager_runner, "is_cp_active", return_value=False),
            patch.object(eager_runner, "maybe_publish_prefill_shared_read_done"),
            patch.object(
                eager_runner,
                "get_req_to_token_pool",
                return_value=SimpleNamespace(req_to_token=torch.zeros((1, 8))),
            ),
            patch.object(
                eager_runner,
                "get_token_to_kv_pool",
                return_value=SimpleNamespace(
                    get_kv_buffer_shape=lambda: [torch.Size([8, 1, 4])]
                ),
            ),
        ):
            output = runner.execute(batch)
        torch.testing.assert_close(output, torch.tensor([6, 8]))
        return batch

    def test_verify_without_extend_prefix_metadata(self):
        """DCP verification must run when no prefill prefix lengths exist."""
        batch = self._forward(ForwardMode.TARGET_VERIFY)
        self.assertIsNone(batch.attn_dcp_metadata)

    def test_prefill_keeps_prefix_metadata(self):
        batch = self._forward(ForwardMode.EXTEND)
        torch.testing.assert_close(batch.attn_dcp_metadata, torch.tensor([4]))


if __name__ == "__main__":
    unittest.main()
