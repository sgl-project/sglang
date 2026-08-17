import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.scheduler_components import dp_attn
from sglang.srt.managers.scheduler_components.dp_attn import MLPSyncBatchInfo
from sglang.srt.model_executor.forward_batch_info import ForwardMode

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestMLPSyncBatchInfo(CustomTestCase):
    def test_all_gather_single_inactive_slot_uses_fallback_without_alias(self):
        sync_info = MLPSyncBatchInfo(
            dp_size=1,
            tp_size=1,
            cp_size=1,
            num_tokens=8,
            num_tokens_for_logprob=3,
            can_run_decode_cuda_graph=False,
            can_run_prefill_cuda_graph=True,
            is_extend_in_batch=True,
            local_can_run_tbo=True,
            local_forward_mode=ForwardMode.DECODE.value,
        )
        tp_group = SimpleNamespace(active_ranks_cpu=torch.zeros(1, dtype=torch.int32))

        def fake_all_gather_into_tensor(output_tensor, input_tensor, group):
            output_tensor.copy_(input_tensor)

        with (
            patch.object(dp_attn, "get_tp_group", return_value=tp_group),
            patch.object(dp_attn, "_ENABLE_METRICS_DP_ATTENTION", False),
            patch.object(
                torch.distributed,
                "all_gather_into_tensor",
                side_effect=fake_all_gather_into_tensor,
            ),
        ):
            sync_info.all_gather(device="cpu", group=None)

        self.assertEqual(
            sync_info.tp0_info_cpu[0].tolist(),
            [
                0,
                0,
                1,
                0,
                1,
                ForwardMode.IDLE.value,
                0,
            ],
        )
        self.assertEqual(sync_info.global_num_tokens, [0])
        self.assertEqual(sync_info.global_num_tokens_for_logprob, [0])
        self.assertTrue(sync_info.can_run_decode_cuda_graph)
        self.assertFalse(sync_info.is_extend_in_batch)
        self.assertFalse(sync_info.can_run_prefill_cuda_graph)


if __name__ == "__main__":
    unittest.main()
