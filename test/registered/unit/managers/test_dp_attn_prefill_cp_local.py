import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.cp.zigzag import ZigzagCPStrategy
from sglang.srt.managers.scheduler_components.dp_attn import (
    MLPSyncBatchInfo,
    _local_prefill_cp_candidate,
    _requires_local_prefill_cp_latch,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestDPPrefillCPLocalLatch(unittest.TestCase):
    def test_policy_is_scoped_to_kimi_k3_cp_v2(self):
        server_args = SimpleNamespace(enable_prefill_cp=True, attn_cp_size=4)
        kimi_config = SimpleNamespace(
            hf_config=SimpleNamespace(architectures=["KimiK3ForConditionalGeneration"]),
            hf_text_config=SimpleNamespace(architectures=["KimiLinearForCausalLM"]),
        )
        qwen_config = SimpleNamespace(
            hf_config=SimpleNamespace(architectures=["Qwen3MoeForCausalLM"]),
            hf_text_config=None,
        )

        with patch(
            "sglang.srt.environ.envs.SGLANG_ENABLE_CP_V2.get", return_value=True
        ):
            self.assertTrue(_requires_local_prefill_cp_latch(server_args, kimi_config))
            self.assertFalse(_requires_local_prefill_cp_latch(server_args, qwen_config))

    def test_each_dp_replica_latches_its_real_local_batch(self):
        eligible = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            extend_lens=[8],
        )
        short = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            extend_lens=[3],
        )
        strategy = ZigzagCPStrategy(cp_size=2)

        with (
            patch(
                "sglang.srt.environ.envs.SGLANG_ENABLE_CP_V2.get",
                return_value=True,
            ),
            patch(
                "sglang.srt.layers.cp.utils.get_cp_strategy",
                return_value=strategy,
            ),
        ):
            self.assertTrue(_local_prefill_cp_candidate(eligible, 8, 4))
            self.assertFalse(_local_prefill_cp_candidate(short, 3, 4))
            self.assertFalse(_local_prefill_cp_candidate(None, 0, 4))

    def test_local_cp_state_is_not_serialized_into_dp_all_gather(self):
        common = dict(
            dp_size=2,
            tp_size=4,
            cp_size=2,
            num_tokens=8,
            num_tokens_for_logprob=1,
            can_run_decode_cuda_graph=False,
            can_run_prefill_cuda_graph=True,
            is_extend_in_batch=True,
            local_can_run_tbo=False,
            local_forward_mode=ForwardMode.EXTEND.value,
        )
        cp_on = MLPSyncBatchInfo(**common, local_prefill_cp_active=True)
        cp_off = MLPSyncBatchInfo(**common, local_prefill_cp_active=False)

        self.assertEqual(cp_on._get_local_tensor("cpu").shape[0], 7)
        self.assertTrue(
            torch.equal(cp_on._get_local_tensor("cpu"), cp_off._get_local_tensor("cpu"))
        )


if __name__ == "__main__":
    unittest.main()
