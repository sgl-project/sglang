import unittest
from unittest.mock import patch

import torch

from sglang.srt.eplb.expert_location_dispatch import ExpertLocationDispatchInfo
from sglang.srt.layers.moe import topk as topk_module
from sglang.srt.layers.moe.topk import TopKConfig, select_experts
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=7, suite="base-a-test-cpu")


class CorrectionBiasCacheTestMixin:
    device: str

    def test_lazy_cache_uses_loaded_value_and_reuses_pointer(self):
        correction_bias = torch.empty(8, dtype=torch.bfloat16, device=self.device)
        config = TopKConfig(top_k=2, correction_bias=correction_bias)

        loaded_value = torch.linspace(-1, 1, 8, dtype=torch.float32).to(
            device=self.device, dtype=torch.bfloat16
        )
        with torch.no_grad():
            correction_bias.copy_(loaded_value)

        converted = config.correction_bias_for_dtype(torch.float32)
        converted_again = config.correction_bias_for_dtype(torch.float32)

        torch.testing.assert_close(converted, loaded_value.float())
        self.assertEqual(converted.data_ptr(), converted_again.data_ptr())
        self.assertNotEqual(converted.data_ptr(), correction_bias.data_ptr())

    def test_cache_refreshes_if_weight_is_reloaded(self):
        correction_bias = torch.zeros(8, dtype=torch.bfloat16, device=self.device)
        config = TopKConfig(top_k=2, correction_bias=correction_bias)
        converted = config.correction_bias_for_dtype(torch.float32)

        reloaded_value = torch.arange(8, dtype=torch.float32).to(
            device=self.device, dtype=torch.bfloat16
        )
        with torch.no_grad():
            correction_bias.copy_(reloaded_value)
        converted_after_reload = config.correction_bias_for_dtype(torch.float32)

        torch.testing.assert_close(converted_after_reload, reloaded_value.float())
        self.assertNotEqual(converted.data_ptr(), converted_after_reload.data_ptr())


class TestCorrectionBiasCacheCPU(CorrectionBiasCacheTestMixin, CustomTestCase):
    device = "cpu"


class TestAiterCorrectionBiasWithFakeDispatch(CustomTestCase):
    """select_experts on the aiter path must not swap the bias that the fake EP
    dispatch zeroed for the config's real bias (#44897)."""

    num_experts = 16

    def _bias_seen_by_aiter(self, bias_dtype, ep_dispatch_algorithm):
        n = self.num_experts
        seen = {}

        def fake_aiter_biased_grouped_topk(gating_output, correction_bias, w, ids, *a):
            seen["bias"] = correction_bias.clone()
            w.zero_()
            ids.zero_()

        info = None
        if ep_dispatch_algorithm is not None:
            info = ExpertLocationDispatchInfo(
                ep_dispatch_algorithm=ep_dispatch_algorithm,
                partial_logical_to_rank_dispatch_physical_map=torch.arange(n),
                partial_logical_to_all_physical_map=torch.arange(n).unsqueeze(1),
                partial_logical_to_all_physical_map_num_valid=torch.ones(
                    n, dtype=torch.int64
                ),
                num_physical_experts=n,
                rank_invariant=True,
            )
        config = TopKConfig(
            top_k=2,
            use_grouped_topk=True,
            topk_group=1,
            num_expert_group=1,
            correction_bias=torch.linspace(1, 2, n).to(bias_dtype),
        )
        with (
            patch.object(topk_module, "_use_aiter", True),
            patch.object(topk_module, "_is_cuda", False),
            patch.object(
                topk_module, "biased_grouped_topk", topk_module.biased_grouped_topk_gpu
            ),
            patch.object(
                topk_module,
                "aiter_biased_grouped_topk",
                fake_aiter_biased_grouped_topk,
                create=True,
            ),
        ):
            select_experts(
                torch.randn(4, 8, dtype=torch.bfloat16),
                torch.randn(4, n, dtype=torch.bfloat16),
                config,
                expert_location_dispatch_info=info,
            )
        return seen["bias"], config.correction_bias

    def test_fake_dispatch_bias_stays_zero(self):
        for bias_dtype in (torch.bfloat16, torch.float32):
            with self.subTest(bias_dtype=bias_dtype):
                seen, _ = self._bias_seen_by_aiter(bias_dtype, "fake")
                self.assertTrue(bool((seen == 0).all()))

    def test_real_dispatch_still_uses_config_bias(self):
        for ep_dispatch_algorithm in (None, "static"):
            with self.subTest(ep_dispatch_algorithm=ep_dispatch_algorithm):
                seen, bias = self._bias_seen_by_aiter(
                    torch.bfloat16, ep_dispatch_algorithm
                )
                torch.testing.assert_close(seen, bias)


if __name__ == "__main__":
    unittest.main()
