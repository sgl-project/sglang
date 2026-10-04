import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.layers.moe.topk import TopKOutputFormat
from sglang.srt.models.deepseek_v2 import DeepseekV2MoE
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _Experts:
    def __init__(self, inplace):
        self.moe_runner_config = SimpleNamespace(inplace=inplace)
        self.quant_method = object()

    def __call__(self, hidden, topk):
        return hidden.mul_(3) if self.moe_runner_config.inplace else hidden * 3


class TestDeepseekSharedExpertInput(unittest.TestCase):
    def test_shared_expert_reads_original_input_after_inplace_routed_write(self):
        for inplace in (True, False):
            with self.subTest(inplace=inplace):
                hidden = torch.tensor([[2.0, 4.0]])
                expected = hidden * 5
                shared = Mock(side_effect=lambda value, *_args, **_kwargs: value * 2)
                model = SimpleNamespace(
                    alt_stream=Mock(),
                    experts=_Experts(inplace),
                    _maybe_quant_moe_input_once=lambda _: None,
                    _should_quant_routed_input_mxfp8=lambda _: False,
                    num_fused_shared_experts=0,
                    is_nextn=False,
                    gate=lambda *_: torch.zeros(1, 1),
                    topk=Mock(
                        return_value=SimpleNamespace(format=TopKOutputFormat.STANDARD)
                    ),
                    _fuse_finalize_all_reduce=False,
                    _shared_expert_tp1=False,
                    routed_scaling_factor=1.0,
                    reduce_results=False,
                    _forward_shared_experts=shared,
                )
                with (
                    patch(
                        "sglang.srt.models.deepseek_v2.get_forward",
                        return_value=SimpleNamespace(flashinfer_trtllm_bypass=False),
                    ),
                    patch(
                        "sglang.srt.models.deepseek_v2.get_exec",
                        return_value=SimpleNamespace(
                            moe=SimpleNamespace(enable_eplb=False)
                        ),
                    ),
                    patch("sglang.srt.models.deepseek_v2._is_hip", False),
                    patch("sglang.srt.models.deepseek_v2._is_cuda", True),
                    patch(
                        "sglang.srt.models.deepseek_v2.torch.cuda.current_stream",
                        return_value=Mock(),
                    ),
                    patch(
                        "sglang.srt.models.deepseek_v2.torch.cuda.stream",
                        side_effect=lambda _: nullcontext(),
                    ),
                    patch(
                        "sglang.srt.models.deepseek_v2.maybe_fuse_routed_scale_and_shared_add",
                        side_effect=lambda _, routed, shared, scale: routed + shared,
                    ),
                ):
                    actual = DeepseekV2MoE.forward_normal_dual_stream(model, hidden)
                torch.testing.assert_close(actual, expected)
                shared_input = shared.call_args.args[0]
                self.assertEqual(
                    shared_input.data_ptr() == hidden.data_ptr(), not inplace
                )


if __name__ == "__main__":
    unittest.main()
