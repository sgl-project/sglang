"""Unsupported speculative modes must not add per-layer padding copies."""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.layers.moe.moe_runner import flashinfer_cutlass as cutlass
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.speculative.ragged_verify import RaggedVerifyMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


class TestHopperPaddingScope(unittest.TestCase):
    def test_only_static_dspark_rows_use_padding_copy(self):
        # Native arithmetic is covered by the eight-rank Humming test. Here
        # only the native boundary and device query are replaced, to observe
        # the real runner's dispatch without CUDA allocations.
        x = torch.empty((48, 5120), dtype=torch.bfloat16)
        ids = torch.zeros((48, 6), dtype=torch.int32)
        dispatch = StandardDispatchOutput(
            hidden_states=x,
            hidden_states_scale=None,
            topk_output=StandardTopKOutput(
                topk_weights=torch.ones((48, 6)),
                topk_ids=ids,
                router_logits=torch.empty((48, 384)),
            ),
        )
        quant = cutlass.FlashInferCutlassMxfp4MoeQuantInfo(
            w13_weight=torch.empty((48, 4), dtype=torch.uint8),
            w2_weight=torch.empty((48, 4), dtype=torch.uint8),
            w13_weight_scale=torch.empty((48, 4), dtype=torch.uint8),
            w2_weight_scale=torch.empty((48, 4), dtype=torch.uint8),
            w13_humming_residual_scale=torch.ones(48),
            w2_humming_residual_scale=torch.ones(48),
            humming_fc2_act_scale=torch.ones(()),
        )
        config = MoeRunnerConfig(
            layer=SimpleNamespace(
                moe_tp_size=1, moe_tp_rank=0, moe_ep_size=8, moe_ep_rank=0
            )
        )
        parallel = SimpleNamespace(
            tp_size=8,
            attn_dp_size=8,
            moe_ep_size=8,
            attn_tp_size=1,
            attn_cp_size=1,
            tp_group=None,
        )
        activation = SimpleNamespace(Swiglu=object(), SwigluStep=object())
        for algorithm, width, mode, wanted in (
            ("DSPARK", 6, RaggedVerifyMode.STATIC, True),
            (None, None, RaggedVerifyMode.STATIC, False),
            ("EAGLE", 6, RaggedVerifyMode.STATIC, False),
            ("DSPARK", 4, RaggedVerifyMode.STATIC, False),
            ("DSPARK", 6, RaggedVerifyMode.CAP_ACCEPT, False),
            ("DSPARK", 6, RaggedVerifyMode.COMPACT, False),
        ):
            with (
                self.subTest(algorithm=algorithm, width=width, mode=mode),
                mock.patch.object(
                    cutlass,
                    "get_spec",
                    return_value=SimpleNamespace(
                        speculative_algorithm=algorithm,
                        speculative_num_draft_tokens=width,
                    ),
                ),
                mock.patch.object(cutlass, "get_parallel", return_value=parallel),
                mock.patch.object(
                    cutlass, "read_ragged_verify_mode", return_value=mode
                ),
                mock.patch.object(
                    cutlass,
                    "_flashinfer_cutlass_fused_moe",
                    return_value=(mock.Mock(), activation),
                ),
                mock.patch.object(
                    cutlass, "use_symmetric_memory", return_value=nullcontext()
                ),
                mock.patch.object(
                    cutlass, "is_allocation_symmetric", return_value=False
                ),
                mock.patch.object(
                    cutlass, "_cutlass_moe_tune_max_num_tokens", return_value=48
                ),
                mock.patch.object(
                    torch.cuda, "get_device_capability", return_value=(9, 0)
                ),
                mock.patch.object(
                    cutlass.envs.SGLANG_FLASHINFER_MOE_FUSED_FINALIZE,
                    "get",
                    return_value=False,
                ),
                mock.patch(
                    "sglang.srt.layers.dp_attention.copy_mask_dp_moe_padding",
                    return_value=ids,
                ) as copied,
            ):
                cutlass._fused_experts_flashinfer_mxfp4_cutlass(dispatch, quant, config)
                self.assertEqual(copied.call_count, int(wanted))


if __name__ == "__main__":
    unittest.main()
