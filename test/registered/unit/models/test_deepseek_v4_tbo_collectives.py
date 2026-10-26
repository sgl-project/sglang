import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.batch_overlap.two_batch_overlap import _model_forward_tbo_merge_key
from sglang.srt.layers.dp_attention import DpPaddingMode
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.models import deepseek_v4
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestDeepseekV4TboGate(unittest.TestCase):
    def test_non_ep_topologies(self):
        model = SimpleNamespace(pp_group=SimpleNamespace(world_size=1))
        for dp, cp, padding, cp_active, expected in (
            (2, 1, DpPaddingMode.SUM_LEN, False, True),
            (1, 1, DpPaddingMode.SUM_LEN, False, False),
            (2, 1, DpPaddingMode.MAX_LEN, False, False),
            (2, 2, DpPaddingMode.SUM_LEN, False, False),
            (2, 2, DpPaddingMode.SUM_LEN, True, False),
            (2, 1, None, False, False),
        ):
            with self.subTest(dp=dp, cp=cp, padding=padding, cp_active=cp_active):
                batch = SimpleNamespace(
                    can_run_tbo=True,
                    tbo_children=[object(), object()],
                    global_forward_mode=ForwardMode.EXTEND,
                    dp_padding_mode=padding,
                )
                with (
                    patch("sglang.srt.layers.moe.is_tbo_enabled", return_value=True),
                    patch.object(deepseek_v4, "is_cp_active", return_value=cp_active),
                    patch.object(
                        deepseek_v4,
                        "get_moe_a2a_backend",
                        return_value=SimpleNamespace(is_none=lambda: True),
                    ),
                    patch.object(
                        deepseek_v4,
                        "get_parallel",
                        return_value=SimpleNamespace(attn_dp_size=dp, attn_cp_size=cp),
                    ),
                ):
                    self.assertEqual(
                        deepseek_v4.DeepseekV4Model._can_run_tbo(model, batch), expected
                    )


class TestDeepseekV4TboMerge(unittest.TestCase):
    def test_mhc_outputs_merge_with_padding_and_no_residual(self):
        chunks = (torch.arange(6).reshape(3, 2), torch.arange(6, 10).reshape(2, 2))
        outputs = [
            dict(
                hidden_states=chunk,
                residual=None,
                forward_batch=SimpleNamespace(tbo_parent_token_range=token_range),
            )
            for chunk, token_range in zip(chunks, ((0, 2), (2, 3)))
        ]
        hidden = _model_forward_tbo_merge_key(*outputs, "hidden_states", original_len=3)
        torch.testing.assert_close(hidden, torch.cat((chunks[0][:2], chunks[1][:1])))


if __name__ == "__main__":
    unittest.main()
