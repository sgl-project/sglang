import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.batch_overlap.two_batch_overlap import _model_forward_tbo_merge_key
from sglang.srt.layers.dp_attention import DpPaddingMode
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.models import deepseek_v4
from sglang.srt.models.deepseek_common import moe_transport
from sglang.srt.models.deepseek_v4 import _tbo_collective_sizes
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestDeepseekV4TboCollectiveSizes(unittest.TestCase):
    def test_attention_tp_shards_preserve_dp_layout(self):
        dp_sizes, tp_sizes = _tbo_collective_sizes(
            [8, 8, 8, 8, 12, 12, 12, 12], attn_tp_size=4
        )

        self.assertEqual(dp_sizes, [8, 12])
        self.assertEqual(tp_sizes, [2, 2, 2, 2, 3, 3, 3, 3])
        self.assertEqual(sum(dp_sizes), sum(tp_sizes))

    def test_rejects_inconsistent_attention_tp_replicas(self):
        with self.assertRaisesRegex(ValueError, "differ within attention TP"):
            _tbo_collective_sizes([8, 8, 4, 8], attn_tp_size=4)

    def test_rejects_unshardable_token_count(self):
        with self.assertRaisesRegex(ValueError, "not divisible"):
            _tbo_collective_sizes([6, 6, 6, 6], attn_tp_size=4)


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


class TestDeepseekV4TboGather(unittest.TestCase):
    def test_tp1_uses_dp_gather_and_tp4_gathers_one_shard(self):
        local = torch.arange(32).reshape(8, 4)
        global_hidden = torch.empty(16, 4)
        layer = SimpleNamespace(
            mlp=SimpleNamespace(shared_experts=None, _shared_expert_tp1=False)
        )
        for tp_size in (1, 4):
            with self.subTest(attn_tp_size=tp_size):
                group = Mock(rank_in_group=0)
                parallel = SimpleNamespace(
                    attn_tp_size=tp_size, attn_tp_rank=0, tp_group=group
                )
                batch = SimpleNamespace(_tbo_tp_sizes=[8 // tp_size] * (2 * tp_size))
                with (
                    patch.object(moe_transport, "get_parallel", return_value=parallel),
                    patch.object(
                        moe_transport, "get_global_dp_buffer_len", return_value=16
                    ),
                    patch.object(
                        moe_transport,
                        "get_tbo_persistent_buffer",
                        return_value=global_hidden,
                    ),
                    patch.object(
                        moe_transport, "get_dp_tbo_comm_stream", return_value=Mock()
                    ),
                    patch.object(moe_transport, "_tbo_event", return_value=Mock()),
                    patch("torch.cuda.current_stream", return_value=Mock()),
                    patch("torch.cuda.stream", return_value=nullcontext()),
                    patch.object(moe_transport, "dp_gather_replicate") as dp_gather,
                ):
                    _, _, _, keepalive = moe_transport._tbo_gather_moe_input(
                        layer.mlp, local, batch, 0, False
                    )
                if tp_size == 1:
                    dp_gather.assert_called_once_with(global_hidden, local, batch)
                    group.all_gatherv.assert_not_called()
                else:
                    dp_gather.assert_not_called()
                    group.all_gatherv.assert_called_once()
                    args, kwargs = group.all_gatherv.call_args
                    torch.testing.assert_close(args[0], local[:2])
                    self.assertEqual(kwargs["sizes"], [2] * 8)
                    self.assertIs(kwargs["output"], global_hidden)
                self.assertIs(keepalive, local if tp_size == 1 else args[0])


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
