import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.kernels.ops.speculative.topk1 import draft_topk1_postprocess
from sglang.srt.layers.mtp_head_tp import (
    _ineligible_reason,
    configure_draft_head_tp,
    head_tp4_rank_groups,
    local_argmax_pair,
    merge_argmax_pairs,
)
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
register_cuda_ci(est_time=15, stage="base-b", runner_config="1-gpu-small")


class TestMTPHeadTopology(unittest.TestCase):
    def test_node_local_groups_and_fallback(self):
        for size in (4, 8, 16):
            hosts = [(r, f"node{r // 4}", r % 4) for r in range(size)]
            expected = [list(range(i, i + 4)) for i in range(0, size, 4)]
            self.assertEqual(head_tp4_rank_groups(hosts), expected)
            hosts[-1] = (size - 1, "wrong-host", 3)
            self.assertIsNone(head_tp4_rank_groups(hosts))
        self.assertIsNone(head_tp4_rank_groups([(r, "node", 0) for r in range(4)]))

    def test_megakernel_draft_skips_head_tp(self):
        for entries, step_fold in ((True, False), (False, True)):
            model = SimpleNamespace()
            with (
                mock.patch(
                    "sglang.srt.runtime_context.get_parallel",
                    return_value=SimpleNamespace(
                        pp_size=1,
                        tp_size=4,
                        tp_group=SimpleNamespace(cpu_group=None, rank=0),
                    ),
                ),
                mock.patch("sglang.srt.runtime_context.get_spec"),
                mock.patch(
                    "sglang.srt.speculative.eagle_megakernel._offers_eagle_entries",
                    return_value=entries,
                ),
                mock.patch(
                    "sglang.srt.kernel_plugins.nvidia.binding.megakernel_entry",
                    side_effect=lambda _, name: (
                        object() if step_fold and name == "draft_step_fold" else None
                    ),
                ),
                mock.patch("torch.distributed.all_reduce"),
                self.assertLogs("sglang.srt.layers.mtp_head_tp", "INFO") as logs,
            ):
                configure_draft_head_tp(model, None)
            self.assertFalse(hasattr(model, "draft_head_tp"))
            self.assertIn("megakernel", logs.output[0])


class TestMTPHeadLayouts(unittest.TestCase):
    def reason(self, enable_dp_lm_head, head_tp_size, attn_tp_size=1):
        parallel = SimpleNamespace(
            pp_size=1,
            tp_size=4,
            attn_dp_size=4 // attn_tp_size,
            attn_tp_size=attn_tp_size,
            enable_dp_lm_head=enable_dp_lm_head,
        )
        spec = SimpleNamespace(
            speculative_eagle_topk=1,
            speculative_use_rejection_sampling=False,
            speculative_adaptive=False,
        )
        lm_head = SimpleNamespace(
            weight=SimpleNamespace(
                is_cuda=True, dtype=torch.bfloat16, ndim=2, shape=(248320, 8)
            ),
            tp_size=head_tp_size,
            num_added_embeddings=0,
        )
        lp = SimpleNamespace(
            use_fp32_lm_head=False,
            logit_scale=None,
            final_logit_softcapping=None,
            return_full_logits=False,
        )
        model = SimpleNamespace(
            lm_head=lm_head, logits_processor=lp, config=SimpleNamespace(vocab_size=248320)
        )
        with (
            mock.patch(
                "sglang.srt.speculative.eagle_megakernel._offers_eagle_entries",
                return_value=False,
            ),
            mock.patch(
                "sglang.srt.kernel_plugins.nvidia.binding.megakernel_entry",
                return_value=None,
            ),
            mock.patch(
                "sglang.srt.runtime_context.get_disagg",
                return_value=SimpleNamespace(disaggregation_mode="decode"),
            ),
            mock.patch(
                "sglang.srt.runtime_context.get_lora",
                return_value=SimpleNamespace(enable_lora=False),
            ),
        ):
            return _ineligible_reason(model, None, parallel, spec)

    def test_replicated_and_stock_sharded_heads_are_eligible(self):
        self.assertIsNone(self.reason(enable_dp_lm_head=True, head_tp_size=1))
        self.assertIsNone(self.reason(enable_dp_lm_head=False, head_tp_size=4))

    def test_other_layouts_decline(self):
        self.assertIn("not replicated", self.reason(True, 4))
        self.assertIn("not sharded", self.reason(False, 1))
        self.assertIn("pure attention DP", self.reason(False, 4, attn_tp_size=2))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestMTPHeadTP(unittest.TestCase):
    def test_four_shard_argmax(self):
        for vocab in (3, 32773):
            torch.manual_seed(168)
            logits = torch.randn((10, vocab), device="cuda", dtype=torch.bfloat16)
            logits[0].fill_(1)
            logits[1].fill_(-float("inf"))
            logits[2, [0, vocab - 1]] = float("inf")
            logits[3, [1, vocab - 1]] = float("nan")
            logits[4].fill_(float("nan"))
            logits[5].fill_(-float("inf"))
            logits[5, -1] = float("nan")
            logits[6].fill_(-1e30)
            logits[6, -1] = float("nan")
            logits[7].fill_(0)
            logits[7, 0] = 1
            logits[7, -1] = torch.nextafter(logits.new_tensor(1), logits.new_tensor(2))
            for sentinel in (False, True):
                expected = torch.argmax(logits, -1, keepdim=True)
                if sentinel:
                    _, expected = draft_topk1_postprocess(
                        logits, torch.zeros(10, device="cuda", dtype=torch.int64)
                    )
                width = (vocab + 3) // 4
                pairs = []
                for rank in range(4):
                    start, end = (
                        min(rank * width, vocab),
                        min((rank + 1) * width, vocab),
                    )
                    padded = torch.full((10, width + 3), float("nan"), device="cuda")
                    padded[:, : end - start] = logits[:, start:end]
                    pairs.append(
                        local_argmax_pair(padded, start, end - start, sentinel)
                    )
                actual = merge_argmax_pairs(torch.stack(pairs))
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                empty = torch.stack(
                    [local_argmax_pair(logits[:0], draft_nan_sentinel=sentinel)] * 4
                )
                self.assertEqual(tuple(merge_argmax_pairs(empty).shape), (0, 1))


if __name__ == "__main__":
    unittest.main()
