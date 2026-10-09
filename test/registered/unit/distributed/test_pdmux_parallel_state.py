"""PDMux uses distinct lane communicators with identical rank topology."""

import contextlib
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.distributed import parallel_state
from sglang.srt.layers.moe.utils import post_experts_all_reduce
from sglang.srt.runtime_context import get_parallel, reset_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import publish_build_topology

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestPDMuxParallelGroups(unittest.TestCase):
    def _initialize(self, *, dp_size, ep_size=1, duplicate=True, rank=0):
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        self.addCleanup(reset_context)
        for name in (
            "_TP",
            "_ATTN_TP",
            "_ATTN_CP",
            "_MOE_EP",
            "_MOE_DP",
            "_MOE_TP",
            "_SHARED_EXPERTS_TP",
            "_DCP",
            "_PP",
            "_SELF_PP",
            "_PDMUX_PREFILL_TP_GROUP",
        ):
            stack.enter_context(patch.object(parallel_state, name, None))
        world = SimpleNamespace(
            ranks=list(range(8)),
            world_size=8,
            rank_in_group=rank,
            local_rank=rank,
            device_group=object(),
        )
        stack.enter_context(patch.object(parallel_state, "_WORLD", world))
        stack.enter_context(
            patch("torch.distributed.is_initialized", return_value=True)
        )
        stack.enter_context(patch("torch.distributed.get_world_size", return_value=8))
        stack.enter_context(patch("torch.distributed.get_rank", return_value=rank))
        created = {}
        arguments = {}

        def initialize(group_ranks, local_rank, backend, **kwargs):
            name = kwargs["group_name"]
            ranks = next(ranks for ranks in group_ranks if rank in ranks)
            group = SimpleNamespace(
                ranks=ranks,
                world_size=len(ranks),
                rank_in_group=ranks.index(rank),
                local_rank=local_rank,
                device_group=object(),
                pynccl_comm=None,
                destroy=Mock(),
            )
            created[name] = (group_ranks, group)
            arguments[name] = kwargs
            return group

        stack.enter_context(
            patch.object(
                parallel_state, "init_model_parallel_group", side_effect=initialize
            )
        )
        publish_build_topology(
            tp_size=8, pp_size=1, ep_size=ep_size, attn_dp_size=dp_size
        )
        parallel_state.initialize_model_parallel(
            backend="nccl", duplicate_tp_group=duplicate
        )
        self.addCleanup(parallel_state.destroy_model_parallel)
        return created, arguments

    def test_tp8_dp2_prefill_keeps_independent_attention_group(self):
        for rank in (0, 4):
            with self.subTest(rank=rank):
                created, _ = self._initialize(dp_size=2, rank=rank)
                self.assertEqual(
                    created["attention_tp"][0], [list(range(4)), list(range(4, 8))]
                )
                self.assertNotIn("pdmux_prefill_attention_tp", created)
                attention = created["attention_tp"][1]
                decode = created["tp"][1]
                prefill = created["pdmux_prefill_tp"][1]
                with parallel_state.pdmux_prefill_tp_group():
                    self.assertIs(get_parallel().tp_group, prefill)
                    self.assertIs(get_parallel().attn_tp_group, attention)
                    self.assertIs(get_parallel().moe_tp_group, prefill)
                self.assertIs(get_parallel().tp_group, decode)
                self.assertIs(get_parallel().attn_tp_group, attention)
                parallel_state.destroy_model_parallel()
                prefill.destroy.assert_called_once_with()

    def test_tp8_dp8_keeps_singleton_attention_groups(self):
        created, _ = self._initialize(dp_size=8)
        self.assertNotIn("pdmux_prefill_attention_tp", created)
        decode_attention = created["attention_tp"][1]
        self.assertEqual(decode_attention.ranks, [0])
        with parallel_state.pdmux_prefill_tp_group():
            self.assertIs(get_parallel().attn_tp_group, decode_attention)
            self.assertIs(get_parallel().moe_tp_group, created["pdmux_prefill_tp"][1])

    def test_tp8_dp8_ep8_public_moe_reduction_follows_prefill_alias(self):
        for rank in range(8):
            with self.subTest(rank=rank):
                created, _ = self._initialize(dp_size=8, ep_size=8, rank=rank)
                decode = created["tp"][1]
                prefill = created["pdmux_prefill_tp"][1]
                decode.all_reduce = Mock(side_effect=lambda tensor: tensor)
                prefill.all_reduce = Mock(side_effect=lambda tensor: tensor)
                hidden = torch.zeros(8, 4)
                if rank in (2, 5, 6, 7):
                    hidden[rank] = 1
                self.assertIs(get_parallel().moe_ep_group, decode)
                with patch(
                    "sglang.srt.layers.moe.utils.should_skip_post_experts_all_reduce",
                    return_value=False,
                ):
                    post_experts_all_reduce(hidden)
                    with parallel_state.pdmux_prefill_tp_group():
                        self.assertIs(get_parallel().moe_ep_group, prefill)
                        post_experts_all_reduce(hidden)
                        with parallel_state.pdmux_prefill_tp_group():
                            self.assertIs(get_parallel().moe_ep_group, prefill)
                        self.assertIs(get_parallel().moe_ep_group, prefill)
                    post_experts_all_reduce(hidden)
                self.assertIs(get_parallel().moe_ep_group, decode)
                self.assertEqual(decode.all_reduce.call_count, 2)
                prefill.all_reduce.assert_called_once_with(hidden)
                parallel_state.destroy_model_parallel()
                prefill.destroy.assert_called_once_with()

    def test_derived_moe_groups_are_shared_without_new_communicators(self):
        created, _ = self._initialize(dp_size=2, ep_size=2)
        ep, tp = get_parallel().moe_ep_group, get_parallel().moe_tp_group
        with parallel_state.pdmux_prefill_tp_group():
            self.assertIs(get_parallel().moe_ep_group, ep)
            self.assertIs(get_parallel().moe_tp_group, tp)
        self.assertEqual(
            [name for name in created if name.startswith("pdmux_prefill_")],
            ["pdmux_prefill_tp"],
        )

    def test_full_tp_aliases_restore_after_exception(self):
        created, _ = self._initialize(dp_size=1)
        decode = created["tp"][1]
        prefill = created["pdmux_prefill_tp"][1]
        self.assertNotIn("pdmux_prefill_attention_tp", created)
        with self.assertRaisesRegex(RuntimeError, "stop lane"):
            with parallel_state.pdmux_prefill_tp_group():
                self.assertIs(get_parallel().tp_group, prefill)
                self.assertIs(get_parallel().attn_tp_group, prefill)
                self.assertIs(get_parallel().moe_tp_group, prefill)
                raise RuntimeError("stop lane")
        self.assertIs(get_parallel().tp_group, decode)
        self.assertIs(get_parallel().attn_tp_group, decode)
        self.assertIs(get_parallel().moe_tp_group, decode)

    def test_independent_attention_alias_and_exception_restore(self):
        created, _ = self._initialize(dp_size=2)
        decode = created["attention_tp"][1]
        with get_parallel().override(
            shared_experts_tp_size=4, shared_experts_tp_group=decode
        ):
            with self.assertRaisesRegex(RuntimeError, "stop lane"):
                with parallel_state.pdmux_prefill_tp_group():
                    self.assertIs(get_parallel().attn_tp_group, decode)
                    self.assertIs(get_parallel().shared_experts_tp_group, decode)
                    raise RuntimeError("stop lane")
            self.assertIs(get_parallel().attn_tp_group, decode)
            self.assertIs(get_parallel().shared_experts_tp_group, decode)

    def test_ordinary_dp_does_not_create_lane_groups(self):
        created, _ = self._initialize(dp_size=2, duplicate=False)
        self.assertNotIn("pdmux_prefill_tp", created)
        self.assertNotIn("pdmux_prefill_attention_tp", created)
        self.assertFalse(parallel_state.is_pdmux_enabled())


if __name__ == "__main__":
    unittest.main()
