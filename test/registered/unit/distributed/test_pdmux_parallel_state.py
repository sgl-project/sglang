"""PDMux uses distinct lane communicators with identical rank topology."""

import contextlib
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.distributed import parallel_state
from sglang.srt.runtime_context import get_parallel, reset_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import publish_build_topology

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestPDMuxParallelGroups(unittest.TestCase):
    def _initialize(self, *, dp_size, duplicate=True, rank=0):
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
            "_PDMUX_PREFILL_ATTN_TP_GROUP",
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
        publish_build_topology(tp_size=8, pp_size=1, ep_size=1, attn_dp_size=dp_size)
        parallel_state.initialize_model_parallel(
            backend="nccl", duplicate_tp_group=duplicate
        )
        self.addCleanup(parallel_state.destroy_model_parallel)
        return created, arguments

    def test_tp8_dp2_duplicates_the_attention_subgroup(self):
        for rank in (0, 4):
            with self.subTest(rank=rank):
                created, arguments = self._initialize(dp_size=2, rank=rank)
                expected = [list(range(4)), list(range(4, 8))]
                self.assertEqual(created["attention_tp"][0], expected)
                self.assertEqual(created["pdmux_prefill_attention_tp"][0], expected)
                decode = created["attention_tp"][1]
                prefill = created["pdmux_prefill_attention_tp"][1]
                self.assertIsNot(decode, prefill)
                for key in (
                    "use_pynccl",
                    "use_custom_allreduce",
                    "use_torch_symm_mem_allreduce",
                ):
                    self.assertEqual(
                        arguments["attention_tp"][key],
                        arguments["pdmux_prefill_attention_tp"][key],
                    )
                with parallel_state.pdmux_prefill_tp_group():
                    self.assertIs(get_parallel().attn_tp_group, prefill)
                    self.assertEqual(get_parallel().attn_tp_group.ranks, decode.ranks)
                    self.assertIs(
                        get_parallel().tp_group, created["pdmux_prefill_tp"][1]
                    )
                self.assertIs(get_parallel().attn_tp_group, decode)
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
        prefill = created["pdmux_prefill_attention_tp"][1]
        with get_parallel().override(
            shared_experts_tp_size=4, shared_experts_tp_group=decode
        ):
            with self.assertRaisesRegex(RuntimeError, "stop lane"):
                with parallel_state.pdmux_prefill_tp_group():
                    self.assertIs(get_parallel().attn_tp_group, prefill)
                    self.assertIs(get_parallel().shared_experts_tp_group, prefill)
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
