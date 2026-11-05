"""CPU fault injection for the TP-voted expert-set transaction."""

import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.eplb.expert_swap import plan_expert_swap
from sglang.srt.eplb.expert_swap_transaction import hot_swap_experts
from sglang.srt.eplb.expert_swap_weights import ExpertRowPatch
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _Metadata:
    def __init__(self, mapping):
        self.mapping = torch.tensor([mapping])
        self.num_logical_experts = 3
        self._mega_gate_maps = {}

    def update(self, other, update_layer_ids):
        self.mapping.copy_(other.mapping)


class TestExpertSwapTransaction(CustomTestCase):
    def fixture(self, stack, peer_prepare_error=False):
        module = SimpleNamespace(
            w13_weight=torch.zeros(2, 4, 2), w2_weight=torch.zeros(2, 2, 2)
        )
        row = ExpertRowPatch(
            module,
            1,
            torch.zeros(4, 2),
            torch.zeros(2, 2),
            torch.ones(4, 2),
            torch.full((2, 2), 2.0),
        )
        metadata = _Metadata([0, 1])
        context = dict(
            plan=plan_expert_swap([[0, 1]], [[0, 2]], 3),
            modules={0: module},
            metadata=metadata,
            old=_Metadata([0, 1]),
            new=_Metadata([0, 2]),
            available=2**30,
            runner=SimpleNamespace(model_config=SimpleNamespace(model_path="unused")),
        )
        scheduler = SimpleNamespace(_engine_paused=False)
        scheduler.pause_generation = Mock(
            side_effect=lambda req: setattr(scheduler, "_engine_paused", True)
        )
        scheduler.continue_generation = Mock(
            side_effect=lambda req: setattr(scheduler, "_engine_paused", False)
        )
        manager = SimpleNamespace(
            tp_cpu_group=object(),
            scheduler=scheduler,
            flush_cache=Mock(return_value=True),
            hot_swap_failed=False,
        )
        votes = []

        def gather(output, error, group):
            votes.append(error)
            output[0] = (
                "peer prepare rejected"
                if peer_prepare_error and len(votes) == 3
                else error
            )

        prefix = "sglang.srt.eplb.expert_swap_transaction."
        stack.enter_context(patch(prefix + "_prepare_context", return_value=context))
        stack.enter_context(
            patch(prefix + "prepare_expert_rows", return_value=([row], 192))
        )
        stack.enter_context(
            patch(
                prefix + "get_parallel",
                return_value=SimpleNamespace(moe_tp_rank=0, moe_tp_size=1),
            )
        )
        stack.enter_context(patch("torch.cuda.synchronize"))
        stack.enter_context(patch("torch.distributed.get_world_size", return_value=1))
        stack.enter_context(
            patch("torch.distributed.all_gather_object", side_effect=gather)
        )
        return manager, context, module, row, votes

    def test_success_changes_real_rows_and_publishes_after_prepare_votes(self):
        with ExitStack() as stack:
            manager, context, module, row, votes = self.fixture(stack)
            pointers = (module.w13_weight.data_ptr(), module.w2_weight.data_ptr())
            hot_swap_experts(manager, [[0, 2]], [[0, 1]], memory_reserve_bytes=0)
            self.assertEqual(context["metadata"].mapping.tolist(), [[0, 2]])
            self.assertTrue(torch.equal(module.w13_weight[1], row.new_w13))
            self.assertTrue(torch.equal(module.w2_weight[1], row.new_w2))
            self.assertEqual(
                (module.w13_weight.data_ptr(), module.w2_weight.data_ptr()), pointers
            )
            self.assertEqual(len(votes), 7)
            manager.scheduler.continue_generation.assert_called_once()

    def test_prepare_rejection_never_changes_weights_or_metadata(self):
        with ExitStack() as stack:
            manager, context, module, row, _ = self.fixture(
                stack, peer_prepare_error=True
            )
            stack.enter_context(
                patch.object(
                    row, "copy_new", side_effect=AssertionError("must not write")
                )
            )
            with self.assertRaisesRegex(RuntimeError, "old weights retained"):
                hot_swap_experts(manager, [[0, 2]], [[0, 1]], memory_reserve_bytes=0)
            self.assertEqual(module.w13_weight.sum().item(), 0.0)
            self.assertEqual(context["metadata"].mapping.tolist(), [[0, 1]])
            manager.flush_cache.assert_not_called()
            manager.scheduler.continue_generation.assert_called_once()

    def test_partial_copy_failure_restores_old_data_and_mapping(self):
        with ExitStack() as stack:
            manager, context, module, row, _ = self.fixture(stack)

            def partial_write():
                module.w13_weight[1].fill_(9)
                raise RuntimeError("injected copy failure")

            stack.enter_context(
                patch.object(row, "copy_new", side_effect=partial_write)
            )
            with self.assertRaisesRegex(RuntimeError, "rolled back"):
                hot_swap_experts(manager, [[0, 2]], [[0, 1]], memory_reserve_bytes=0)
            self.assertEqual(module.w13_weight.sum().item(), 0.0)
            self.assertEqual(module.w2_weight.sum().item(), 0.0)
            self.assertEqual(context["metadata"].mapping.tolist(), [[0, 1]])
            self.assertFalse(manager.hot_swap_failed)
            manager.scheduler.continue_generation.assert_called_once()

    def test_rollback_failure_prevents_resume(self):
        with ExitStack() as stack:
            manager, _, _, row, _ = self.fixture(stack)
            stack.enter_context(
                patch.object(row, "copy_new", side_effect=RuntimeError("copy failed"))
            )
            stack.enter_context(
                patch.object(
                    row, "restore", side_effect=RuntimeError("rollback failed")
                )
            )
            with self.assertRaisesRegex(RuntimeError, "HOT_SWAP_UNSAFE"):
                hot_swap_experts(manager, [[0, 2]], [[0, 1]], memory_reserve_bytes=0)
            self.assertTrue(manager.hot_swap_failed)
            manager.scheduler.continue_generation.assert_not_called()

    def test_every_consensus_round_failure_holds_generation(self):
        for fail_round in range(1, 8):
            with self.subTest(fail_round=fail_round), ExitStack() as stack:
                manager, context, module, _, _ = self.fixture(stack)
                rounds = []

                def fail_gather(output, error, group):
                    rounds.append(error)
                    if len(rounds) == fail_round:
                        raise RuntimeError("injected Gloo failure")
                    output[0] = error

                stack.enter_context(
                    patch(
                        "torch.distributed.all_gather_object", side_effect=fail_gather
                    )
                )
                with self.assertRaisesRegex(RuntimeError, "HOT_SWAP_UNSAFE") as raised:
                    hot_swap_experts(
                        manager, [[0, 2]], [[0, 1]], memory_reserve_bytes=0
                    )
                self.assertTrue(manager.hot_swap_failed)
                self.assertTrue(manager.scheduler._engine_paused)
                if fail_round < 7:
                    manager.scheduler.continue_generation.assert_not_called()
                if fail_round == 1:
                    self.assertIn("preflight-before-mutation", str(raised.exception))
                    self.assertEqual(module.w13_weight.sum().item(), 0.0)
                    self.assertEqual(context["metadata"].mapping.tolist(), [[0, 1]])
                if fail_round == 7:
                    manager.scheduler.continue_generation.assert_called_once()

    def test_rollback_consensus_failure_also_holds_generation(self):
        with ExitStack() as stack:
            manager, _, _, row, _ = self.fixture(stack)
            stack.enter_context(
                patch.object(row, "copy_new", side_effect=RuntimeError("copy failed"))
            )
            rounds = []

            def fail_gather(output, error, group):
                rounds.append(error)
                if len(rounds) == 6:
                    raise RuntimeError("rollback vote failed")
                output[0] = error

            stack.enter_context(
                patch("torch.distributed.all_gather_object", side_effect=fail_gather)
            )
            with self.assertRaisesRegex(RuntimeError, "HOT_SWAP_UNSAFE: rollback"):
                hot_swap_experts(manager, [[0, 2]], [[0, 1]], memory_reserve_bytes=0)
            self.assertTrue(manager.hot_swap_failed)
            manager.scheduler.continue_generation.assert_not_called()

    def test_pause_bookkeeping_failure_is_fail_closed_before_any_weight_write(self):
        with ExitStack() as stack:
            manager, context, module, _, _ = self.fixture(stack)

            def bad_pause(req):
                manager.scheduler._engine_paused = True
                raise RuntimeError("pause bookkeeping failed")

            manager.scheduler.pause_generation.side_effect = bad_pause
            with self.assertRaisesRegex(RuntimeError, "HOT_SWAP_UNSAFE: pause"):
                hot_swap_experts(manager, [[0, 2]], [[0, 1]], memory_reserve_bytes=0)
            self.assertTrue(manager.hot_swap_failed)
            self.assertTrue(manager.scheduler._engine_paused)
            manager.scheduler.continue_generation.assert_not_called()
            self.assertEqual(module.w13_weight.sum().item(), 0.0)
            self.assertEqual(context["metadata"].mapping.tolist(), [[0, 1]])


if __name__ == "__main__":
    unittest.main()
