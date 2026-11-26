import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.eplb import expert_distribution as records
from sglang.srt.eplb.expert_location import ExpertLocationMetadata
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


class TestIntervalRecording(unittest.TestCase):
    def test_install_updates_selected_rows_in_place(self):
        # Independent CPU mirrors exercise the same writes as a device table.
        def metadata(offset):
            physical = (torch.arange(4).repeat(6, 1) + offset) % 4
            replicas = torch.full((6, 4, 4), -1)
            replicas[..., 0] = physical.argsort()
            return ExpertLocationMetadata(
                physical,
                physical.clone(),
                replicas,
                replicas.clone(),
                (replicas != -1).sum(-1),
                2,
                replicas[..., 0].clone(),
            )

        for selected in (list(range(6)), [4, 2, 3, 4], [0, 2, 5], [], [-1, 8]):
            with self.subTest(selected=selected):
                active, proposal = metadata(0), metadata(1)
                # Capture the cached tensors before installation, as a graph would.
                maps = [active.mega_gate_maps(key) for key in ("static", "dynamic")]
                fields = [
                    key for key, value in vars(active).items() if torch.is_tensor(value)
                ]
                before = {key: getattr(active, key).clone() for key in fields}
                ptrs = {key: getattr(active, key).data_ptr() for key in fields}
                active.update(proposal, selected)
                for key in fields:
                    result = getattr(active, key)
                    self.assertEqual(result.data_ptr(), ptrs[key])
                    for row in range(6):
                        expected = (
                            getattr(proposal, key) if row in selected else before[key]
                        )
                        torch.testing.assert_close(result[row], expected[row])
                torch.testing.assert_close(
                    maps[0][0][..., 0],
                    active.logical_to_rank_dispatch_physical_map.int(),
                )
                torch.testing.assert_close(
                    maps[1][0], active.logical_to_all_physical_map.int()
                )
                torch.testing.assert_close(
                    maps[1][1], active.logical_to_all_physical_map_num_valid.int()
                )

    def make_accumulator(self, policy):
        exec_config = SimpleNamespace(
            moe=SimpleNamespace(
                eplb_interval_policy=policy,
                expert_distribution_recorder_buffer_size=3,
                eplb_min_rebalancing_utilization_threshold=1.0,
            )
        )
        metadata = SimpleNamespace(
            num_layers=1,
            num_physical_experts=3,
            num_logical_experts=2,
            physical_to_logical_map=torch.tensor([[0, 1, 0]]),
        )
        with (
            patch.object(records, "get_exec", return_value=exec_config),
            patch.object(
                records,
                "get_device_namespace",
                return_value=SimpleNamespace(device="cpu"),
            ),
            patch.object(records, "get_eplb_group", return_value=None),
            patch.object(records, "reports_expert_balancedness", return_value=False),
        ):
            return records._StatAccumulator(metadata, 0)

    def test_peek_across_install_and_wrap(self):
        acc = self.make_accumulator("auto")
        recorder = records._ExpertDistributionRecorderReal.__new__(
            records._ExpertDistributionRecorderReal
        )
        recorder._accumulator = acc
        for t in range(4):
            if t == 2:
                acc._expert_location_metadata.physical_to_logical_map.copy_(
                    torch.tensor([[1, 0, 1]])
                )
            acc.append(
                t,
                "primary",
                {
                    "global_physical_count": torch.tensor(
                        [[t + 1, 0, 0]], dtype=torch.int32
                    )
                },
                {},
            )
        window = recorder.peek_interval_window()
        expected = torch.tensor([[[2, 0]], [[0, 3]], [[0, 4]]])
        torch.testing.assert_close(window.sample_recent(3, 3).long(), expected)
        # Simulate SUM with an identical peer. A second peek must not sum the peer twice.
        with (
            patch.object(
                torch.distributed, "all_reduce", side_effect=lambda t, **kw: t.mul_(2)
            ),
            patch.object(records, "_dump_to_file") as export,
            patch.object(
                torch,
                "get_device_module",
                return_value=SimpleNamespace(empty_cache=lambda: None),
            ),
        ):
            first = recorder.dump_record(output_mode="object")["logical_count"].clone()
            compact = recorder.counts_for_interval_rebalance()
            torch.testing.assert_close(compact, first.sum(dim=0))
            recorder.dump_record(output_mode="file")
            second = recorder.dump_record(output_mode="object")["logical_count"]
        self.assertEqual(export.call_args.args[1]["valid_steps"], 3)
        torch.testing.assert_close(export.call_args.args[1]["logical_count"], second)
        self.assertTrue(torch.equal(first, second))
        torch.testing.assert_close(second.long(), expected * 2)
        self.assertEqual(second.sum().item(), 18)
        self.assertTrue(torch.equal(window.sample_recent(3, 3).long(), expected))

    def test_window_sum_exceeds_int32(self):
        acc = self.make_accumulator("auto")
        for t in range(2):
            acc.append(
                t,
                "primary",
                {
                    "global_physical_count": torch.tensor(
                        [[2**30, 0, 0]], dtype=torch.int32
                    )
                },
                {},
            )
        self.assertEqual(acc.interval_window().sum_steps()[0, 0].item(), 2**31)

    def test_manager_reuses_live_plan(self):
        from sglang.srt.eplb import eplb_manager

        manager = eplb_manager.EPLBManager.__new__(eplb_manager.EPLBManager)
        manager._rebalance_disabled_reason = manager._rebalance_layers_per_chunk = None
        manager.interval_controller = Mock()
        self.assertIs(manager.active_interval_controller, manager.interval_controller)
        manager._rebalance_disabled_reason = "paused"
        self.assertIsNone(manager.active_interval_controller)
        manager._rebalance_disabled_reason = None
        manager.interval_controller.advance.return_value = True
        manager._get_model = lambda: SimpleNamespace(
            routed_experts_weights_of_layer={0: ()}
        )
        manager._get_expert_backup_client = manager._get_expert_location_updater = (
            lambda: None
        )
        manager._get_weight_updater = Mock()
        manager._compute_expert_location_metadata = Mock()
        recorder = Mock()
        with (
            patch.object(
                eplb_manager,
                "get_global_expert_distribution_recorder",
                return_value=recorder,
            ),
            patch.object(
                eplb_manager.ElasticEPStateManager, "instance", return_value=None
            ),
            patch.object(
                eplb_manager, "get_parallel", return_value=SimpleNamespace(tp_rank=0)
            ),
            patch.object(
                eplb_manager,
                "get_exec",
                return_value=SimpleNamespace(
                    moe=SimpleNamespace(ep_dispatch_algorithm="static")
                ),
            ),
            patch.object(torch, "get_device_module", return_value=Mock()),
            patch.object(
                eplb_manager, "update_expert_location_with_recovery"
            ) as install,
        ):
            next(manager._adaptive_entrypoint())
        recorder.counts_for_interval_rebalance.assert_called_once()
        recorder.dump_record.assert_not_called()
        manager._compute_expert_location_metadata.assert_called_once_with(
            recorder.counts_for_interval_rebalance.return_value,
            broadcast_over_world=False,
        )
        ms, used_recorder, proposed = (
            manager.interval_controller.finish_check.call_args.args
        )
        self.assertGreaterEqual(ms, 0)
        self.assertIs(used_recorder, recorder)
        self.assertIs(
            proposed, install.call_args.kwargs["new_expert_location_metadata"]
        )

    def test_fixed_dump_resets(self):
        acc = self.make_accumulator("fixed")
        recorder = records._ExpertDistributionRecorderReal.__new__(
            records._ExpertDistributionRecorderReal
        )
        recorder._accumulator = acc
        recorder._current_layer_idx = SimpleNamespace(value=None)
        recorder._single_pass_gatherers = {}
        acc.append(
            0, "primary", {"global_physical_count": torch.tensor([[1, 2, 3]])}, {}
        )
        with (
            patch.object(torch.distributed, "all_reduce"),
            patch.object(
                torch,
                "get_device_module",
                return_value=SimpleNamespace(empty_cache=lambda: None),
            ),
        ):
            self.assertEqual(
                recorder.dump_record(output_mode="object")["logical_count"]
                .sum()
                .item(),
                6,
            )
            self.assertEqual(
                recorder.dump_record(output_mode="object")["logical_count"]
                .sum()
                .item(),
                0,
            )


if __name__ == "__main__":
    unittest.main()
