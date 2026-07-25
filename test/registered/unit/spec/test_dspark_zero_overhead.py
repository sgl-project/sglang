import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.kernels.ops.speculative.dspark.dspark_schedule import (
    ScheduleVerifyLensTopk,
)
from sglang.srt.environ import envs
from sglang.srt.speculative.dflash_info import DFlashVerifyInput
from sglang.srt.speculative.dspark_components.dspark_draft import DraftBlockProposer
from sglang.srt.speculative.dspark_components.dspark_observability import (
    DsparkStepObservers,
    InfoComponent,
)
from sglang.srt.speculative.dspark_components.dspark_planner import (
    DSparkScheduleConfig,
    DSparkVerifyPlanner,
)
from sglang.srt.speculative.dspark_components.dspark_verify import (
    TargetVerifyExecutor,
)
from sglang.srt.speculative.dspark_components.dspark_worker_v2 import (
    DSparkWorkerV2,
)
from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout, RaggedVerifyMode
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=10, stage="stage-b", runner_config="1-gpu-small-amd")


class _FakeBroadcastGroup:
    def __init__(self, payload: torch.Tensor, *, rank_in_group: int = 1):
        self.rank_in_group = rank_in_group
        self.payload = payload
        self.calls = 0
        self.ranks = [0, 1]
        self.cpu_group = object()

    def broadcast(self, tensor: torch.Tensor, src: int) -> None:
        self.calls += 1
        self.assert_source = src
        if self.rank_in_group != src:
            tensor.copy_(self.payload)


class TestDSparkPlannerZeroOverhead(unittest.TestCase):
    def test_tp_dynamic_tier_is_deterministic_without_cpu_collective(self):
        planner = object.__new__(DSparkVerifyPlanner)
        planner.server_args = SimpleNamespace(tp_size=2)
        planner.verify_num_draft_tokens = 4
        group = _FakeBroadcastGroup(torch.empty(0, dtype=torch.int32))
        batch = SimpleNamespace(
            spec_verify_tier_num_tokens=-1, batch_size=lambda: 2
        )

        with (
            mock.patch(
                "sglang.srt.speculative.dspark_components.dspark_planner."
                "verify_lens_broadcast_group",
                return_value=(group, 2),
            ),
            mock.patch(
                "sglang.srt.speculative.dspark_components.dspark_planner."
                "is_dp_attention_enabled",
                return_value=False,
            ),
            mock.patch(
                "sglang.srt.speculative.dspark_components.dspark_planner."
                "torch.distributed.broadcast",
                side_effect=AssertionError("pure TP must not coordinate on CPU"),
            ),
            mock.patch.object(planner, "_maybe_gather_dp_verify_tier") as gather,
        ):
            planner._coordinate_verify_tier(
                batch=batch, local_tier_num_tokens=3
            )

        self.assertEqual(batch.spec_verify_tier_num_tokens, 8)
        gather.assert_called_once_with(batch=batch, local_tier_num_tokens=8)

    def test_tp_prepare_path_uses_no_cpu_collective(self):
        planner = object.__new__(DSparkVerifyPlanner)
        planner.server_args = SimpleNamespace(tp_size=4)
        planner.verify_num_draft_tokens = 4
        planner._budget_planner = SimpleNamespace(forced_budget_frac=0.5)
        planner._is_verify_all = True
        planner._dp_tier_gather_enabled = False
        group = _FakeBroadcastGroup(torch.empty(0, dtype=torch.int32))
        batch = SimpleNamespace(
            spec_info=None,
            spec_verify_tier_num_tokens=-1,
            batch_size=lambda: 2,
        )

        with (
            mock.patch(
                "sglang.srt.speculative.dspark_components.dspark_planner."
                "verify_lens_broadcast_group",
                return_value=(group, 4),
            ),
            mock.patch(
                "sglang.srt.speculative.dspark_components.dspark_planner."
                "is_dp_attention_enabled",
                return_value=False,
            ),
            mock.patch(
                "sglang.srt.speculative.dspark_components.dspark_planner."
                "torch.distributed.broadcast",
                side_effect=AssertionError("pure TP must not broadcast on CPU"),
            ),
            mock.patch(
                "sglang.srt.speculative.dspark_components.dspark_planner."
                "torch.distributed.all_gather_into_tensor",
                side_effect=AssertionError("pure TP must not all-gather on CPU"),
            ),
        ):
            planner.prepare_verify_budget(batch, future_map=mock.Mock())

        self.assertEqual(batch.spec_verify_tier_num_tokens, 8)

    def test_dp_attention_keeps_cpu_tier_coordination(self):
        planner = object.__new__(DSparkVerifyPlanner)
        planner.server_args = SimpleNamespace(tp_size=2)
        group = _FakeBroadcastGroup(torch.empty(0, dtype=torch.int32))
        batch = SimpleNamespace(spec_verify_tier_num_tokens=-1)

        def broadcast(tensor, *, src, group):
            self.assertEqual(src, 0)
            self.assertIs(group, group_ref.cpu_group)
            tensor.fill_(3)

        group_ref = group
        with (
            mock.patch(
                "sglang.srt.speculative.dspark_components.dspark_planner."
                "verify_lens_broadcast_group",
                return_value=(group, 2),
            ),
            mock.patch(
                "sglang.srt.speculative.dspark_components.dspark_planner."
                "is_dp_attention_enabled",
                return_value=True,
            ),
            mock.patch(
                "sglang.srt.speculative.dspark_components.dspark_planner."
                "torch.distributed.broadcast",
                side_effect=broadcast,
            ) as tier_broadcast,
            mock.patch.object(planner, "_maybe_gather_dp_verify_tier") as gather,
        ):
            planner._coordinate_verify_tier(
                batch=batch, local_tier_num_tokens=-1
            )

        self.assertEqual(batch.spec_verify_tier_num_tokens, 3)
        tier_broadcast.assert_called_once()
        gather.assert_called_once_with(batch=batch, local_tier_num_tokens=3)

    def test_conservative_tp_tier_preserves_non_decode_zero(self):
        planner = object.__new__(DSparkVerifyPlanner)
        planner.verify_num_draft_tokens = 4
        batch = SimpleNamespace(batch_size=lambda: 2)
        self.assertEqual(
            planner._conservative_tp_verify_tier(
                batch=batch, local_tier_num_tokens=0
            ),
            0,
        )

        empty_batch = SimpleNamespace(batch_size=lambda: 0)
        self.assertEqual(
            planner._conservative_tp_verify_tier(
                batch=empty_batch, local_tier_num_tokens=-1
            ),
            0,
        )

    def test_verify_all_skips_tp_tier_collective(self):
        planner = object.__new__(DSparkVerifyPlanner)
        planner.server_args = SimpleNamespace(tp_size=2)
        planner._is_verify_all = True
        planner._budget_planner = SimpleNamespace(forced_budget_frac=None)
        group = _FakeBroadcastGroup(torch.empty(0, dtype=torch.int32))
        batch = SimpleNamespace(spec_verify_tier_num_tokens=-1)

        with (
            mock.patch(
                "sglang.srt.speculative.dspark_components.dspark_planner."
                "verify_lens_broadcast_group",
                return_value=(group, 2),
            ),
            mock.patch(
                "sglang.srt.speculative.dspark_components.dspark_planner."
                "torch.distributed.broadcast",
                side_effect=AssertionError("verify-all must not coordinate TP tier"),
            ),
            mock.patch.object(planner, "_maybe_gather_dp_verify_tier") as gather,
        ):
            planner._coordinate_verify_tier(
                batch=batch, local_tier_num_tokens=8
            )

        self.assertEqual(batch.spec_verify_tier_num_tokens, 8)
        gather.assert_called_once_with(batch=batch, local_tier_num_tokens=8)

    def test_verify_all_forced_budget_uses_conservative_tp_tier(self):
        planner = object.__new__(DSparkVerifyPlanner)
        planner.server_args = SimpleNamespace(tp_size=2)
        planner.verify_num_draft_tokens = 4
        planner._is_verify_all = True
        planner._budget_planner = SimpleNamespace(forced_budget_frac=0.5)
        group = _FakeBroadcastGroup(torch.empty(0, dtype=torch.int32))
        batch = SimpleNamespace(
            spec_verify_tier_num_tokens=-1, batch_size=lambda: 2
        )

        with (
            mock.patch(
                "sglang.srt.speculative.dspark_components.dspark_planner."
                "verify_lens_broadcast_group",
                return_value=(group, 2),
            ),
            mock.patch(
                "sglang.srt.speculative.dspark_components.dspark_planner."
                "is_dp_attention_enabled",
                return_value=False,
            ),
            mock.patch(
                "sglang.srt.speculative.dspark_components.dspark_planner."
                "torch.distributed.broadcast",
                side_effect=AssertionError("pure TP must not coordinate on CPU"),
            ),
            mock.patch.object(planner, "_maybe_gather_dp_verify_tier") as gather,
        ):
            planner._coordinate_verify_tier(
                batch=batch, local_tier_num_tokens=-1
            )

        self.assertEqual(batch.spec_verify_tier_num_tokens, 8)
        gather.assert_called_once_with(batch=batch, local_tier_num_tokens=8)

    def test_tp_non_source_broadcasts_without_local_confidence(self):
        planner = object.__new__(DSparkVerifyPlanner)
        planner._budget_planner = object()
        planner.verify_num_draft_tokens = 4
        source_lens = torch.tensor([2, 3], dtype=torch.int32)
        group = _FakeBroadcastGroup(source_lens)

        with mock.patch.object(
            ScheduleVerifyLensTopk,
            "execute",
            side_effect=AssertionError("non-source rank must not schedule"),
        ):
            verify_lens = planner._schedule_verify_lens(
                req_pool_indices=torch.tensor([0, 1], dtype=torch.int64),
                prefix_lens=torch.tensor([10, 20], dtype=torch.int64),
                device=torch.device("cpu"),
                confidence=None,
                budget=None,
                broadcast_group=group,
                broadcast_group_size=2,
            )

        self.assertTrue(torch.equal(verify_lens, source_lens))
        self.assertEqual(group.calls, 1)
        self.assertEqual(group.assert_source, 0)

    def test_tp_source_missing_readiness_broadcasts_full_verify(self):
        planner = object.__new__(DSparkVerifyPlanner)
        planner._budget_planner = object()
        planner.verify_num_draft_tokens = 4
        group = _FakeBroadcastGroup(
            torch.empty(0, dtype=torch.int32), rank_in_group=0
        )

        with mock.patch.object(
            ScheduleVerifyLensTopk,
            "execute",
            side_effect=AssertionError("missing readiness must use full verify"),
        ):
            verify_lens = planner._schedule_verify_lens(
                req_pool_indices=torch.tensor([0, 1], dtype=torch.int64),
                prefix_lens=torch.tensor([10, 20], dtype=torch.int64),
                device=torch.device("cpu"),
                confidence=None,
                budget=None,
                broadcast_group=group,
                broadcast_group_size=2,
            )

        self.assertTrue(
            torch.equal(verify_lens, torch.tensor([4, 4], dtype=torch.int32))
        )
        self.assertEqual(group.calls, 1)

    def test_dynamic_compact_tp_layout_stays_device_side(self):
        planner = object.__new__(DSparkVerifyPlanner)
        planner._align_verify_tokens_to_graph_tier = False
        planner._budget_planner = object()
        planner._dynamic_graph_tier = True
        planner._is_verify_all = False
        planner._ragged_verify_mode = RaggedVerifyMode.COMPACT
        planner._schedule_cfg = DSparkScheduleConfig(gamma=3)
        planner._uniform_layout_cache = {}
        planner.verify_num_draft_tokens = 4
        planner.model_runner = SimpleNamespace(
            decode_cuda_graph_runner=SimpleNamespace(
                ragged_verify_mode=True,
                capture_num_tokens=[4, 8, 16],
                max_bs=8,
            )
        )
        planner.server_args = SimpleNamespace(tp_size=2)
        source_lens = torch.tensor([1, 2], dtype=torch.int32)
        group = _FakeBroadcastGroup(source_lens)
        layout = object()

        with (
            mock.patch(
                "sglang.srt.speculative.dspark_components.dspark_planner."
                "verify_lens_broadcast_group",
                return_value=(group, 2),
            ),
            mock.patch.object(
                ScheduleVerifyLensTopk,
                "execute",
                side_effect=AssertionError("non-source rank must not schedule"),
            ),
            mock.patch.object(
                RaggedVerifyLayout,
                "from_verify_lens",
                side_effect=AssertionError("dynamic compact graph materialized lengths"),
            ),
            mock.patch.object(
                RaggedVerifyLayout,
                "from_verify_lens_device",
                return_value=layout,
            ) as assemble,
        ):
            result = planner.schedule_layout(
                req_pool_indices=torch.tensor([0, 1], dtype=torch.int64),
                prefix_lens=torch.tensor([10, 20], dtype=torch.int64),
                device=torch.device("cpu"),
                confidence=None,
                budget=None,
                tp_tier_num_tokens=3,
            )

        self.assertIs(result, layout)
        self.assertTrue(
            torch.equal(assemble.call_args.kwargs["verify_lens"], source_lens)
        )
        self.assertEqual(assemble.call_args.kwargs["graph_num_tokens"], 4)

        with (
            mock.patch(
                "sglang.srt.speculative.dspark_components.dspark_planner."
                "verify_lens_broadcast_group",
                return_value=(group, 2),
            ),
            mock.patch.object(
                ScheduleVerifyLensTopk,
                "execute",
                side_effect=AssertionError("non-source rank must not schedule"),
            ),
            mock.patch.object(
                RaggedVerifyLayout,
                "from_verify_lens_device",
                return_value=layout,
            ) as unavailable_assemble,
        ):
            result = planner.schedule_layout(
                req_pool_indices=torch.tensor([0, 1], dtype=torch.int64),
                prefix_lens=torch.tensor([10, 20], dtype=torch.int64),
                device=torch.device("cpu"),
                confidence=None,
                budget=0,
                tp_tier_num_tokens=-1,
            )

        self.assertIs(result, layout)
        self.assertEqual(
            unavailable_assemble.call_args.kwargs["graph_num_tokens"], 8
        )


    def test_verify_all_skips_confidence_until_forced_budget_needs_it(self):
        planner = object.__new__(DSparkVerifyPlanner)
        planner._budget_planner = SimpleNamespace(forced_budget_frac=None)
        planner._is_verify_all = True
        self.assertFalse(planner.needs_confidence_publication)

        planner._budget_planner.forced_budget_frac = 0.5
        self.assertTrue(planner.needs_confidence_publication)

        planner._budget_planner.forced_budget_frac = None
        planner._is_verify_all = False
        self.assertTrue(planner.needs_confidence_publication)

        planner._budget_planner = None
        self.assertFalse(planner.needs_confidence_publication)

    def test_observability_modes_preserve_confidence_publication(self):
        observers = object.__new__(DsparkStepObservers)
        observers._info_components = set()
        with (
            mock.patch.object(
                envs.SGLANG_DSPARK_LOG_SPS_PRED_INTERVAL, "get", return_value=0
            ),
            mock.patch.object(
                envs.SGLANG_DSPARK_DEBUG_CONFIDENCE_PREFIX_SCHEDULER,
                "get",
                return_value=False,
            ),
        ):
            self.assertFalse(observers.needs_budget_telemetry)
            observers._info_components = {InfoComponent.REQS}
            self.assertTrue(observers.needs_budget_telemetry)
            observers._info_components = set()

        with mock.patch.object(
            envs.SGLANG_DSPARK_LOG_SPS_PRED_INTERVAL, "get", return_value=8
        ):
            self.assertTrue(observers.needs_budget_telemetry)

        with mock.patch.object(
            envs.SGLANG_DSPARK_DEBUG_CONFIDENCE_PREFIX_SCHEDULER,
            "get",
            return_value=True,
        ):
            self.assertTrue(observers.needs_budget_telemetry)

    def test_worker_confidence_publication_gate(self):
        worker = object.__new__(DSparkWorkerV2)
        worker._verify_planner = SimpleNamespace(
            needs_confidence_publication=False
        )
        worker._observers = SimpleNamespace(needs_budget_telemetry=False)
        self.assertFalse(worker._should_publish_confidence(None))
        self.assertFalse(worker._should_publish_confidence(torch.ones(1)))

        worker._verify_planner.needs_confidence_publication = True
        self.assertTrue(worker._should_publish_confidence(torch.ones(1)))
        worker._verify_planner.needs_confidence_publication = False
        worker._observers.needs_budget_telemetry = True
        self.assertTrue(worker._should_publish_confidence(torch.ones(1)))

    def test_planner_reset_forwards_to_budget_state(self):
        planner = object.__new__(DSparkVerifyPlanner)
        planner._budget_planner = mock.Mock()

        planner.reset_runtime_state()

        planner._budget_planner.reset_runtime_state.assert_called_once_with()

    def test_worker_cache_clear_resets_planner_state(self):
        worker = object.__new__(DSparkWorkerV2)
        worker._verify_planner = mock.Mock()

        worker.clear_cache_pool()

        worker._verify_planner.reset_runtime_state.assert_called_once_with()


class _FakeTargetWorker:
    def __init__(self):
        self.model_runner = SimpleNamespace(attn_backend=SimpleNamespace())

    def forward_batch_generation(self, **kwargs):
        return SimpleNamespace(
            logits_output=SimpleNamespace(next_token_logits=torch.empty(0)),
            can_run_cuda_graph=False,
        )


class TestDSparkDeviceOnlyLengths(unittest.TestCase):
    def _run_draft(self, *, seq_lens_cpu):
        seen = {}
        gamma = 4
        bs = 2

        class FakeDraftRunner:
            device = "cpu"

            def forward(self, forward_batch):
                seen["seq_lens_cpu"] = forward_batch.seq_lens_cpu
                seen["seq_lens_sum"] = forward_batch.seq_lens_sum
                return SimpleNamespace(
                    logits_output=SimpleNamespace(
                        hidden_states=torch.empty((bs * gamma, 16))
                    ),
                    can_run_graph=False,
                )

        proposer = DraftBlockProposer(
            draft_model=SimpleNamespace(),
            draft_model_runner=FakeDraftRunner(),
            gamma=gamma,
            mask_token_id=0,
            draft_block_spec_info=SimpleNamespace(),
        )
        batch = SimpleNamespace(
            seq_lens=torch.tensor([10, 20], dtype=torch.int32),
            seq_lens_cpu=seq_lens_cpu,
            seq_lens_sum=None if seq_lens_cpu is None else int(seq_lens_cpu.sum()),
            req_pool_indices=torch.tensor([0, 1], dtype=torch.int64),
            global_num_tokens=None,
        )
        proposer._run_forward(
            batch=batch,
            draft_input=SimpleNamespace(
                bonus_tokens=torch.tensor([7, 8], dtype=torch.int64),
                reserved_seq_lens_cpu=torch.tensor([18, 28], dtype=torch.int32),
                reserved_seq_lens_sum=46,
            ),
            verify_window=SimpleNamespace(
                positions_2d=torch.arange(bs * gamma).view(bs, gamma),
                verify_cache_loc_2d=torch.arange(bs * gamma).view(bs, gamma),
            ),
            bs=bs,
            device="cpu",
            embed_module=torch.nn.Embedding(16, 16),
        )
        return seen

    def test_draft_ignores_reserved_host_bound_without_cpu_mirror(self):
        seen = self._run_draft(seq_lens_cpu=None)
        self.assertIsNone(seen["seq_lens_cpu"])
        self.assertIsNone(seen["seq_lens_sum"])

    def test_draft_retains_eager_cpu_mirror_fallback(self):
        seen = self._run_draft(
            seq_lens_cpu=torch.tensor([10, 20], dtype=torch.int32)
        )
        self.assertEqual(seen["seq_lens_cpu"].tolist(), [14, 24])
        self.assertEqual(seen["seq_lens_sum"], 38)

    def test_target_verify_ignores_reserved_bound_without_cpu_mirror(self):
        target_worker = _FakeTargetWorker()
        executor = TargetVerifyExecutor(
            target_worker=target_worker,
            gamma=3,
            verify_num_draft_tokens=4,
            model_runner=target_worker.model_runner,
            kv_injector=SimpleNamespace(),
        )
        batch = SimpleNamespace(
            seq_lens=torch.tensor([10, 20], dtype=torch.int32),
            seq_lens_cpu=None,
            seq_lens_sum=None,
            out_cache_loc=None,
        )
        seen = {}

        def capture_prepare(_verify_input, verify_batch, _target_worker):
            seen["seq_lens_cpu"] = verify_batch.seq_lens_cpu
            seen["seq_lens_sum"] = verify_batch.seq_lens_sum
            return SimpleNamespace(), False

        with mock.patch.object(
            DFlashVerifyInput, "prepare_for_verify", new=capture_prepare
        ):
            executor.run_non_compact(
                batch=batch,
                draft_input=SimpleNamespace(
                    reserved_seq_lens_cpu=torch.tensor([18, 28], dtype=torch.int32),
                    reserved_seq_lens_sum=46,
                ),
                verify_ids_2d=torch.ones((2, 4), dtype=torch.int64),
                verify_window=SimpleNamespace(
                    positions_2d=torch.arange(8).view(2, 4),
                    verify_cache_loc=torch.arange(8),
                ),
                sampling_info=None,
            )

        self.assertIsNone(seen["seq_lens_cpu"])
        self.assertIsNone(seen["seq_lens_sum"])
        self.assertIsNone(batch.seq_lens_cpu)
        self.assertIsNone(batch.seq_lens_sum)


if __name__ == "__main__":
    unittest.main()
