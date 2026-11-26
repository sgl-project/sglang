import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.scheduler_components.batch_result_processor import (  # noqa: E402
    SchedulerBatchResultProcessor,
)
from sglang.srt.managers.scheduler_components.pp_dspark_draft import (  # noqa: E402
    PPDSparkDraftCoordinator,
)
from sglang.srt.managers.scheduler_pp_mixin import (  # noqa: E402
    PPBatchMetadata,
    SchedulerPPMixin,
)
from sglang.srt.managers.utils import GenerationBatchResult  # noqa: E402
from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2  # noqa: E402
from sglang.srt.speculative.dspark_components.dspark_pp import (  # noqa: E402
    PPDSparkCandidate,
    PPDSparkIdentity,
)
from sglang.srt.speculative.dspark_components.dspark_worker_v2 import (  # noqa: E402
    DSparkWorkerV2,
)

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _Req(SimpleNamespace):
    __hash__ = object.__hash__


class _Outputs:
    def __init__(self, tensors):
        self.tensors = tensors

    def __getitem__(self, key):
        return self.tensors[key]


def _req(rid, room, retractions=0):
    return _Req(
        rid=rid,
        bootstrap_room=room,
        retraction_count=retractions,
        spec_verify_ct=0,
    )


def _draft_input(bonus_tokens, seq_lens=None):
    rows = len(bonus_tokens)
    return DFlashDraftInputV2(
        topk_p=torch.empty((rows, 0)),
        topk_index=torch.empty((rows, 0), dtype=torch.int64),
        bonus_tokens=torch.tensor(bonus_tokens),
        new_seq_lens=torch.tensor(seq_lens or range(10, 10 + rows)),
        hidden_states=torch.empty((rows, 0)),
    )


def _relay_outputs(identities):
    rows = len(identities)
    return _Outputs(
        {
            "next_token_ids": torch.arange(1, rows + 1),
            "dspark_projected_context": torch.empty((rows, 0)),
            "dspark_identities": [identity.to_wire() for identity in identities],
            "dspark_new_seq_lens": torch.tensor([11, 21]),
            "dspark_accept_lens": torch.ones(rows, dtype=torch.int64),
            "dspark_context_token_counts": tuple([1] * rows),
            "dspark_bonus_tokens": torch.arange(1, rows + 1),
            "dspark_block_accept_lens": torch.ones(rows, dtype=torch.int64),
        }
    )


class TestSchedulerPPRelayOrder(unittest.TestCase):
    def test_enqueue_freezes_batch_owned_draft_input(self):
        batch = SimpleNamespace(
            reqs=[_req("a", 7)],
            req_pool_indices=torch.tensor([3]),
            spec_info=_draft_input([11]),
        )
        coordinator = PPDSparkDraftCoordinator(SimpleNamespace())

        coordinator.enqueue(batch, batch.spec_info)
        work = coordinator._pending[0]
        batch.spec_info.bonus_tokens[0] = 99

        self.assertIsNot(work.draft_input, batch.spec_info)
        self.assertEqual(work.draft_input.bonus_tokens.tolist(), [11])

    def test_committed_snapshot_updates_only_matching_live_generation(self):
        committed = SimpleNamespace(
            reqs=[_req("b", 9), _req("a", 7)],
            spec_info=_draft_input([20, 10], [220, 110]),
            seq_lens=torch.tensor([220, 110]),
            seq_lens_cpu=torch.tensor([220, 110]),
            seq_lens_sum=330,
            orig_seq_lens=torch.tensor([200, 100], dtype=torch.int32),
            input_ids=None,
        )
        live = SimpleNamespace(
            reqs=[_req("a", 7), _req("c", 8), _req("b", 9, 1)],
            spec_info=_draft_input([1, 3, 2], [11, 33, 22]),
            seq_lens=torch.tensor([11, 33, 22]),
            seq_lens_cpu=torch.tensor([11, 33, 22]),
            seq_lens_sum=66,
            orig_seq_lens=torch.tensor([10, 30, 20], dtype=torch.int32),
            input_ids=torch.tensor([1, 3, 2]),
        )
        live.spec_info.nxt_kv_lens_cpu = torch.tensor([12, 34, 23])
        live.spec_info.future_indices = torch.tensor([4, 5, 6])
        scheduler = SchedulerPPMixin()
        scheduler.running_mbs = [live]
        identities = (
            SimpleNamespace(rid="b", generation=(9, 0)),
            SimpleNamespace(rid="a", generation=(7, 0)),
        )

        scheduler._pp_sync_committed_snapshot_to_running_slots(committed, identities)

        self.assertEqual(live.spec_info.bonus_tokens.tolist(), [10, 3, 2])
        self.assertEqual(live.seq_lens.tolist(), [110, 33, 22])
        self.assertEqual(live.seq_lens_sum, 165)
        self.assertEqual(live.orig_seq_lens.tolist(), [100, 30, 20])
        self.assertIsNone(live.spec_info.nxt_kv_lens_cpu)
        self.assertIsNone(live.spec_info.future_indices)
        self.assertIsNone(live.input_ids)

    def test_forward_snapshot_mask_excludes_retracted_result_row(self):
        snapshot = SimpleNamespace(reqs=[_req("a", 7), _req("b", 9, 1)])
        live = SimpleNamespace(reqs=[_req("b", 9, 2), _req("a", 7)])
        metadata = PPBatchMetadata(
            can_run_cuda_graph=True,
            fwd_batch=snapshot,
            dspark_identities=(
                SimpleNamespace(rid="a", generation=(7, 0)),
                SimpleNamespace(rid="b", generation=(9, 1)),
            ),
        )
        scheduler = SchedulerPPMixin()
        scheduler.mbs = [live]
        scheduler.mb_metadata = [metadata]

        process_batch = scheduler._pp_result_process_batch(0)
        processor = SimpleNamespace(still_in_batch=lambda req: True)
        mask = SchedulerBatchResultProcessor._rows_still_in_batch(
            processor, process_batch
        )

        self.assertIs(process_batch, snapshot)
        self.assertEqual(process_batch.pp_result_commit_mask, (True, False))
        self.assertEqual(mask, [True, False])

    def test_stale_candidate_falls_back_to_warmup_anchor(self):
        worker = DSparkWorkerV2.__new__(DSparkWorkerV2)
        worker.device = "cpu"
        worker.query_token_num = 2
        worker.gamma = 2
        worker._greedy_draft = True
        request = _req("a", 7, 1)
        worker._pp_candidates = {
            request: PPDSparkCandidate(
                identity=PPDSparkIdentity("a", (7, 0), 0),
                draft_block_ids=torch.tensor([91, 92]),
                draft_tokens=torch.tensor([93, 94]),
            )
        }
        batch = SimpleNamespace(
            reqs=[request],
            spec_info=SimpleNamespace(bonus_tokens=torch.tensor([7])),
        )

        proposal = worker.consume_pp_draft(batch)

        self.assertEqual(proposal.draft_block_ids.tolist(), [[7, 7]])
        self.assertEqual(proposal.draft_block.draft_tokens.tolist(), [[7, 7]])

    def test_relayed_result_skips_commit_when_generation_changed(self):
        identities = (
            PPDSparkIdentity("a", (7, 0), 0),
            PPDSparkIdentity("b", (9, 0), 0),
        )
        fwd_batch = SimpleNamespace(
            reqs=[_req("a", 7), _req("b", 9, 1)],
            spec_info=None,
            seq_lens=torch.tensor([10, 20]),
            seq_lens_cpu=torch.tensor([10, 20]),
            seq_lens_sum=30,
        )
        live_batch = SimpleNamespace(
            reqs=[_req("a", 7), _req("b", 9, 1)],
            return_logprob=False,
            return_hidden_states=False,
        )
        metadata = PPBatchMetadata(
            can_run_cuda_graph=True,
            fwd_batch=fwd_batch,
            dspark_commit_state=SimpleNamespace(identities=identities),
            dspark_identities=identities,
            speculative_num_draft_tokens=5,
        )
        draft_coordinator = Mock()
        scheduler = SimpleNamespace(
            _pp_spec_relay=False,
            pp_group=SimpleNamespace(is_first_rank=False, is_last_rank=False),
            model_worker=SimpleNamespace(commit_pp_draft_context=Mock()),
            pp_dspark_draft=draft_coordinator,
            device_module=SimpleNamespace(Event=Mock(return_value=Mock())),
        )

        with patch.object(GenerationBatchResult, "copy_to_cpu", autospec=True):
            result = SchedulerPPMixin._pp_prep_batch_result(
                scheduler, live_batch, metadata, _relay_outputs(identities)
            )

        self.assertEqual(result.new_seq_lens.tolist(), [11, 21])
        scheduler.model_worker.commit_pp_draft_context.assert_not_called()
        draft_coordinator.on_relayed_proposals.assert_not_called()


if __name__ == "__main__":
    unittest.main()
