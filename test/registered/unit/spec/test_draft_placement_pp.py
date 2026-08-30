"""Where the draft worker lands when the PD-prefill role runs pipeline parallel.

The draft head is a ``pp_size == 1`` runner that reads the target's
``lm_head`` / ``embed_tokens``, and a pipeline-split target holds those on the
last stage only. So on a PD-prefill role with ``pp_size > 1`` exactly one stage
builds a draft worker and the rest run target-only.

``maybe_init_draft_worker`` is the single decision point for that placement, and
it reads only a handful of fields off the scheduler, so this drives it on a
stand-in record (same pattern as ``test_draft_construction_isolation.py``'s
``ModelRunner.__new__``) rather than standing up a real scheduler.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _StandInDraftWorker:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class TestDraftPlacementUnderPipelineParallel(CustomTestCase):
    def _decide(self, *, pp_size, pp_rank, disaggregation_mode, algorithm="DSPARK"):
        """Run the placement decision; return the scheduler it decided for."""
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.spec_algorithm = SpeculativeAlgorithm.from_string(algorithm)
        scheduler.ps = SimpleNamespace(pp_size=pp_size, pp_rank=pp_rank, gpu_id=0)
        scheduler.server_args = SimpleNamespace()
        scheduler.nccl_port = 0
        scheduler.tp_worker = object()
        scheduler.ipc_channels = SimpleNamespace(
            send_to_tokenizer=SimpleNamespace(send_output=None)
        )

        with patch.object(
            SpeculativeAlgorithm, "create_worker", return_value=_StandInDraftWorker
        ), patch(
            "sglang.srt.managers.scheduler.get_disagg",
            return_value=SimpleNamespace(disaggregation_mode=disaggregation_mode),
        ):
            scheduler.maybe_init_draft_worker()

        return scheduler

    def test_the_last_prefill_stage_hosts_the_draft(self):
        scheduler = self._decide(pp_size=2, pp_rank=1, disaggregation_mode="prefill")
        self.assertIsInstance(scheduler.draft_worker, _StandInDraftWorker)
        # It drafts off this stage's own target, not a separately loaded one.
        self.assertIs(
            scheduler.draft_worker.kwargs["target_worker"], scheduler.tp_worker
        )

    def test_earlier_prefill_stages_run_target_only(self):
        scheduler = self._decide(pp_size=2, pp_rank=0, disaggregation_mode="prefill")
        self.assertIsNone(scheduler.draft_worker)
        # The ngram corpus manager hangs off the draft worker, so it has to go
        # too; leaving a stale one behind would be a null dereference later.
        self.assertIsNone(scheduler.external_corpus_manager)

    def test_only_the_last_of_four_prefill_stages_hosts_the_draft(self):
        placement = [
            self._decide(
                pp_size=4, pp_rank=rank, disaggregation_mode="prefill"
            ).draft_worker
            is not None
            for rank in range(4)
        ]
        self.assertEqual(placement, [False, False, False, True])

    def test_a_single_stage_prefill_role_is_unaffected(self):
        self.assertIsNotNone(
            self._decide(
                pp_size=1, pp_rank=0, disaggregation_mode="prefill"
            ).draft_worker
        )

    def test_the_skip_is_scoped_to_the_prefill_role(self):
        # The decode role and the non-disaggregated path still reject pp > 1 with
        # speculative decoding at the arg layer; if that ever relaxes, this skip
        # must not silently apply to them, because only prefill was validated.
        for mode in ("decode", "null"):
            with self.subTest(disaggregation_mode=mode):
                self.assertIsNotNone(
                    self._decide(
                        pp_size=2, pp_rank=0, disaggregation_mode=mode
                    ).draft_worker
                )

    def test_no_speculative_algorithm_still_short_circuits_first(self):
        self.assertIsNone(
            self._decide(
                pp_size=2, pp_rank=1, disaggregation_mode="prefill", algorithm="NONE"
            ).draft_worker
        )


if __name__ == "__main__":
    unittest.main()
