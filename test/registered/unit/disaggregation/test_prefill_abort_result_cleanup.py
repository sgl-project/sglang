from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from sglang.srt.disaggregation.req_state import DisaggReqState
from sglang.srt.managers.request_lists import CoreLists
from sglang.srt.managers.schedule_batch import FINISH_ABORT, ReqKvInfo
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.mem_cache.base_prefix_cache import CacheRequestHandle
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.disagg_owner_shells import (
    prefill_owner_shell,
    result_processor_shell,
)
from sglang.test.scheduler_request_lists import make_request_retirer
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


class _Req:
    def __init__(self, *, allocated: bool = True):
        self.rid = "aborted-prefill"
        self.cache_request_handle = CacheRequestHandle(self.rid, 0)
        self.kv = ReqKvInfo(
            req_pool_idx=1 if allocated else None,
            kv_allocated_len=1 if allocated else 0,
            mamba_pool_idx=object() if allocated else None,
        )
        self.disagg = DisaggReqState(
            metadata_buffer_index=7 if allocated else -1, pending_bootstrap=allocated
        )
        self.sender = Mock()
        self.to_finish = FINISH_ABORT() if allocated else None
        self.finished_reason = None if allocated else FINISH_ABORT()
        self.discard_output_reason = None
        self.return_logprob = False
        self.return_sampling_mask = False
        self.grammar = None
        self.output_ids = []
        self.origin_input_ids = list(range(100))
        self.session = None
        self.media_released = 0
        self.extend_end = None
        self.time_stats = SimpleNamespace(
            set_prefill_finished_time=Mock(),
            set_last_chunked_prefill_finish_time=Mock(),
            set_completion_time=Mock(),
        )

    def finished(self):
        return self.finished_reason is not None

    def release_mm_inputs_on_abort(self):
        self.media_released += 1

    def update_finish_state(self):
        self.finished_reason = self.to_finish
        self.to_finish = None


def _owner():
    tree_cache = Mock()
    return prefill_owner_shell(
        batch_result_processor=result_processor_shell(),
        spec_algorithm=SimpleNamespace(is_eagle=lambda: False),
        tree_cache=tree_cache,
        request_retirer=make_request_retirer(kv=SimpleNamespace(tree_cache=tree_cache)),
        pending_chunk_rids={"aborted-prefill"},
        req_to_metadata_buffer_idx_allocator=Mock(),
        send_kv_chunk=Mock(),
        output_streamer=Mock(),
        core_lists=CoreLists(),
    )


def _batch(req, *, middle_chunk=False):
    return SimpleNamespace(
        reqs=[req],
        chunked_req=req if middle_chunk else None,
        spec_info=None,
        prefill_stats=None,
        dp_cooperation_info=None,
        extend_lens=None,
        extend_logprob_start_lens=None,
        return_logprob=False,
    )


def _result():
    return GenerationBatchResult(next_token_ids=torch.tensor([11]))


def _free_req(req, _tree_cache, *, checkpoint):
    assert checkpoint is False
    req.kv.req_pool_idx = None
    req.kv.mark_kv_released()
    req.kv.mamba_pool_idx = None


class TestPrefillCompleteResult(CustomTestCase):
    def setUp(self):
        super().setUp()
        override = get_context().override_server_args(
            disaggregation_decode_allocation_policy="prefill_complete"
        )
        override.install()
        self.addCleanup(override.restore)

    @patch("sglang.srt.disaggregation.prefill.checkpoint_kv_cache")
    def test_ready_observes_accepted_token_and_speculative_sidecars(self, cache):
        owner = _owner()
        owner.spec_algorithm = SimpleNamespace(is_eagle=lambda: True)
        req = _Req()
        owner.kv_senders[req.rid] = req.sender
        req.to_finish = None
        req.time_stats.set_prefill_transfer_queue_entry_time = Mock()
        draft = SimpleNamespace(
            hidden_states=torch.tensor([[1.0, 2.0]]),
            topk_p=torch.tensor([[0.75, 0.25]]),
            topk_index=torch.tensor([[3, 4]]),
            dsa_topk_indices=torch.tensor([[5, 6]]),
        )
        # The launched step keeps no draft input; the result carries it.
        batch = _batch(req)
        result = GenerationBatchResult(
            next_token_ids=torch.tensor([11]), next_draft_input=draft
        )
        observed = []

        def ready():
            observed.append(
                (
                    list(req.output_ids),
                    req.hidden_states_tensor.tolist(),
                    req.output_dsa_topk_indices.tolist(),
                    req in owner.transfer,
                )
            )

        req.sender.mark_prefill_complete.side_effect = ready
        owner.commit_extend(batch, result)
        self.assertEqual(observed, [([11], [1.0, 2.0], [5, 6], True)])
        self.assertTrue(req.disagg.pending_bootstrap)
        owner.send_kv_chunk.assert_not_called()

    @patch(
        "sglang.srt.managers.scheduler_components.request_retirer.release_kv_cache",
        side_effect=_free_req,
    )
    def test_aborted_final_result_never_publishes_ready(self, release):
        owner = _owner()
        req = _Req()
        owner.kv_senders[req.rid] = req.sender
        owner.commit_extend(_batch(req), _result())
        self.assertTrue(req.finished())
        self.assertFalse(req.kv.holds_kv)
        self.assertEqual(owner.transfer, [])
        req.sender.mark_prefill_complete.assert_not_called()

    @patch("sglang.srt.disaggregation.prefill.should_force_retry", return_value=True)
    def test_discarded_attempt_never_publishes_ready(self, force_retry):
        owner = _owner()
        req = _Req()
        owner.kv_senders[req.rid] = req.sender
        req.to_finish = None
        retried = []
        owner.optimistic_release_and_requeue = lambda r, waiting: (
            retried.append(r) or waiting
        )
        owner.commit_extend(_batch(req), _result())
        self.assertEqual(retried, [req])
        self.assertEqual(req.output_ids, [])
        self.assertEqual(owner.transfer, [])
        req.sender.mark_prefill_complete.assert_not_called()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
