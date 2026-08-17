import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.srt.runtime_context import get_context
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


def _make_batch() -> tuple[Req, ScheduleBatch]:
    sampling_params = SamplingParams(max_new_tokens=32)
    sampling_params.normalize(None)
    req = Req(
        rid="req",
        origin_input_text="",
        origin_input_ids=array("q", [1, 2]),
        sampling_params=sampling_params,
        vocab_size=128,
    )
    req.output_ids.append(3)
    req.kv_committed_len = 2

    batch = ScheduleBatch(reqs=[req])
    batch.device = "cpu"
    batch.model_config = SimpleNamespace(is_encoder_decoder=False)
    batch.enable_overlap = True
    batch.spec_algorithm = SimpleNamespace(is_none=lambda: True)
    batch.sampling_info = SimpleNamespace(
        penalizer_orchestrator=SimpleNamespace(is_required=False)
    )
    batch.hisparse_coordinator = None
    batch.seq_lens = torch.tensor([2], dtype=torch.int64)
    batch.seq_lens_cpu = torch.tensor([2], dtype=torch.int64)
    batch.orig_seq_lens = torch.tensor([2], dtype=torch.int32)
    return req, batch


def _make_processor() -> SchedulerBatchResultProcessor:
    metrics_reporter = MagicMock()
    metrics_reporter.num_generated_tokens = 0
    metrics_reporter.forward_ct_decode = 0
    return SchedulerBatchResultProcessor(
        is_generation=True,
        disaggregation_mode=None,
        enable_overlap=True,
        enable_overlap_mlx=False,
        server_args=SimpleNamespace(),
        model_config=SimpleNamespace(think_end_ids=None),
        token_to_kv_pool_allocator=MagicMock(),
        tree_cache=None,
        hisparse_coordinator=None,
        req_to_token_pool=None,
        decode_offload_manager=None,
        metrics_collector=None,
        metrics_reporter=metrics_reporter,
        draft_worker=None,
        model_worker=MagicMock(),
        logprob_result_processor=None,
        output_streamer=MagicMock(),
        abort_request=lambda *args, **kwargs: None,
    )


def _make_result():
    return SimpleNamespace(
        copy_done=None,
        routed_experts_output=None,
        indexer_topk_output=None,
        logits_output=SimpleNamespace(hidden_states=None, customized_info=None),
        next_token_ids=[4],
        can_run_cuda_graph=False,
        num_correct_drafts=0,
        num_block_accept_tokens=0,
        num_cap_tokens=0,
        speculative_num_draft_tokens=0,
    )


class TestMambaBoundaryMaskReuse(unittest.TestCase):
    def test_overlap_scheduler_handles_zero_and_one_batch_lookahead(self):
        for schedule_next_decode, expected_lookahead in ((False, 0), (True, 1)):
            with self.subTest(schedule_next_decode=schedule_next_decode):
                req, batch = _make_batch()
                processor = _make_processor()
                result = _make_result()

                scheduler = Scheduler.__new__(Scheduler)
                scheduler.gracefully_exit = False
                scheduler.request_receiver = MagicMock()
                scheduler.request_receiver.recv_requests.side_effect = [
                    [],
                    [],
                    StopIteration,
                ]
                scheduler.process_input_requests = MagicMock()
                scheduler._engine_paused = False
                scheduler.running_batch = batch
                scheduler.is_disable_overlap_for_batch = MagicMock(return_value=False)
                scheduler.run_batch = MagicMock(return_value=result)
                scheduler._apply_war_barrier = MagicMock()
                scheduler.is_generation = False
                scheduler.last_batch = None

                plan_count = 0

                def get_next_batch_to_run(*, running_batch, last_batch):
                    nonlocal plan_count
                    del running_batch, last_batch
                    plan_count += 1
                    if plan_count == 1:
                        batch.prepare_for_decode()
                        return SimpleNamespace(
                            running_batch=batch,
                            batch_to_run=batch,
                        )
                    if plan_count == 2 and schedule_next_decode:
                        batch.prepare_for_decode()
                        return SimpleNamespace(
                            running_batch=batch,
                            batch_to_run=batch,
                        )
                    return SimpleNamespace(
                        running_batch=batch,
                        batch_to_run=None,
                    )

                scheduler.get_next_batch_to_run = get_next_batch_to_run
                observed_lookahead = []

                def process_batch_result(result_batch, batch_result):
                    observed_lookahead.append(
                        req.decode_batch_idx
                        - result_batch.mamba_decode_batch_idx_cpu[0]
                    )
                    processor.process_batch_result_decode(result_batch, batch_result)

                scheduler.process_batch_result = process_batch_result

                with (
                    # The mamba predicates and the track interval read the
                    # published bags, so publish the configuration under test
                    # (non-lazy extra buffer, interval 4); observability and
                    # disagg reads are served by the same publish at their
                    # defaults.
                    get_context().override_server_args(
                        mamba_radix_cache_strategy="extra_buffer",
                        mamba_track_interval=4,
                    ),
                    patch(
                        "sglang.srt.managers.schedule_batch.alloc_for_decode",
                        return_value=torch.tensor([3], dtype=torch.int64),
                    ),
                    patch(
                        "sglang.srt.managers.schedule_batch.set_mamba_track_indices_from_reqs"
                    ),
                    patch.object(torch.Tensor, "pin_memory", lambda tensor: tensor),
                    patch.object(
                        SchedulerBatchResultProcessor,
                        "_mamba_prefix_cache_update",
                    ) as cache_update,
                ):
                    with self.assertRaises(StopIteration):
                        scheduler.event_loop_overlap()

                self.assertEqual(observed_lookahead, [expected_lookahead])
                if expected_lookahead == 0:
                    cache_update.assert_not_called()
                else:
                    self.assertTrue(cache_update.call_args.kwargs["known_boundary"])


if __name__ == "__main__":
    unittest.main()
