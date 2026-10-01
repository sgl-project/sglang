"""PP result relay preserves accepted prefixes, local slots and draft state."""

import copy
import multiprocessing as mp
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import torch.distributed as dist
from sglang.srt.managers.scheduler_pp_mixin import (
    SchedulerPPMixin,
    _pp_can_skip_output_comm,
)
from sglang.srt.speculative.dspark_components.dspark_pp_result import (
    pack_dspark_pp_result,
    unpack_dspark_pp_result,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.dspark_pp_result_utils import (
    check_pp_result,
    make_pp_result_fixture,
    receive_pp_result,
    relay_pp_result,
)
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=45, suite="base-a-test-cpu")

RUNTIME = "sglang.srt.managers.scheduler_pp_mixin.get_disagg"


class TestDSparkPPResult(CustomTestCase):
    def setUp(self):
        runtime = patch(
            RUNTIME, return_value=SimpleNamespace(disaggregation_mode="null")
        )
        runtime.start()
        self.addCleanup(runtime.stop)

    @unittest.skipUnless(
        dist.is_available() and dist.is_gloo_available(), "Gloo required"
    )
    def test_four_stage_transport_commits_only_accepted_tokens(self):
        with tempfile.TemporaryDirectory() as root:
            context = mp.get_context("spawn")
            workers = [
                context.Process(target=relay_pp_result, args=(rank, root))
                for rank in range(4)
            ]
            try:
                for worker in workers:
                    worker.start()
                deadline = time.monotonic() + 120
                for worker in workers:
                    worker.join(timeout=max(0, deadline - time.monotonic()))
                self.assertEqual([worker.exitcode for worker in workers], [0] * 4)
            finally:
                for worker in workers:
                    if worker.pid is not None and worker.is_alive():
                        worker.terminate()
                        worker.join(timeout=5)
                        if worker.is_alive():
                            worker.kill()
                            worker.join(timeout=5)

    def test_stale_reordered_and_malformed_frames_are_rejected(self):
        _, batch, source, _ = make_pp_result_fixture()
        packet = pack_dspark_pp_result(source, batch)
        cases = []
        for key, value in (
            ("version", 2),
            ("phase", "prefill"),
            ("requests", list(reversed(packet["dspark_result"]["requests"]))),
            ("stride", 3),
            ("has_cap_lens", False),
        ):
            altered = copy.deepcopy(packet)
            altered["dspark_result"][key] = value
            cases.append(altered)
        for name in ("dspark_result", "dspark_accept_lens", "next_token_ids"):
            altered = dict(packet)
            del altered[name]
            cases.append(altered)
        cases.extend(
            (
                packet | {"dspark_unknown": torch.tensor(0)},
                packet | {"dspark_accept_lens": torch.ones(3)},
                packet | {"next_token_ids": torch.ones(3, 4, dtype=torch.int64)},
            )
        )
        for index, altered in enumerate(cases):
            with self.subTest(index=index), self.assertRaises(ValueError):
                unpack_dspark_pp_result(altered, batch, can_run_cuda_graph=False)
        batch.reqs[0].kv_committed_len += 1
        with self.assertRaisesRegex(ValueError, "different request/step"):
            unpack_dspark_pp_result(packet, batch, can_run_cuda_graph=False)
        batch.reqs[0].kv_committed_len -= 1
        batch.reqs[0].output_ids.append(123)
        with self.assertRaisesRegex(ValueError, "different request/step"):
            unpack_dspark_pp_result(packet, batch, can_run_cuda_graph=False)
        batch.spec_algorithm = SpeculativeAlgorithm.NONE
        with self.assertRaisesRegex(ValueError, "algorithm"):
            unpack_dspark_pp_result(packet, batch, can_run_cuda_graph=False)

    def test_invalid_commits_do_not_advance_batch_or_request_state(self):
        for name, value in (
            ("dspark_accept_lens", [0, 3, 4]),
            ("dspark_accept_lens", [1, 3, 5]),
            ("dspark_bonus_tokens", [999, 43, 54]),
            ("dspark_new_seq_lens", [6, 12, 18]),
            ("dspark_block_accept_lens", [1, 2, 4]),
            ("dspark_cap_lens", [2, 3, 5]),
        ):
            with self.subTest(name=name, value=value):
                scheduler, batch, source, observed = make_pp_result_fixture()
                packet = pack_dspark_pp_result(source, batch)
                packet[name] = torch.tensor(value)
                result = receive_pp_result(scheduler, batch, packet)
                with self.assertRaises(ValueError):
                    SchedulerPPMixin._pp_process_batch_result(scheduler, batch, result)
                self.assertIsNone(batch.spec_info)
                self.assertEqual(batch.seq_lens.tolist(), [5, 9, 13])
                self.assertEqual([r.kv_committed_len for r in batch.reqs], [5, 9, 13])
                self.assertEqual(observed, [])

    def test_host_payload_waits_for_its_d2h_completion(self):
        scheduler, batch, source, observed = make_pp_result_fixture()
        completed_tokens = source.next_token_ids.clone()
        source.next_token_ids.fill_(-999)
        source.copy_done = SimpleNamespace(
            synchronize=Mock(
                side_effect=lambda: source.next_token_ids.copy_(completed_tokens)
            )
        )
        packet = pack_dspark_pp_result(source, batch)
        source.copy_done.synchronize.assert_called_once_with()
        result = receive_pp_result(scheduler, batch, packet)
        check_pp_result(scheduler, batch, result, observed, prefill=False)

    def test_prefill_mismatch_and_missing_next_draft_state_fail(self):
        for name, value in (
            ("dspark_bonus_tokens", [31, 41, 999]),
            ("dspark_new_seq_lens", [5, 9, 14]),
        ):
            with self.subTest(name=name):
                scheduler, batch, source, observed = make_pp_result_fixture(
                    prefill=True
                )
                packet = pack_dspark_pp_result(source, batch)
                packet[name] = torch.tensor(value)
                result = receive_pp_result(scheduler, batch, packet)
                with self.assertRaisesRegex(ValueError, "prefill state"):
                    SchedulerPPMixin._pp_process_batch_result(scheduler, batch, result)
                self.assertEqual(observed, [])
        source.next_draft_input = None
        with self.assertRaisesRegex(ValueError, "next-draft state"):
            pack_dspark_pp_result(source, batch)

    def test_retracted_request_is_not_committed_again(self):
        scheduler, batch, source, _ = make_pp_result_fixture()
        packet = pack_dspark_pp_result(source, batch)
        batch.reqs[1].is_retracted = True
        result = receive_pp_result(scheduler, batch, packet)
        SchedulerPPMixin._pp_process_batch_result(scheduler, batch, result)
        self.assertEqual([r.kv_committed_len for r in batch.reqs], [6, 9, 17])
        self.assertEqual([r.spec_verify_ct for r in batch.reqs], [1, 0, 1])

    def test_chunked_prefill_keeps_draft_state_and_pd_teacher_handoff(self):
        scheduler, batch, source, observed = make_pp_result_fixture(prefill=True)
        batch.reqs = batch.reqs[:1]
        batch.contains_last_prefill_chunk = False
        with patch(
            "sglang.srt.managers.scheduler_pp_mixin.envs.SGLANG_PP_SKIP_PURE_CHUNKED_OUTPUT_COMM.get",
            return_value=True,
        ):
            self.assertFalse(_pp_can_skip_output_comm(batch))
            batch.spec_algorithm = SpeculativeAlgorithm.NONE
            self.assertTrue(_pp_can_skip_output_comm(batch))
        scheduler, batch, source, observed = make_pp_result_fixture(prefill=True)
        capture = SimpleNamespace(
            pack_pp_handoffs=Mock(return_value=[None, b"teacher", None]),
            accept_pp_handoffs=Mock(),
        )
        scheduler.tp_worker.training_capture = capture
        with patch(
            RUNTIME, return_value=SimpleNamespace(disaggregation_mode="prefill")
        ):
            packet = SchedulerPPMixin._pp_prepare_tensor_dict(scheduler, source, batch)
            result = receive_pp_result(scheduler, batch, packet)
        capture.accept_pp_handoffs.assert_called_once_with(
            batch, [None, b"teacher", None]
        )
        check_pp_result(scheduler, batch, result, observed, prefill=True)


if __name__ == "__main__":
    unittest.main()
