import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.srt.speculative.dspark_components.dspark_worker_v2 import DSparkWorkerV2
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=1, stage="stage-b", runner_config="1-gpu-small-amd")


class TestDSparkC128Cleanup(unittest.TestCase):
    @staticmethod
    def _worker(token_to_kv_pool):
        worker = object.__new__(DSparkWorkerV2)
        worker.model_runner = SimpleNamespace(token_to_kv_pool=token_to_kv_pool)
        worker.verify_num_draft_tokens = 6
        return worker

    @staticmethod
    def _batch(*, idle=False):
        return SimpleNamespace(
            forward_mode=SimpleNamespace(is_idle=lambda: idle),
            req_pool_indices=torch.tensor([2, 5], dtype=torch.int64),
        )

    def test_cleanup_uses_final_committed_lengths(self):
        cleanup = MagicMock()
        worker = self._worker(
            SimpleNamespace(clear_unaccepted_c128_draft_states=cleanup)
        )
        batch = self._batch()
        prefix_lens = torch.tensor([8192, 4096], dtype=torch.int64)
        commit_lens = torch.tensor([2, 5], dtype=torch.int32)

        worker._clear_unaccepted_target_c128_states(
            batch=batch,
            seq_lens_pre_verify=prefix_lens,
            commit_lens=commit_lens,
        )

        cleanup.assert_called_once_with(
            batch.req_pool_indices,
            prefix_lens,
            commit_lens,
            6,
        )

    def test_cleanup_is_optional_and_skips_idle(self):
        worker = self._worker(SimpleNamespace())
        worker._clear_unaccepted_target_c128_states(
            batch=self._batch(),
            seq_lens_pre_verify=torch.tensor([8192, 4096]),
            commit_lens=torch.tensor([1, 1]),
        )

        cleanup = MagicMock()
        worker.model_runner.token_to_kv_pool = SimpleNamespace(
            clear_unaccepted_c128_draft_states=cleanup
        )
        worker._clear_unaccepted_target_c128_states(
            batch=self._batch(idle=True),
            seq_lens_pre_verify=torch.tensor([8192, 4096]),
            commit_lens=torch.tensor([1, 1]),
        )

        cleanup.assert_not_called()

    def test_final_war_event_follows_prefill_or_decode(self):
        for is_extend in (True, False):
            with self.subTest(is_extend=is_extend):
                calls = []
                event = MagicMock()
                worker = object.__new__(DSparkWorkerV2)
                worker.device = torch.device("cpu")
                worker.model_runner = SimpleNamespace(shared_read_done_event=object())
                worker._verify_planner = MagicMock()
                worker._observers = MagicMock()
                worker._forward_prefill = MagicMock(
                    side_effect=lambda *_: calls.append("forward") or "prefill"
                )
                worker._forward_decode = MagicMock(
                    side_effect=lambda *_: calls.append("forward") or "decode"
                )
                batch = SimpleNamespace(
                    forward_mode=SimpleNamespace(is_extend=lambda: is_extend),
                    is_extend_in_batch=False,
                )

                with unittest.mock.patch(
                    "torch.get_device_module",
                    return_value=SimpleNamespace(
                        Event=lambda: calls.append("event") or event
                    ),
                ):
                    result = worker.forward_batch_generation(batch)

                self.assertEqual(calls, ["forward", "event"])
                self.assertEqual(result, "prefill" if is_extend else "decode")
                event.record.assert_called_once_with()
                self.assertIs(worker.model_runner.shared_read_done_event, event)


if __name__ == "__main__":
    unittest.main()
