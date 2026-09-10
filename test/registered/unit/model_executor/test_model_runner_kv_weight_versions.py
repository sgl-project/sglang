import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.mem_cache.kv_weight_version_tracker import KvWeightVersionRecord
from sglang.srt.model_executor.model_runner import ModelRunner, ModelRunnerOutput
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestModelRunnerKvWeightVersions(CustomTestCase):
    def test_only_enabled_target_forwards_capture_slots(self) -> None:
        """Draft forwards and disabled tracking never publish target provenance."""
        for enabled, draft, has_slots in (
            (False, False, True),
            (True, True, True),
            (True, False, False),
            (True, False, True),
        ):
            with self.subTest(enabled=enabled, draft=draft, has_slots=has_slots):
                runner = ModelRunner.__new__(ModelRunner)
                runner.is_draft_worker = draft
                runner.server_args = SimpleNamespace(
                    enable_prefill_weight_versions=enabled
                )
                batch = SimpleNamespace(
                    out_cache_loc=torch.tensor([4, 5]) if has_slots else None
                )
                with patch(
                    "sglang.srt.mem_cache.kv_weight_version_tracker.get_serving",
                    return_value=SimpleNamespace(weight_version="v0"),
                ):
                    record = KvWeightVersionRecord.maybe_capture(
                        model_runner=runner, forward_batch=batch
                    )

                if enabled and not draft and has_slots:
                    batch.out_cache_loc.fill_(9)
                    self.assertEqual(record.slot_indices.tolist(), [4, 5])
                    self.assertEqual(record.version, "v0")
                else:
                    self.assertIsNone(record)

    def test_target_worker_preserves_forward_provenance(self) -> None:
        """TP result construction retains the target forward record."""
        record = KvWeightVersionRecord.capture(
            slot_indices=torch.tensor([4, 5]), version="v0"
        )
        output = ModelRunnerOutput(
            logits_output=None,
            can_run_graph=False,
            kv_weight_version_record=record,
        )
        worker = SimpleNamespace(
            is_dllm=lambda: False,
            pp_group=SimpleNamespace(is_last_rank=True),
            model_runner=SimpleNamespace(forward=lambda *args, **kwargs: output),
        )
        batch = SimpleNamespace(
            apply_deprecated_skip_attn_backend_init=lambda value: None
        )

        result = TpModelWorker.forward_batch_generation(
            worker, batch=None, forward_batch=batch, is_verify=True
        )

        self.assertIs(result.kv_weight_version_record, record)

    def test_result_copies_record_before_publishing_completion(self) -> None:
        """The shared result-copy path includes KV slots before its completion event."""
        record = KvWeightVersionRecord.capture(
            slot_indices=torch.tensor([4, 5]), version="v0"
        )
        source = record.slot_indices
        completed: list[bool] = []

        def publish_completion() -> None:
            self.assertIsNot(record.slot_indices, source)
            completed.append(True)

        result = GenerationBatchResult(
            next_token_ids=torch.tensor([1]),
            kv_weight_version_record=record,
            copy_done=SimpleNamespace(record=publish_completion),
        )
        with patch(
            "sglang.srt.managers.utils._async_d2h",
            side_effect=lambda tensor: tensor.clone(),
        ):
            result.copy_to_cpu(return_logprob=False, return_hidden_states=False)

        self.assertEqual(record.slot_indices.tolist(), [4, 5])
        self.assertEqual(completed, [True])


if __name__ == "__main__":
    unittest.main()
