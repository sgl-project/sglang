import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.mem_cache.kv_weight_version_tracker import KvWeightVersionRecord
from sglang.srt.model_executor.model_runner import ModelRunner, ModelRunnerOutput
from sglang.srt.speculative import eagle_worker_common
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
                    out_cache_loc=torch.tensor([4, 5]) if has_slots else None,
                    out_cache_loc_virtual=None,
                    origin_out_cache_loc=None,
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


class TestKvProvenanceForwardPaths(CustomTestCase):
    def test_translated_forward_records_request_table_slot_ids(self) -> None:
        """Physical or rank-local writes retain allocator-global KV provenance."""
        for translated in (False, True):
            with self.subTest(translated=translated):
                runner = SimpleNamespace(
                    is_draft_worker=False,
                    server_args=SimpleNamespace(enable_prefill_weight_versions=True),
                )
                logical_slots = torch.tensor([7, 9])
                batch = SimpleNamespace(
                    out_cache_loc=torch.tensor([1, 2]),
                    out_cache_loc_virtual=logical_slots if translated else None,
                    origin_out_cache_loc=torch.tensor([3, 4])
                    if translated
                    else logical_slots,
                )

                with patch(
                    "sglang.srt.mem_cache.kv_weight_version_tracker.get_serving",
                    return_value=SimpleNamespace(weight_version="v1"),
                ):
                    record = KvWeightVersionRecord.maybe_capture(
                        model_runner=runner, forward_batch=batch
                    )

                logical_slots.fill_(0)
                assert record.slot_indices.tolist() == [7, 9]
                assert record.version == "v1"

    def test_eagle_post_verify_graph_preserves_kv_provenance(self) -> None:
        """Graph-based acceptance retains the target forward's KV version record."""
        record = KvWeightVersionRecord.capture(
            slot_indices=torch.tensor([7, 9]), version="v1"
        )
        output = GenerationBatchResult(kv_weight_version_record=record)
        batch = SimpleNamespace(
            spec_info=SimpleNamespace(),
            seq_lens=torch.tensor([2]),
            input_ids=None,
            out_cache_loc=None,
            has_grammar=False,
            return_logprob=False,
        )
        tokens = torch.tensor([1])
        post = (tokens, tokens, tokens, tokens, tokens, None)
        target_worker = SimpleNamespace(
            forward_batch_generation=lambda **_: output,
            model_runner=SimpleNamespace(canary_manager=None),
        )

        with (
            patch.object(eagle_worker_common, "record_stream_for_v2_verify"),
            patch.object(eagle_worker_common, "record_stream_each"),
            patch.object(
                eagle_worker_common,
                "eagle_prepare_for_verify",
                return_value=(SimpleNamespace(), True),
            ),
        ):
            result = eagle_worker_common.run_eagle_verify(
                batch=batch,
                target_worker=target_worker,
                req_to_token_pool=None,
                token_to_kv_pool_allocator=SimpleNamespace(get_kvcache=lambda: None),
                plan_stream=None,
                plan_stream_ctx=nullcontext(),
                topk=1,
                num_draft_tokens=2,
                device="cpu",
                metadata_ready_pre_pad=False,
                finalize_tree_path=True,
                post_verify_graphs=SimpleNamespace(run=lambda **_: post),
            )

        assert result.kv_weight_version_record is record


if __name__ == "__main__":
    unittest.main()
