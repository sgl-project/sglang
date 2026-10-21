"""Deferred PD draft initialization must consume the complete target prefix."""

import contextlib
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.speculative import eagle_disaggregation
from sglang.srt.speculative import eagle_worker_v2 as eagle
from sglang.srt.speculative.eagle_worker_v2 import EAGLEWorkerV2
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


class TestDeferredDraftPrefillWorker(CustomTestCase):
    def _decode_case(self):
        worker = object.__new__(EAGLEWorkerV2)
        worker._draft_worker = SimpleNamespace(
            draft_owns_attention=False,
            _draft_extend_for_prefill=Mock(return_value=object()),
        )
        batch = SimpleNamespace(
            reqs=[
                SimpleNamespace(
                    origin_input_ids=[3, 4, 5], output_ids=[9], multimodal_inputs=None
                ),
                SimpleNamespace(
                    origin_input_ids=[6, 7], output_ids=[10], multimodal_inputs=None
                ),
            ],
            prefix_lens=[0, 0],
            forward_mode=ForwardMode.PREBUILT,
            input_ids=None,
            spec_info=None,
            device="cpu",
        )
        scope = contextlib.ExitStack()
        scope.enter_context(
            patch.object(
                eagle,
                "get_disagg",
                return_value=SimpleNamespace(
                    disaggregation_defer_draft_prefill=True,
                    disaggregation_mode="decode",
                ),
            )
        )
        for name in (
            "draft_tp_context",
            "speculative_moe_backend_context",
            "speculative_moe_a2a_backend_context",
            "spec_stage_span",
        ):
            scope.enter_context(
                patch.object(
                    eagle, name, side_effect=lambda *args: contextlib.nullcontext()
                )
            )
        self.addCleanup(scope.close)
        return worker, batch

    def test_complete_prefix_is_extended_without_mutating_prebuilt_batch(self):
        worker, batch = self._decode_case()
        hidden = torch.arange(20).reshape(5, 4).float()
        result = worker.initialize_deferred_draft_prefill(batch, hidden)
        draft_call = worker._draft_worker._draft_extend_for_prefill
        self.assertIs(result, draft_call.return_value)
        draft_batch, actual_hidden, first_tokens = draft_call.call_args.args
        self.assertIsNot(draft_batch, batch)
        self.assertIs(actual_hidden, hidden)
        self.assertEqual(draft_batch.input_ids.tolist(), [3, 4, 5, 6, 7])
        self.assertEqual(draft_batch.extend_lens, [3, 2])
        self.assertEqual(draft_batch.prefix_lens, [0, 0])
        self.assertEqual(first_tokens.tolist(), [9, 10])
        self.assertEqual(draft_batch.forward_mode, ForwardMode.EXTEND)
        self.assertTrue(draft_batch.is_extend_in_batch)
        self.assertEqual(batch.forward_mode, ForwardMode.PREBUILT)
        self.assertIsNone(batch.input_ids)
        self.assertIsNone(batch.spec_info)

    def test_incomplete_prefix_never_enters_draft_forward(self):
        worker, batch = self._decode_case()
        with self.assertRaisesRegex(RuntimeError, "Incomplete target hidden"):
            worker.initialize_deferred_draft_prefill(batch, torch.zeros(1, 4))
        worker._draft_worker._draft_extend_for_prefill.assert_not_called()

    def test_cached_prefix_is_rejected(self):
        worker, batch = self._decode_case()
        batch.prefix_lens[0] = 1
        with self.assertRaisesRegex(RuntimeError, "cached draft prefix"):
            worker.initialize_deferred_draft_prefill(batch, torch.zeros(5, 4))
        worker._draft_worker._draft_extend_for_prefill.assert_not_called()

    def test_d_local_dsa_seed_is_not_remapped_as_a_p_wire_position(self):
        seed = torch.tensor([801, 990, -1], dtype=torch.int32)
        batch = SimpleNamespace(
            reqs=[
                SimpleNamespace(
                    output_topk_p=torch.tensor([1.0]),
                    output_topk_index=torch.tensor([3]),
                    hidden_states_tensor=torch.zeros(4),
                    output_dsa_topk_indices=seed,
                )
            ],
            device="cpu",
            enable_overlap=False,
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(
                    architectures=["GlmMoeDsaForCausalLM"],
                    index_share_for_mtp_iteration=True,
                    index_topk=3,
                )
            ),
        )
        spec = SimpleNamespace(
            speculative_eagle_topk=1,
            speculative_num_steps=4,
            enable_multi_layer_eagle=False,
            speculative_use_rejection_sampling=False,
        )
        with (
            patch.object(eagle_disaggregation, "get_spec", return_value=spec),
            patch.object(
                eagle_disaggregation,
                "get_disagg",
                return_value=SimpleNamespace(disaggregation_defer_draft_prefill=True),
            ),
            patch.object(
                eagle_disaggregation,
                "should_remap_pd_dsa_seed_to_local_slots",
                return_value=True,
            ),
        ):
            draft_input = eagle_disaggregation.build_eagle_disagg_draft_input(
                batch, torch.tensor([2]), None
            )
        torch.testing.assert_close(draft_input.dsa_topk_indices, seed[None])

    def test_p_stages_full_hidden_before_publishing_target_end(self):
        worker = object.__new__(EAGLEWorkerV2)
        hidden = torch.arange(20).reshape(5, 4).float()
        result = SimpleNamespace(
            logits_output=SimpleNamespace(hidden_states=hidden), kv_loc_plan=None
        )
        worker._target_worker = SimpleNamespace(
            forward_batch_generation=Mock(return_value=result)
        )
        worker._draft_worker = None
        worker.speculative_algorithm = SimpleNamespace(is_standalone=lambda: False)
        pools = [Mock(), Mock()]
        reqs = [SimpleNamespace(deferred_draft_prefill_pool=pool) for pool in pools]
        batch = SimpleNamespace(
            reqs=reqs,
            prefix_lens=[0, 2],
            extend_lens=[3, 2],
            seq_lens=torch.tensor([3, 4]),
        )

        def publish(_):
            # The transfer's existing target-end fence must cover hidden copies.
            self.assertEqual([pool.write_chunk.call_count for pool in pools], [1, 1])

        with (
            patch.object(
                eagle,
                "get_disagg",
                return_value=SimpleNamespace(
                    disaggregation_defer_draft_prefill=True,
                    disaggregation_mode="prefill",
                ),
            ),
            patch.object(
                eagle,
                "get_parallel",
                return_value=SimpleNamespace(
                    pp_group=SimpleNamespace(is_last_rank=True)
                ),
            ),
        ):
            self.assertIs(
                worker._forward_prefill_batch(batch, on_publish=publish), result
            )
        first_call, second_call = [pool.write_chunk.call_args.args for pool in pools]
        self.assertEqual(first_call[2:], (0, 3))
        self.assertEqual(second_call[2:], (2, 4))
        torch.testing.assert_close(first_call[1], hidden[:3])
        torch.testing.assert_close(second_call[1], hidden[3:])


if __name__ == "__main__":
    unittest.main()
