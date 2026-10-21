"""Core target-only P and draft-only D handoff semantics."""

import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.configs.draft_parallel import DraftParallelLayout
from sglang.srt.disaggregation.deferred_draft_prefill import DeferredDraftPrefillPool
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import get_context
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.speculative import eagle_disaggregation
from sglang.srt.speculative import eagle_worker_v2 as eagle
from sglang.srt.speculative.eagle_worker_v2 import EAGLEWorkerV2
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


def _request(rid, tokens):
    return Req(rid, "", array("q", tokens), SamplingParams(max_new_tokens=8))


class TestDeferredDraftPrefillWorker(CustomTestCase):
    def test_p_captures_both_chunks_before_publishing_target_end(self):
        req = _request("prefill", [3, 4, 5, 6, 7])
        pool = DeferredDraftPrefillPool(1, 8, 4, torch.float32, "cpu")
        pool.alloc(req)
        req.deferred_draft_prefill_pool = pool
        hidden = torch.arange(20, dtype=torch.float32).reshape(5, 4)
        outputs = [
            SimpleNamespace(
                logits_output=SimpleNamespace(hidden_states=chunk), kv_loc_plan=None
            )
            for chunk in (hidden[:2], hidden[2:])
        ]
        worker = object.__new__(EAGLEWorkerV2)
        worker._target_worker = SimpleNamespace(
            forward_batch_generation=Mock(side_effect=outputs)
        )
        worker._draft_worker = None
        worker.speculative_algorithm = SimpleNamespace(is_standalone=lambda: False)
        published = []

        def publish(seq_lens):
            end = seq_lens.item()
            torch.testing.assert_close(pool.buffer[0, :end], hidden[:end])
            if end < len(req.origin_input_ids):
                with self.assertRaises(RuntimeError):
                    pool.get(req)
            published.append(end)

        with (
            get_context().override_server_args(
                disaggregation_defer_draft_prefill=True, disaggregation_mode="prefill"
            ),
            patch.object(
                eagle,
                "get_parallel",
                return_value=SimpleNamespace(
                    pp_group=SimpleNamespace(is_last_rank=True)
                ),
            ),
        ):
            for output, start, end in zip(outputs, (0, 2), (2, 5), strict=True):
                batch = SimpleNamespace(
                    reqs=[req],
                    prefix_lens=[start],
                    extend_lens=[end - start],
                    seq_lens=torch.tensor([end]),
                )
                self.assertIs(worker._forward_prefill_batch(batch, publish), output)
        self.assertEqual(published, [2, 5])
        torch.testing.assert_close(pool.get(req), hidden)
        self.assertEqual(pool.transfer_indices(req), list(range(5)))

    def test_d_initializes_only_draft_without_changing_target_batch(self):
        reqs = [_request("a", [3, 4, 5]), _request("b", [6, 7])]
        for index, req in enumerate(reqs):
            req.output_ids = [9 + index]
            req.kv.req_pool_idx = 20 + index
        worker = object.__new__(EAGLEWorkerV2)
        target_forward = Mock(side_effect=AssertionError("Target replayed on D"))
        worker._target_worker = SimpleNamespace(forward_batch_generation=target_forward)
        draft_forward = Mock(return_value=object())
        worker._draft_worker = SimpleNamespace(
            parallel_layout=DraftParallelLayout(tp_group=None, attn_tp_size=4),
            _draft_extend_for_prefill=draft_forward,
        )
        relay = object()
        batch = SimpleNamespace(
            reqs=reqs,
            prefix_lens=[0, 0],
            forward_mode=ForwardMode.PREBUILT,
            input_ids=None,
            spec_info=relay,
            device="cpu",
        )
        hidden = torch.arange(20, dtype=torch.float32).reshape(5, 4)
        with get_context().override_server_args(
            disaggregation_defer_draft_prefill=True, disaggregation_mode="decode"
        ):
            result = worker.initialize_deferred_draft_prefill(batch, hidden)
        self.assertIs(result, draft_forward.return_value)
        draft_batch, actual_hidden, first_tokens = draft_forward.call_args.args
        self.assertIsNot(draft_batch, batch)
        self.assertIs(actual_hidden, hidden)
        self.assertEqual(draft_batch.input_ids.tolist(), [3, 4, 5, 6, 7])
        self.assertEqual(draft_batch.extend_lens, [3, 2])
        self.assertEqual(first_tokens.tolist(), [9, 10])
        self.assertEqual(draft_batch.forward_mode, ForwardMode.EXTEND)
        target_forward.assert_not_called()
        self.assertEqual(batch.forward_mode, ForwardMode.PREBUILT)
        self.assertIsNone(batch.input_ids)
        self.assertIs(batch.spec_info, relay)
        self.assertEqual([req.kv.req_pool_idx for req in reqs], [20, 21])
        self.assertEqual([req.output_ids for req in reqs], [[9], [10]])

    def test_d_local_dsa_seed_is_not_remapped_again(self):
        req = _request("decode", [3, 4])
        seed = torch.tensor([801, 990, -1], dtype=torch.int32)
        req.output_topk_p = torch.tensor([1.0])
        req.output_topk_index = torch.tensor([3])
        req.hidden_states_tensor = torch.zeros(4)
        req.output_dsa_topk_indices = seed
        batch = SimpleNamespace(
            reqs=[req],
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
        with (
            get_context().override_server_args(
                disaggregation_defer_draft_prefill=True,
                speculative_eagle_topk=1,
                speculative_num_steps=4,
                enable_multi_layer_eagle=False,
                speculative_use_rejection_sampling=False,
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


if __name__ == "__main__":
    unittest.main()
