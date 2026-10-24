import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.speculative.eagle_info import EagleDraftInput
from sglang.srt.speculative.multi_layer_eagle_worker_v2 import (
    MultiLayerEagleDraftWorker,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

MODULE = "sglang.srt.speculative.multi_layer_eagle_worker_v2"


class TestMultiLayerDraftBatch(CustomTestCase):
    def test_per_depth_padding_preserves_chain_and_target_state(self):
        for prefill in (True, False):
            for chain in (True, False):
                for idle in (True, False):
                    with self.subTest(prefill=prefill, chain=chain, idle=idle):
                        self.check_depths(prefill, chain, idle)

    def check_depths(self, prefill, chain, idle):
        bs, tokens = (0, 0) if idle else (1, 4)
        mode = (
            ForwardMode.IDLE
            if idle
            else ForwardMode.EXTEND
            if prefill
            else ForwardMode.DRAFT_EXTEND_V2
        )
        hidden = torch.arange(tokens * 2, dtype=torch.float32).reshape(tokens, 2)
        batch = SimpleNamespace(
            forward_mode=mode,
            input_ids=torch.arange(tokens),
            seq_lens=torch.full((bs,), 4, dtype=torch.int32),
        )
        fb = ForwardBatch(
            forward_mode=mode,
            batch_size=bs,
            input_ids=batch.input_ids,
            req_pool_indices=torch.arange(bs),
            seq_lens=batch.seq_lens,
            out_cache_loc=torch.arange(tokens),
            seq_lens_sum=tokens,
            positions=torch.arange(tokens),
        )
        result = SimpleNamespace(
            logits_output=SimpleNamespace(hidden_states=hidden),
            next_token_ids=torch.zeros(bs, dtype=torch.int64),
            accept_lens=torch.ones(bs, dtype=torch.int32),
            next_draft_input=EagleDraftInput(),
        )
        seen = []

        def make_runner(depth):
            def plan(padded):
                self.assertEqual(padded.input_ids.shape[0], 8)
                self.assertEqual(padded.spec_info.hidden_states.shape[0], 8)

            runner = SimpleNamespace(
                req_to_token_pool=object(),
                token_to_kv_pool=object(),
                attn_backend=SimpleNamespace(
                    get_cuda_graph_seq_len_fill_value=lambda: 1,
                    get_cpu_graph_seq_len_fill_value=lambda: 1,
                    init_forward_metadata=plan,
                ),
            )

            def forward(step_batch):
                self.assertEqual(step_batch.input_ids.shape[0], tokens)
                expected = hidden + (depth if chain else 0)
                torch.testing.assert_close(step_batch.spec_info.hidden_states, expected)
                if not prefill:
                    torch.testing.assert_close(
                        step_batch.spec_info.num_accept_tokens, result.accept_lens
                    )
                seen.append(step_batch)
                step_batch._pad_inputs_to_size(runner, 8, 2)
                if not idle:
                    runner.attn_backend.init_forward_metadata(step_batch)
                logits = torch.zeros((bs, 4))
                logits[:, depth] = 1
                return SimpleNamespace(
                    logits_output=SimpleNamespace(
                        next_token_logits=logits,
                        hidden_states=expected + 1,
                    )
                )

            runner.forward = forward
            return runner

        worker = SimpleNamespace(
            draft_runner_list=[make_runner(i) for i in range(3)],
            model_config=SimpleNamespace(vocab_size=4),
            speculative_algorithm=SpeculativeAlgorithm.EAGLE,
            speculative_num_steps=3,
            speculative_num_draft_tokens=4,
            draft_extend_num_front_tokens=0,
            chain_mtp_hidden_states=chain,
            use_rejection_sampling=False,
            topk=1,
            device="cpu",
            cuda_graph_runner_for_draft_extend=None,
            plan_stream=None,
            plan_stream_ctx=nullcontext(),
            _apply_deferred_mamba_init_to_draft_pools=Mock(),
            _seed_boundary_kv_stash=Mock(),
            _fill_boundary_kv_front_and_update_stash=Mock(),
            _compute_boundary_kv_locs_positions=lambda _: (None, None),
        )

        def prepare(spec, *_args, **_kwargs):
            fb.spec_info = spec
            return fb

        def init_new(source, *_args, **_kwargs):
            fb.spec_info = source.spec_info
            return fb

        with (
            patch.object(ForwardBatch, "init_new", side_effect=init_new),
            patch(f"{MODULE}.prepare_for_draft_extend", side_effect=prepare),
            patch(f"{MODULE}.require_gathered_buffer", return_value=False),
            patch(f"{MODULE}.rotate_input_ids"),
        ):
            if prefill:
                output = MultiLayerEagleDraftWorker._draft_extend_for_prefill(
                    worker, batch, hidden, result.next_token_ids
                )
            else:
                MultiLayerEagleDraftWorker._draft_extend_for_decode(
                    worker, batch, result
                )
                output = result.next_draft_input
        self.assertEqual(len(seen), 3)
        self.assertEqual(fb.input_ids.shape[0], tokens)
        torch.testing.assert_close(
            fb.spec_info.hidden_states, hidden + (2 if chain else 0)
        )
        torch.testing.assert_close(output.topk_index, torch.arange(3).expand(bs, 3))
        torch.testing.assert_close(
            hidden, torch.arange(tokens * 2, dtype=torch.float32).reshape(tokens, 2)
        )


if __name__ == "__main__":
    unittest.main()
