"""Eager draft metadata precedes DP padding; cache writes must exclude dummy rows."""

import unittest
from contextlib import nullcontext
from types import MethodType, SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import torch

from sglang.srt.layers.attention.dsa import dsa_indexer_kpool as indexer_module
from sglang.srt.layers.attention.dsa import kpool_fp8_index, kpool_plan
from sglang.srt.layers.attention.dsa.dsa_indexer_kpool import IndexerKPool
from sglang.srt.model_executor import forward_batch_info as fbi
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestKPoolSpeculativePadding(unittest.TestCase):
    def _run(self, *, padded, dual_stream, return_indices, draft_tokens=2):
        # One request on this rank, two on its peers. With steps=1/topk=1,
        # draft_tokens=2, MAX_LEN grows the physical input from two to four rows.
        counts = [draft_tokens] + [draft_tokens * (2 if padded else 1)] * 7
        batch = ForwardBatch(
            forward_mode=ForwardMode.DRAFT_EXTEND_V2,
            batch_size=1,
            input_ids=torch.arange(draft_tokens),
            req_pool_indices=torch.tensor([2]),
            seq_lens=torch.tensor([10]),
            seq_lens_sum=10,
            out_cache_loc=torch.arange(20, 20 + draft_tokens),
            positions=torch.arange(draft_tokens),
            spec_info=SimpleNamespace(
                num_tokens_per_req=draft_tokens, is_draft_input=lambda: False
            ),
            is_extend_in_batch=False,
            global_num_tokens_cpu=counts,
            global_num_tokens_for_logprob_cpu=list(counts),
            global_num_tokens_gpu=torch.tensor(counts),
        )
        plan = kpool_plan._alloc_kpool_write_plan_buffers(
            max_bs=batch.batch_size,
            num_draft_tokens=draft_tokens,
            pool_size=4,
            device=torch.device("cpu"),
            is_verify=True,
            is_v2=True,
        )
        plan.req.copy_(batch.req_pool_indices)
        plan.effective_n_per_batch.fill_(draft_tokens)
        metadata = SimpleNamespace(
            attn_metadata=SimpleNamespace(kpool_write_plan=plan),
            get_seqlens_expanded=lambda: plan.seqlens_per_q,
        )
        batch.mark_forward_metadata_ready()
        runner = SimpleNamespace(
            is_draft_worker=True,
            model_config=SimpleNamespace(),
            attn_tp_sequence_sharded=lambda num_tokens: False,
            attn_backend=SimpleNamespace(get_cpu_graph_seq_len_fill_value=lambda: 1),
        )
        no_prefill_graph = SimpleNamespace(
            graph=SimpleNamespace(
                cuda_graph_config=SimpleNamespace(prefill=SimpleNamespace(bs=[]))
            )
        )
        with (
            patch.multiple(
                fbi,
                get_parallel=lambda: SimpleNamespace(attn_tp_size=1),
                get_exec=lambda: no_prefill_graph,
                _elastic_should_preserve_local_token_counts=lambda **kwargs: False,
                dp_slot_in=lambda per_rank: 0,
                set_dp_buffer_len_from_batch=lambda *args: None,
                set_is_extend_in_batch=lambda *args: None,
                _is_cpu=True,
            ),
            patch.object(fbi, "mambaish_config", return_value=object()),
            patch("sglang.srt.layers.dp_attention.dp_gather_width", return_value=8),
            patch(
                "sglang.srt.batch_overlap.two_batch_overlap.TboForwardBatchPreparer.prepare"
            ),
        ):
            batch.prepare_mlp_sync_batch(runner)
        physical_rows = draft_tokens * (2 if padded else 1)
        self.assertEqual(batch.input_ids.shape[0], physical_rows)
        self.assertFalse(batch.needs_forward_metadata_init())
        self.assertEqual(plan.write_loc.shape, (1, 1))

        x = torch.arange(physical_rows * 128, dtype=torch.float32).reshape(-1, 128)
        key = x.to(torch.bfloat16)
        gate_score = x.clone() if dual_stream else None
        pool = SimpleNamespace(
            index_kpool=4,
            tail_extra_slots=draft_tokens,
            slots_per_page=16,
            index_head_dim=128,
            get_compress_tail_buffers=lambda layer: (
                torch.zeros(4, 4 + draft_tokens, 128, dtype=torch.bfloat16),
                torch.zeros(4, 4 + draft_tokens, 128, dtype=torch.float32),
            ),
            get_index_k_with_scale_buffer=lambda **kwargs: torch.zeros(
                1, 16 * 132, dtype=torch.uint8
            ),
        )
        topk = Mock(side_effect=lambda batch, layer, q, *args: q[:, 0, :1])
        indexer = SimpleNamespace(
            alt_stream=Mock(),
            compress_gate_stream=Mock(),
            index_kpool_compress_ape=torch.zeros(4, 128),
            index_kpool_compress_gate=torch.eye(128),
            block_size=128,
            scale_fmt=None,
            _get_q_k_bf16=Mock(return_value=(key.unsqueeze(1), key, gate_score, None)),
            _resolve_head_gate_weights=lambda x, *args: x[:, :1].unsqueeze(1),
            _get_topk_paged=topk,
        )
        indexer._compute_gate_score_if_missing = MethodType(
            IndexerKPool._compute_gate_score_if_missing, indexer
        )
        # Exercise the real writer's host checks and launch geometry. Only the
        # GPU launch is mocked, so the original bug raises its shape assertion.
        kernel = MagicMock()
        with (
            patch.object(indexer_module, "is_cuda", return_value=True),
            patch.object(indexer_module, "get_token_to_kv_pool", return_value=pool),
            patch.object(torch.cuda, "current_stream", return_value=Mock()),
            patch.object(torch.cuda, "stream", return_value=nullcontext()),
            patch.object(
                kpool_fp8_index, "_kpool_write_tail_and_maybe_compress_kernel", kernel
            ),
        ):
            output = IndexerKPool._forward_cuda_target_verify(
                indexer,
                x=x,
                q_lora=x,
                positions=batch.positions,
                forward_batch=batch,
                layer_id=0,
                act_quant=lambda q, *args: (q, torch.ones(physical_rows, 1, 1)),
                metadata=metadata,
                enable_dual_stream=dual_stream,
                return_indices=return_indices,
            )
        kernel.__getitem__.assert_called_once_with((1,))
        launch = kernel.__getitem__.return_value
        launch.assert_called_once()
        args = launch.call_args.args
        torch.testing.assert_close(args[0], key[:draft_tokens])
        torch.testing.assert_close(args[1], x[:draft_tokens])
        torch.testing.assert_close(args[5], torch.tensor([2]))
        torch.testing.assert_close(args[9], torch.arange(20, 20 + draft_tokens))
        self.assertIs(args[10], plan.effective_n_per_batch)
        self.assertEqual(launch.call_args.kwargs["N"], draft_tokens)
        if return_indices:
            # The query/top-k path retains the physical rows for DP collectives.
            self.assertEqual(output.shape[0], physical_rows)
            self.assertIs(topk.call_args.args[-1], metadata)
        else:
            self.assertIsNone(output)
            topk.assert_not_called()

    def test_draft_extend_writes_only_real_requests_after_dp_padding(self):
        for draft_tokens in (2, 4):
            for dual_stream in (False, True):
                for return_indices in (False, True):
                    with self.subTest(
                        draft_tokens=draft_tokens,
                        dual_stream=dual_stream,
                        return_indices=return_indices,
                    ):
                        self._run(
                            padded=True,
                            dual_stream=dual_stream,
                            return_indices=return_indices,
                            draft_tokens=draft_tokens,
                        )

    def test_unpadded_draft_extend_keeps_all_writes(self):
        self._run(padded=False, dual_stream=False, return_indices=True)


if __name__ == "__main__":
    unittest.main()
