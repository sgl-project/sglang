"""CPU checks for DSV4 interleave BCG padding and the eager logits tail."""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.layers.cp.base import get_cp_strategy, init_cp_strategy
from sglang.srt.layers.cp.bcg import (
    PrefillCPBCGInput,
    execute_prefill_cp_bcg,
    filter_prefill_cp_bcg_capture_num_tokens,
    supports_prefill_cp_bcg,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestDSV4PrefillCPBCG(unittest.TestCase):
    def tearDown(self):
        init_cp_strategy(enable_prefill_cp=False, cp_size=1, cp_strategy="zigzag")

    def test_sparse_prefill_never_uses_cp_local_queries(self):
        from sglang.srt.layers.attention.deepseek_v4_backend import (
            DeepseekV4AttnBackend,
        )

        batch = SimpleNamespace(forward_mode=ForwardMode.EXTEND)
        for cp_active in (False, True):
            for forced in (False, True):
                for num_tokens in (256, 11673, 11674, 16384, 32768):
                    with (
                        patch(
                            "sglang.srt.layers.attention.deepseek_v4_backend.is_cp_v2_active",
                            return_value=cp_active,
                        ),
                        patch(
                            "sglang.srt.layers.attention.deepseek_v4_backend.get_platform",
                            return_value=SimpleNamespace(is_sm120=False),
                        ),
                        patch(
                            "sglang.srt.layers.attention.deepseek_v4_backend.envs"
                        ) as envs,
                    ):
                        envs.SGLANG_OPT_FLASHMLA_SPARSE_PREFILL.get.return_value = (
                            forced
                        )
                        selected = DeepseekV4AttnBackend._use_sparse_prefill(
                            None, batch, num_qo_tokens=num_tokens
                        )
                    self.assertEqual(
                        selected, not cp_active and (forced or num_tokens > 11673)
                    )

    def test_sparse_metadata_rejects_short_query_buffer_before_launch(self):
        from sglang.srt.layers.attention.deepseek_v4_backend import (
            DeepseekV4AttnBackend,
        )

        for lengths in ([32768], [8192, 24576]):
            batch = SimpleNamespace(
                seq_lens_cpu=torch.tensor(lengths),
                extend_seq_lens_cpu=lengths,
            )
            # No device metadata or pool is present: the invariant must fail
            # before reading those fields or launching the index combiner.
            with self.assertRaisesRegex(AssertionError, "allocated 16384.*32768"):
                DeepseekV4AttnBackend._build_sparse_prefill_chunk_cache(
                    None, batch, num_qo_tokens=16384
                )

    def test_inactive_cp_batch_cannot_replay_cp_body(self):
        from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import (
            PrefillCudaGraphRunner,
        )

        init_cp_strategy(enable_prefill_cp=True, cp_size=2, cp_strategy="interleave")
        runner = SimpleNamespace(
            can_replay_locally=lambda **kwargs: True,
            _has_inactive_dp_rank=lambda batch: False,
            enable_cp_v2_bcg_capture=True,
            enable_lora=False,
        )
        batch = SimpleNamespace(
            global_num_tokens_cpu=None,
            batch_size=1,
            input_ids=[1],
            input_embeds=None,
            replace_embeds=None,
            extend_prefix_lens_cpu=[0],
            forward_mode=ForwardMode.EXTEND,
            capture_hidden_mode=None,
            return_logprob=False,
            extend_seq_lens_cpu=[1],
        )
        self.assertFalse(PrefillCudaGraphRunner.can_run_graph(runner, batch))

    def test_attention_keeps_global_write_bucket_and_live_sequence_lengths(self):
        from sglang.srt.layers.attention.deepseek_v4_backend import (
            DeepseekV4AttnBackend,
        )

        for use_graph, write_tokens in ((False, 11), (True, 16)):
            core = Mock()
            backend = SimpleNamespace(
                expand_prefill_casually=Mock(
                    return_value=(torch.ones(16), torch.zeros(16))
                ),
                make_core_attn_metadata=Mock(return_value=core),
                init_forward_metadata_indexer=Mock(return_value=None),
                req_to_token=object(),
                token_to_kv_pool=object(),
            )
            out_loc = torch.arange(write_tokens)
            batch = SimpleNamespace(
                attn_cp_metadata=SimpleNamespace(per_rank_actual_token=[8, 8])
            )
            with (
                patch(
                    "sglang.srt.layers.attention.deepseek_v4_backend.is_cp_v2_active",
                    return_value=True,
                ),
                patch(
                    "sglang.srt.layers.attention.deepseek_v4_backend.create_paged_compressor_data",
                    return_value=None,
                ) as planner,
                patch("sglang.srt.layers.attention.deepseek_v4_backend.envs") as envs,
            ):
                envs.SGLANG_OPT_USE_ONLINE_COMPRESS.get.return_value = False
                DeepseekV4AttnBackend.init_forward_metadata_prefill(
                    backend,
                    max_seq_len=128,
                    req_pool_indices=torch.tensor([2, 3]),
                    seq_lens=torch.tensor([22, 29]),
                    seq_lens_cpu=[22, 29],
                    out_cache_loc=out_loc,
                    num_tokens=11,
                    extend_seq_lens=torch.tensor([2, 9]),
                    extend_seq_lens_cpu=[2, 9],
                    use_prefill_cuda_graph=use_graph,
                    forward_batch=batch,
                )
            self.assertEqual(
                backend.expand_prefill_casually.call_args.kwargs["num_tokens"], 11
            )
            self.assertEqual(
                backend.expand_prefill_casually.call_args.kwargs["padded_num_tokens"],
                16,
            )
            self.assertEqual(
                backend.make_core_attn_metadata.call_args.kwargs["num_tokens"],
                write_tokens,
            )
            core.apply_cp_reindex.assert_called_once_with(num_tokens=write_tokens)
            if use_graph:
                for call in planner.call_args_list:
                    self.assertEqual(call.kwargs["num_q_tokens"], 16)
                    self.assertIsNone(call.kwargs["seq_lens_cpu"])

    def test_supported_configurations_and_capture_buckets(self):
        cfg = SimpleNamespace(
            enable_prefill_cp=True,
            pp_size=1,
            attn_cp_size=2,
            tp_size=2,
            cp_strategy="interleave",
        )
        with (
            patch("sglang.srt.layers.cp.bcg.resolving_view", return_value=cfg),
            patch("sglang.srt.layers.cp.bcg.resolved_view", return_value=cfg),
            patch(
                "sglang.srt.layers.cp.bcg.attention_backends_of",
                return_value=("dsv4", "dsv4"),
            ),
        ):
            self.assertTrue(supports_prefill_cp_bcg(cfg))
            self.assertEqual(
                filter_prefill_cp_bcg_capture_num_tokens([1, 2, 4, 8], cfg), [2, 4, 8]
            )
            cfg.pp_size = 2
            self.assertFalse(supports_prefill_cp_bcg(cfg))
            cfg.pp_size = 1
            cfg.tp_size = 4
            self.assertFalse(supports_prefill_cp_bcg(cfg))
            cfg.tp_size = 2
            cfg.cp_strategy = "zigzag"
            self.assertFalse(supports_prefill_cp_bcg(cfg))
            self.assertEqual(
                filter_prefill_cp_bcg_capture_num_tokens([1, 2, 4, 8], cfg), [4, 8]
            )

    def test_interleave_capacity_is_independent_of_request_boundaries(self):
        init_cp_strategy(enable_prefill_cp=True, cp_size=4, cp_strategy="interleave")
        inputs = PrefillCPBCGInput(torch.empty(0), torch.empty(0), {16: 4, 32: 8})
        with patch(
            "sglang.srt.layers.cp.bcg.get_cp_padding_align_size", return_value=4
        ):
            for lengths in ([13], [1, 3, 9], [4, 4, 4, 1]):
                self.assertEqual(inputs.required_local_tokens(lengths), 4)
                self.assertEqual(
                    inputs.select_replay_bucket_for_batch(
                        num_tokens=13,
                        extend_seq_lens=lengths,
                        capture_num_tokens=[16, 32],
                        max_padding_factor=2,
                    ),
                    16,
                )
            self.assertEqual(inputs.required_local_tokens([1, 16]), 8)

    @staticmethod
    def _batch(lengths, bucket):
        n = sum(lengths)
        ids = torch.zeros(bucket, dtype=torch.int64)
        ids[:n] = torch.arange(1, n + 1)
        positions = torch.zeros(bucket, dtype=torch.int64)
        positions[:n] = torch.cat([torch.arange(length) + 20 for length in lengths])
        out_cache_loc = torch.zeros(bucket, dtype=torch.int64)
        out_cache_loc[:n] = torch.arange(256, 256 + n)
        return SimpleNamespace(
            input_ids=ids,
            positions=positions,
            out_cache_loc=out_cache_loc,
            forward_mode=ForwardMode.EXTEND,
            extend_num_tokens=n,
            extend_seq_lens_cpu=list(lengths),
            seq_lens_cpu=[length + 20 for length in lengths],
            global_num_tokens_cpu=None,
            attn_cp_metadata=None,
        )

    def test_repeated_padded_replay_preserves_addresses_and_global_order(self):
        for cp_size, bucket in ((2, 16), (4, 16), (4, 18)):
            init_cp_strategy(
                enable_prefill_cp=True, cp_size=cp_size, cp_strategy="interleave"
            )
            inputs = [
                PrefillCPBCGInput(
                    torch.zeros(bucket, 1), torch.zeros(bucket, dtype=torch.int64)
                )
                for _ in range(cp_size)
            ]
            model = SimpleNamespace(
                get_input_embeddings=lambda: lambda ids: ids[:, None].float()
            )
            runner = SimpleNamespace(model_runner=SimpleNamespace(model=model))
            for lengths, capture in (
                ([bucket], True),
                ([2, 9], False),
                ([1, 3, 5], False),
                ([bucket], False),
            ):
                batches, shards = [], []
                for rank in range(cp_size):
                    batch = self._batch(lengths, bucket)
                    cache_loc = batch.out_cache_loc
                    global_positions = batch.positions.clone()
                    with (
                        get_parallel().override(
                            attn_cp_rank=rank, attn_cp_size=cp_size
                        ),
                        patch(
                            "sglang.srt.layers.cp.padding.get_cp_padding_align_size",
                            return_value=cp_size,
                        ),
                    ):
                        inputs[rank].prepare(
                            runner, batch, static_num_tokens=bucket, capture=capture
                        )
                    n = sum(lengths)
                    local_ids = torch.arange(1, n + 1)[rank::cp_size].float()
                    torch.testing.assert_close(
                        batch.input_embeds[: len(local_ids), 0], local_ids
                    )
                    torch.testing.assert_close(
                        batch.positions[: len(local_ids)],
                        global_positions[:n][rank::cp_size],
                    )
                    self.assertEqual(
                        torch.count_nonzero(
                            batch.input_embeds[len(local_ids) :]
                        ).item(),
                        0,
                    )
                    self.assertEqual(
                        torch.count_nonzero(batch.positions[len(local_ids) :]).item(), 0
                    )
                    self.assertEqual(
                        batch.input_embeds.data_ptr(),
                        inputs[rank].input_embeds.data_ptr(),
                    )
                    self.assertEqual(
                        batch.positions.data_ptr(), inputs[rank].positions.data_ptr()
                    )
                    self.assertIs(batch.out_cache_loc, cache_loc)
                    self.assertEqual(batch.out_cache_loc.shape[0], bucket)
                    self.assertEqual(batch.attn_cp_metadata.total_seq_lens, bucket)
                    self.assertEqual(batch.extend_seq_lens_cpu, list(lengths))
                    self.assertEqual(
                        batch.input_embeds.shape[0],
                        inputs[rank].bucket_local_tokens[bucket],
                    )
                    batches.append(batch)
                    shards.append(batch.input_embeds.clone())

                def gather(output, local):
                    output.copy_(torch.cat(shards))

                with (
                    get_parallel().override(
                        attn_cp_rank=0, attn_cp_size=cp_size, attn_cp_group=object()
                    ),
                    patch(
                        "sglang.srt.layers.cp.interleave.attn_cp_all_gather_into_tensor",
                        side_effect=gather,
                    ),
                    patch(
                        "sglang.srt.layers.cp.interleave.is_allocation_symmetric",
                        return_value=False,
                    ),
                    patch(
                        "sglang.srt.layers.cp.interleave.use_symmetric_memory",
                        return_value=nullcontext(),
                    ),
                ):
                    gathered = get_cp_strategy().gather_kv_cache(shards[0], batches[0])
                self.assertEqual(gathered.shape, (bucket, 1))
                torch.testing.assert_close(
                    gathered[:n, 0], torch.arange(1, n + 1).float()
                )
                self.assertEqual(torch.count_nonzero(gathered[n:]).item(), 0)

    def test_logits_tail_gathers_dspark_aux_and_trims_padding(self):
        for capture_aux in (False, True):
            hidden = torch.arange(8).view(8, 1).float()
            pre_head = hidden + 100
            aux = [hidden + 200, hidden + 300]
            body_output = (
                ((hidden, pre_head), aux) if capture_aux else (hidden, pre_head)
            )
            logits = Mock(return_value="logits")
            model = SimpleNamespace(
                capture_aux_hidden_states=capture_aux,
                pp_group=SimpleNamespace(is_last_rank=True),
                logits_processor=logits,
                lm_head=object(),
            )
            runner = SimpleNamespace(
                prefill_cp_bcg_input=SimpleNamespace(live_local_tokens=8),
                model_runner=SimpleNamespace(model=model),
                _prefill_forward_context=lambda *args, **kwargs: nullcontext(),
                backend=SimpleNamespace(replay=lambda *args, **kwargs: body_output),
            )
            batch = SimpleNamespace(input_ids=torch.arange(11))

            def gather(value, *args):
                if isinstance(value, tuple):
                    return tuple(gather(item) for item in value)
                return value.repeat_interleave(2, dim=0)

            with (
                patch(
                    "sglang.srt.layers.cp.bcg.cp_gather_after_forward",
                    side_effect=gather,
                ),
                patch(
                    "sglang.srt.layers.cp.bcg.torch.cuda.current_stream",
                    return_value=None,
                ),
            ):
                self.assertEqual(
                    execute_prefill_cp_bcg(runner, batch, batch, 16, 11), "logits"
                )
            args, kwargs = logits.call_args
            torch.testing.assert_close(args[1], gather(hidden)[:11])
            if capture_aux:
                self.assertNotIn("hidden_states_before_norm", kwargs)
                for actual, expected in zip(args[4], aux):
                    torch.testing.assert_close(actual, gather(expected)[:11])
            else:
                self.assertIsNone(args[4])
                torch.testing.assert_close(
                    kwargs["hidden_states_before_norm"], gather(pre_head)[:11]
                )


if __name__ == "__main__":
    unittest.main()
