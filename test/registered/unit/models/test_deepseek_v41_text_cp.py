"""Pure-language V4.1 CP input, padding and DSpark state regressions."""

import unittest
from contextlib import nullcontext
from functools import partial
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import torch

from sglang.srt.layers.attention.deepseek_v4_backend import (
    DeepseekV4AttnBackend,
    DSV4Metadata,
    LateLayerTail,
)
from sglang.srt.layers.cp.bcg import execute_prefill_cp_bcg
from sglang.srt.layers.cp.utils import (
    cp_gather_after_forward,
    cp_shard_model_inputs,
    is_cp_active,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.runner.eager_runner import EagerRunner
from sglang.srt.models.deepseek_v4 import (
    DeepseekV4ForCausalLM,
    deepseek_v4_low_ratio_sources,
)
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.dsv41_cp_test_utils import cp_context, simulated_collective
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


RUNNER = "sglang.srt.model_executor.runner.eager_runner"


class TestDSV41TextCP(CustomTestCase):
    def test_cp_graph_tail_build_refreshes_live_rows_without_changing_capacity(self):
        """Upstream rejects CP at graph-tail metadata construction; padding must
        instead own dummy KV slots while local row/position addresses stay fixed."""
        for rank in range(4):
            backend = object.__new__(DeepseekV4AttnBackend)
            backend.encoder_row_floor = None
            backend._bcg_tail_capacity = 256
            backend._bcg_tail_buffers = None
            backend._bcg_cp_tail_buffers = {}
            backend.token_to_kv_pool = NS(
                request_window=None,
                translate_loc_from_full_to_swa=lambda loc: loc,
            )
            built = []

            def metadata_kernel(**kwargs):
                built.append(kwargs)
                return DSV4Metadata(NS(), None)

            backend.init_forward_metadata_prefill = metadata_kernel
            addresses = None
            for lengths in ([257, 300], [5, 9]):
                with (
                    cp_context(4, rank, lengths) as (_, batch),
                    get_parallel().override(attn_cp_rank=rank, attn_cp_size=4),
                ):
                    batch.attn_cp_metadata.per_rank_actual_token = [160] * 4
                    batch.req_pool_indices = torch.tensor([2, 7])
                    batch.out_cache_loc = torch.zeros(640, dtype=torch.int64)
                    batch.out_cache_loc[: sum(lengths)] = torch.arange(
                        256, 256 + sum(lengths)
                    )
                    batch.seq_lens_cpu = torch.tensor([7 + lengths[0], 13 + lengths[1]])
                    batch.seq_lens = batch.seq_lens_cpu.to(torch.int32)
                    tail_metadata = backend._build_late_layer_tail_metadata(
                        batch, for_prefill_graph=True, max_seq_len_override=1024
                    )
                    tail = tail_metadata.late_layer_tail
                    expected = torch.cat(
                        [
                            torch.arange(lengths[0] - min(128, lengths[0]), lengths[0]),
                            torch.arange(
                                sum(lengths) - min(128, lengths[1]), sum(lengths)
                            ),
                        ]
                    )
                    local = expected[expected % 4 == rank]
                    self.assertEqual(tail.token_indices.shape, (64,))
                    self.assertEqual(tail.real_rows_count, len(local))
                    torch.testing.assert_close(tail.global_token_indices, expected)
                    torch.testing.assert_close(
                        tail.token_indices[: len(local)], local // 4
                    )
                    torch.testing.assert_close(
                        tail.positions[: len(local)], batch.positions[local]
                    )
                    torch.testing.assert_close(
                        tail.pos_global[: len(expected)], batch.positions[expected]
                    )
                    self.assertTrue(
                        torch.all(tail.swa_out_cache_loc[len(expected) :] == 0)
                    )
                    self.assertEqual(built[-1]["out_cache_loc"].shape, (256,))
                    self.assertTrue(built[-1]["use_prefill_cuda_graph"])
                    pointers = (
                        tail.token_indices.data_ptr(),
                        tail.positions.data_ptr(),
                        tail.scatter_indices.data_ptr(),
                        tail.cp_metadata.gather_index.data_ptr(),
                        tail.cp_metadata.moe_local_token_count.data_ptr(),
                    )
                    if addresses is not None:
                        self.assertEqual(addresses, pointers)
                    addresses = pointers

    def test_cp_sources_trim_capture_padding_before_request_expansion(self):
        """A short batch in a larger BCG bucket must not become request rows."""
        for lengths, bucket in (((1, 3), 32), ((5, 9), 16), ((5003, 9777), 14848)):
            total = sum(lengths)
            full = torch.zeros(bucket, 3)
            full[:total] = torch.arange(total * 3).reshape(total, 3)
            for rank in range(4):
                with (
                    self.subTest(lengths=lengths, rank=rank),
                    get_context().override_server_args(tp_size=4, attn_cp_size=4),
                    get_parallel().override(attn_cp_rank=rank),
                    cp_context(4, rank, lengths) as (strategy, batch),
                ):
                    batch.req_pool_indices = torch.tensor([2, 7])
                    batch.extend_seq_lens = torch.tensor(lengths)
                    positions = torch.zeros(bucket, dtype=torch.int64)
                    positions[:total] = batch.positions
                    batch.positions = positions
                    metadata = batch.attn_cp_metadata
                    metadata.total_seq_lens = bucket
                    metadata.per_rank_actual_token = [bucket // 4] * 4
                    metadata.per_rank_logical_token = [bucket // 4] * 4
                    outputs = {}

                    def compress(layer, x, req, pos):
                        outputs["compressed"] = (x.clone(), req.clone(), pos.clone())

                    def index(layer, x, q, req, pos, fb, *, rows_per_request):
                        outputs["indexed"] = (x.clone(), q.clone(), pos.clone())
                        outputs["lengths"] = rows_per_request

                    backend = NS(
                        forward_metadata=NS(late_layer_tail=None),
                        _low_ratio_compress_torch=compress,
                        _low_ratio_index_topk=index,
                    )
                    backend.forward_low_ratio_sources = partial(
                        DeepseekV4AttnBackend._forward_low_ratio_sources_cp,
                        backend,
                        run_compressor=True,
                        run_indexer=True,
                    )
                    batch.global_num_token_non_padded_cpu = total
                    local = full[rank::4].contiguous()
                    with (
                        simulated_collective(strategy, batch, full),
                        patch(
                            "sglang.srt.layers.attention.deepseek_v4_backend.torch.cuda.current_stream",
                            return_value=None,
                        ),
                        patch(
                            "sglang.srt.models.deepseek_v4.get_tc_piecewise_forward_context",
                            return_value=NS(forward_batch=batch),
                        ),
                        patch(
                            "sglang.srt.models.deepseek_v4.get_attn_backend",
                            return_value=backend,
                        ),
                    ):
                        deepseek_v4_low_ratio_sources(
                            NS(compressor=object(), indexer=object()),
                            x=local,
                            q_lora=local + 100,
                            positions=positions[rank::4],
                        )
                    x, req, pos = outputs["compressed"]
                    torch.testing.assert_close(x, full[:total])
                    torch.testing.assert_close(
                        req, torch.tensor([2] * lengths[0] + [7] * lengths[1])
                    )
                    torch.testing.assert_close(pos, positions[:total])
                    x, q, pos = outputs["indexed"]
                    torch.testing.assert_close(x, full[:total][rank::4])
                    torch.testing.assert_close(q, full[:total][rank::4] + 100)
                    torch.testing.assert_close(pos, positions[:total][rank::4])
                    expected_lens = [
                        sum(row % 4 == rank for row in range(start, start + length))
                        for start, length in ((0, lengths[0]), (lengths[0], lengths[1]))
                    ]
                    self.assertEqual(outputs["lengths"], expected_lens)
                    self.assertEqual(metadata.total_seq_lens, bucket)

    def test_tail_moe_count_is_prepared_per_batch_without_lazy_device_copy(self):
        for rank in range(4):
            with (
                self.subTest(rank=rank),
                cp_context(4, rank, (5, 9)) as (
                    strategy,
                    batch,
                ),
            ):
                batch.batch_size = 2
                full_metadata = batch.attn_cp_metadata
                full_count = strategy.moe_num_token_non_padded(batch)
                for tail_lens, rows in (
                    ([2, 3], [3, 4, 11, 12, 13]),
                    ([1, 1], [4, 13]),
                ):
                    with patch(
                        "sglang.srt.layers.attention.deepseek_v4_backend.get_parallel",
                        return_value=NS(attn_cp_rank=rank, attn_cp_size=4),
                    ):
                        layout = DeepseekV4AttnBackend._late_layer_tail_cp_layout(
                            None,
                            batch,
                            tail_lens,
                        )
                    tail_metadata = layout["cp_metadata"]
                    count = tail_metadata.moe_local_token_count
                    self.assertIsNotNone(count)
                    self.assertEqual(count.shape, torch.Size([]))
                    self.assertEqual(count.dtype, torch.int32)
                    self.assertEqual(count.item(), sum(row % 4 == rank for row in rows))
                    self.assertNotEqual(count.data_ptr(), full_count.data_ptr())
                    batch.attn_cp_metadata = tail_metadata
                    try:
                        with patch(
                            "sglang.srt.layers.cp.interleave.torch.tensor",
                            side_effect=AssertionError("lazy tail count copy"),
                        ):
                            self.assertIs(
                                strategy.moe_num_token_non_padded(batch), count
                            )
                            self.assertIs(
                                strategy.moe_num_token_non_padded(batch), count
                            )
                    finally:
                        batch.attn_cp_metadata = full_metadata
                    self.assertEqual(full_count.item(), len(batch.input_ids[rank::4]))

    def test_compact_tail_gather_and_exception_restore_full_cp_layout(self):
        global_rows = torch.tensor([3, 4, 11, 12, 13])
        full = torch.arange(42, dtype=torch.float32).reshape(14, 3)
        for rank in range(4):
            for fail in (False, True):
                with (
                    self.subTest(rank=rank, fail=fail),
                    cp_context(4, rank, (5, 9)) as (strategy, cp_batch),
                ):
                    batch = ForwardBatch(
                        forward_mode=cp_batch.forward_mode,
                        batch_size=2,
                        input_ids=cp_batch.input_ids,
                        req_pool_indices=torch.tensor([0, 1]),
                        seq_lens=torch.tensor([12, 22]),
                        out_cache_loc=torch.arange(14),
                        seq_lens_sum=34,
                        positions=cp_batch.positions,
                        extend_seq_lens=torch.tensor([5, 9]),
                        extend_seq_lens_cpu=[5, 9],
                        extend_prefix_lens_cpu=[7, 13],
                        extend_logprob_start_lens_cpu=[5, 9],
                        attn_cp_metadata=cp_batch.attn_cp_metadata,
                    )
                    full_metadata = batch.attn_cp_metadata
                    with patch(
                        "sglang.srt.layers.attention.deepseek_v4_backend.get_parallel",
                        return_value=NS(attn_cp_rank=rank, attn_cp_size=4),
                    ):
                        layout = DeepseekV4AttnBackend._late_layer_tail_cp_layout(
                            None, batch, [2, 3]
                        )
                    tail = LateLayerTail(
                        token_indices=layout["local_token_indices"],
                        positions=layout["local_positions"],
                        extend_seq_lens=torch.tensor([2, 3]),
                        extend_seq_lens_cpu=[2, 3],
                        swa_out_cache_loc=torch.zeros(2, dtype=torch.int64),
                        cp_metadata=layout["cp_metadata"],
                        pad_rows=layout["pad_rows"],
                        global_token_indices=global_rows,
                    )
                    compact = tail.rows(strategy.shard_hidden_states(full, batch))
                    model = NS(
                        get_input_embeddings=lambda: lambda ids: full,
                        model=Mock(
                            return_value=((compact, compact.clone()), [compact.clone()])
                        ),
                        capture_aux_hidden_states=True,
                        pp_group=NS(is_last_rank=True),
                        lm_head=object(),
                        logits_processor=Mock(return_value=NS()),
                    )
                    model.model.late_layer_start = 21
                    peers = []
                    for peer in range(4):
                        peer_rows = full[global_rows[global_rows % 4 == peer]]
                        padded = full.new_zeros((2, 3))
                        padded[: len(peer_rows)] = peer_rows
                        peers.append(padded)

                    def gather(out, local):
                        self.assertIs(batch.attn_cp_metadata, tail.cp_metadata)
                        torch.testing.assert_close(local, peers[rank])
                        if fail:
                            raise RuntimeError("collective failed")
                        out.copy_(torch.cat(peers))

                    with (
                        patch(RUNNER + ".torch.cuda.current_stream", return_value=None),
                        patch(
                            "sglang.srt.layers.cp.interleave.use_symmetric_memory",
                            side_effect=lambda *a, **k: nullcontext(),
                        ),
                        patch(
                            "sglang.srt.layers.cp.interleave.is_allocation_symmetric",
                            return_value=False,
                        ),
                        patch(
                            "sglang.srt.layers.cp.interleave.attn_cp_all_gather_into_tensor",
                            side_effect=gather,
                        ),
                    ):
                        runner = NS(
                            model_runner=NS(
                                model=model,
                                attn_backend=NS(
                                    tail_forward_metadata=NS(late_layer_tail=tail)
                                ),
                            )
                        )
                        runner.prefill_cp_bcg_input = NS(live_local_tokens=4)
                        runner._prefill_forward_context = lambda *a, **kw: nullcontext()
                        runner.backend = NS(
                            replay=lambda *a, **kw: model.model.return_value
                        )
                        for execute in (
                            lambda: EagerRunner._execute_extend_cp(runner, batch, {}),
                            lambda: execute_prefill_cp_bcg(
                                runner, batch, batch, 16, 14
                            ),
                        ):
                            if fail:
                                with self.assertRaisesRegex(
                                    RuntimeError, "collective failed"
                                ):
                                    execute()
                                self.assertIs(batch.attn_cp_metadata, full_metadata)
                                continue
                            output = execute()
                            args = model.logits_processor.call_args.args
                            torch.testing.assert_close(
                                args[0], batch.input_ids[global_rows]
                            )
                            torch.testing.assert_close(args[1], full[global_rows])
                            torch.testing.assert_close(args[4][0], full[global_rows])
                            self.assertEqual(args[3].extend_seq_lens_cpu, [2, 3])
                            torch.testing.assert_close(
                                output.hidden_states_token_indices, global_rows
                            )
                            self.assertIs(batch.attn_cp_metadata, full_metadata)
                    self.assertIs(batch.attn_cp_metadata, full_metadata)

    def test_interleave_roundtrip_mixed_lengths_prefix_and_padding(self):
        for size in (2, 4):
            for length in (4, 5, 9, 127, 128, 129):
                for rank in range(size):
                    with (
                        self.subTest(size=size, length=length, rank=rank),
                        cp_context(size, rank, (1, length - 1), (0, 16384)) as (
                            strategy,
                            batch,
                        ),
                    ):
                        embeddings = torch.arange(
                            length * 3, dtype=torch.float32
                        ).reshape(length, 3)
                        original_ids = batch.input_ids.clone()
                        with cp_shard_model_inputs(
                            embeddings, batch.positions, batch, batch.input_ids
                        ) as (local, positions, ids):
                            count = len(embeddings[rank::size])
                            torch.testing.assert_close(
                                local[:count], embeddings[rank::size]
                            )
                            torch.testing.assert_close(
                                positions[:count], batch.positions[rank::size]
                            )
                            torch.testing.assert_close(
                                ids[:count], batch.input_ids[rank::size]
                            )
                            self.assertEqual(
                                torch.count_nonzero(local[count:]).item(), 0
                            )
                            self.assertEqual(torch.count_nonzero(ids[count:]).item(), 0)
                            with simulated_collective(strategy, batch, embeddings):
                                restored = cp_gather_after_forward(local, batch)
                            torch.testing.assert_close(
                                restored, embeddings, rtol=0, atol=0
                            )
                        torch.testing.assert_close(batch.input_ids, original_ids)
                        self.assertFalse(hasattr(batch, "input_ids_global"))

    def test_speculative_state_and_global_ids_restored_on_exception(self):
        for size in (2, 4):
            for rank in range(size):
                for had_global in (False, True):
                    with (
                        self.subTest(size=size, rank=rank, had_global=had_global),
                        cp_context(size, rank) as (_, batch),
                    ):
                        full = torch.arange(36, dtype=torch.float32).reshape(9, 4)
                        batch.spec_info = NS(hidden_states=full)
                        previous = object()
                        if had_global:
                            batch.input_ids_global = previous
                        with self.assertRaisesRegex(RuntimeError, "injected"):
                            with cp_shard_model_inputs(
                                full, batch.positions, batch, batch.input_ids
                            ):
                                n = len(full[rank::size])
                                torch.testing.assert_close(
                                    batch.spec_info.hidden_states[:n], full[rank::size]
                                )
                                # Global MoE IDs are in rank-major order, with padding.
                                physical = sum(
                                    batch.attn_cp_metadata.per_rank_actual_token
                                )
                                padded = batch.input_ids.new_zeros(physical)
                                padded[:9] = batch.input_ids
                                expected = torch.cat(
                                    [padded[r::size] for r in range(size)]
                                )
                                torch.testing.assert_close(
                                    batch.input_ids_global, expected
                                )
                                raise RuntimeError("injected")
                        self.assertIs(batch.spec_info.hidden_states, full)
                        if had_global:
                            self.assertIs(batch.input_ids_global, previous)
                        else:
                            self.assertFalse(hasattr(batch, "input_ids_global"))

    def test_short_prompt_falls_back_from_cp(self):
        with cp_context(4, 0, (1, 2), (0, 0)) as (_, batch):
            self.assertFalse(is_cp_active(batch))

    def test_runner_text_embedding_and_preembedded_paths(self):
        for preembedded in (False, True):
            for rank in range(4):
                with (
                    self.subTest(preembedded=preembedded, rank=rank),
                    cp_context(4, rank) as (strategy, batch),
                ):
                    full = torch.arange(27, dtype=torch.float32).reshape(9, 3)
                    embedding = Mock(return_value=full)
                    model = NS(
                        vision=None,
                        _prepare_mm_embeddings=Mock(
                            side_effect=AssertionError("Text must not invoke vision")
                        ),
                        get_input_embeddings=Mock(return_value=embedding),
                        pp_group=NS(is_last_rank=True),
                        lm_head=object(),
                        capture_aux_hidden_states=False,
                        logits_processor=Mock(return_value="ok"),
                    )
                    model.prepare_language_model_inputs = lambda ids, fb, emb: (
                        DeepseekV4ForCausalLM.prepare_language_model_inputs(
                            model, ids, fb, emb
                        )
                    )

                    def body(ids, positions, fb, input_embeds):
                        n = len(full[rank::4])
                        torch.testing.assert_close(ids[:n], batch.input_ids[rank::4])
                        torch.testing.assert_close(
                            positions[:n], batch.positions[rank::4]
                        )
                        torch.testing.assert_close(input_embeds[:n], full[rank::4])
                        return input_embeds

                    model.model = body
                    with (
                        simulated_collective(strategy, batch, full),
                        patch(RUNNER + ".torch.cuda.current_stream", return_value=None),
                    ):
                        result = EagerRunner._execute_extend_cp(
                            NS(model_runner=NS(model=model)),
                            batch,
                            {"input_embeds": full} if preembedded else {},
                        )
                    self.assertEqual(result, "ok")
                    if preembedded:
                        model.get_input_embeddings.assert_not_called()
                    else:
                        embedding.assert_called_once_with(batch.input_ids)
                    model._prepare_mm_embeddings.assert_not_called()
                    args = model.logits_processor.call_args.args
                    torch.testing.assert_close(args[0], batch.input_ids)
                    torch.testing.assert_close(args[1], full)

    def test_dspark_aux_tensor_and_list_gathered_without_pre_norm_override(self):
        for as_list in (False, True):
            with self.subTest(as_list=as_list), cp_context(4, 2) as (strategy, batch):
                full = torch.arange(27, dtype=torch.float32).reshape(9, 3)
                local = strategy.shard_hidden_states(full, batch)
                aux = [local.clone(), local.clone()] if as_list else local.clone()
                model = NS(
                    get_input_embeddings=lambda: lambda ids: full,
                    model=Mock(return_value=((local, local.clone()), aux)),
                    capture_aux_hidden_states=True,
                    pp_group=NS(is_last_rank=True),
                    lm_head=object(),
                    logits_processor=Mock(return_value="ok"),
                )
                model.model.late_layer_start = None
                with (
                    simulated_collective(strategy, batch, full),
                    patch(RUNNER + ".torch.cuda.current_stream", return_value=None),
                ):
                    EagerRunner._execute_extend_cp(
                        NS(model_runner=NS(model=model)), batch, {}
                    )
                args, kwargs = model.logits_processor.call_args
                torch.testing.assert_close(args[1], full)
                for tensor in args[4] if as_list else [args[4]]:
                    torch.testing.assert_close(tensor, full)
                self.assertNotIn("hidden_states_before_norm", kwargs)

    def test_target_hidden_states_before_norm_preserved_without_dspark_aux(self):
        with cp_context(4, 1) as (strategy, batch):
            full = torch.arange(27, dtype=torch.float32).reshape(9, 3)
            local = strategy.shard_hidden_states(full, batch)
            model = NS(
                get_input_embeddings=lambda: lambda ids: full,
                model=Mock(return_value=(local, local.clone())),
                capture_aux_hidden_states=False,
                pp_group=NS(is_last_rank=True),
                lm_head=object(),
                logits_processor=Mock(return_value="ok"),
            )
            with (
                simulated_collective(strategy, batch, full),
                patch(RUNNER + ".torch.cuda.current_stream", return_value=None),
            ):
                EagerRunner._execute_extend_cp(
                    NS(model_runner=NS(model=model)), batch, {}
                )
            torch.testing.assert_close(
                model.logits_processor.call_args.kwargs["hidden_states_before_norm"],
                full,
            )


if __name__ == "__main__":
    unittest.main()
