"""CPU contracts for V4 zigzag: query rows, global writes and routing order."""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.layers.attention import deepseek_v4_backend as be
from sglang.srt.layers.attention.dsv4.compressor import Compressor
from sglang.srt.layers.cp import utils as cp_utils
from sglang.srt.layers.cp.base import init_cp_strategy
from sglang.srt.layers.cp.interleave import InterleaveCPStrategy
from sglang.srt.layers.cp.padding import pad_logical_token_to_physical
from sglang.srt.layers.cp.zigzag import ZigzagCPStrategy
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestDeepseekV4Zigzag(CustomTestCase):
    def tearDown(self):
        init_cp_strategy(enable_prefill_cp=False, cp_size=1, cp_strategy="zigzag")

    def _batches(self, cp_size, lengths, prefixes, strategy_name="zigzag"):
        init_cp_strategy(
            enable_prefill_cp=True, cp_size=cp_size, cp_strategy=strategy_name
        )
        strategy = (
            ZigzagCPStrategy if strategy_name == "zigzag" else InterleaveCPStrategy
        )(cp_size)
        batches = []
        for rank in range(cp_size):
            with (
                get_parallel().override(attn_cp_rank=rank, attn_cp_size=cp_size),
                mock.patch(
                    "sglang.srt.layers.cp.zigzag.get_device",
                    return_value=SimpleNamespace(device="cpu"),
                ),
            ):
                metadata = strategy.build_metadata(
                    sum(lengths), [a + b for a, b in zip(lengths, prefixes)], lengths
                )
                pad_logical_token_to_physical(metadata)
                batches.append(
                    SimpleNamespace(
                        input_ids=torch.arange(1, sum(lengths) + 1),
                        forward_mode=ForwardMode.EXTEND,
                        extend_seq_lens_cpu=lengths,
                        attn_cp_metadata=metadata,
                    )
                )
        return strategy, batches

    def _core(self, lengths, prefixes, physical, present_ratios=(4, 128)):
        count = sum(lengths)
        positions = torch.cat(
            [
                torch.arange(p, p + n, dtype=torch.int32)
                for n, p in zip(lengths, prefixes)
            ]
        )
        positions = torch.cat(
            (positions, torch.zeros(physical - count, dtype=torch.int32))
        )
        rows = torch.arange(physical, dtype=torch.int32)
        core = be.DSV4AttnMetadata(
            page_size=256,
            page_table=rows[:, None],
            raw_out_loc=torch.arange(count) + 1000,
            cuda_int32_kwargs={"dtype": torch.int32},
            seq_lens_casual=positions + 1,
            positions_casual=positions,
            swa_page_indices=rows[:, None].expand(-1, 64).clone(),
            swa_topk_lengths=(positions + 1).clamp(max=128),
            index_topk=512,
            present_ratios=present_ratios,
        )
        for i, field in enumerate(core._CP_GLOBAL_FIELDS):
            setattr(core, field, torch.arange(count) + 1000 * (i + 1))
        for field in core._CP_REINDEX_OPTIONAL_FIELDS:
            setattr(core, field, rows.clone())
        core.c128_page_indices = rows[:, None].expand(-1, 64).clone()
        if 128 not in present_ratios:
            core.c128_page_indices = core.c128_topk_lengths_clamp1 = None
        return core

    def test_prefill_metadata_matches_physical_query_shards_and_keeps_global_writes(
        self,
    ):
        for cp_size, lengths, prefixes in ((4, [9, 13], [5, 129]), (2, [17], [127])):
            strategy, batches = self._batches(cp_size, lengths, prefixes)
            for rank, batch in enumerate(batches):
                for ratios in ((4,), (4, 128)):
                    with self.subTest(cp_size=cp_size, rank=rank, ratios=ratios):
                        physical = sum(batch.attn_cp_metadata.per_rank_actual_token)
                        core = self._core(lengths, prefixes, physical, ratios)
                        originals = {
                            name: getattr(core, name)
                            for name in core._CP_REINDEX_FIELDS
                            + core._CP_REINDEX_OPTIONAL_FIELDS
                            + core._CP_GLOBAL_FIELDS
                        }
                        backend = be.DeepseekV4AttnBackend.__new__(
                            be.DeepseekV4AttnBackend
                        )
                        backend.req_to_token = torch.empty(0)
                        backend.has_c4 = backend.has_c128 = False
                        with (
                            get_parallel().override(
                                attn_cp_rank=rank, attn_cp_size=cp_size
                            ),
                            mock.patch.object(
                                backend,
                                "expand_prefill_casually",
                                return_value=(
                                    core.seq_lens_casual,
                                    torch.zeros(physical),
                                ),
                            ),
                            mock.patch.object(
                                backend, "make_core_attn_metadata", return_value=core
                            ),
                            mock.patch.object(
                                be, "_create_flashmla_metadata", side_effect=object
                            ),
                        ):
                            metadata = backend.init_forward_metadata_prefill(
                                max_seq_len=max(
                                    a + b for a, b in zip(lengths, prefixes)
                                ),
                                req_pool_indices=torch.arange(len(lengths)),
                                seq_lens=torch.tensor(
                                    [a + b for a, b in zip(lengths, prefixes)]
                                ),
                                seq_lens_cpu=[a + b for a, b in zip(lengths, prefixes)],
                                out_cache_loc=core.raw_out_loc,
                                num_tokens=sum(lengths),
                                extend_seq_lens=torch.tensor(lengths),
                                extend_seq_lens_cpu=lengths,
                                need_compress=False,
                                forward_batch=batch,
                            )
                            local_ids = strategy.shard_hidden_states(
                                batch.input_ids, batch
                            )
                        self.assertIs(metadata.core_attn_metadata, core)
                        valid = local_ids != 0
                        for field in (
                            core._CP_REINDEX_FIELDS + core._CP_REINDEX_OPTIONAL_FIELDS
                        ):
                            actual, original = getattr(core, field), originals[field]
                            if original is None:
                                self.assertIsNone(actual)
                                continue
                            self.assertEqual(actual.shape[0], local_ids.shape[0])
                            torch.testing.assert_close(
                                actual[valid], original[local_ids[valid] - 1]
                            )
                        # ExpandPrefillCausally pads causal lengths with 1 and
                        # positions with 0. Select those prebuilt sentinel rows,
                        # not the first logical query (which has a prefix).
                        self.assertTrue(torch.all(core.seq_lens_casual[~valid] == 1))
                        self.assertTrue(torch.all(core.positions_casual[~valid] == 0))
                        self.assertTrue(
                            torch.all(core.page_table[~valid] >= sum(lengths))
                        )
                        for field in core._CP_GLOBAL_FIELDS:
                            self.assertIs(getattr(core, field), originals[field])

    def test_rank_major_routing_ids_and_model_input_restoration(self):
        for strategy_name in ("zigzag", "interleave"):
            for cp_size in (2, 4):
                strategy, batches = self._batches(
                    cp_size, [9, 13], [5, 129], strategy_name
                )
                rank_ids = []
                for rank, batch in enumerate(batches):
                    with get_parallel().override(
                        attn_cp_rank=rank, attn_cp_size=cp_size
                    ):
                        rank_ids.append(
                            strategy.shard_hidden_states(batch.input_ids, batch)
                        )
                gathered_hidden = torch.cat(rank_ids).float().unsqueeze(1)
                for rank, batch in enumerate(batches):
                    for a2a in (False, True):
                        with (
                            self.subTest(
                                strategy=strategy_name,
                                cp_size=cp_size,
                                rank=rank,
                                a2a=a2a,
                            ),
                            get_parallel().override(
                                attn_cp_rank=rank, attn_cp_size=cp_size
                            ),
                            mock.patch.object(
                                cp_utils,
                                "get_moe_a2a_backend",
                                return_value=SimpleNamespace(is_none=lambda: not a2a),
                            ),
                        ):
                            expected_ids = (
                                rank_ids[rank] if a2a else torch.cat(rank_ids)
                            )
                            torch.testing.assert_close(
                                cp_utils.cp_interleave_input_ids(
                                    batch.input_ids, batch
                                ),
                                expected_ids,
                            )
                            for previous in (None, torch.tensor([-1])):
                                batch.input_ids_global = previous
                                with cp_utils.cp_shard_model_inputs(
                                    batch.input_ids[:, None].float(),
                                    batch.input_ids,
                                    batch,
                                    batch.input_ids,
                                ):
                                    routed_ids = batch.input_ids_global
                                    hidden = (
                                        rank_ids[rank].float().unsqueeze(1)
                                        if a2a
                                        else gathered_hidden
                                    )
                                    # Token-dependent expert routing must align with raw rank-major FFN gather.
                                    routed = (
                                        hidden * (routed_ids.remainder(3) + 1)[:, None]
                                    )
                                    local = (
                                        routed if a2a else routed.chunk(cp_size)[rank]
                                    )
                                    torch.testing.assert_close(
                                        local[:, 0],
                                        rank_ids[rank].float()
                                        * (rank_ids[rank].remainder(3) + 1),
                                    )
                                self.assertIs(batch.input_ids_global, previous)
                            del batch.input_ids_global
                            with cp_utils.cp_shard_model_inputs(
                                batch.input_ids[:, None], batch.input_ids, batch, None
                            ):
                                self.assertFalse(hasattr(batch, "input_ids_global"))
                            self.assertFalse(hasattr(batch, "input_ids_global"))

    def test_compressor_projection_gather_removes_padding_before_grouping(self):
        # Both c4 and c128 boundaries cross shards; prefixes complete partial groups.
        lengths, prefixes, cp_size = [129, 137], [3, 127], 4
        strategy, batches = self._batches(cp_size, lengths, prefixes)
        global_rows = torch.arange(sum(lengths) * 2).view(-1, 2).float()
        rank_rows = []
        for rank, batch in enumerate(batches):
            with get_parallel().override(attn_cp_rank=rank, attn_cp_size=cp_size):
                rank_rows.append(strategy.shard_hidden_states(global_rows, batch))

        def gather(output, _):
            wire_rows = output.shape[0] // cp_size
            torch.cat([rows[:wire_rows] for rows in rank_rows], out=output)

        compressor = SimpleNamespace(_compute_wkv_gate=lambda x: x)
        for rank, batch in enumerate(batches):
            with (
                get_parallel().override(
                    attn_cp_rank=rank,
                    attn_cp_size=cp_size,
                    attn_cp_group=SimpleNamespace(all_gather_into_tensor=gather),
                ),
                mock.patch(
                    "sglang.srt.layers.attention.dsv4.compressor.dsa_use_prefill_cp",
                    return_value=True,
                ),
                mock.patch("torch.cuda.current_stream", return_value=None),
                mock.patch(
                    "sglang.srt.distributed.device_communicators.pynccl_allocator.use_symmetric_memory",
                    return_value=torch.no_grad(),
                ),
            ):
                projected = Compressor.compute_kv_score(
                    compressor, rank_rows[rank], batch
                )
                restored = strategy.gather_hidden_states(rank_rows[rank], batch)
            torch.testing.assert_close(projected, global_rows)
            torch.testing.assert_close(restored, global_rows)
            # Exercise the active v2 attention/indexer writer, with only the
            # fused CUDA compression/store replaced by a row-sentinel store.
            from sglang.srt.layers.attention.dsv4.compressor_v2 import (
                CompressorBackendMixin,
            )
            from sglang.kernels.ops.attention.dsv4.kv_layout import KVLayout

            for ratio, indexer in ((4, True), (4, False), (128, False)):
                locations = torch.arange(sum(lengths)).flip(0)
                destination = torch.zeros_like(global_rows)
                backend = CompressorBackendMixin()
                backend.enable_deepseek_v4_fp4_indexer = False
                backend.forward_metadata = SimpleNamespace(
                    core_metadata=SimpleNamespace(**{f"c{ratio}_out_loc": locations})
                )
                backend.token_to_kv_pool = SimpleNamespace(
                    uniform_fp8=False,
                    layer_mapping={0: (None, None, object())},
                    get_extra_key_buffer=lambda _: torch.empty(0, dtype=torch.uint8),
                    get_extra_key_page_size=lambda _: 1,
                    get_extra_key_layout=lambda _: KVLayout.V4,
                    get_index_k_page_size=lambda _: 1,
                    get_index_k_with_scale_buffer=lambda _: torch.empty(
                        0, dtype=torch.uint8
                    ),
                )
                writer_compressor = SimpleNamespace(
                    compute_kv_score=lambda x, fb: Compressor.compute_kv_score(
                        compressor, x, fb
                    ),
                    get_state_pool=lambda _: SimpleNamespace(
                        kv_score_buffer=SimpleNamespace(kv_score=torch.empty(0))
                    ),
                    ratio=ratio,
                    is_in_indexer=indexer,
                    ape=torch.empty(0),
                    head_dim=2,
                    norm=None,
                    freqs_cis=None,
                    rotate=indexer,
                )

                def sentinel_store(*, kv_score_input, out_loc, **kwargs):
                    destination[out_loc] = kv_score_input

                with (
                    get_parallel().override(
                        attn_cp_rank=rank,
                        attn_cp_size=cp_size,
                        attn_cp_group=SimpleNamespace(all_gather_into_tensor=gather),
                    ),
                    mock.patch(
                        "sglang.srt.layers.attention.dsv4.compressor.dsa_use_prefill_cp",
                        return_value=True,
                    ),
                    mock.patch("torch.cuda.current_stream", return_value=None),
                    mock.patch.object(
                        backend,
                        "_forward_compress_all_in_one",
                        side_effect=sentinel_store,
                    ),
                    mock.patch(
                        "sglang.kernels.ops.attention.dsv4.unified_kv_kernels.env_gate.is_unified_kv_triton",
                        return_value=False,
                    ),
                ):
                    writer = (
                        backend.forward_indexer_compressor
                        if indexer
                        else backend.forward_core_compressor
                    )
                    writer(rank_rows[rank], batch, 0, writer_compressor)
                torch.testing.assert_close(destination[locations], global_rows)

    def test_sparse_prefill_dispatch_falls_back_for_zigzag(self):
        from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool

        for strategy_name in ("zigzag", "interleave"):
            _, batches = self._batches(2, [17], [127], strategy_name)
            for enabled, query_count, q8 in (
                (True, 8, False),
                (True, 8, True),
                (False, be._LARGE_INDEXER_QUERY_THRESHOLD + 1, False),
                (False, be._LARGE_INDEXER_QUERY_THRESHOLD + 1, True),
            ):
                core = self._core([query_count], [0], query_count)
                core.c0_flashmla_metadata = object()
                backend = be.DeepseekV4AttnBackend.__new__(be.DeepseekV4AttnBackend)
                backend.mtp_enabled = backend.trtllm_attn = backend.is_dsv41 = False
                backend.head_dim_v = 2
                backend.softmax_scale = 1.0
                backend.dsv4_prefill_backend = "bf16"
                backend.forward_metadata = SimpleNamespace(
                    core_attn_metadata=core, late_layer_tail=None
                )
                pool = mock.Mock(spec=DeepSeekV4TokenToKVPool)
                pool.request_window = None
                pool.get_swa_key_buffer_radix.return_value = torch.zeros(1, 2)
                pool.get_swa_key_page_size.return_value = 1
                pool.get_swa_key_bytes_per_token.return_value = 2
                backend.token_to_kv_pool = pool
                q = torch.ones(query_count, 1, 2)
                with (
                    get_parallel().override(attn_cp_rank=0, attn_cp_size=2),
                    mock.patch.object(
                        be, "get_platform", return_value=SimpleNamespace(is_sm120=False)
                    ),
                    mock.patch.object(
                        be.envs.SGLANG_OPT_FLASHMLA_SPARSE_PREFILL,
                        "get",
                        return_value=enabled,
                    ),
                    mock.patch.object(
                        be, "use_dsv4_q8kv8_sparse_prefill", return_value=q8
                    ),
                    mock.patch.object(
                        backend, "_forward_prefill_sparse_q8kv8", return_value=q * 3
                    ),
                    mock.patch.object(
                        backend, "_forward_prefill_sparse", return_value=q * 3
                    ),
                    mock.patch(
                        "sgl_kernel.flash_mla.flash_mla_with_kvcache",
                        return_value=(q.unsqueeze(1) * 2, None),
                    ),
                ):
                    output = backend._forward_attention(
                        q,
                        None,
                        None,
                        SimpleNamespace(layer_id=0),
                        batches[0],
                        compress_ratio=0,
                        save_kv_cache=False,
                        attn_sink=torch.zeros(1),
                    )
                torch.testing.assert_close(
                    output, q * (2 if strategy_name == "zigzag" else 3)
                )


if __name__ == "__main__":
    unittest.main()
