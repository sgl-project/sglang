"""CPU checks for DSV4 interleave BCG padding, the eager logits tail and the HIP V4.1 path."""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.layers.attention.deepseek_v4_backend_hip_radix import (
    DeepseekV4HipRadixBackend,
)
from sglang.srt.layers.cp.base import get_cp_strategy, init_cp_strategy
from sglang.srt.layers.cp.bcg import (
    PrefillCPBCGInput,
    execute_prefill_cp_bcg,
    filter_prefill_cp_bcg_capture_num_tokens,
    supports_prefill_cp_bcg,
)
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardMode,
)
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestDSV4PrefillCPBCG(unittest.TestCase):
    def setUp(self):
        # The layouts below use fewer tokens than the HIP CP minimum per rank;
        # the HIP rule has its own test (test_dsv41_prefill_cp_bcg_hip.py).
        for target, kwargs in (
            ("sglang.srt.layers.cp.interleave._is_hip", {"new": False}),
            ("sglang.srt.layers.cp.bcg.is_hip", {"return_value": False}),
        ):
            patcher = patch(target, **kwargs)
            patcher.start()
            self.addCleanup(patcher.stop)

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
                            "sglang.srt.layers.attention.deepseek_v4_backend.is_cp_active",
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

        backend = SimpleNamespace(
            forward_metadata=SimpleNamespace(late_layer_tail=None)
        )
        for lengths in ([32768], [8192, 24576]):
            batch = SimpleNamespace(
                seq_lens_cpu=torch.tensor(lengths),
                extend_seq_lens=torch.tensor(lengths, dtype=torch.int32),
                extend_seq_lens_cpu=lengths,
            )
            # No core attention metadata or pool is present: the invariant
            # must fail before accessing either or launching the index combiner.
            with self.assertRaisesRegex(AssertionError, "allocated 16384.*32768"):
                DeepseekV4AttnBackend._build_sparse_prefill_chunk_cache(
                    backend, batch, core_attn_metadata=None, num_qo_tokens=16384
                )

    def test_inactive_cp_batch_cannot_replay_cp_body(self):
        from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import (
            PrefillCudaGraphRunner,
        )

        init_cp_strategy(enable_prefill_cp=True, cp_size=2, cp_strategy="interleave")
        runner = SimpleNamespace(
            can_replay_locally=lambda **kwargs: True,
            _has_inactive_dp_rank=lambda batch: False,
            enable_cp_bcg_capture=True,
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
                has_c4=True,
                has_c128=True,
                low_ratio_prefill_graph=False,
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
                attn_cp_metadata=SimpleNamespace(
                    per_rank_actual_token=[8, 8], local_index=None
                )
            )
            with (
                patch(
                    "sglang.srt.layers.attention.deepseek_v4_backend.is_cp_active",
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
            core.apply_cp_reindex.assert_called_once_with(
                num_tokens=write_tokens, local_index=batch.attn_cp_metadata.local_index
            )
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
            # Interleave + DSV4 is enabled on ROCm only.
            self.assertEqual(supports_prefill_cp_bcg(cfg), is_hip())
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
                    torch.zeros(bucket, 1),
                    torch.zeros(bucket, dtype=torch.int64),
                    input_ids=torch.zeros(bucket, dtype=torch.int64),
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
                            attn_cp_rank=rank,
                            attn_cp_size=cp_size,
                            enable_cp_tp_group_sharing=False,
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
                    # Attention backends keep reading global positions; the
                    # captured body gets the CP-local positions and ids.
                    torch.testing.assert_close(batch.positions, global_positions)
                    torch.testing.assert_close(
                        batch._cp_positions[: len(local_ids)],
                        global_positions[:n][rank::cp_size],
                    )
                    torch.testing.assert_close(
                        batch._cp_input_ids[: len(local_ids)].float(), local_ids
                    )
                    self.assertIs(batch.input_ids_global, batch._cp_input_ids)
                    self.assertEqual(
                        torch.count_nonzero(
                            batch.input_embeds[len(local_ids) :]
                        ).item(),
                        0,
                    )
                    self.assertEqual(
                        torch.count_nonzero(
                            batch._cp_positions[len(local_ids) :]
                        ).item(),
                        0,
                    )
                    self.assertEqual(
                        torch.count_nonzero(
                            batch._cp_input_ids[len(local_ids) :]
                        ).item(),
                        0,
                    )
                    self.assertEqual(
                        batch.input_embeds.data_ptr(),
                        inputs[rank].input_embeds.data_ptr(),
                    )
                    self.assertEqual(
                        batch._cp_positions.data_ptr(),
                        inputs[rank].positions.data_ptr(),
                    )
                    self.assertEqual(
                        batch._cp_input_ids.data_ptr(),
                        inputs[rank].input_ids.data_ptr(),
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
            for capture_mode in (CaptureHiddenMode.NULL, CaptureHiddenMode.LAST):
                with self.subTest(capture_aux=capture_aux, capture_mode=capture_mode):
                    self._check_logits_tail(capture_aux, capture_mode)

    def _check_logits_tail(self, capture_aux, capture_mode):
        hidden = torch.arange(8).view(8, 1).float()
        pre_head = hidden + 100
        aux = [hidden + 200, hidden + 300]
        body_output = ((hidden, pre_head), aux) if capture_aux else (hidden, pre_head)
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
        batch = SimpleNamespace(
            input_ids=torch.arange(11), capture_hidden_mode=capture_mode
        )
        gathered = []

        def gather(value, *args):
            if isinstance(value, tuple):
                return tuple(gather(item) for item in value)
            gathered.append(value)
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
        # Pre-mHC-head states are gathered only when logits store them (MTP/EAGLE).
        needs_pre_head = capture_mode.need_capture() and not capture_aux
        self.assertEqual(
            any(torch.equal(value, pre_head) for value in gathered), needs_pre_head
        )
        torch.testing.assert_close(args[1], gather(hidden)[:11])
        if capture_aux:
            for actual, expected in zip(args[4], aux):
                torch.testing.assert_close(actual, gather(expected)[:11])
        else:
            self.assertIsNone(args[4])
        if needs_pre_head:
            torch.testing.assert_close(
                kwargs["hidden_states_before_norm"], gather(pre_head)[:11]
            )
        else:
            self.assertNotIn("hidden_states_before_norm", kwargs)


_HIP = "sglang.srt.layers.attention.deepseek_v4_backend_hip_radix"


class TestDSV41PrefillCPBCGHip(unittest.TestCase):
    def tearDown(self):
        init_cp_strategy(enable_prefill_cp=False, cp_size=1, cp_strategy="zigzag")

    def test_capture_buckets_follow_hip_min_tokens_per_rank(self):
        cfg = SimpleNamespace(attn_cp_size=4, cp_strategy="interleave")
        buckets = [16, 32, 63, 64, 128]
        with (
            patch("sglang.srt.layers.cp.bcg.resolving_view", return_value=cfg),
            patch("sglang.srt.layers.cp.bcg.resolved_view", return_value=cfg),
        ):
            with patch("sglang.srt.layers.cp.bcg.is_hip", return_value=True):
                self.assertEqual(
                    filter_prefill_cp_bcg_capture_num_tokens(buckets, cfg), [64, 128]
                )
                with self.assertRaisesRegex(ValueError, ">= 64"):
                    filter_prefill_cp_bcg_capture_num_tokens([16, 32], cfg)
                cfg.cp_strategy = "zigzag"
                self.assertEqual(
                    filter_prefill_cp_bcg_capture_num_tokens(buckets, cfg),
                    [16, 32, 63, 64, 128],
                )
                cfg.cp_strategy = "interleave"
            with patch("sglang.srt.layers.cp.bcg.is_hip", return_value=False):
                self.assertEqual(
                    filter_prefill_cp_bcg_capture_num_tokens(buckets, cfg), buckets
                )

    def test_smallest_hip_bucket_is_cp_active(self):
        # Every captured bucket must shard; prepare_cp_forward asserts it does.
        init_cp_strategy(enable_prefill_cp=True, cp_size=4, cp_strategy="interleave")
        cfg = SimpleNamespace(attn_cp_size=4, cp_strategy="interleave")
        with (
            patch("sglang.srt.layers.cp.bcg.resolving_view", return_value=cfg),
            patch("sglang.srt.layers.cp.bcg.resolved_view", return_value=cfg),
            patch("sglang.srt.layers.cp.bcg.is_hip", return_value=True),
            patch("sglang.srt.layers.cp.interleave._is_hip", True),
        ):
            smallest = filter_prefill_cp_bcg_capture_num_tokens([8, 63, 64, 96], cfg)[0]
            for num_tokens, expected in ((smallest - 1, False), (smallest, True)):
                batch = SimpleNamespace(
                    forward_mode=ForwardMode.EXTEND,
                    extend_seq_lens_cpu=[num_tokens],
                )
                self.assertEqual(
                    get_cp_strategy().can_apply(num_tokens, batch), expected
                )

    def test_low_ratio_capture_accepts_attention_cp(self):
        metadata = object()
        backend = SimpleNamespace(
            low_ratios=(1, 2),
            has_c4=False,
            has_c128=False,
            _graph_metadata_max_seq_len=Mock(return_value=None),
            _prefill_metadata_for_batch=Mock(return_value=metadata),
            init_forward_metadata_in_graph=Mock(),
            _refresh_fp4_prefill_workspace=Mock(),
        )
        batch = SimpleNamespace(forward_mode=ForwardMode.EXTEND)
        with (
            get_parallel().override(attn_cp_size=4, attn_cp_rank=1),
            patch(
                "sglang.kernels.ops.attention.dsv4.unified_kv_kernels.env_gate."
                "is_unified_kv_triton",
                return_value=False,
            ),
        ):
            result = DeepseekV4HipRadixBackend.init_forward_metadata_for_breakable_cuda_graph_capture(
                backend, batch
            )
        self.assertIs(result, metadata)
        self.assertIs(backend.forward_metadata, metadata)
        backend.init_forward_metadata_in_graph.assert_called_once_with(batch)
        backend._refresh_fp4_prefill_workspace.assert_called_once_with(batch)

    def test_low_ratio_cp_graph_metadata_writes_only_live_rows(self):
        # A CP replay builds from the static batch, whose out_cache_loc is padded to
        # the bucket (32). The eager low-ratio sources write the 11 live rows, so the
        # write fields (raw/c1/c2 out_loc) must be built from the live prefix only.
        bucket, lengths = 32, [5, 6]
        for cp_active in (False, True):
            backend = SimpleNamespace(
                _init_forward_metadata_prefill_from_batch=Mock(return_value="meta")
            )
            out_cache_loc = torch.zeros(bucket, dtype=torch.int64)
            out_cache_loc[: sum(lengths)] = torch.arange(1, sum(lengths) + 1)
            batch = SimpleNamespace(
                seq_lens_cpu=torch.tensor([30, 41]),
                seq_lens=torch.tensor([30, 41]),
                req_pool_indices=torch.tensor([1, 2]),
                max_seq_len_override=None,
                extend_seq_lens_cpu=lengths,
                out_cache_loc=out_cache_loc,
            )
            with patch(f"{_HIP}.is_cp_active", return_value=cp_active):
                self.assertEqual(
                    DeepseekV4HipRadixBackend._prefill_metadata_for_batch(
                        backend, batch
                    ),
                    "meta",
                )
            kwargs = backend._init_forward_metadata_prefill_from_batch.call_args.kwargs
            self.assertEqual(kwargs["max_seq_len"], 41)
            if cp_active:
                torch.testing.assert_close(
                    kwargs["out_cache_loc"], out_cache_loc[: sum(lengths)]
                )
            else:
                self.assertIsNone(kwargs["out_cache_loc"])

    def test_low_ratio_sources_read_live_rows_under_a_bucket_total(self):
        # CP BCG sets total_seq_lens to the capture bucket (16); the compressor and
        # the indexer must still see the 11 live tokens of the [5, 6] extend.
        bucket, cp_size, lengths = 16, 2, [5, 6]
        live = sum(lengths)
        for rank in range(cp_size):
            positions = torch.zeros(bucket, dtype=torch.int64)
            positions[:live] = torch.cat([torch.arange(n) + 100 for n in lengths])
            batch = SimpleNamespace(
                attn_cp_metadata=SimpleNamespace(total_seq_lens=bucket),
                extend_seq_lens_cpu=list(lengths),
                positions=positions,
            )
            local_rows = bucket // cp_size
            x = torch.arange(local_rows, dtype=torch.float32)[:, None]
            gathered = torch.arange(bucket, dtype=torch.float32)[:, None]
            backend = SimpleNamespace(
                forward_metadata=SimpleNamespace(late_layer_tail=None),
                _low_ratio_compress=Mock(),
                _low_ratio_index_topk=Mock(),
            )
            layer = SimpleNamespace(compressor=object(), indexer=object())
            local_positions = positions[:live][rank::cp_size]
            with (
                get_parallel().override(attn_cp_rank=rank, attn_cp_size=cp_size),
                patch(
                    f"{_HIP}.cp_materialize_global_token_order",
                    return_value=gathered,
                ),
                patch(
                    f"{_HIP}.token_req_indices",
                    side_effect=lambda fb, num_tokens: torch.zeros(num_tokens),
                ) as req_indices,
            ):
                DeepseekV4HipRadixBackend._forward_low_ratio_sources_cp(
                    backend,
                    layer=layer,
                    x=x,
                    q_lora=torch.ones(local_rows, 4),
                    positions=torch.nn.functional.pad(
                        local_positions, (0, local_rows - local_positions.shape[0])
                    ),
                    forward_batch=batch,
                    run_compressor=True,
                    run_indexer=True,
                )
            self.assertEqual(req_indices.call_args.kwargs["num_tokens"], live)
            _, x_global, req, pos, _ = backend._low_ratio_compress.call_args.args
            torch.testing.assert_close(x_global, gathered[:live])
            self.assertEqual(req.shape[0], live)
            torch.testing.assert_close(pos, positions[:live])

            args = backend._low_ratio_index_topk.call_args
            rows = args.kwargs["rows_per_request"]
            self.assertEqual(sum(rows), local_positions.shape[0])
            self.assertEqual(args.args[1].shape[0], sum(rows))
            torch.testing.assert_close(args.args[4], local_positions)

    def test_cp_engram_hash_runs_at_an_eager_break_in_breakable_graphs(self):
        from sglang.srt.models import deepseek_v4

        bucket, cp_size, local_rows = 16, 2, 8
        for in_graph in (False, True):
            hash_ids = torch.arange(bucket)[:, None, None].expand(bucket, 2, 3)
            hasher = Mock(return_value=hash_ids)
            eager_break = Mock(return_value=hash_ids)
            batch = SimpleNamespace(
                forward_mode=ForwardMode.EXTEND,
                attn_cp_metadata=SimpleNamespace(total_seq_lens=bucket),
                input_ids=torch.arange(bucket + 4),
            )
            model = SimpleNamespace(
                pp_group=SimpleNamespace(world_size=1),
                engram_hasher=hasher,
                late_layer_start=None,
                start_layer=1,
                end_layer=1,
                layers=[SimpleNamespace(hc_cfg=None)],
            )
            state = Mock()
            state.materialized.return_value = SimpleNamespace(residual="r", pre="p")
            with (
                get_parallel().override(attn_cp_rank=1, attn_cp_size=cp_size),
                patch.object(deepseek_v4, "is_cp_active", return_value=True),
                patch.object(
                    deepseek_v4, "is_in_breakable_cuda_graph", return_value=in_graph
                ),
                patch.object(
                    deepseek_v4, "bcg_deepseek_v4_engram_hash_ids", eager_break
                ),
                patch.object(
                    deepseek_v4,
                    "get_forward",
                    return_value=SimpleNamespace(sp_active=False),
                ),
                patch.object(deepseek_v4.mhc, "HcState", return_value=state),
            ):
                out = deepseek_v4.DeepseekV4Model._forward_layers_hc_pre_from_prev(
                    model,
                    positions=torch.zeros(local_rows, dtype=torch.int64),
                    hidden_states=torch.zeros(local_rows, 4),
                    forward_batch=batch,
                    input_ids=torch.zeros(local_rows, dtype=torch.int64),
                    input_ids_global=torch.zeros(local_rows, dtype=torch.int64),
                    capture_dspark=False,
                    dspark_aux_hidden_states=[],
                )
            self.assertEqual(out, ("r", "p", None))
            called, idle = (eager_break, hasher) if in_graph else (hasher, eager_break)
            idle.assert_not_called()
            called.assert_called_once()
            ids = called.call_args.args[1 if in_graph else 0]
            torch.testing.assert_close(ids, batch.input_ids[:bucket])
            if in_graph:
                self.assertIs(called.call_args.args[0], hasher)


if __name__ == "__main__":
    unittest.main()
