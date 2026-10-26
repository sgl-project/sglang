"""Focused CPU contract tests for AFD cross-component invariants."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.afd.afd_config import (
    AFDRole,
    check_afd_server_args,
    finalize_afd_speculative_args,
)
from sglang.srt.afd.afd_handshake import (
    AFDGraphCaptureVariant,
    AFDHandshakeInfo,
    AFDHandshakeResult,
    _check_compatible,
    build_afd_ffn_capacity_plan,
    build_afd_graph_capture_plan,
)
from sglang.srt.afd.afd_scheduler import (
    _prepare_afd_attn_step,
    _tokens_per_rank,
    init_afd_mlp_sync_metadata,
    prepare_afd_attn_forward_batch,
)
from sglang.srt.afd.afd_sync import (
    AFDDirectProposal,
    AFDSyncMsg,
)
from sglang.srt.afd.comm.base import AFDCommConfig
from sglang.srt.afd.direct_deepep import AFDDirectDeepEPAdapter
from sglang.srt.afd.dispatcher import AFDDispatcher
from sglang.srt.afd.ffn.executor import AFDFFNExecutor
from sglang.srt.afd.ffn.scheduler import (
    _AFDFFNCohortSelector,
    _build_direct_union_action,
    _select_direct_union_sessions,
)
from sglang.srt.afd.layers import AFDMoELayer
from sglang.srt.afd.microbatch_overlap import (
    AFDMicrobatchAttentionBackend,
    _build_afd_microbatch_child_replay_views,
    prepare_afd_microbatch_forward_batch,
)
from sglang.srt.arg_groups.overrides import resolving_view
from sglang.srt.layers.compute_sm import get_compute_num_sms
from sglang.srt.managers.scheduler_components.dp_attn import _update_gather_batch
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    default_cuda_graph_config,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
    build_replay_fb_view,
)
from sglang.srt.runtime_context import get_context, get_disagg, get_parallel
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.eagle_info import EagleVerifyInput
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=30, suite="base-a-test-cpu")


class TestAFDDPMetadata(unittest.TestCase):
    def setUp(self):
        override = get_context().override_server_args(
            afd_role="attn",
            afd_ffn_moe_backend="none",
            afd_decode_num_microbatches=1,
            disable_cuda_graph_padding=False,
            cuda_graph_config=default_cuda_graph_config(),
        )
        override.install()
        self.addCleanup(override.restore)
        ranks = get_parallel().override(tp_rank=0)
        ranks.__enter__()
        self.addCleanup(ranks.__exit__, None, None, None)

    def test_target_prepare_uses_physical_rows_and_logical_graph_batch(self):
        begin_calls = []
        control_calls = []
        dispatcher = SimpleNamespace(
            begin_batch=lambda **kwargs: begin_calls.append(kwargs) or 7
        )
        model_runner = SimpleNamespace(afd_cuda_graph_batch_size=None)
        scheduler = SimpleNamespace(
            afd_handshake=SimpleNamespace(
                local=SimpleNamespace(
                    num_ranks=2,
                    decode_num_tokens_per_bs=4,
                    decode_token_buckets=[16, 32, 48],
                    decode_graph_token_buckets=[16, 32],
                )
            ),
            ps=SimpleNamespace(tp_rank=0),
            tp_worker=SimpleNamespace(model_runner=model_runner),
            _afd_attn_coordinator=None,
            _afd_sync_sender=SimpleNamespace(
                prepare_for_batch=lambda **kwargs: control_calls.append(kwargs)
            ),
        )
        spec_info = SimpleNamespace(
            is_verify_input=lambda: True,
            num_tokens_per_req=4,
            num_tokens_for_logprob_per_req=4,
        )
        forward_batch = SimpleNamespace(
            spec_info=spec_info,
            global_num_tokens_cpu=[8],
            original_global_num_tokens_cpu=[2, 3],
            is_extend_in_batch=False,
            input_ids=torch.empty(8, dtype=torch.int32),
            can_run_decode_cuda_graph=True,
        )
        max_len_padding = SimpleNamespace(is_max_len=lambda: True)

        for (
            layout,
            model_tokens,
            expected_dp_capacity,
            expected_wire_capacity,
            expected_tokens,
        ) in (
            ("none", [8, 12], 24, 32, [12, 12]),
            ("deepep", [8], 8, 16, [8, 12]),
        ):
            with self.subTest(layout=layout):
                begin_calls.clear()
                control_calls.clear()
                forward_batch.global_num_tokens_cpu = model_tokens
                with (
                    get_disagg().override(afd_ffn_moe_backend=layout),
                    patch(
                        "sglang.srt.afd.afd_scheduler.get_afd_dispatcher",
                        return_value=dispatcher,
                    ),
                    patch(
                        "sglang.srt.layers.dp_attention."
                        "DpPaddingMode.get_dp_padding_mode",
                        return_value=max_len_padding,
                    ),
                ):
                    can_run_paired_graph = prepare_afd_attn_forward_batch(
                        scheduler, forward_batch
                    )

                self.assertTrue(can_run_paired_graph)
                begin = begin_calls[0]
                control = control_calls[0]
                self.assertEqual(
                    (
                        model_runner.afd_cuda_graph_batch_size,
                        begin["forward_mode"],
                        begin["dp_capacity"],
                        begin["wire_capacity"],
                        control["tokens_per_rank"],
                        control["wire_capacity"],
                    ),
                    (
                        3,
                        ForwardMode.TARGET_VERIFY,
                        expected_dp_capacity,
                        expected_wire_capacity,
                        expected_tokens,
                        expected_wire_capacity,
                    ),
                )

        dispatcher.select_graph_variant = lambda _capacity: SimpleNamespace(
            child_wire_capacities=(16, 16)
        )
        with (
            get_disagg().override(
                afd_ffn_moe_backend="none", afd_decode_num_microbatches=2
            ),
            patch(
                "sglang.srt.afd.afd_scheduler.get_afd_dispatcher",
                return_value=dispatcher,
            ),
        ):
            for original_tokens, can_run_graph, expected in (
                ([8, 12], True, (True, (16, 16))),
                ([8, 12], False, (False, (16, 16))),
                ([4, 12], True, (True, (16, 16))),
                ([4, 12], False, (False, ())),
            ):
                with self.subTest(
                    original_tokens=original_tokens,
                    can_run_graph=can_run_graph,
                ):
                    begin_calls.clear()
                    can_run_paired_graph = _prepare_afd_attn_step(
                        scheduler,
                        tokens_per_rank=[12, 12],
                        forward_mode=ForwardMode.TARGET_VERIFY,
                        can_run_decode_cuda_graph=can_run_graph,
                        microbatch_tokens_per_rank=original_tokens,
                    )
                    self.assertEqual(
                        (
                            can_run_paired_graph,
                            begin_calls[0]["child_wire_capacities"],
                        ),
                        expected,
                    )

    def test_target_microbatch_replay_views_split_requests_and_tokens(self):
        seq_lens = torch.arange(10, 15)
        tokens = torch.arange(15)
        tree_mask = torch.arange(int((seq_lens + 3).sum()) * 3)
        spec_info = EagleVerifyInput(
            draft_token=tokens,
            custom_mask=tree_mask,
            positions=tokens,
            retrieve_index=tokens.view(5, 3),
            retrieve_next_token=None,
            retrieve_next_sibling=None,
            retrieve_cum_len=None,
            spec_steps=2,
            topk=1,
            draft_token_num=3,
            capture_hidden_mode=None,
            seq_lens_sum=None,
            seq_lens_cpu=None,
        )
        forward_batch = SimpleNamespace(
            batch_size=5,
            forward_mode=ForwardMode.TARGET_VERIFY,
            input_ids=tokens,
            positions=tokens,
            req_pool_indices=torch.arange(5),
            seq_lens=seq_lens,
            seq_lens_sum=None,
            seq_lens_cpu=None,
            encoder_lens=None,
            out_cache_loc=tokens,
            out_cache_loc_virtual=None,
            out_cache_loc_is_physical=True,
            kv_loc_plan=None,
            kv_loc_cols=None,
            max_seq_len_override=None,
            mamba_track_indices=torch.arange(20, 25),
            lora_ids=None,
            can_run_decode_cuda_graph=True,
            spec_info=spec_info,
        )
        for num_children, sizes in ((2, (3, 2)), (3, (2, 2, 1))):
            for host_lengths, needs_mask in (
                (None, False),
                (seq_lens, False),
                (seq_lens, True),
            ):
                with self.subTest(
                    children=num_children,
                    lengths=host_lengths is not None,
                    mask=needs_mask,
                ):
                    forward_batch.seq_lens_cpu = spec_info.seq_lens_cpu = host_lengths
                    replay_view = build_replay_fb_view(
                        forward_batch,
                        buffers=forward_batch,
                        bs=5,
                        raw_bs=4,
                        num_tokens=15,
                        global_num_tokens_cpu=None,
                        seq_len_fill_value=1,
                        capture_forward_mode=ForwardMode.TARGET_VERIFY,
                        is_encoder_decoder=False,
                    )
                    children = _build_afd_microbatch_child_replay_views(
                        replay_view,
                        torch.tensor(sizes) * 3,
                        num_children,
                        3,
                        needs_custom_mask=needs_mask,
                    )
                    start = 0
                    for child, size in zip(children, sizes, strict=True):
                        end = start + size
                        child_spec = child.spec_info
                        self.assertEqual(child.batch_size, size)
                        self.assertEqual(child.num_padding, max(0, end - 4))
                        self.assertEqual(
                            child.mamba_track_indices.tolist(),
                            list(range(20 + start, 20 + end)),
                        )
                        self.assertTrue(child.out_cache_loc_is_physical)
                        self.assertEqual(
                            child.input_ids.tolist(), list(range(start * 3, end * 3))
                        )
                        self.assertEqual(
                            child_spec.retrieve_index.flatten().tolist(),
                            list(range(start * 3, end * 3)),
                        )
                        if host_lengths is None:
                            self.assertIsNone(child.seq_lens_cpu)
                            self.assertIsNone(child.seq_lens_sum)
                            self.assertIsNone(child_spec.seq_lens_cpu)
                        else:
                            self.assertEqual(
                                child.seq_lens_cpu.tolist(),
                                seq_lens[start:end].tolist(),
                            )
                        if needs_mask:
                            lo = int((seq_lens[:start] + 3).sum()) * 3
                            hi = int((seq_lens[:end] + 3).sum()) * 3
                            self.assertEqual(
                                child_spec.custom_mask.tolist(), list(range(lo, hi))
                            )
                        else:
                            self.assertIsNone(child_spec.custom_mask)
                        start = end
                    self.assertIs(spec_info.custom_mask, tree_mask)

    def test_cross_dp_metadata_preserves_target_and_draft_views(self):
        batch = SimpleNamespace(is_extend_in_batch=False, spec_info=None)
        sync_info = SimpleNamespace(
            num_tokens=4,
            num_tokens_for_logprob=4,
            global_num_tokens=[4, 8, 6, 7],
            global_num_tokens_for_logprob=[4, 8, 6, 7],
            is_extend_in_batch=False,
            tbo_split_seq_index=None,
            global_forward_mode=ForwardMode.DECODE,
            can_run_decode_cuda_graph=True,
            can_run_prefill_cuda_graph=False,
            can_run_draft_cuda_graph=True,
            prefill_cuda_graph_max_prefix_len=None,
        )

        _update_gather_batch(batch, sync_info, require_mlp_tp_gather=False)

        self.assertEqual(batch.global_num_tokens, [4])
        self.assertEqual(
            batch.global_num_tokens_for_logprob_across_dp,
            [4, 8, 6, 7],
        )
        with patch(
            "sglang.srt.layers.dp_attention.DpPaddingMode.get_dp_padding_mode"
        ) as get_padding_mode:
            self.assertEqual(_tokens_per_rank(batch, 4), [4, 8, 6, 7])
            get_padding_mode.assert_not_called()

        draft = SimpleNamespace(
            is_draft_worker=True,
            device=torch.device("cpu"),
        )
        forward_batch = SimpleNamespace()
        init_afd_mlp_sync_metadata(forward_batch, batch, draft)
        self.assertEqual(
            (
                forward_batch.original_global_num_tokens_cpu,
                forward_batch.global_num_tokens_cpu,
                forward_batch.global_num_tokens_for_logprob_cpu,
            ),
            ([4, 8, 6, 7], [4, 8, 6, 7], [4, 8, 6, 7]),
        )

    def test_microbatch_residual_handoff_completes_each_sum_once(self):
        from sglang.srt.afd.microbatch_overlap import (
            merge_afd_residual_streams,
            split_afd_residual_stream,
        )
        from sglang.srt.layers.layer_boundary import PLAIN_ADD
        from sglang.srt.layers.layer_boundary.output import UnreducedOutput
        from sglang.srt.layers.layer_boundary.residual import batch as residual_batch
        from sglang.srt.layers.layer_boundary.residual.stream import ResidualStream

        for ranges in (((0, 3), (3, 5)), ((0, 2), (2, 4), (4, 5))):
            with self.subTest(ranges=ranges):
                partial = torch.arange(10, dtype=torch.float32).reshape(5, 2)
                group = SimpleNamespace(all_reduce=Mock(side_effect=lambda x: x * 2))
                parent = SimpleNamespace(
                    residual_stream=ResidualStream(torch.ones_like(partial)),
                    afd_microbatch_children=[
                        SimpleNamespace(tbo_parent_token_range=rows) for rows in ranges
                    ],
                )
                hidden = parent.residual_stream.record(
                    UnreducedOutput(partial, group=group), PLAIN_ADD
                )
                children = split_afd_residual_stream(hidden, parent)
                self.assertIsNone(parent.residual_stream)
                for index, child in enumerate(parent.afd_microbatch_children):
                    stream = child.residual_stream
                    stream.write(
                        stream.pending.update.update(children[index], stream.residual)
                    )
                    children[index] = stream.record(
                        UnreducedOutput(torch.ones_like(children[index]), group=group),
                        PLAIN_ADD,
                    )
                hidden = merge_afd_residual_streams(children, parent)
                result = residual_batch.final_norm(
                    hidden, parent, lambda value, residual: (value + residual, residual)
                )
                torch.testing.assert_close(result, partial * 2 + 3)
                self.assertEqual(group.all_reduce.call_count, 1 + len(ranges))
                self.assertIsNone(parent.residual_stream)
                self.assertTrue(
                    all(
                        child.residual_stream is None
                        for child in parent.afd_microbatch_children
                    )
                )


class TestAFDDSparkVerify(unittest.TestCase):
    def test_verify_graph_preparation_respects_afd_callback(self):
        from sglang.srt.speculative.dflash_info import DFlashVerifyInput

        for callback_result in (None, False, True):
            with self.subTest(callback_result=callback_result):
                events = []
                prepared = SimpleNamespace(
                    mark_forward_metadata_ready=lambda: events.append("ready")
                )
                runner = SimpleNamespace(
                    decode_cuda_graph_runner=SimpleNamespace(
                        can_run_graph=lambda _: events.append("can_run") or True,
                        load_batch=lambda _: events.append("load"),
                    ),
                    afd_target_verify_prepare_callback=(
                        None
                        if callback_result is None
                        else lambda _: events.append("afd") or callback_result
                    ),
                )
                verify = DFlashVerifyInput(torch.empty(0), torch.empty(0), 8)
                with patch(
                    "sglang.srt.speculative.dflash_info.ForwardBatch.init_new",
                    return_value=prepared,
                ):
                    _, can_run = verify.prepare_for_verify(
                        SimpleNamespace(forward_mode=ForwardMode.IDLE),
                        SimpleNamespace(model_runner=runner),
                    )
                expected = [] if callback_result is None else ["afd"]
                if callback_result is not False:
                    expected += ["can_run", "load", "ready"]
                self.assertEqual(can_run, callback_result is not False)
                self.assertEqual(events, expected)

    def test_microbatch_slices_keep_whole_verify_blocks(self):
        from sglang.srt.afd.microbatch_overlap import _split_afd_spec_info
        from sglang.srt.speculative.dflash_info import DFlashVerifyInput

        verify = DFlashVerifyInput(
            draft_token=torch.arange(40),
            positions=torch.arange(40) + 100,
            draft_token_num=8,
            live_seq_lens_cpu=torch.arange(5),
        )
        for sizes in ((5,), (3, 2), (2, 2, 1), (1, 0, 0)):
            offset = 0
            for size in sizes:
                child = _split_afd_spec_info(
                    verify, offset, offset + size, offset * 8, (offset + size) * 8
                )
                torch.testing.assert_close(
                    child.draft_token,
                    verify.draft_token[offset * 8 : (offset + size) * 8],
                )
                torch.testing.assert_close(
                    child.positions,
                    verify.positions[offset * 8 : (offset + size) * 8],
                )
                torch.testing.assert_close(
                    child.live_seq_lens_cpu,
                    verify.live_seq_lens_cpu[offset : offset + size],
                )
                self.assertEqual(child.num_tokens_per_req, 8)
                offset += size

    def test_static_verify_capture_without_draft_tensors(self):
        from sglang.srt.afd.microbatch_overlap import _split_afd_spec_info
        from sglang.srt.speculative.dflash_info import DFlashVerifyInput

        verify = DFlashVerifyInput(None, None, 8)
        child = _split_afd_spec_info(verify, 0, 2, 0, 16)
        self.assertIsNone(child.draft_token)
        self.assertIsNone(child.positions)
        self.assertEqual(child.num_tokens_per_req, 8)

    def test_microbatch_rejects_non_linear_verify_layouts(self):
        from sglang.srt.afd.microbatch_overlap import _split_afd_spec_info
        from sglang.srt.speculative.dflash_info import DFlashVerifyInput

        for field in ("custom_mask", "ragged_verify_layout"):
            with self.subTest(field=field):
                verify = DFlashVerifyInput(None, None, 8)
                setattr(verify, field, object())
                with self.assertRaisesRegex(ValueError, "static linear verify"):
                    _split_afd_spec_info(verify, 0, 2, 0, 16)

    def test_target_verify_restores_cross_dp_counts_before_scaling(self):
        from sglang.srt.speculative.dflash_info import DFlashVerifyInput

        verify = DFlashVerifyInput(
            draft_token=torch.empty(0, dtype=torch.int64),
            positions=torch.empty(0, dtype=torch.int64),
            draft_token_num=8,
        )
        forward = SimpleNamespace()
        init_afd_mlp_sync_metadata(
            forward,
            SimpleNamespace(
                global_num_tokens_across_dp=[0, 3],
                global_num_tokens_for_logprob_across_dp=[0, 3],
                is_extend_in_batch=False,
                draft_global_num_tokens=None,
                draft_global_num_tokens_for_logprob=None,
                dp_spec_prefill_coordination_applied=False,
                spec_info=verify,
            ),
            SimpleNamespace(is_draft_worker=False, device="cpu"),
        )
        self.assertEqual(forward.original_global_num_tokens_cpu, [0, 3])
        self.assertEqual(forward.global_num_tokens_cpu, [0, 24])

    def test_local_draft_preserves_cross_dp_counts_for_graph_replay(self):
        from sglang.srt.speculative.dflash_info import DFlashVerifyInput
        from sglang.srt.speculative.dspark_components.dspark_draft import (
            DraftBlockProposer,
        )

        proposer = DraftBlockProposer.__new__(DraftBlockProposer)
        proposer._dp_moe_sync = True
        proposer._num_token_non_padded = None
        proposer._draft_block_spec_info = DFlashVerifyInput(
            torch.empty(0), torch.empty(0), 8
        )
        batch = SimpleNamespace(
            can_run_decode_cuda_graph=True,
            global_num_tokens=[0],
            global_num_tokens_for_logprob=[0],
            global_num_tokens_across_dp=[0, 3],
            global_num_tokens_for_logprob_across_dp=[0, 3],
            is_extend_in_batch=False,
            draft_global_num_tokens=None,
            draft_global_num_tokens_for_logprob=None,
            dp_spec_prefill_coordination_applied=False,
        )
        for role, counts in (("attn", [0, 24]), (None, [0])):
            with (
                self.subTest(role=role),
                get_context().override_server_args(afd_role=role),
                patch(
                    "sglang.srt.speculative.dspark_components.dspark_draft._make_num_token_non_padded",
                    return_value=None,
                ),
            ):
                proposer.draft_model_runner = SimpleNamespace(
                    device="cpu", afd_cuda_graph_batch_size=None
                )
                forward = SimpleNamespace(input_ids=torch.empty(0, dtype=torch.int64))
                proposer._fill_dp_moe_sync_metadata(forward, batch)
                self.assertEqual(forward.global_num_tokens_cpu, counts)
                self.assertEqual(
                    forward.original_global_num_tokens_cpu,
                    (
                        batch.global_num_tokens_across_dp
                        if role == "attn"
                        else batch.global_num_tokens
                    ),
                )
                self.assertIsNone(proposer.draft_model_runner.afd_cuda_graph_batch_size)

    def test_local_draft_selects_configured_fp4_runner_and_restores_target(self):
        from sglang.srt.layers.moe.utils import (
            MoeA2ABackend,
            MoeRunnerBackend,
            get_moe_a2a_backend,
            get_moe_runner_backend,
        )
        from sglang.srt.layers.quantization.fp8 import Fp8MoEMethod
        from sglang.srt.runtime_context import get_flags
        from sglang.srt.speculative.dspark_components.dspark_worker_v2 import (
            DSparkWorkerV2,
        )

        worker = DSparkWorkerV2.__new__(DSparkWorkerV2)
        worker._draft_dp_context_enabled = False
        method = Fp8MoEMethod.__new__(Fp8MoEMethod)
        method.is_fp4_expert = True
        runner_config = SimpleNamespace()
        for role in ("attn", None):
            with (
                self.subTest(role=role),
                get_context().override_server_args(afd_role=role),
                patch.object(get_flags().moe, "runner_backend", MoeRunnerBackend.AUTO),
                patch.object(
                    get_flags().moe,
                    "speculative_runner_backend",
                    MoeRunnerBackend.DEEP_GEMM,
                ),
                patch.object(get_flags().moe, "a2a_backend", MoeA2ABackend.AFD_ATTN),
                patch.object(
                    get_flags().moe, "speculative_a2a_backend", MoeA2ABackend.NONE
                ),
                patch("sglang.srt.layers.quantization.fp8.MoeRunner") as runner,
                patch("sglang.srt.layers.quantization.fp8._is_hip", False),
            ):
                with self.assertRaisesRegex(RuntimeError, "draft failure"):
                    with worker._draft_context():
                        method.create_moe_runner(SimpleNamespace(), runner_config)
                        runner.assert_called_once_with(
                            (
                                MoeRunnerBackend.DEEP_GEMM
                                if role == "attn"
                                else MoeRunnerBackend.TRITON
                            ),
                            runner_config,
                        )
                        raise RuntimeError("draft failure")
                self.assertEqual(get_moe_runner_backend(), MoeRunnerBackend.AUTO)
                self.assertEqual(get_moe_a2a_backend(), MoeA2ABackend.AFD_ATTN)

    def test_idle_draft_enters_local_moe_context_and_restores_target(self):
        from sglang.srt.layers.moe.utils import (
            MoeA2ABackend,
            MoeRunnerBackend,
            get_moe_a2a_backend,
            get_moe_runner_backend,
        )
        from sglang.srt.runtime_context import get_flags
        from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2
        from sglang.srt.speculative.dspark_components.dspark_worker_v2 import (
            DSparkWorkerV2,
        )

        worker = DSparkWorkerV2.__new__(DSparkWorkerV2)
        worker._draft_is_moe = True
        worker._draft_dp_context_enabled = False
        worker._observers = SimpleNamespace(note_idle_decode_step=lambda: None)
        observed = []
        worker._proposer = SimpleNamespace(
            run_idle_participation=lambda _: observed.append(
                (get_moe_a2a_backend().value, get_moe_runner_backend().value)
            )
        )
        worker._verify_executor = SimpleNamespace(
            run_idle_participation=lambda **_: observed.append(
                (get_moe_a2a_backend().value, get_moe_runner_backend().value)
            )
        )
        worker._idle_verify_ragged_layout = lambda _: None
        worker._decode_idle_result = lambda **_: None
        batch = SimpleNamespace(
            spec_info=DFlashDraftInputV2.create_idle_input(device="cpu"),
            forward_mode=ForwardMode.IDLE,
            global_num_tokens=None,
        )
        with (
            get_context().override_server_args(
                afd_role="attn",
                attn_dp_size=2,
            ),
            patch.object(get_flags().moe, "runner_backend", MoeRunnerBackend.AUTO),
            patch.object(
                get_flags().moe,
                "speculative_runner_backend",
                MoeRunnerBackend.DEEP_GEMM,
            ),
            patch.object(get_flags().moe, "a2a_backend", MoeA2ABackend.AFD_ATTN),
            patch.object(
                get_flags().moe, "speculative_a2a_backend", MoeA2ABackend.NONE
            ),
        ):
            worker._forward_decode(batch, on_publish=None)
            self.assertEqual(observed, [("none", "deep_gemm"), ("afd", "auto")])
            self.assertEqual(get_moe_a2a_backend().value, "afd")
            self.assertEqual(get_moe_runner_backend(), MoeRunnerBackend.AUTO)

    def test_afd_eager_verify_defers_attention_metadata(self):
        from sglang.srt.speculative.dflash_info import DFlashVerifyInput

        events = []
        graph_runner = SimpleNamespace(can_run_graph=lambda _: events.append("graph"))
        model_runner = SimpleNamespace(
            decode_cuda_graph_runner=graph_runner,
            afd_target_verify_prepare_callback=lambda _: events.append("afd") or False,
            attn_backend=SimpleNamespace(
                init_forward_metadata=lambda _: events.append("metadata")
            ),
        )
        verify = DFlashVerifyInput(torch.empty(0), torch.empty(0), 8)
        with (
            patch(
                "sglang.srt.speculative.dflash_info.ForwardBatch.init_new",
                return_value=SimpleNamespace(),
            ),
            patch("sglang.srt.speculative.spec_utils.prepare_mamba_track_for_verify"),
        ):
            _, can_run_graph = verify.prepare_for_verify(
                SimpleNamespace(forward_mode=ForwardMode.DECODE),
                SimpleNamespace(model_runner=model_runner),
            )
        self.assertFalse(can_run_graph)
        self.assertEqual(events, ["afd"])

    def test_non_afd_eager_verify_preserves_metadata_preparation(self):
        from sglang.srt.speculative.dflash_info import DFlashVerifyInput

        prepared = SimpleNamespace(mark_forward_metadata_ready=Mock())
        attn_backend = SimpleNamespace(init_forward_metadata=Mock())
        runner = SimpleNamespace(
            decode_cuda_graph_runner=SimpleNamespace(can_run_graph=lambda _: False),
            attn_backend=attn_backend,
        )
        verify = DFlashVerifyInput(torch.empty(0), torch.empty(0), 8)
        with (
            patch(
                "sglang.srt.speculative.dflash_info.ForwardBatch.init_new",
                return_value=prepared,
            ),
            patch("sglang.srt.speculative.spec_utils.prepare_mamba_track_for_verify"),
        ):
            _, can_run = verify.prepare_for_verify(
                SimpleNamespace(forward_mode=ForwardMode.DECODE),
                SimpleNamespace(model_runner=runner),
            )
        self.assertFalse(can_run)
        attn_backend.init_forward_metadata.assert_called_once_with(prepared)
        prepared.mark_forward_metadata_ready.assert_called_once_with(
            replan_equivalent=True
        )


class _FakeAFDCommBackend:
    def __init__(self):
        self.config = AFDCommConfig(
            role=AFDRole.ATTN,
            num_attn_ranks=1,
            num_ffn_ranks=1,
            num_local_attn_ranks=1,
            instance_rank=0,
            store_host="127.0.0.1",
            store_port=1,
            device="cpu",
        )
        self.decode_transport = None
        self.send_shapes = []
        self.recv_shapes = []

    def send_to(self, tensors, _peer_global_rank):
        self.send_shapes.append([tuple(tensor.shape) for tensor in tensors])

    def recv_from(self, tensors, _peer_global_rank):
        self.recv_shapes.append([tuple(tensor.shape) for tensor in tensors])
        tensors[0].fill_(len(self.recv_shapes))


class _FakeDirectResult:
    def __init__(self, lane_id, hidden_states, events, wait_event):
        self.lane_id = lane_id
        self.hidden_states = hidden_states
        self.events = events
        self.wait_event = wait_event

    def wait(self):
        self.events.append((self.wait_event, self.lane_id))
        return self.hidden_states


class _FakeDirectLane:
    def __init__(self, lane_id, events):
        self.lane_id = lane_id
        self.events = events
        self.dispatch_calls = []
        self.async_calls = []
        self.scaling_factors = []

    def _result(self, hidden_states, wait_event):
        return _FakeDirectResult(self.lane_id, hidden_states, self.events, wait_event)

    def dispatch(self, _layer_id, hidden_states, topk_ids, topk_weights, **kwargs):
        self.events.append(("dispatch", self.lane_id))
        self.dispatch_calls.append((hidden_states, kwargs))
        self.async_calls.append(
            kwargs.get("async_with_compute_stream") is True
            and kwargs.get("do_cpu_sync") is False
        )
        return self._result(hidden_states, "wait_dispatch")

    def combine_attn(self, dispatch_output, **kwargs):
        self.events.append(("combine_attn", self.lane_id))
        self.async_calls.append(kwargs.get("async_with_compute_stream") is True)
        return self._result(
            torch.full_like(dispatch_output.hidden_states, 7),
            "wait_combine",
        )

    def combine(self, expert_output, _dispatch_output, **kwargs):
        self.events.append(("combine", self.lane_id))
        self.scaling_factors.append(kwargs["routed_scaling_factor"])
        self.async_calls.append(kwargs.get("async_with_compute_stream") is True)
        return self._result(expert_output, "wait_combine")


class TestAFDDispatcher(unittest.TestCase):
    def test_captured_direct_combines_follow_lane_reuse(self):
        events = []
        lanes = tuple(_FakeDirectLane(i, events) for i in range(2))
        for lane in lanes:
            lane.overlap_communication_num_sms = 8
        dispatcher = AFDDispatcher(
            comm=_FakeAFDCommBackend(),
            layers=[
                SimpleNamespace(layer_id=i, hidden_size=4, top_k=2) for i in range(3)
            ],
            token_buckets=(4,),
            dtype=torch.float32,
            device=torch.device("cpu"),
            scattered=True,
            num_slots=2,
            direct_deepep_lanes=lanes,
            compute_num_sms=70,
        )
        payload = (
            torch.ones((4, 4)),
            torch.zeros((4, 2), dtype=torch.int32),
            torch.ones((4, 2)),
        )
        for capturing in (False, True):
            with (
                self.subTest(capturing=capturing),
                patch("torch.cuda.is_current_stream_capturing", return_value=capturing),
            ):
                events.clear()
                dispatcher.begin_batch(
                    ForwardMode.DECODE,
                    dp_capacity=8,
                    wire_capacity=8,
                    child_wire_capacities=(4, 4),
                )
                dispatcher._capturing_parent_graph = True
                handles = [dispatcher.dispatch(0, *payload) for _ in lanes]
                for layer in range(3):
                    for i in range(2):
                        self.assertTrue(torch.all(dispatcher.combine(handles[i]) == 7))
                        if layer < 2:
                            handles[i] = dispatcher.dispatch(layer + 1, *payload)
                self.assertFalse(dispatcher._active_handles)
                self.assertFalse(dispatcher._pending_direct_combines)
                self.assertEqual(sum(e[0] == "combine_attn" for e in events), 6)
                self.assertEqual(sum(e[0] == "wait_combine" for e in events), 6)
                self.assertEqual(dispatcher.compute_num_sms, 70)
                self.assertEqual(dispatcher.shared_expert_compute_num_sms(2), 62)
                if capturing:
                    for index, event in enumerate(events):
                        if event[0] == "dispatch":
                            self.assertEqual(
                                events[index + 1], ("combine_attn", event[1])
                            )
                else:
                    self.assertEqual(events[:2], [("dispatch", 0), ("dispatch", 1)])
                dispatcher._capturing_parent_graph = False

    def test_inactive_direct_step_preserves_layer_and_lane_order(self):
        layers = [
            SimpleNamespace(layer_id=3, hidden_size=4, top_k=2),
            SimpleNamespace(layer_id=7, hidden_size=4, top_k=2),
        ]
        child_capacities = (4, 3, 2)
        events = []
        lanes = tuple(_FakeDirectLane(lane_id, events) for lane_id in range(3))
        dispatcher = AFDDispatcher(
            comm=_FakeAFDCommBackend(),
            layers=layers,
            token_buckets=child_capacities,
            dtype=torch.float32,
            device=torch.device("cpu"),
            scattered=True,
            num_slots=3,
            direct_deepep_lanes=lanes,
        )

        dispatcher.run_inactive_direct_step(
            ForwardMode.DECODE,
            wire_capacity=sum(child_capacities),
            child_wire_capacities=child_capacities,
            do_cpu_sync=False,
        )

        self.assertEqual(
            events,
            [
                ("dispatch", 0),
                ("dispatch", 1),
                ("dispatch", 2),
                ("combine_attn", 0),
                ("combine_attn", 1),
                ("combine_attn", 2),
                ("wait_combine", 0),
                ("dispatch", 0),
                ("wait_combine", 1),
                ("dispatch", 1),
                ("wait_combine", 2),
                ("dispatch", 2),
                ("combine_attn", 0),
                ("combine_attn", 1),
                ("combine_attn", 2),
                ("wait_combine", 0),
                ("wait_combine", 1),
                ("wait_combine", 2),
            ],
        )
        self.assertTrue(
            all(
                call[0].shape == (0, 4)
                for lane in lanes
                for call in lane.dispatch_calls
            )
        )
        self.assertTrue(all(all(lane.async_calls) for lane in lanes))

    def test_staged_fixed_buffer_padding_reuse_and_prefill_exact_size(self):
        layer = AFDMoELayer(
            layer_id=3,
            experts=torch.nn.Identity(),
            hidden_size=4,
            intermediate_size=8,
            top_k=2,
            num_experts=4,
            num_fused_shared_experts=0,
            routed_scaling_factor=None,
        )
        comm = _FakeAFDCommBackend()
        dispatcher = AFDDispatcher(
            comm=comm,
            layers=[layer],
            token_buckets=[4, 8, 12],
            dtype=torch.float32,
            device=torch.device("cpu"),
            scattered=True,
            num_slots=3,
        )

        def payload(num_tokens):
            return (
                torch.arange(num_tokens * 4, dtype=torch.float32).view(num_tokens, 4),
                torch.zeros((num_tokens, 2), dtype=torch.int32),
                torch.ones((num_tokens, 2), dtype=torch.float32),
            )

        dispatcher.begin_batch(
            ForwardMode.TARGET_VERIFY, dp_capacity=3, wire_capacity=4
        )
        self.assertTrue(dispatcher.needs_empty_decode_dispatch)
        first = dispatcher.dispatch(3, *payload(3))
        first_slot_ptr = first.hidden_states.data_ptr()
        self.assertEqual(first.capacity, 4)
        self.assertEqual(torch.count_nonzero(first.hidden_states[3:]).item(), 0)
        self.assertTrue(torch.all(first.topk_ids[3:] == -1))
        self.assertEqual(torch.count_nonzero(first.topk_weights[3:]).item(), 0)
        self.assertEqual(comm.send_shapes[-1], [(4, 4), (4, 2), (4, 2)])
        first_output = dispatcher.combine(first)
        self.assertEqual(comm.recv_shapes[-1], [(4, 4)])
        self.assertEqual(tuple(first_output.shape), (3, 4))
        self.assertTrue(torch.all(first_output == 1))

        dispatcher.begin_batch(ForwardMode.DECODE, dp_capacity=5, wire_capacity=8)
        second = dispatcher.dispatch(3, *payload(5))
        self.assertEqual(second.capacity, 8)
        self.assertEqual(second.hidden_states.data_ptr(), first_slot_ptr)
        self.assertEqual(comm.send_shapes[-1], [(8, 4), (8, 2), (8, 2)])
        second_output = dispatcher.combine(second)
        self.assertEqual(comm.recv_shapes[-1], [(8, 4)])
        self.assertEqual(tuple(second_output.shape), (5, 4))
        self.assertTrue(torch.all(second_output == 2))

        dispatcher.begin_batch(ForwardMode.EXTEND, dp_capacity=5, wire_capacity=5)
        prefill = dispatcher.dispatch(3, *payload(5))
        self.assertIsNone(prefill.slot_id)
        self.assertEqual(prefill.capacity, 5)
        self.assertEqual(comm.send_shapes[-1], [(5, 4), (5, 2), (5, 2)])
        prefill_output = dispatcher.combine(prefill)
        self.assertEqual(comm.recv_shapes[-1], [(5, 4)])
        self.assertTrue(torch.all(prefill_output == 3))


class TestAFDFFNExecutor(unittest.TestCase):
    def test_direct_pipeline_follows_child_lane_order(self):
        layers = [
            SimpleNamespace(
                layer_id=3, hidden_size=4, top_k=2, routed_scaling_factor=None
            ),
            SimpleNamespace(
                layer_id=4, hidden_size=4, top_k=2, routed_scaling_factor=2.5
            ),
        ]
        events = []
        lanes = tuple(_FakeDirectLane(lane_id, events) for lane_id in range(3))
        executor = AFDFFNExecutor.__new__(AFDFFNExecutor)
        executor._direct_deepep_lanes = lanes
        executor._layers = layers
        executor._dtype = torch.float32
        executor._device = torch.device("cpu")
        executor._compute_num_sms = 136
        compute_num_sms = []
        executor._compute = SimpleNamespace(
            run_direct_deepep=lambda layer_index, output: (
                events.append(("compute", layer_index, output.lane_id))
                or compute_num_sms.append(get_compute_num_sms())
                or output.hidden_states
            )
        )
        with (
            patch("sglang.srt.afd.ffn.executor.set_is_extend_in_batch"),
            patch("torch.cuda.is_current_stream_capturing", return_value=True),
            patch(
                "sglang.srt.layers.deep_gemm_wrapper.configure_deep_gemm_num_sms"
            ) as configure_num_sms,
        ):
            executor._run_direct_deepep_step((4, 3, 2), do_cpu_sync=False)

        expected = [("dispatch", lane_id) for lane_id in range(3)]
        for layer_index in range(2):
            for lane_id in range(3):
                expected.extend(
                    [
                        ("wait_dispatch", lane_id),
                        ("compute", layer_index, lane_id),
                        ("combine", lane_id),
                    ]
                )
                if layer_index == 0:
                    expected.append(("dispatch", lane_id))
        expected.extend(("wait_combine", lane_id) for lane_id in range(3))
        self.assertEqual(events, expected)
        configure_num_sms.assert_called_once_with(136)
        self.assertEqual(compute_num_sms, [136] * 6)
        self.assertIsNone(get_compute_num_sms())
        self.assertTrue(all(all(lane.async_calls) for lane in lanes))
        self.assertTrue(all(lane.scaling_factors == [1.0, 2.5] for lane in lanes))

        events.clear()
        with (
            patch("sglang.srt.afd.ffn.executor.set_is_extend_in_batch"),
            patch("torch.cuda.is_current_stream_capturing", return_value=False),
            patch("sglang.srt.layers.deep_gemm_wrapper.configure_deep_gemm_num_sms"),
        ):
            executor._run_direct_deepep_step((4, 3, 2), do_cpu_sync=False)
        # First-use CUDA module/library setup must finish before any receive for
        # the next layer can be queued during eager execution and graph warmup.
        self.assertLess(events.index(("combine", 2)), events.index(("dispatch", 0), 3))

        compute_num_sms.clear()
        with patch("sglang.srt.afd.ffn.executor.set_is_extend_in_batch"):
            executor._run_direct_deepep_step((4,), do_cpu_sync=False)
        self.assertEqual(compute_num_sms, [None] * 2)
        self.assertIsNone(get_compute_num_sms())


def _make_args(**overrides) -> ServerArgs:
    args = dict(
        model_path="dummy",
        afd_role="attn",
        afd_ffn_addr="127.0.0.1:30001",
        afd_backend="nccl",
        afd_decode_num_microbatches=1,
        afd_comm_sm_budget=None,
        afd_direct_deepep_disable_hybrid=False,
        afd_attn_instances=1,
        port=30000,
        disaggregation_mode="null",
        pp_size=1,
        speculative_algorithm=None,
        speculative_adaptive=False,
        speculative_moe_a2a_backend=None,
        speculative_skip_dp_mlp_sync=False,
        enable_multi_layer_eagle=False,
        decoupled_spec_role="null",
        cuda_graph_config=default_cuda_graph_config(),
        disable_overlap_schedule=False,
        debug_cuda_graph=False,
        moe_a2a_backend="none",
        afd_ffn_moe_backend="none",
        moe_runner_backend="auto",
        quantization=None,
        tp_size=1,
        ep_size=1,
        moe_dp_size=1,
        attn_dp_size=1,
        dp_size=1,
        attn_cp_size=1,
        enable_prefill_cp=False,
        enable_eplb=False,
        ep_num_redundant_experts=0,
        init_expert_location="trivial",
        enable_waterfill=False,
        enable_two_batch_overlap=False,
        enable_single_batch_overlap=False,
        enable_pdmux=False,
        enable_torch_compile=False,
        enable_lora=False,
        disable_shared_experts_fusion=False,
        enforce_shared_experts_fusion=False,
        skip_server_warmup=False,
    )
    args.update(overrides)
    return ServerArgs(**args)


def _make_eagle_args(**overrides) -> ServerArgs:
    eagle_args = {
        "speculative_algorithm": "EAGLE",
        "speculative_eagle_topk": 1,
        "speculative_num_steps": 3,
        "speculative_num_draft_tokens": 4,
    }
    eagle_args.update(overrides)
    return _make_args(**eagle_args)


class TestAFDServerArgsCheck(unittest.TestCase):
    def test_static_topk1_eagle_is_supported(self):
        for backend, layout in (
            ("nccl", "none"),
            ("direct_deepep", "deepep"),
        ):
            with self.subTest(backend=backend, layout=layout):
                args = _make_eagle_args(
                    afd_backend=backend,
                    moe_a2a_backend=layout,
                )
                check_afd_server_args(args)
                finalize_afd_speculative_args(args)

                self.assertEqual(resolving_view(args).afd_ffn_moe_backend, layout)
                self.assertEqual(
                    resolving_view(args).speculative_moe_a2a_backend, "none"
                )

    def test_dspark_all_afd_backends_and_overlap_modes(self):
        for backend in ("nccl", "nvshmem", "direct_deepep", "direct_megamoe"):
            for microbatches in (1, 2, 3):
                with self.subTest(backend=backend, microbatches=microbatches):
                    args = _make_eagle_args(
                        speculative_algorithm="DSPARK",
                        speculative_moe_runner_backend="deep_gemm",
                        speculative_num_steps=1,
                        speculative_num_draft_tokens=8,
                        afd_backend=backend,
                        afd_decode_num_microbatches=microbatches,
                        tp_size=2,
                        attn_dp_size=2,
                        enable_dp_lm_head=True,
                    )
                    check_afd_server_args(args)
                    finalize_afd_speculative_args(args)
                    self.assertEqual(
                        resolving_view(args).speculative_moe_a2a_backend, "none"
                    )

    def test_dspark_fp4_resolves_deferred_auto_runner(self):
        from sglang.srt.arg_groups.overrides import declare_resolution

        args = _make_eagle_args(
            speculative_algorithm="DSPARK",
            speculative_moe_runner_backend=None,
            speculative_num_steps=1,
            speculative_num_draft_tokens=8,
        )
        declare_resolution(
            args, "speculative_default", speculative_moe_runner_backend="auto"
        )
        check_afd_server_args(args)
        with patch(
            "sglang.srt.afd.afd_config.model_config_of",
            return_value=SimpleNamespace(is_fp4_experts=True),
        ):
            finalize_afd_speculative_args(args)
        self.assertEqual(
            resolving_view(args).speculative_moe_runner_backend, "deep_gemm"
        )

    def test_rejects_unsupported_speculative_combinations(self):
        disabled_graph = default_cuda_graph_config()
        disabled_graph.decode.backend = Backend.DISABLED
        cases = (
            ({"speculative_algorithm": "MTP"}, "speculative_algorithm in"),
            ({"speculative_eagle_topk": 2}, "speculative-eagle-topk 1"),
            (
                {"cuda_graph_config": disabled_graph},
                "CUDA graph backend 'full'",
            ),
            ({"speculative_moe_a2a_backend": "deepep"}, "draft-local"),
        )
        for overrides, pattern in cases:
            with self.subTest(pattern=pattern):
                with self.assertRaisesRegex(ValueError, pattern):
                    args = _make_eagle_args(**overrides)
                    check_afd_server_args(args)
                    finalize_afd_speculative_args(args)


class TestAFDCohortSelection(unittest.TestCase):
    @staticmethod
    def _msg(step_id: int, wire_capacity: int = 1) -> AFDSyncMsg:
        return AFDSyncMsg(
            step_id=step_id,
            tokens_per_rank=[1],
            forward_mode=int(ForwardMode.DECODE),
            wire_capacity=wire_capacity,
        )

    def test_step_boundary_policy(self):
        selector = _AFDFFNCohortSelector(join_timeout_us=10)

        def can_cohort(_session_id, _msg):
            return True

        def cohort_cost(_session_id, msg):
            return (msg.wire_capacity,)

        def select(now_ns, *, capacity=None):
            return selector.select(
                now_ns,
                can_cohort=can_cohort,
                cohort_cost=cohort_cost,
                cohort_capacity=capacity,
            )

        selector.add(0, self._msg(0))
        self.assertTrue(selector.has_pending_session(0))
        self.assertFalse(selector.has_pending_session(1))
        self.assertIsNone(select(0))
        action = select(10_000)
        self.assertEqual([session_id for session_id, _ in action], [0])
        self.assertFalse(selector.has_pending_session(0))

        selector.add(0, self._msg(1))
        action = select(10_001)
        self.assertEqual([session_id for session_id, _ in action], [0])

        selector.add(1, self._msg(0))
        self.assertIsNone(select(10_002))
        selector.add(0, self._msg(2))
        action = select(10_003)
        self.assertEqual([session_id for session_id, _ in action], [1, 0])

        selector.add(1, self._msg(1))
        self.assertIsNone(select(10_004))
        self.assertIsNone(select(20_003))
        action = select(20_004)
        self.assertEqual([session_id for session_id, _ in action], [1])

        selector.add(0, self._msg(3))
        selector.add(1, self._msg(2))
        selector.add(2, self._msg(0))
        action = select(20_005)
        self.assertEqual([session_id for session_id, _ in action], [0, 1, 2])

        selector.add(0, self._msg(4, wire_capacity=2))
        selector.add(1, self._msg(3, wire_capacity=2))
        selector.add(2, self._msg(1, wire_capacity=1))
        action = select(20_006, capacity=3)
        self.assertEqual([session_id for session_id, _ in action], [0, 2])

        self.assertIsNone(select(30_005, capacity=3))
        action = select(30_006, capacity=3)
        self.assertEqual([session_id for session_id, _ in action], [1])


def _handshake_info(
    role: str,
    num_ranks: int = 2,
    backend: str = "nccl",
    decode_token_buckets=None,
    decode_graph_token_buckets=None,
    afd_ffn_moe_backend: str = "none",
    decode_num_tokens_per_bs: int = 1,
    decode_num_microbatches: int = 1,
):
    return AFDHandshakeInfo(
        role=role,
        model_identity="m:abc",
        num_ranks=num_ranks,
        rank_node_ids=["node-0"] * num_ranks,
        afd_backend=backend,
        afd_ffn_moe_backend=afd_ffn_moe_backend,
        wire_dtype="bfloat16",
        moe_layers=[],
        decode_token_buckets=(
            [] if decode_token_buckets is None else decode_token_buckets
        ),
        decode_graph_token_buckets=(
            [] if decode_graph_token_buckets is None else decode_graph_token_buckets
        ),
        decode_num_tokens_per_bs=decode_num_tokens_per_bs,
        decode_num_microbatches=decode_num_microbatches,
    )


class TestAFDHandshake(unittest.TestCase):
    def test_staged_ffn_capacity_uses_graph_children_and_transport_max(self):
        ffn = _handshake_info(
            "ffn",
            num_ranks=2,
            afd_ffn_moe_backend="deepep",
            decode_graph_token_buckets=[1],
        )
        attn0 = _handshake_info(
            "attn",
            num_ranks=4,
            afd_ffn_moe_backend="deepep",
            decode_token_buckets=[32, 64, 96],
            decode_graph_token_buckets=[64],
            decode_num_microbatches=2,
        )
        attn1 = _handshake_info(
            "attn",
            num_ranks=2,
            afd_ffn_moe_backend="deepep",
            decode_token_buckets=[16, 32, 64],
            decode_graph_token_buckets=[32],
            decode_num_microbatches=2,
        )

        single = build_afd_ffn_capacity_plan(ffn, (attn0,), True)
        self.assertEqual(single.graph_token_buckets, (64,))
        self.assertEqual(single.deepep_dispatch_limit, 192)

        cohort = build_afd_ffn_capacity_plan(ffn, (attn0, attn1), True)
        self.assertEqual(cohort.graph_token_buckets, (16, 64, 72, 80))
        self.assertEqual(cohort.deepep_dispatch_limit, 256)

        ffn.disable_cuda_graph_padding = True
        exact = build_afd_ffn_capacity_plan(ffn, (attn0, attn1), True)
        self.assertEqual(exact.graph_token_buckets, (16, 64, 80))

        attn0.decode_token_buckets[-1] = 2048
        normal = build_afd_ffn_capacity_plan(ffn, (attn0,), False)
        self.assertIsNone(normal.deepep_dispatch_limit)

    def test_fixed_multi_token_graph_supports_asymmetric_layouts(self):
        topologies = (
            ("full_a_gt_f", "none", 4, 2),
            ("full_a_lt_f", "none", 2, 4),
            ("scattered_a_gt_f", "deepep", 4, 2),
            ("scattered_a_lt_f", "deepep", 2, 4),
        )
        for backend in ("nccl", "nvshmem"):
            for (
                name,
                layout,
                num_attn_ranks,
                num_ffn_ranks,
            ) in topologies:
                with self.subTest(backend=backend, topology=name):
                    scattered = layout == "deepep"
                    wire_multiplier = 4 * (1 if scattered else num_attn_ranks)
                    fan_in = (
                        (num_attn_ranks + num_ffn_ranks - 1) // num_ffn_ranks
                        if scattered
                        else 1
                    )
                    wire_capacities = [
                        batch_size * wire_multiplier for batch_size in (4, 8)
                    ]
                    compute_capacities = [
                        capacity * fan_in for capacity in wire_capacities
                    ]
                    attn = _handshake_info(
                        "attn",
                        num_ranks=num_attn_ranks,
                        backend=backend,
                        afd_ffn_moe_backend=layout,
                        decode_token_buckets=wire_capacities,
                        decode_graph_token_buckets=wire_capacities,
                        decode_num_tokens_per_bs=4,
                    )
                    ffn = _handshake_info(
                        "ffn",
                        num_ranks=num_ffn_ranks,
                        backend=backend,
                        afd_ffn_moe_backend=layout,
                        decode_graph_token_buckets=compute_capacities,
                    )
                    handshake = AFDHandshakeResult(
                        local=ffn,
                        peer=attn,
                        dist_host="127.0.0.1",
                        sync_port=1,
                        dist_port=2,
                    )

                    _check_compatible(ffn, attn)
                    plan = build_afd_graph_capture_plan(handshake)

                    self.assertEqual(
                        [
                            (
                                variant.attn_batch_size,
                                variant.attn_wire_capacity,
                                variant.child_compute_capacities,
                            )
                            for variant in plan.attn_graph_variants
                        ],
                        [
                            (8, wire_capacities[1], (compute_capacities[1],)),
                            (4, wire_capacities[0], (compute_capacities[0],)),
                        ],
                    )
                    self.assertEqual(
                        plan.ffn_graph_variants,
                        (
                            (wire_capacities[1], compute_capacities[1]),
                            (wire_capacities[0], compute_capacities[0]),
                        ),
                    )

    def test_unequal_scattered_overlap_graph_uses_max_fan_in_capacity(self):
        attn = _handshake_info(
            "attn",
            num_ranks=4,
            afd_ffn_moe_backend="deepep",
            decode_token_buckets=[32, 64],
            decode_graph_token_buckets=[64],
            decode_num_microbatches=3,
        )
        ffn = _handshake_info(
            "ffn",
            num_ranks=2,
            afd_ffn_moe_backend="deepep",
            decode_graph_token_buckets=[64, 128],
        )
        plan = build_afd_graph_capture_plan(
            AFDHandshakeResult(
                local=attn,
                peer=ffn,
                dist_host="127.0.0.1",
                sync_port=1,
                dist_port=2,
            )
        )

        variant = plan.attn_graph_variants[0]
        self.assertEqual(variant.child_wire_capacities, (32, 32, 32))
        self.assertEqual(
            variant.child_compute_capacities,
            (64, 64, 64),
        )


class TestAFDDirectDeepEPScaleFormat(unittest.TestCase):
    def test_dispatch_uses_ffn_encoding_even_when_local_default_differs(self):
        for ue8m0 in (False, True):
            for role, tokens in ((AFDRole.ATTN, 1), (AFDRole.FFN, 0)):
                adapter = SimpleNamespace(
                    _max_tokens=1,
                    _role=role,
                    _use_nvfp4=False,
                    _fp8_scale_ue8m0=ue8m0,
                )
                with (
                    self.subTest(ue8m0=ue8m0, role=role),
                    patch(
                        "sglang.srt.layers.deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0",
                        not ue8m0,
                    ),
                    patch(
                        "sglang.kernels.ops.quantization.fp8_kernel.sglang_per_token_group_quant_fp8",
                        side_effect=RuntimeError("quantization boundary"),
                    ) as quantize,
                ):
                    # Stop before the GPU-only dispatch; inspect its quantizer inputs.
                    with self.assertRaisesRegex(RuntimeError, "quantization boundary"):
                        AFDDirectDeepEPAdapter.dispatch(
                            adapter,
                            0,
                            torch.zeros(tokens, 128),
                            None,
                            None,
                            num_max_dispatch_tokens_per_rank=1,
                        )
                    self.assertEqual(
                        quantize.call_args.kwargs,
                        dict(
                            column_major_scales=ue8m0,
                            scale_tma_aligned=ue8m0,
                            scale_ue8m0=ue8m0,
                        ),
                    )


class TestAFDDirectUnion(unittest.TestCase):
    def test_direct_union_uses_common_bucket_for_full_and_partial_graph(self):
        peer = _handshake_info(
            "attn",
            num_ranks=2,
            backend="direct_deepep",
            decode_token_buckets=[16, 32, 64],
            decode_graph_token_buckets=[64],
            decode_num_microbatches=2,
        )
        handshake = SimpleNamespace(peer=peer)
        session = SimpleNamespace(
            handshake=handshake,
            plan=SimpleNamespace(
                attn_graph_variants=(
                    AFDGraphCaptureVariant(128, 128, (64, 64), (None, None)),
                    AFDGraphCaptureVariant(64, 64, (32, 32), (None, None)),
                )
            ),
        )
        proposals = [
            AFDDirectProposal(3, (24, 32), int(ForwardMode.DECODE), 32, True, (16, 16)),
            AFDDirectProposal(
                5, (40, 48), int(ForwardMode.TARGET_VERIFY), 64, True, (32, 32)
            ),
        ]

        plan, msg = _build_direct_union_action(session, proposals, [0, 1], 7)
        self.assertTrue(plan.communication_graph)
        self.assertEqual((plan.parent_wire_capacity, msg.wire_capacity), (64, 64))
        self.assertEqual(plan.child_wire_capacities, (32, 32))
        self.assertEqual(msg.tokens_per_rank, [24, 32, 40, 48])

        peer.disable_cuda_graph_padding = True
        eager_plan, _ = _build_direct_union_action(session, proposals, [0, 1], 8)
        self.assertFalse(eager_plan.communication_graph)
        peer.disable_cuda_graph_padding = False

        partial_plan, partial_msg = _build_direct_union_action(
            session, proposals, [0], 9
        )
        self.assertTrue(partial_plan.communication_graph)
        self.assertEqual(partial_plan.child_wire_capacities, (32, 32))
        self.assertEqual(partial_msg.tokens_per_rank, [24, 32, 0, 0])

        prefill_proposals = [
            AFDDirectProposal(7, (9, 7), int(ForwardMode.MIXED), 9),
            proposals[0],
            AFDDirectProposal(11, (3, 5), int(ForwardMode.EXTEND), 5),
        ]
        prefill_sessions = _select_direct_union_sessions(prefill_proposals)
        self.assertEqual(prefill_sessions, [0, 2])

        prefill_plan, prefill_msg = _build_direct_union_action(
            session,
            prefill_proposals,
            prefill_sessions,
            10,
        )
        self.assertEqual(prefill_plan.active_mask, (True, False, True))
        self.assertEqual(ForwardMode(prefill_plan.forward_mode), ForwardMode.EXTEND)
        self.assertEqual(prefill_msg.wire_capacity, 9)
        self.assertEqual(prefill_msg.tokens_per_rank, [9, 7, 0, 0, 3, 5])


class TestAFDHybridState(CustomTestCase):
    def setUp(self):
        override = get_context().override_server_args(
            afd_role="attn", afd_decode_num_microbatches=2
        )
        override.install()
        self.addCleanup(override.restore)

    def test_microbatches_keep_request_tracking_aligned(self):
        batch = ForwardBatch(
            forward_mode=ForwardMode.DECODE,
            batch_size=5,
            input_ids=torch.arange(5),
            req_pool_indices=torch.tensor([7, 2, 8, 4, 6]),
            seq_lens=torch.arange(10, 15),
            seq_lens_sum=60,
            seq_lens_cpu=torch.arange(10, 15),
            out_cache_loc=torch.arange(5),
            positions=torch.arange(5),
            mamba_track_indices=torch.arange(20, 25),
            mamba_track_mask=torch.tensor([True, False, True, False, True]),
            mamba_track_seqlens=torch.arange(30, 35),
            mamba_prefill_track_mask_cpu=[True, False, True, False, True],
            mamba_track_seqlens_cpu=list(range(30, 35)),
            global_num_tokens_cpu=[5],
            global_num_tokens_gpu=torch.tensor([5]),
            global_num_token_non_padded_cpu=5,
        )
        prepare_afd_microbatch_forward_batch(batch, 2)
        for field in (
            "req_pool_indices",
            "mamba_track_indices",
            "mamba_track_mask",
            "mamba_track_seqlens",
        ):
            torch.testing.assert_close(
                torch.cat(
                    [getattr(child, field) for child in batch.afd_microbatch_children]
                ),
                getattr(batch, field),
            )
        second = batch.afd_microbatch_children[1]
        self.assertEqual(second.mamba_prefill_track_mask_cpu, [False, True])
        self.assertEqual(second.mamba_track_seqlens_cpu, [33, 34])

    def test_mtp_commit_preserves_scratch_across_microbatches(self):
        from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
            HybridLinearAttnBackend,
        )
        from sglang.srt.layers.attention.linear.utils import (
            build_verify_intermediate_state_indices,
            select_verify_intermediate_state_indices,
        )

        with (
            patch("sglang.srt.utils.common.get_eager_max_batch_size", return_value=3),
            patch(
                "sglang.srt.utils.common.get_cuda_graph_max_batch_size", return_value=8
            ),
        ):
            default_indices = build_verify_intermediate_state_indices(3, "cpu")
        scratch = torch.zeros(4, 2)
        for offset, rows, values in (
            (0, [8, 5], [[80, 81], [50, 51]]),
            (2, [7, 0], [[70, 71], [-1, -1]]),
            (4, [0, 0], [[-1, -1], [-1, -1]]),
        ):
            rows = torch.tensor(rows)
            indices = select_verify_intermediate_state_indices(
                default_indices,
                rows,
                rows != 0,
                pool_size=3,
                intermediate_state_offset=offset,
            )
            scratch[indices.long()] = torch.tensor(values, dtype=scratch.dtype)
        torch.testing.assert_close(
            scratch, torch.tensor([[80, 81], [50, 51], [70, 71], [-1, -1.0]])
        )

        state = torch.zeros(9)
        mapping = torch.tensor([0, 0, 0, 0, 0, 2, 0, 6, 4])
        req_pool = SimpleNamespace(
            get_mamba_indices=lambda rows: mapping[rows],
            get_speculative_mamba2_params_all_layers=lambda: state,
            mamba_pool=SimpleNamespace(),
        )
        primary = HybridLinearAttnBackend.__new__(HybridLinearAttnBackend)
        primary.linear_attn_backend = SimpleNamespace(
            req_to_token_pool=req_pool,
            _translate_mamba_indices=lambda rows: rows,
            forward_metadata=SimpleNamespace(mamba_cache_indices=torch.tensor([1])),
            accept_lens_pool=None,
        )
        primary._update_ple_state_after_mtp_verify = lambda *args: None
        wrapper = AFDMicrobatchAttentionBackend.__new__(AFDMicrobatchAttentionBackend)
        wrapper.primary = primary

        def scatter(destination, indices, steps, *args, src_indices_raw):
            source_rows = (
                torch.arange(steps.numel())
                if src_indices_raw is None
                else src_indices_raw
            )
            destination[indices] = scratch[source_rows, steps]

        with patch(
            "sglang.srt.layers.attention.hybrid_linear_attn_backend."
            "scatter_mamba_states_after_mtp_verify",
            side_effect=scatter,
        ):
            wrapper.update_mamba_state_after_mtp_verify(
                torch.tensor([1, 0, 1]),
                None,
                None,
                model=None,
                req_pool_indices=torch.tensor([8, 5, 7]),
            )
        torch.testing.assert_close(
            state, torch.tensor([0, 0, 50, 0, 81, 0, 71, 0, 0.0])
        )


if __name__ == "__main__":
    unittest.main()
