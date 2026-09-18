"""Regression tests for DFlash attention-TP scopes and mixed DP batches."""

import unittest
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.arg_groups import speculative_hook
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardMode
from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
    DecodeCudaGraphRunner,
)
from sglang.srt.speculative import dflash_worker_v2 as dflash
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")


class TestDFlashDPAttention(CustomTestCase):
    def test_dp_lm_head_resolution(self):
        for device, dp_attention, already_enabled in (
            ("cuda", True, False),
            ("npu", True, False),
            ("cuda", True, True),
            ("cuda", False, False),
            ("xpu", False, False),
        ):
            with self.subTest(device=device, dp_attention=dp_attention):
                cfg = SimpleNamespace(
                    device=device,
                    enable_dp_attention=dp_attention,
                    enable_dp_lm_head=already_enabled,
                    pp_size=1,
                    speculative_draft_model_path="draft",
                    speculative_num_steps=1,
                    speculative_eagle_topk=1,
                    speculative_dflash_block_size=None,
                    speculative_num_draft_tokens=7,
                    speculative_draft_window_size=None,
                    max_running_requests=48,
                )
                with (
                    patch.object(speculative_hook, "resolving_view", return_value=cfg),
                    patch.object(speculative_hook, "declare_resolution") as resolve,
                    patch.object(
                        speculative_hook, "_resolve_dflash_draft_attention_backend"
                    ),
                ):
                    speculative_hook._handle_dflash(object())
                if dp_attention and not already_enabled:
                    self.assertEqual(resolve.call_count, 1)
                    self.assertEqual(
                        resolve.call_args.kwargs, {"enable_dp_lm_head": True}
                    )
                else:
                    resolve.assert_not_called()

    def test_draft_scope_restores_target_state(self):
        initial_group, runtime_group, target_group = object(), object(), object()
        state = SimpleNamespace(group=target_group, enabled=True)

        @contextmanager
        def tp_scope(group):
            self.assertIs(state.group, target_group, "nested TP patch")
            state.group = group
            try:
                yield
            finally:
                state.group = target_group

        @contextmanager
        def dp_scope(*, enabled):
            old = state.enabled
            state.enabled = enabled
            try:
                yield
            finally:
                state.enabled = old

        worker = SimpleNamespace(
            enable_dp_attention=True,
            _draft_tp_group=initial_group,
            draft_model_runner=SimpleNamespace(tp_group=runtime_group),
        )
        flags = SimpleNamespace(dp=SimpleNamespace(override=dp_scope))
        with (
            patch.object(dflash, "draft_tp_context", side_effect=tp_scope),
            patch.object(dflash, "get_flags", return_value=flags),
        ):
            for initializing in (False, True):
                with self.subTest(initializing=initializing):
                    with self.assertRaisesRegex(RuntimeError, "draft failure"):
                        with dflash.DFlashWorkerV2._draft_context(
                            worker, initializing=initializing
                        ):
                            self.assertIs(
                                state.group,
                                initial_group if initializing else runtime_group,
                            )
                            self.assertFalse(state.enabled)
                            raise RuntimeError("draft failure")
                    self.assertIs(state.group, target_group)
                    self.assertTrue(state.enabled)
            worker.enable_dp_attention = False
            with dflash.DFlashWorkerV2._draft_context(worker):
                self.assertIs(state.group, target_group)
                self.assertTrue(state.enabled)

    def test_non_extend_rank_skips_prompt_kv(self):
        for mode in (ForwardMode.IDLE, ForwardMode.DECODE):
            with self.subTest(mode=mode):
                batch = SimpleNamespace(
                    forward_mode=mode,
                    is_extend_in_batch=True,
                    seq_lens=torch.tensor([10]) if mode.is_decode() else torch.empty(0),
                )
                output = SimpleNamespace(
                    logits_output=SimpleNamespace(hidden_states=torch.ones(4, 2)),
                    next_token_ids=torch.tensor([2])
                    if mode.is_decode()
                    else torch.empty(0, dtype=torch.long),
                )
                worker = SimpleNamespace(
                    device="cpu",
                    _validate_phase1_sampling_support=Mock(),
                    target_worker=SimpleNamespace(
                        forward_batch_generation=Mock(return_value=output)
                    ),
                    _tp_sync=SimpleNamespace(sync=Mock()),
                    _make_next_draft_input_prefill=Mock(return_value="next-decode"),
                    _append_target_hidden_to_draft_kv_by_loc=Mock(),
                )
                publish = Mock()
                result = dflash.DFlashWorkerV2.forward_batch_generation(
                    worker, batch, on_publish=publish
                )
                self.assertIs(result, output)
                self.assertIsNone(result.logits_output.hidden_states)
                worker._append_target_hidden_to_draft_kv_by_loc.assert_not_called()
                publish.assert_called_once_with(batch.seq_lens)
                if mode.is_decode():
                    self.assertEqual(result.next_draft_input, "next-decode")
                    worker._make_next_draft_input_prefill.assert_called_once()
                else:
                    worker._make_next_draft_input_prefill.assert_not_called()

    def test_idle_rank_joins_target_verify(self):
        for dp_attention in (False, True):
            with self.subTest(dp_attention=dp_attention):
                batch = SimpleNamespace(
                    forward_mode=ForwardMode.IDLE,
                    is_extend_in_batch=False,
                    spec_info=None,
                )
                target = SimpleNamespace(forward_batch_generation=Mock())
                next_input = SimpleNamespace(
                    new_seq_lens=torch.empty(0, dtype=torch.long)
                )
                worker = SimpleNamespace(
                    device="cpu",
                    block_size=7,
                    _validate_phase1_sampling_support=Mock(),
                    _target_worker=target,
                    _make_next_draft_input_decode=Mock(return_value=next_input),
                )
                forward_batch = SimpleNamespace(can_run_decode_cuda_graph=True)
                with (
                    patch.object(
                        dflash,
                        "get_parallel",
                        return_value=SimpleNamespace(enable_dp_attention=dp_attention),
                    ),
                    patch.object(dflash, "DFlashVerifyInput") as verify,
                ):
                    verify.return_value.prepare_for_verify.return_value = (
                        forward_batch,
                        None,
                    )
                    result = dflash.DFlashWorkerV2.forward_batch_generation(
                        worker, batch
                    )
                    if dp_attention:
                        self.assertEqual(
                            verify.call_args.kwargs["capture_hidden_mode"],
                            CaptureHiddenMode.FULL,
                        )
                        self.assertEqual(verify.call_args.kwargs["draft_token_num"], 7)
                        self.assertFalse(forward_batch.can_run_decode_cuda_graph)
                        target.forward_batch_generation.assert_called_once_with(
                            batch=None,
                            forward_batch=forward_batch,
                            is_verify=True,
                            skip_attn_backend_init=True,
                        )
                    else:
                        target.forward_batch_generation.assert_not_called()
                self.assertEqual(result.next_token_ids.numel(), 0)
                self.assertFalse(result.can_run_cuda_graph)

    def test_padded_hidden_rows_are_not_materialized(self):
        worker = SimpleNamespace(
            model_runner=SimpleNamespace(device=torch.device("cpu")),
            _draft_context=nullcontext,
            draft_model=SimpleNamespace(project_target_hidden=lambda x: x),
            _use_fused_kv_materialize=False,
            _append_target_hidden_sequential=Mock(),
        )
        hidden = torch.arange(10, dtype=torch.float32).reshape(5, 2)
        cache_loc = torch.arange(3)
        positions = torch.arange(3)
        dflash.DFlashWorkerV2._append_target_hidden_to_draft_kv_by_loc(
            worker,
            target_hidden=hidden,
            cache_loc=cache_loc,
            positions=positions,
        )
        torch.testing.assert_close(
            worker._append_target_hidden_sequential.call_args.kwargs["ctx_hidden"],
            hidden[:3],
        )
        worker._append_target_hidden_sequential.reset_mock()
        with self.assertRaisesRegex(ValueError, "cache_loc length mismatch"):
            dflash.DFlashWorkerV2._append_target_hidden_to_draft_kv_by_loc(
                worker,
                target_hidden=hidden[:2],
                cache_loc=cache_loc,
                positions=positions,
            )
        worker._append_target_hidden_sequential.assert_not_called()

    def test_target_graphs_keep_global_dp_metadata(self):
        for is_draft in (False, True):
            runner = SimpleNamespace(
                is_draft_worker=is_draft,
                spec_algorithm=SimpleNamespace(is_dflash=lambda: True),
            )
            self.assertEqual(
                DecodeCudaGraphRunner._forward_is_dp_local(runner), is_draft
            )
        other = SimpleNamespace(
            is_draft_worker=True,
            spec_algorithm=SimpleNamespace(
                is_dflash=lambda: False, is_dspark=lambda: False
            ),
        )
        self.assertFalse(DecodeCudaGraphRunner._forward_is_dp_local(other))

    def test_embedding_cache_respects_sharding_and_padding(self):
        full = torch.arange(14, dtype=torch.float32).reshape(7, 2)
        for layout in ("replicated", "replicated-padded", "attention-tp", "global-tp"):
            for rank in (0, 1):
                with self.subTest(layout=layout, rank=rank):
                    global_group = SimpleNamespace(
                        world_size=2 if layout == "global-tp" else 4,
                        device_group=object(),
                    )
                    attn_group = SimpleNamespace(world_size=2, device_group=object())
                    shard = None
                    parts = [full]
                    if layout != "replicated":
                        sharded = layout != "replicated-padded"
                        num_org_padded = 3 if sharded else 6
                        num_added_padded = 1 if sharded else 2
                        parts = []
                        for shard_rank in range(2 if sharded else 1):
                            part = torch.full(
                                (num_org_padded + num_added_padded, 2), -100.0
                            )
                            start = shard_rank * num_org_padded
                            rows = min(num_org_padded, 5 - start)
                            part[:rows] = full[start : start + rows]
                            if sharded:
                                part[num_org_padded] = full[5 + shard_rank]
                            else:
                                part[num_org_padded:] = full[5:]
                            parts.append(part)
                        shard = SimpleNamespace(
                            num_org_elements_padded=num_org_padded,
                            num_added_elements_padded=num_added_padded,
                        )
                    tp_size = 2 if layout in ("attention-tp", "global-tp") else 1
                    embedding = SimpleNamespace(
                        weight=parts[rank if tp_size > 1 else 0],
                        tp_size=tp_size,
                        use_attn_tp_group=layout == "attention-tp",
                        shard_indices=shard,
                        org_vocab_size=5,
                        num_added_embeddings=2,
                    )
                    worker = SimpleNamespace(
                        _target_worker=SimpleNamespace(
                            model_runner=SimpleNamespace(
                                model=SimpleNamespace(
                                    get_input_embeddings=lambda: embedding
                                ),
                                model_config=SimpleNamespace(vocab_size=7),
                            )
                        ),
                        ps=SimpleNamespace(tp_rank=rank),
                    )

                    def gather(outputs, local, *, group):
                        expected_group = (
                            attn_group if layout == "attention-tp" else global_group
                        )
                        self.assertIs(group, expected_group.device_group)
                        self.assertEqual(tuple(local.shape), (4, 2))
                        for out, part in zip(outputs, parts):
                            out.copy_(part)

                    with (
                        patch.object(
                            dflash,
                            "get_parallel",
                            return_value=SimpleNamespace(
                                enable_dp_attention=True, attn_tp_group=attn_group
                            ),
                        ),
                        patch.object(dflash, "get_tp_group", return_value=global_group),
                        patch.object(
                            dflash.dist, "all_gather", side_effect=gather
                        ) as all_gather,
                    ):
                        dflash.DFlashWorkerV2._cache_full_embed_weight(worker)
                        self.assertEqual(all_gather.call_count, int(tp_size > 1))
                    torch.testing.assert_close(worker._full_embed_gpu, full)


if __name__ == "__main__":
    unittest.main()
