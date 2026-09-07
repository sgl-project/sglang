import unittest
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.layers.attention.verify_mask import VerifyMask
from sglang.srt.model_executor.cuda_graph_buffer_registry import (
    CudaGraphBufferRegistry,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.model_runner_components.layer_setup import (
    _assert_pp_mtp_compat,
)
from sglang.srt.model_executor.pool_configurator import DefaultPoolConfigurator
from sglang.srt.model_executor.runner.eager_runner import EagerRunner
from sglang.srt.speculative.eagle_info import EaglePPVerifyInputRaw
from sglang.srt.speculative.eagle_utils import TreeMaskMode
from sglang.srt.speculative.eagle_worker_v2 import (
    EagleDraftWorker,
    EAGLEWorkerV2,
    _maybe_sync_eagle_cuda_debug,
    _resolve_eagle_cuda_sync_debug_checkpoints,
    _sync_eagle_cuda_debug,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils.async_probe import maybe_sync_eagle_cuda_debug
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestEaglePPVerifyInputRaw(unittest.TestCase):
    @staticmethod
    def _raw():
        return EaglePPVerifyInputRaw(
            draft_tokens=[[10, 11, 12, 13], [20, 21, 22, 23]],
            bonus_tokens=[10, 20],
            top_scores_index=[[0, 1, 2], [0, 1, 2]],
            parent_list=[[-1, 0, 1], [-1, 0, 1]],
            accept_lens=[2, 3],
            accept_index=[[0, 1], [0, 1, 2]],
        )

    def test_tensor_dict_round_trip_preserves_fields(self):
        raw = self._raw()
        restored = EaglePPVerifyInputRaw.from_pp_outputs(raw.to_tensor_dict())
        self.assertEqual(restored.draft_tokens, raw.draft_tokens)
        self.assertEqual(restored.bonus_tokens, raw.bonus_tokens)
        self.assertEqual(restored.top_scores_index, raw.top_scores_index)
        self.assertEqual(restored.parent_list, raw.parent_list)
        self.assertEqual(restored.accept_lens, raw.accept_lens)
        self.assertEqual(restored.accept_index, raw.accept_index)

    def test_dummy_filter_and_merge_keep_rows_aligned(self):
        raw = self._raw()
        raw.filter_batch(torch.tensor([1]), new_indices_cpu=[1])
        raw.merge_batch(
            EaglePPVerifyInputRaw.build_dummy_from_bonus_tokens(
                torch.tensor([30]), num_draft=4
            )
        )
        self.assertEqual(raw.bonus_tokens, [20, 30])
        self.assertEqual(raw.draft_tokens[1], [30, 30, 30, 30])
        self.assertEqual(raw.parent_list[1], [-1, 0, 1])
        self.assertEqual(raw.accept_lens, [3, 1])
        self.assertIsNone(raw.accept_index)

    def test_filter_rejects_missing_required_field(self):
        raw = self._raw()
        raw.parent_list = None
        with self.assertRaisesRegex(RuntimeError, "requires a relayed or dummy"):
            raw.filter_batch(torch.tensor([0]))


class TestEagleCudaSyncDebug(unittest.TestCase):
    @staticmethod
    def _forward_batch(**overrides):
        values = dict(
            forward_mode=ForwardMode.DECODE,
            batch_size=1,
            input_ids=torch.tensor([1]),
            req_pool_indices=torch.tensor([0]),
            seq_lens=torch.tensor([1]),
            out_cache_loc=torch.tensor([0]),
            seq_lens_sum=1,
        )
        values.update(overrides)
        return ForwardBatch(**values)

    @staticmethod
    def _decode_batch():
        return SimpleNamespace(
            forward_mode=ForwardMode.IDLE,
            is_extend_in_batch=False,
            spec_info=None,
            seq_lens=torch.tensor([1]),
            batch_size=lambda: 1,
        )

    @staticmethod
    def _decode_output():
        return SimpleNamespace(
            next_draft_input=SimpleNamespace(bonus_tokens=torch.tensor([7])),
            new_seq_lens=torch.tensor([2]),
            accept_lens=torch.tensor([1]),
            accept_index=None,
        )

    @classmethod
    def _pp_worker(cls, *, is_last_rank, checkpoints):
        output = cls._decode_output()
        draft_worker = SimpleNamespace(
            draft_runner=SimpleNamespace(tp_group=object()),
            draft_tp_context=MagicMock(return_value=nullcontext()),
            _draft_extend_for_decode=MagicMock(),
            draft=MagicMock(
                return_value=(
                    torch.tensor([7, 8]),
                    torch.tensor([-1]),
                    torch.tensor([0]),
                )
            ),
        )
        worker = SimpleNamespace(
            _pp_enabled=True,
            _pp_is_last_rank=is_last_rank,
            _eagle_cuda_sync_debug_checkpoints=frozenset(checkpoints),
            device="cuda:1",
            speculative_algorithm=SimpleNamespace(is_standalone=lambda: False),
            speculative_num_steps=3,
            speculative_num_draft_tokens=2,
            topk=1,
            draft_worker=draft_worker,
            _build_idle_verify_input=MagicMock(
                return_value=SimpleNamespace(is_verify_input=lambda: True)
            ),
            verify=MagicMock(return_value=output),
            _prepare_pp_next_draft_batch=MagicMock(),
        )
        return worker, output

    def test_unset_resolves_empty(self):
        from sglang.srt.environ import envs

        with envs.SGLANG_EAGLE_CUDA_SYNC_DEBUG.override(""):
            self.assertEqual(
                _resolve_eagle_cuda_sync_debug_checkpoints("cuda:0"), frozenset()
            )

    @patch("sglang.srt.speculative.eagle_worker_v2.torch.cuda.current_stream")
    def test_selected_checkpoint_synchronizes_current_stream(self, current_stream):
        from sglang.srt.environ import envs

        with (
            envs.SGLANG_EAGLE_CUDA_SYNC_DEBUG.override(
                "after_draft_extend,after_draft"
            ),
            patch(
                "sglang.srt.speculative.eagle_worker_v2._is_cuda",
                True,
            ),
        ):
            selected = _resolve_eagle_cuda_sync_debug_checkpoints("cuda:1")

        self.assertEqual(selected, frozenset({"after_draft_extend", "after_draft"}))
        _sync_eagle_cuda_debug("after_draft", "cuda:1")

        current_stream.assert_called_once_with(device="cuda:1")
        current_stream.return_value.synchronize.assert_called_once_with()

    def test_internal_draft_checkpoints_are_selectable(self):
        from sglang.srt.environ import envs

        expected = frozenset(
            {
                "after_draft_prepare",
                "after_draft_metadata",
                "after_draft_forward",
                "after_draft_topk",
                "after_draft_pp_tree",
                "after_nextn_embed",
                "after_nextn_attention",
                "after_nextn_moe",
                "after_nextn_decoder",
                "after_nextn_norm",
                "after_nextn_logits",
                "after_megamoe_shared",
                "after_megamoe_topk",
                "after_megamoe_pre_dispatch",
                "after_megamoe_routed",
            }
        )
        with (
            envs.SGLANG_EAGLE_CUDA_SYNC_DEBUG.override(",".join(expected)),
            patch(
                "sglang.srt.speculative.eagle_worker_v2._is_cuda",
                True,
            ),
        ):
            selected = _resolve_eagle_cuda_sync_debug_checkpoints("cuda:1")

        self.assertEqual(selected, expected)

    def test_owning_layer_callback_defaults_to_noop(self):
        forward_batch = SimpleNamespace()

        maybe_sync_eagle_cuda_debug(forward_batch, "after_nextn_attention")

        self.assertFalse(hasattr(forward_batch, "_eagle_cuda_sync_debug_callback"))

    def test_owning_layer_callback_forwards_draft_step(self):
        callback = MagicMock()
        forward_batch = SimpleNamespace(
            _eagle_cuda_sync_debug_callback=callback,
            _eagle_cuda_sync_debug_detail="step=1",
        )

        maybe_sync_eagle_cuda_debug(forward_batch, "after_megamoe_routed")

        callback.assert_called_once_with("after_megamoe_routed", detail="step=1")

    def test_owning_layer_callback_survives_forward_batch_clone(self):
        callback = MagicMock()
        forward_batch = self._forward_batch(
            _eagle_cuda_sync_debug_callback=callback,
            _eagle_cuda_sync_debug_detail="step=1",
        )

        cloned_batch = replace(forward_batch)
        maybe_sync_eagle_cuda_debug(cloned_batch, "after_nextn_decoder")

        callback.assert_called_once_with("after_nextn_decoder", detail="step=1")

    def test_owning_layer_callback_survives_both_eager_load_paths(self):
        from sglang.srt.environ import envs

        for no_copy in (True, False):
            with self.subTest(no_copy=no_copy):
                callback = MagicMock()
                forward_batch = self._forward_batch(
                    _eagle_cuda_sync_debug_callback=callback,
                    _eagle_cuda_sync_debug_detail="step=1",
                )
                runner = SimpleNamespace(
                    _eager_registry=CudaGraphBufferRegistry(
                        device=torch.device("cpu"),
                        max_bs=1,
                        max_num_tokens=1,
                    )
                )

                with envs.SGLANG_EAGER_INPUT_NO_COPY.override(no_copy):
                    loaded_batch = EagerRunner.load_batch(runner, forward_batch)
                maybe_sync_eagle_cuda_debug(loaded_batch, "after_megamoe_pre_dispatch")

                callback.assert_called_once_with(
                    "after_megamoe_pre_dispatch", detail="step=1"
                )

    def test_owning_layer_callback_is_scoped_to_model_forward(self):
        worker = object.__new__(EagleDraftWorker)
        worker.device = "cuda:1"
        worker._eagle_cuda_sync_debug_checkpoints = frozenset({"after_nextn_attention"})
        previous_callback = MagicMock()
        forward_batch = SimpleNamespace(
            _eagle_cuda_sync_debug_callback=previous_callback,
            _eagle_cuda_sync_debug_detail="previous",
        )

        with worker._model_forward_cuda_debug(forward_batch, "step=1"):
            self.assertEqual(forward_batch._eagle_cuda_sync_debug_detail, "step=1")
            self.assertEqual(
                forward_batch._eagle_cuda_sync_debug_callback,
                worker._maybe_sync_cuda_debug,
            )

        self.assertIs(forward_batch._eagle_cuda_sync_debug_callback, previous_callback)
        self.assertEqual(forward_batch._eagle_cuda_sync_debug_detail, "previous")

    def test_owning_layer_callback_scope_restores_after_error(self):
        worker = object.__new__(EagleDraftWorker)
        worker.device = "cuda:1"
        worker._eagle_cuda_sync_debug_checkpoints = frozenset({"after_nextn_attention"})
        forward_batch = self._forward_batch()

        with self.assertRaisesRegex(RuntimeError, "model failure"):
            with worker._model_forward_cuda_debug(forward_batch, "step=1"):
                raise RuntimeError("model failure")

        self.assertIsNone(forward_batch._eagle_cuda_sync_debug_callback)
        self.assertIsNone(forward_batch._eagle_cuda_sync_debug_detail)

    @patch("sglang.srt.speculative.eagle_worker_v2._sync_eagle_cuda_debug")
    def test_maybe_sync_forwards_step_detail_only_when_selected(self, sync_debug):
        selected = frozenset({"after_draft_forward"})

        _maybe_sync_eagle_cuda_debug(
            selected, "after_draft_forward", "cuda:1", detail="step=0"
        )
        _maybe_sync_eagle_cuda_debug(
            selected, "after_draft_topk", "cuda:1", detail="step=0"
        )

        sync_debug.assert_called_once_with(
            "after_draft_forward", "cuda:1", detail="step=0"
        )

    @patch("sglang.srt.speculative.eagle_worker_v2._sync_eagle_cuda_debug")
    def test_minimal_draft_worker_defaults_cuda_sync_off(self, sync_debug):
        worker = object.__new__(EagleDraftWorker)
        worker.device = "cuda:1"

        worker._maybe_sync_cuda_debug("after_draft_forward", detail="idle_step=0")

        sync_debug.assert_not_called()

    def test_unknown_checkpoint_fails_closed(self):
        from sglang.srt.environ import envs

        with envs.SGLANG_EAGLE_CUDA_SYNC_DEBUG.override("after_typo"):
            with self.assertRaisesRegex(
                RuntimeError, "Unknown SGLANG_EAGLE_CUDA_SYNC_DEBUG"
            ):
                _resolve_eagle_cuda_sync_debug_checkpoints("cuda:0")

    def test_non_cuda_enablement_fails_closed(self):
        from sglang.srt.environ import envs

        with (
            envs.SGLANG_EAGLE_CUDA_SYNC_DEBUG.override("after_verify"),
            patch(
                "sglang.srt.speculative.eagle_worker_v2._is_cuda",
                False,
            ),
            self.assertRaisesRegex(RuntimeError, "supported only on CUDA"),
        ):
            _resolve_eagle_cuda_sync_debug_checkpoints("cpu")

    @patch("sglang.srt.speculative.eagle_worker_v2.torch.cuda.current_stream")
    def test_sync_error_identifies_checkpoint(self, current_stream):
        current_stream.return_value.synchronize.side_effect = RuntimeError(
            "illegal memory access"
        )
        with self.assertRaisesRegex(
            RuntimeError,
            "EAGLE CUDA sync debug failed: checkpoint=after_draft_extend",
        ):
            _sync_eagle_cuda_debug("after_draft_extend", "cuda:0")

    @patch("sglang.srt.speculative.eagle_worker_v2.torch.cuda.current_stream")
    def test_sync_error_identifies_draft_step(self, current_stream):
        current_stream.return_value.synchronize.side_effect = RuntimeError(
            "illegal memory access"
        )
        with self.assertRaisesRegex(
            RuntimeError,
            "checkpoint=after_draft_topk detail=step=1",
        ):
            _sync_eagle_cuda_debug("after_draft_topk", "cuda:0", detail="step=1")

    @patch("sglang.srt.speculative.eagle_worker_v2.torch.cuda.current_stream")
    def test_sync_error_identifies_owning_layer_step(self, current_stream):
        current_stream.return_value.synchronize.side_effect = RuntimeError(
            "illegal memory access"
        )
        with self.assertRaisesRegex(
            RuntimeError,
            "checkpoint=after_megamoe_pre_dispatch detail=step=1",
        ):
            _sync_eagle_cuda_debug(
                "after_megamoe_pre_dispatch", "cuda:0", detail="step=1"
            )

    @patch("sglang.srt.speculative.eagle_worker_v2._sync_eagle_cuda_debug")
    def test_pp_non_last_syncs_after_verify_before_return(self, sync_debug):
        worker, output = self._pp_worker(
            is_last_rank=False,
            checkpoints={"after_verify", "after_draft_extend", "after_draft"},
        )
        events = MagicMock()
        events.attach_mock(worker.verify, "verify")
        sync_debug.side_effect = events.sync

        result = EAGLEWorkerV2.forward_batch_generation(worker, self._decode_batch())

        self.assertIs(result, output)
        self.assertEqual(
            [str(call).split("(", maxsplit=1)[0] for call in events.mock_calls],
            ["call.verify", "call.sync"],
        )
        sync_debug.assert_called_once_with("after_verify", "cuda:1")
        worker.draft_worker._draft_extend_for_decode.assert_not_called()
        worker.draft_worker.draft.assert_not_called()

    @patch("sglang.srt.speculative.eagle_worker_v2.spec_stage_span", nullcontext)
    @patch(
        "sglang.srt.speculative.eagle_worker_v2.speculative_moe_a2a_backend_context",
        nullcontext,
    )
    @patch(
        "sglang.srt.speculative.eagle_worker_v2.speculative_moe_backend_context",
        nullcontext,
    )
    @patch("sglang.srt.speculative.eagle_worker_v2._sync_eagle_cuda_debug")
    def test_pp_last_syncs_three_boundaries_in_order(self, sync_debug):
        worker, output = self._pp_worker(
            is_last_rank=True,
            checkpoints={"after_verify", "after_draft_extend", "after_draft"},
        )
        events = MagicMock()
        events.attach_mock(worker.verify, "verify")
        events.attach_mock(worker.draft_worker._draft_extend_for_decode, "draft_extend")
        events.attach_mock(worker._prepare_pp_next_draft_batch, "prepare")
        events.attach_mock(worker.draft_worker.draft, "draft")
        sync_debug.side_effect = events.sync

        result = EAGLEWorkerV2.forward_batch_generation(worker, self._decode_batch())

        self.assertIs(result, output)
        self.assertEqual(
            [call.args[0] for call in sync_debug.call_args_list],
            ["after_verify", "after_draft_extend", "after_draft"],
        )
        self.assertEqual(
            [str(call).split("(", maxsplit=1)[0] for call in events.mock_calls],
            [
                "call.verify",
                "call.sync",
                "call.draft_extend",
                "call.sync",
                "call.prepare",
                "call.draft",
                "call.sync",
            ],
        )

    @patch("sglang.srt.speculative.eagle_worker_v2.spec_stage_span", nullcontext)
    @patch(
        "sglang.srt.speculative.eagle_worker_v2.speculative_moe_a2a_backend_context",
        nullcontext,
    )
    @patch(
        "sglang.srt.speculative.eagle_worker_v2.speculative_moe_backend_context",
        nullcontext,
    )
    @patch("sglang.srt.speculative.eagle_worker_v2._sync_eagle_cuda_debug")
    def test_empty_checkpoint_set_never_synchronizes(self, sync_debug):
        worker, output = self._pp_worker(is_last_rank=True, checkpoints=set())

        result = EAGLEWorkerV2.forward_batch_generation(worker, self._decode_batch())

        self.assertIs(result, output)
        sync_debug.assert_not_called()


class TestEaglePPVerifyRebuild(unittest.TestCase):
    @staticmethod
    def _worker(verify_mask=None):
        return SimpleNamespace(
            topk=1,
            speculative_num_steps=3,
            speculative_num_draft_tokens=4,
            tree_mask_mode=TreeMaskMode.FULL_MASK,
            target_worker=SimpleNamespace(
                model_runner=SimpleNamespace(
                    attn_backend=SimpleNamespace(
                        verify_mask=verify_mask, max_context_len=4096
                    )
                )
            ),
        )

    @staticmethod
    def _batch(raw):
        return SimpleNamespace(
            spec_info=raw,
            seq_lens=torch.tensor([10, 12], dtype=torch.int64),
            seq_lens_cpu=torch.tensor([10, 12], dtype=torch.int64),
            seq_lens_sum=22,
            input_ids=None,
        )

    def test_rebuild_strips_bonus_column_before_tree_kernel(self):
        raw = TestEaglePPVerifyInputRaw._raw()
        batch = self._batch(raw)
        arranged = torch.tensor([10, 11, 12, 13, 20, 21, 22, 23])
        kernel_result = tuple(torch.tensor([i]) for i in range(1, 6)) + (arranged,)
        with patch(
            "sglang.srt.speculative.eagle_worker_v2.build_tree_kernel_efficient",
            return_value=kernel_result,
        ) as build_tree:
            verify = EAGLEWorkerV2._build_verify_input_from_pp_raw(
                self._worker(), batch
            )
        self.assertEqual(
            build_tree.call_args.args[3].tolist(), [[11, 12, 13], [21, 22, 23]]
        )
        self.assertIs(batch.input_ids, arranged)
        self.assertEqual(verify.draft_token_num, 4)

    def test_rebuild_uses_rank_local_verify_mask(self):
        raw = TestEaglePPVerifyInputRaw._raw()
        batch = self._batch(raw)
        mask_buffer = torch.empty(128, dtype=torch.bool)
        verify_mask = VerifyMask(
            buffer=mask_buffer,
            mode=TreeMaskMode.QLEN_ONLY,
            max_bs=2,
            is_read=False,
        )
        arranged = torch.arange(8)
        kernel_result = (
            (mask_buffer,) + tuple(torch.tensor([i]) for i in range(2, 6)) + (arranged,)
        )
        with patch(
            "sglang.srt.speculative.eagle_worker_v2.build_tree_kernel_efficient",
            return_value=kernel_result,
        ) as build_tree:
            EAGLEWorkerV2._build_verify_input_from_pp_raw(
                self._worker(verify_mask), batch
            )
        self.assertEqual(build_tree.call_args.args[9], TreeMaskMode.QLEN_ONLY)
        self.assertIs(build_tree.call_args.args[10], mask_buffer)
        self.assertFalse(build_tree.call_args.kwargs["fill_prefix_mask"])

    def test_rebuild_rejects_parent_shape_mismatch(self):
        raw = TestEaglePPVerifyInputRaw._raw()
        raw.parent_list = [[-1, 0], [-1, 0]]
        with self.assertRaisesRegex(AssertionError, "topology shape mismatch"):
            EAGLEWorkerV2._build_verify_input_from_pp_raw(
                self._worker(), self._batch(raw)
            )


class TestEaglePPLastRankDraftOwnership(unittest.TestCase):
    @staticmethod
    def _target(is_last_rank):
        return SimpleNamespace(
            pp_group=SimpleNamespace(is_last_rank=is_last_rank),
            model_runner=SimpleNamespace(
                model_config=SimpleNamespace(context_len=4096),
                attn_backend=object(),
            ),
        )

    @patch(
        "sglang.srt.speculative.eagle_worker_v2.get_plan_stream",
        return_value=(object(), nullcontext()),
    )
    @patch("sglang.srt.speculative.eagle_worker_v2.get_pp_group")
    @patch("sglang.srt.speculative.eagle_worker_v2.EagleDraftWorker")
    def test_non_last_rank_does_not_construct_draft_worker(
        self, draft_worker_cls, get_pp_group, _get_plan_stream
    ):
        get_pp_group.return_value.is_last_rank = False
        target = self._target(is_last_rank=False)
        server_args = SimpleNamespace(pp_size=2)
        with (
            patch(
                "sglang.srt.speculative.eagle_worker_v2.get_parallel",
                return_value=SimpleNamespace(pp_size=2),
            ),
            patch(
                "sglang.srt.speculative.eagle_worker_v2.get_spec",
                return_value=SimpleNamespace(
                    speculative_eagle_topk=1,
                    speculative_num_steps=3,
                    speculative_num_draft_tokens=4,
                    speculative_algorithm="EAGLE",
                    speculative_adaptive=False,
                ),
            ),
            patch(
                "sglang.srt.speculative.eagle_worker_v2.get_device",
                return_value=SimpleNamespace(device="cpu"),
            ),
            patch(
                "sglang.srt.speculative.eagle_worker_v2.get_schedule",
                return_value=SimpleNamespace(page_size=1),
            ),
        ):
            worker = EAGLEWorkerV2(server_args, 0, object(), 1234, target_worker=target)
        draft_worker_cls.assert_not_called()
        self.assertIsNone(worker.draft_worker)
        self.assertEqual(
            worker.spec_v2_attn_backends, (target.model_runner.attn_backend,)
        )

    def test_pp_idle_build_does_not_call_draft_worker(self):
        draft_worker = SimpleNamespace(draft=MagicMock())
        worker = SimpleNamespace(
            _pp_enabled=True,
            draft_worker=draft_worker,
            topk=1,
            speculative_num_steps=3,
            speculative_num_draft_tokens=4,
            device="cpu",
        )
        verify_input = EAGLEWorkerV2._build_idle_verify_input(worker, SimpleNamespace())
        draft_worker.draft.assert_not_called()
        self.assertTrue(verify_input.is_verify_input())

    def test_non_pp_idle_build_runs_draft_collectives(self):
        verify_input = object()
        draft_runner = SimpleNamespace(tp_group=object())
        draft_worker = SimpleNamespace(
            draft_runner=draft_runner,
            draft_tp_context=lambda _group: nullcontext(),
            draft=MagicMock(return_value=verify_input),
        )
        worker = SimpleNamespace(
            _pp_enabled=False,
            draft_worker=draft_worker,
            speculative_algorithm=SimpleNamespace(is_standalone=lambda: False),
            target_worker=SimpleNamespace(model_config=SimpleNamespace(vocab_size=123)),
            topk=1,
            speculative_num_steps=3,
            speculative_num_draft_tokens=4,
            device="cpu",
        )
        batch = SimpleNamespace(spec_info=None)
        idle_draft_input = object()

        with (
            patch(
                "sglang.srt.speculative.eagle_worker_v2.get_draft_recurrent_hidden_state_spec",
                return_value=(64, torch.float32),
            ),
            patch(
                "sglang.srt.speculative.eagle_worker_v2.EagleDraftInput.create_idle_input",
                return_value=idle_draft_input,
            ),
            patch(
                "sglang.srt.speculative.eagle_worker_v2.speculative_moe_backend_context",
                return_value=nullcontext(),
            ),
            patch(
                "sglang.srt.speculative.eagle_worker_v2.speculative_moe_a2a_backend_context",
                return_value=nullcontext(),
            ),
            patch(
                "sglang.srt.speculative.eagle_worker_v2.spec_stage_span",
                return_value=nullcontext(),
            ),
        ):
            result = EAGLEWorkerV2._build_idle_verify_input(worker, batch)

        self.assertIs(result, verify_input)
        self.assertIs(batch.spec_info, idle_draft_input)
        draft_worker.draft.assert_called_once_with(batch)

    def test_prepare_next_draft_preserves_idle_companion_mode(self):
        next_draft_input = object()
        new_seq_lens = torch.empty((0,), dtype=torch.int64)
        batch = SimpleNamespace(
            forward_mode=ForwardMode.IDLE, spec_info=None, seq_lens=None
        )

        EAGLEWorkerV2._prepare_pp_next_draft_batch(
            batch,
            SimpleNamespace(
                next_draft_input=next_draft_input, new_seq_lens=new_seq_lens
            ),
        )

        self.assertEqual(batch.forward_mode, ForwardMode.IDLE)
        self.assertIs(batch.spec_info, next_draft_input)
        self.assertIs(batch.seq_lens, new_seq_lens)

    def test_prepare_next_draft_promotes_active_batch_to_decode(self):
        next_draft_input = object()
        new_seq_lens = torch.tensor([17], dtype=torch.int64)
        batch = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND, spec_info=None, seq_lens=None
        )

        EAGLEWorkerV2._prepare_pp_next_draft_batch(
            batch,
            SimpleNamespace(
                next_draft_input=next_draft_input, new_seq_lens=new_seq_lens
            ),
        )

        self.assertEqual(batch.forward_mode, ForwardMode.DECODE)
        self.assertIs(batch.spec_info, next_draft_input)
        self.assertIs(batch.seq_lens, new_seq_lens)

    def test_pp_draft_keeps_checkpoint_embedding_and_shares_only_head(self):
        target_model = SimpleNamespace(
            get_head=MagicMock(return_value=object()),
            get_embed_and_head=MagicMock(side_effect=AssertionError("missing embed")),
            lm_head=None,
        )
        draft_model = SimpleNamespace(
            set_head=MagicMock(),
            set_embed_and_head=MagicMock(),
            hot_token_id=None,
        )
        worker = SimpleNamespace(
            target_worker=SimpleNamespace(
                pp_group=SimpleNamespace(world_size=2),
                model_runner=SimpleNamespace(model=target_model),
            ),
            draft_runner=SimpleNamespace(model=draft_model),
            speculative_algorithm=SpeculativeAlgorithm.EAGLE,
            hot_token_id=None,
        )

        EagleDraftWorker.init_lm_head(worker)

        target_model.get_head.assert_called_once_with()
        target_model.get_embed_and_head.assert_not_called()
        draft_model.set_head.assert_called_once_with(target_model.get_head.return_value)
        draft_model.set_embed_and_head.assert_not_called()


class TestPPMTPCompatibility(unittest.TestCase):
    def test_eagle_and_dspark_allow_partitioned_mtp_target(self):
        for algorithm in (SpeculativeAlgorithm.EAGLE, SpeculativeAlgorithm.DSPARK):
            with self.subTest(algorithm=algorithm):
                _assert_pp_mtp_compat(
                    model_has_mtp_layers=True,
                    spec_algorithm=algorithm,
                    num_effective_layers=39,
                    model_num_layers=78,
                )

    def test_other_spec_algorithm_still_rejects_partitioned_mtp_target(self):
        with self.assertRaisesRegex(AssertionError, "not compatible with MTP"):
            _assert_pp_mtp_compat(
                model_has_mtp_layers=True,
                spec_algorithm=SpeculativeAlgorithm.NGRAM,
                num_effective_layers=39,
                model_num_layers=78,
            )


class TestPPDraftKVAccounting(unittest.TestCase):
    @staticmethod
    def _config(pp_rank):
        spec_algorithm = MagicMock()
        spec_algorithm.is_eagle.return_value = True
        spec_algorithm.is_standalone.return_value = False
        spec_algorithm.is_dflash_family.return_value = False
        return SimpleNamespace(
            kv_cache_dtype_str="auto",
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(architectures=["LlamaForCausalLM"]),
                context_len=4096,
            ),
            layer_info=SimpleNamespace(
                start_layer=pp_rank * 16,
                end_layer=(pp_rank + 1) * 16,
                num_effective_layers=16,
            ),
            ps=SimpleNamespace(pp_size=2),
            pp_group=SimpleNamespace(is_last_rank=pp_rank == 1),
            spec_algorithm=spec_algorithm,
            spec_aux_config=SimpleNamespace(eagle_draft_num_layers=4),
            is_draft_worker=False,
        )

    def test_only_last_target_stage_reserves_draft_kv(self):
        with (
            patch.object(
                DefaultPoolConfigurator, "_compute_cell_size", return_value=1600
            ),
            patch(
                "sglang.srt.model_executor.pool_configurator.mambaish_config",
                return_value=None,
            ),
            patch(
                "sglang.srt.model_executor.pool_configurator.get_schedule",
                return_value=SimpleNamespace(max_total_tokens=4096),
            ),
        ):
            first = DefaultPoolConfigurator(self._config(0))
            last = DefaultPoolConfigurator(self._config(1))

        self.assertEqual(first._cell_size, 1600)
        self.assertEqual(last._cell_size, 2000)


if __name__ == "__main__":
    unittest.main()
