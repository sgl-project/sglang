"""Frozen-specific adaptive state and zero-step contracts."""

import unittest
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import torch

from sglang.srt.speculative.adaptive_runtime_state import SpecRuntimeState
from sglang.srt.speculative.adaptive_spec_params import adaptive_unsupported_reason
from sglang.srt.speculative.frozen_kv_mtp_info import FrozenKVMTPVerifyInput
from sglang.srt.speculative.frozen_kv_mtp_worker_v2 import (
    FrozenKVMTPDraftWorker,
    FrozenKVMTPWorkerV2,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")

MODULE = "sglang.srt.speculative.frozen_kv_mtp_worker_v2"
EAGLE = "sglang.srt.speculative.eagle_worker_v2"
PARAMS = "sglang.srt.speculative.adaptive_spec_params"


class TestFrozenAdaptive(unittest.TestCase):
    def test_eager_states_clear_stale_draft_graph(self):
        for steps in (0, 1):
            worker = NS(speculative_num_steps=steps, cuda_graph_runner=object())
            with patch(f"{MODULE}.cuda_graph_fully_disabled", return_value=False):
                FrozenKVMTPDraftWorker._capture_cuda_graphs(worker)
            self.assertIsNone(worker.cuda_graph_runner)

    def test_fresh_backend_binding_and_workspace_restore(self):
        backend = object()
        runner = NS(init_new_workspace=False)

        def build(**kwargs):
            self.assertTrue(kwargs["init_new_workspace"])
            runner.init_new_workspace = True
            return backend

        runner._get_attention_backend = build
        worker = NS(draft_model_runner=runner)
        FrozenKVMTPDraftWorker.init_attention_backend(worker)
        self.assertIs(worker.draft_attn_backend, backend)
        self.assertIs(runner.attn_backend, backend)
        self.assertIs(runner.draft_attn_backend, backend)
        self.assertFalse(runner.init_new_workspace)

        def failed(**kwargs):
            runner.init_new_workspace = True
            raise RuntimeError("capture setup")

        runner._get_attention_backend = failed
        with self.assertRaisesRegex(RuntimeError, "capture setup"):
            FrozenKVMTPDraftWorker.init_attention_backend(worker)
        self.assertFalse(runner.init_new_workspace)
        self.assertIs(runner.attn_backend, backend)

    def make_worker(self):
        worker = object.__new__(FrozenKVMTPWorkerV2)
        worker.speculative_num_steps = -1
        worker.speculative_num_draft_tokens = 0
        worker._draft_worker = NS(
            draft_runner=NS(), _rebuild_topk1_chain_buffers=Mock()
        )
        worker._target_worker = NS(model_runner=NS())
        return worker

    def test_switch_rebinds_frozen_seed_backend_and_preserves_kv(self):
        worker = self.make_worker()
        pool = object()
        worker._draft_worker.kv_context = pool
        states = {
            step: SpecRuntimeState(
                speculative_num_steps=step,
                speculative_num_draft_tokens=step + 1,
                draft_attn_backend=object(),
                cuda_graph_runner=object() if step > 1 else None,
                target_attn_backend=object(),
                target_graph_runner=object(),
                draft_extend_attn_backend=None,
                cuda_graph_runner_for_draft_extend=None,
            )
            for step in (0, 1, 2)
        }
        with (
            patch(f"{EAGLE}.get_context", return_value=NS(override=Mock())),
            patch(f"{EAGLE}.log_info_on_rank0"),
        ):
            for step in (2, 1, 0, 1, 2):
                state = states[step]
                worker.apply_runtime_state(state)
                dw = worker._draft_worker
                self.assertEqual(worker.speculative_num_steps, step)
                self.assertEqual(dw.speculative_num_draft_tokens, step + 1)
                self.assertIs(dw.draft_runner.attn_backend, state.draft_attn_backend)
                self.assertIs(
                    dw.draft_runner.draft_attn_backend, state.draft_attn_backend
                )
                self.assertIs(dw.cuda_graph_runner, state.cuda_graph_runner)
                self.assertIs(
                    worker._target_worker.model_runner.attn_backend,
                    state.target_attn_backend,
                )
                self.assertIs(
                    worker._target_worker.model_runner.decode_cuda_graph_runner,
                    state.target_graph_runner,
                )
                self.assertIs(dw.kv_context, pool)

    def test_capture_failure_restores_all_runtime_bindings(self):
        worker = self.make_worker()
        worker.speculative_num_steps = 2
        worker.speculative_num_draft_tokens = 3
        dw = worker._draft_worker
        dw.speculative_num_steps = 2
        dw.speculative_num_draft_tokens = 3
        dw.draft_attn_backend = object()
        dw.draft_extend_attn_backend = None
        dw.draft_runner.draft_attn_backend = dw.draft_attn_backend
        dw.draft_runner.attn_backend = dw.draft_attn_backend
        dw.cuda_graph_runner = object()
        dw.cuda_graph_runner_for_draft_extend = None
        original = dict(vars(dw))
        original_runner = dict(vars(dw.draft_runner))
        spec = NS(speculative_num_steps=2, speculative_num_draft_tokens=3)
        graph = NS(cuda_graph_bs_decode=[1, 8], disable_cuda_graph=False)

        def override(reason, **kwargs):
            for key, value in kwargs.items():
                setattr(spec if key.startswith("speculative_") else graph, key, value)

        with (
            patch(f"{EAGLE}.get_spec", return_value=spec),
            patch(f"{EAGLE}.get_exec", return_value=NS(graph=graph)),
            patch(f"{EAGLE}.get_context", return_value=NS(override=override)),
            self.assertRaisesRegex(RuntimeError, "capture failed"),
        ):
            with worker._override_worker_state(0, 1, cuda_graph_bs=[]):
                self.assertTrue(graph.disable_cuda_graph)
                dw.draft_attn_backend = object()
                dw.draft_runner.attn_backend = object()
                dw.draft_runner.draft_attn_backend = object()
                dw.cuda_graph_runner = None
                raise RuntimeError("capture failed")
        self.assertEqual(vars(dw), original)
        self.assertEqual(vars(dw.draft_runner), original_runner)
        self.assertEqual(worker.speculative_num_steps, 2)
        self.assertEqual(worker.speculative_num_draft_tokens, 3)
        self.assertEqual(
            vars(spec), dict(speculative_num_steps=2, speculative_num_draft_tokens=3)
        )
        self.assertEqual(
            vars(graph), dict(cuda_graph_bs_decode=[1, 8], disable_cuda_graph=False)
        )

    def test_zero_step_uses_frozen_tag_and_one_node(self):
        worker = self.make_worker()
        worker.device = "cpu"
        worker.topk = 1
        worker._target_worker.model_runner.attn_backend = NS(
            verify_mask=None, max_context_len=8
        )
        batch = NS(
            forward_mode=NS(is_idle=lambda: False),
            seq_lens=torch.tensor([3, 5], dtype=torch.int64),
            seq_lens_sum=8,
            seq_lens_cpu=None,
            spec_info=NS(bonus_tokens=torch.tensor([11, 12])),
        )
        result = worker._build_trivial_verify_input(batch)
        self.assertIsInstance(result, FrozenKVMTPVerifyInput)
        self.assertEqual(result.spec_steps, 0)
        self.assertEqual(result.draft_token_num, 1)
        self.assertEqual(result.max_tree_depth, 1)
        self.assertEqual(result.retrieve_index.tolist(), [[0], [1]])
        self.assertTrue(torch.equal(result.draft_token, batch.spec_info.bonus_tokens))
        self.assertTrue(result.custom_mask.all())
        batch.forward_mode = NS(is_idle=lambda: True)
        self.assertIsInstance(
            worker._build_trivial_verify_input(batch), FrozenKVMTPVerifyInput
        )

    def test_frozen_configuration_gate(self):
        cfg = NS(
            speculative_algorithm="FROZEN_KV_MTP",
            device="cuda",
            pp_size=1,
            attention_backend="triton",
            prefill_attention_backend=None,
            decode_attention_backend=None,
            speculative_draft_attention_backend=None,
            speculative_eagle_topk=1,
            enable_multi_layer_eagle=False,
            enable_two_batch_overlap=False,
            enable_pdmux=False,
        )
        with (
            patch(f"{PARAMS}.resolving_view", side_effect=lambda x: x),
            patch(f"{PARAMS}.resolved_view", side_effect=lambda x: x),
            patch(f"{PARAMS}.attn_dp_enabled_of", return_value=False),
        ):
            self.assertIsNone(adaptive_unsupported_reason(cfg))
            for field, value in [
                ("device", "cpu"),
                ("pp_size", 2),
                ("attention_backend", "flashinfer"),
                ("prefill_attention_backend", "flashinfer"),
                ("speculative_eagle_topk", 2),
                ("enable_two_batch_overlap", True),
            ]:
                old = getattr(cfg, field)
                setattr(cfg, field, value)
                self.assertIsNotNone(adaptive_unsupported_reason(cfg))
                setattr(cfg, field, old)


if __name__ == "__main__":
    unittest.main()
