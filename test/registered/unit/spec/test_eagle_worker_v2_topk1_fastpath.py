"""Equivalence and ownership tests for EAGLE topk=1 fast paths.

For topk=1 the draft tree degenerates to a chain, so `draft_forward` skips the
cat/topk/sort/gather of the slow path and returns pre-allocated constants. These
tests check that the pre-allocated `parent_list` / `top_scores_index` match the
slow path (`organize_draft_results`) for num_steps in {1, 2, 3, 4}.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.speculative.adaptive_runtime_state import SpecRuntimeState
from sglang.srt.speculative.eagle_target_verify import (
    maybe_eagle_sample_target_verify_topk1,
)
from sglang.srt.speculative.eagle_utils import organize_draft_results
from sglang.srt.speculative.eagle_worker_common import EagleVerifyStepResult
from sglang.srt.speculative.eagle_worker_v2 import (
    EagleDraftWorker,
    EAGLEWorkerV2,
)
from sglang.srt.speculative.multi_layer_eagle_worker_v2 import (
    MultiLayerEagleWorkerV2,
)
from sglang.srt.speculative.spec_info import SpecInputType
from sglang.srt.utils import is_cuda
from sglang.test.ci.ci_register import (
    register_amd_ci,
    register_cpu_ci,
    register_cuda_ci,
)
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=20, stage="stage-b", runner_config="1-gpu-small-amd")
register_cpu_ci(est_time=20, suite="base-a-test-cpu")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _fake_server_args(**fields):
    """server_args stand-in: carries fields and the override() entry point."""
    ns = SimpleNamespace(**fields)

    def _override(source, **updates):
        for key, value in updates.items():
            setattr(ns, key, value)

    ns.override = _override
    return ns


def _make_chain_lists(num_steps: int, bs: int):
    """Build the (score, token, parents) lists a topk=1 chain produces.

    Shapes/values mirror `select_top_k_tokens` for topk=1: each step yields one
    token; the first step's parents are [-1, 0], later steps' parents are [i].
    """
    score_list, token_list, parents_list = [], [], []
    for i in range(num_steps):
        # Strictly decreasing scores, as a real chain produces (cumulative probs).
        score_list.append(torch.full((bs, 1, 1), float(num_steps - i), device=DEVICE))
        token_list.append(
            torch.arange(i * bs, (i + 1) * bs, device=DEVICE).unsqueeze(1)
        )
        if i == 0:
            parents_list.append(
                torch.tensor([-1, 0], dtype=torch.long, device=DEVICE).repeat(bs, 1)
            )
        else:
            parents_list.append(torch.full((bs, 1), i, dtype=torch.long, device=DEVICE))
    return score_list, token_list, parents_list


def _make_worker(num_steps: int, num_draft_tokens: int):
    worker = object.__new__(EagleDraftWorker)
    worker.topk = 1
    worker.device = DEVICE
    worker.speculative_num_steps = num_steps
    worker.speculative_num_draft_tokens = num_draft_tokens
    worker.server_args = _fake_server_args(
        cuda_graph_config=SimpleNamespace(decode=SimpleNamespace(max_bs=8)),
        max_running_requests=8,
    )
    return worker


def _make_backend_factory(decode_backend, draft_extend_backend, captured_kwargs=None):
    class FakeDraftBackendFactory:
        def __init__(self, *args, **kwargs):
            if captured_kwargs is not None:
                captured_kwargs.update(kwargs)

        def create_decode_backend(self):
            return decode_backend

        def create_draft_extend_backend(self):
            return draft_extend_backend

    return FakeDraftBackendFactory


def _make_target_verify_selection_case(device: torch.device):
    batch_size, num_tokens, vocab_size = 1, 2, 16
    logits = torch.zeros(
        (batch_size * num_tokens, vocab_size), device=device, dtype=torch.float32
    )
    additive_penalty = torch.zeros((batch_size, vocab_size), device=device)
    additive_penalty[:, 3] = 10.0
    sampling_info = SimpleNamespace(
        is_all_greedy=True,
        acc_additive_penalties=additive_penalty,
        acc_scaling_penalties=None,
        logit_bias=None,
    )
    retrieve_index = torch.arange(
        batch_size * num_tokens, dtype=torch.long, device=device
    ).view(batch_size, num_tokens)
    retrieve_next_token = torch.tensor([[1, -1]], dtype=torch.long, device=device)
    verify_input = SimpleNamespace(
        spec_input_type=SpecInputType.EAGLE_VERIFY,
        tree_topk=1,
        draft_token_num=num_tokens,
        max_tree_depth=num_tokens,
        draft_token=torch.tensor([0, 3], dtype=torch.long, device=device),
        retrieve_index=retrieve_index,
        retrieve_next_token=retrieve_next_token,
    )
    batch = SimpleNamespace(
        forward_mode=ForwardMode.DECODE,
        sampling_info=sampling_info,
        seq_lens=torch.tensor([32], dtype=torch.int32, device=device),
    )
    logits_output = SimpleNamespace(next_token_logits=logits)
    return verify_input, batch, logits_output


class TestEagleWorkerV2Topk1FastPath(CustomTestCase):
    def test_fast_path_matches_slow_path(self):
        bs = 3
        for num_steps in (1, 2, 3, 4):
            with self.subTest(num_steps=num_steps):
                num_draft_tokens = num_steps + 1
                worker = _make_worker(num_steps, num_draft_tokens)
                worker._rebuild_topk1_chain_buffers()

                score_list, token_list, parents_list = _make_chain_lists(num_steps, bs)
                ref_parent, ref_index, ref_tokens = organize_draft_results(
                    score_list, token_list, parents_list, num_draft_tokens
                )

                fast_parent = worker._topk1_parents_prealloc[:bs]
                fast_index = worker._topk1_score_indices_prealloc[:bs]
                fast_tokens = torch.cat(token_list, dim=1)

                self.assertEqual(fast_parent.shape, ref_parent.shape)
                self.assertEqual(fast_parent.tolist(), ref_parent.long().tolist())
                self.assertEqual(fast_index.tolist(), ref_index.long().tolist())
                self.assertEqual(fast_tokens.tolist(), ref_tokens.tolist())

                # The kernel reads these via data_ptr() as contiguous int64.
                self.assertEqual(fast_parent.dtype, torch.long)
                self.assertEqual(fast_index.dtype, torch.long)
                self.assertTrue(fast_parent.is_contiguous())
                self.assertTrue(fast_index.is_contiguous())

    def test_assert_on_inconsistent_steps_and_draft_tokens(self):
        # num_draft_tokens must equal num_steps + 1 for topk=1.
        worker = _make_worker(num_steps=3, num_draft_tokens=3)
        with self.assertRaises(AssertionError):
            worker._rebuild_topk1_chain_buffers()


class TestTargetVerifyFusionOwnership(CustomTestCase):
    @staticmethod
    def _make_verify_worker(worker_cls):
        worker = object.__new__(worker_cls)
        worker._target_worker = object()
        worker.req_to_token_pool = object()
        worker.token_to_kv_pool_allocator = object()
        worker.plan_stream = object()
        worker.plan_stream_ctx = object()
        worker.topk = 1
        worker.speculative_num_steps = 5
        worker.speculative_num_draft_tokens = 6
        worker.device = "cpu"
        return worker

    def test_target_verify_topk1_is_enabled_only_for_internal_single_layer(self):
        batch = object()
        batch_result = GenerationBatchResult()
        verify_step = EagleVerifyStepResult(batch_result, None)
        worker = self._make_verify_worker(EAGLEWorkerV2)

        with patch(
            "sglang.srt.speculative.eagle_worker_v2.run_eagle_verify",
            return_value=verify_step,
        ) as run_verify:
            worker.verify(batch)
            worker._verify_for_draft_extend(batch)

        generic_call, fused_call = run_verify.call_args_list
        self.assertFalse(generic_call.kwargs["enable_target_verify_topk1"])
        self.assertTrue(fused_call.kwargs["enable_target_verify_topk1"])

        multi_layer = self._make_verify_worker(MultiLayerEagleWorkerV2)
        with patch(
            "sglang.srt.speculative.multi_layer_eagle_worker_v2.run_eagle_verify",
            return_value=verify_step,
        ) as run_multi_layer_verify:
            multi_layer.verify(batch)
        self.assertFalse(
            run_multi_layer_verify.call_args.kwargs["enable_target_verify_topk1"]
        )


@unittest.skipUnless(
    is_cuda() and torch.cuda.is_available(), "NVIDIA CUDA is required for this test."
)
class TestTargetVerifyTopk1Selection(CustomTestCase):
    def setUp(self):
        self.device = torch.device("cuda")

    def test_applies_logits_transform_once(self):
        verify_input, batch, logits_output = _make_target_verify_selection_case(
            self.device
        )

        result = maybe_eagle_sample_target_verify_topk1(
            verify_input, batch, logits_output
        )

        self.assertIsNotNone(result)
        torch.testing.assert_close(
            result.predict,
            torch.tensor([3, 3], dtype=torch.int32, device=self.device),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            logits_output.next_token_logits[:, 3],
            torch.full((2,), 10.0, device=self.device),
            rtol=0,
            atol=0,
        )

    def test_fallbacks(self):
        cases = (
            "non_greedy",
            "input_type",
            "topk",
            "idle",
            "simulation",
            "logits_stride",
            "draft_layout",
        )
        for case in cases:
            with self.subTest(case=case):
                verify_input, batch, logits_output = _make_target_verify_selection_case(
                    self.device
                )
                simulation = 0
                if case == "non_greedy":
                    batch.sampling_info.is_all_greedy = False
                elif case == "input_type":
                    verify_input.spec_input_type = SpecInputType.FROZEN_KV_MTP_VERIFY
                elif case == "topk":
                    verify_input.tree_topk = 2
                elif case == "idle":
                    batch.forward_mode = ForwardMode.IDLE
                elif case == "simulation":
                    simulation = 1
                elif case == "logits_stride":
                    logits_output.next_token_logits = torch.empty(
                        (16, 2), device=self.device
                    ).t()
                elif case == "draft_layout":
                    verify_input.draft_token = torch.tensor(
                        [[0, -1], [3, -1]], dtype=torch.long, device=self.device
                    )[:, 0]

                with patch(
                    "sglang.srt.speculative.spec_utils.SIMULATE_ACC_LEN", simulation
                ):
                    result = maybe_eagle_sample_target_verify_topk1(
                        verify_input,
                        batch,
                        logits_output,
                    )
                self.assertIsNone(result)


class TestEagleWorkerV2BackendFallback(CustomTestCase):
    def test_missing_seed_cuda_graph_fallback(self):
        graph_result = (
            [],
            torch.zeros((1, 1), dtype=torch.long, device=DEVICE),
            torch.zeros((1, 1), dtype=torch.long, device=DEVICE),
            None,
        )
        tree_result = (
            torch.empty((0,), dtype=torch.bool, device=DEVICE),
            torch.zeros((1,), dtype=torch.long, device=DEVICE),
            torch.zeros((1, 2), dtype=torch.long, device=DEVICE),
            torch.zeros((1, 2), dtype=torch.long, device=DEVICE),
            torch.zeros((1, 2), dtype=torch.long, device=DEVICE),
            torch.zeros((2,), dtype=torch.long, device=DEVICE),
        )

        for seed_enabled, seed_present, expect_graph in (
            (True, False, False),
            (True, True, True),
            (False, False, True),
        ):
            with self.subTest(
                seed_enabled=seed_enabled,
                seed_present=seed_present,
            ):
                worker = object.__new__(EagleDraftWorker)
                worker.req_to_token_pool = None
                worker.cuda_graph_runner = SimpleNamespace(
                    execute=MagicMock(return_value=graph_result)
                )
                worker.draft_runner = SimpleNamespace(canary_manager=None)
                worker.topk = 1
                worker.speculative_num_steps = 1
                worker.speculative_num_draft_tokens = 2
                worker.device = DEVICE
                worker.tree_mask_mode = None
                worker.seed_dsa_topk_from_draft_extend = seed_enabled
                worker.index_share_for_mtp_iteration = True
                forward_batch = SimpleNamespace(forward_mode=ForwardMode.DECODE)
                worker.draft_forward = MagicMock(return_value=graph_result)
                attn_backend = SimpleNamespace(
                    get_verify_buffers_to_fill_after_draft=lambda: (None, None),
                    max_context_len=1,
                )
                worker.target_worker = SimpleNamespace(
                    model_runner=SimpleNamespace(attn_backend=attn_backend)
                )
                draft_input = SimpleNamespace(
                    bonus_tokens=torch.zeros((1,), dtype=torch.long, device=DEVICE),
                    dsa_topk_indices=(
                        torch.ones((1, 1), dtype=torch.int32, device=DEVICE)
                        if seed_present
                        else None
                    ),
                )
                batch = SimpleNamespace(
                    spec_info=draft_input,
                    forward_mode=ForwardMode.DECODE,
                    seq_lens_sum=1,
                    seq_lens=torch.ones((1,), dtype=torch.int32, device=DEVICE),
                )

                with patch(
                    "sglang.srt.speculative.eagle_worker_common.build_tree_kernel_efficient",
                    return_value=tree_result,
                ), patch(
                    "sglang.srt.speculative.eagle_worker_v2.prepare_for_draft",
                    return_value=(forward_batch, True),
                ):
                    worker.draft(batch)

                self.assertEqual(worker.cuda_graph_runner.execute.called, expect_graph)
                self.assertEqual(worker.draft_forward.called, not expect_graph)

    def test_preserves_initialized_backend_when_draft_extend_backend_is_unset(self):
        worker = object.__new__(EagleDraftWorker)
        existing_backend = object()
        decode_backend = object()
        worker.server_args = _fake_server_args()
        worker.draft_runner = SimpleNamespace(attn_backend=existing_backend)
        worker.topk = 1
        worker.speculative_num_steps = 2
        worker.seed_dsa_topk_from_draft_extend = False

        with patch(
            "sglang.srt.speculative.eagle_worker_v2.DraftBackendFactory",
            _make_backend_factory(decode_backend, None),
        ):
            worker.init_attention_backend()

        self.assertIs(worker.draft_attn_backend, decode_backend)
        self.assertIsNone(worker.draft_extend_attn_backend)
        self.assertIs(worker.draft_runner.draft_attn_backend, decode_backend)
        self.assertIs(worker.draft_runner.attn_backend, existing_backend)

    def test_uses_draft_extend_backend_when_available(self):
        worker = object.__new__(EagleDraftWorker)
        existing_backend = object()
        decode_backend = object()
        draft_extend_backend = object()
        worker.server_args = _fake_server_args()
        worker.draft_runner = SimpleNamespace(attn_backend=existing_backend)
        worker.topk = 1
        worker.speculative_num_steps = 2
        worker.seed_dsa_topk_from_draft_extend = True
        factory_kwargs = {}

        with patch(
            "sglang.srt.speculative.eagle_worker_v2.DraftBackendFactory",
            _make_backend_factory(
                decode_backend, draft_extend_backend, captured_kwargs=factory_kwargs
            ),
        ):
            worker.init_attention_backend()

        self.assertIs(worker.draft_attn_backend, decode_backend)
        self.assertIs(worker.draft_extend_attn_backend, draft_extend_backend)
        self.assertIs(worker.draft_runner.draft_attn_backend, decode_backend)
        self.assertIs(worker.draft_runner.attn_backend, draft_extend_backend)
        self.assertTrue(factory_kwargs["seed_dsa_topk_from_draft_extend"])

    def _make_adaptive_worker(self, runner_attn_backend):
        """An EAGLEWorkerV2 with a draft worker whose state-machine fields are
        filled with sentinels, sufficient to drive _override_worker_state /
        apply_runtime_state without touching the GPU."""
        draft_runner = SimpleNamespace(
            draft_attn_backend=object(),
            attn_backend=runner_attn_backend,
        )
        draft_worker = SimpleNamespace(
            speculative_num_steps=2,
            speculative_num_draft_tokens=3,
            draft_attn_backend=object(),
            draft_extend_attn_backend=object(),
            cuda_graph_runner=object(),
            cuda_graph_runner_for_draft_extend=object(),
            draft_runner=draft_runner,
            # _override_worker_state / apply_runtime_state call this hook; the
            # topk=1 buffers are exercised by the fast-path tests above.
            _rebuild_topk1_chain_buffers=lambda: None,
        )
        worker = object.__new__(EAGLEWorkerV2)
        worker._draft_worker = draft_worker
        worker._target_worker = SimpleNamespace(
            model_runner=SimpleNamespace(
                attn_backend=object(), decode_cuda_graph_runner=object()
            )
        )
        worker.speculative_num_steps = 2
        worker.speculative_num_draft_tokens = 3
        worker.server_args = _fake_server_args(
            speculative_num_steps=2,
            speculative_num_draft_tokens=3,
            cuda_graph_bs_decode=None,
            disable_cuda_graph=False,
        )
        return worker, draft_worker

    def test_override_worker_state_restores_runner_attn_backend(self):
        # build_adaptive_runtime_state runs init_attention_backend inside this
        # context for each candidate step; the runner backend it assigns must
        # not leak into the live worker.
        initial_backend = object()
        candidate_backend = object()
        worker, dw = self._make_adaptive_worker(initial_backend)

        with worker._override_worker_state(3, 4):
            dw.draft_runner.attn_backend = candidate_backend
            self.assertIs(dw.draft_runner.attn_backend, candidate_backend)

        self.assertIs(dw.draft_runner.attn_backend, initial_backend)

    def test_apply_runtime_state_updates_runner_attn_backend(self):
        # Switching to another step config must repoint the runner backend at
        # that config's draft-extend backend (read by the draft-extend forward).
        new_extend_backend = object()
        worker, dw = self._make_adaptive_worker(object())

        state = SpecRuntimeState(
            speculative_num_steps=3,
            speculative_num_draft_tokens=4,
            draft_attn_backend=object(),
            cuda_graph_runner=object(),
            target_attn_backend=object(),
            target_graph_runner=object(),
            draft_extend_attn_backend=new_extend_backend,
            cuda_graph_runner_for_draft_extend=object(),
        )
        worker.apply_runtime_state(state)

        self.assertIs(dw.draft_runner.attn_backend, new_extend_backend)

    def test_spec_v2_attn_backends_include_draft_extend_fallback(self):
        target_backend = object()
        decode_backend = object()
        fallback_backend = object()

        worker = object.__new__(EAGLEWorkerV2)
        worker._target_worker = SimpleNamespace(
            model_runner=SimpleNamespace(attn_backend=target_backend)
        )
        worker._draft_worker = SimpleNamespace(
            draft_attn_backend=decode_backend,
            draft_extend_attn_backend=None,
            draft_runner=SimpleNamespace(attn_backend=fallback_backend),
        )

        self.assertEqual(
            worker.spec_v2_attn_backends,
            (target_backend, decode_backend, fallback_backend),
        )


if __name__ == "__main__":
    unittest.main()
