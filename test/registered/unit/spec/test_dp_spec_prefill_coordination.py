import contextlib
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.dp_attention import DpPaddingMode
from sglang.srt.managers.scheduler_components.dp_attn import _update_gather_batch
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.speculative.dp_spec_prefill_coordination import (
    DPSpecPrefillCoordinationPlan,
)
from sglang.srt.speculative.eagle_worker_v2 import EAGLEWorkerV2
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

WORKER_MODULE = "sglang.srt.speculative.eagle_worker_v2"


def make_plan():
    return DPSpecPrefillCoordinationPlan(
        counts=[4096, 32, 0, 512, 7, 1, 19, 0],
        logprob_counts=[1, 32, 0, 2, 7, 1, 19, 0],
        prefills=torch.tensor([1, 0, 0, 1, 0, 0, 0, 0]),
        draft_width=1,
        verify_width=4,
    )


def make_batch(counts, logprobs):
    return SimpleNamespace(
        global_num_tokens=list(counts),
        global_num_tokens_for_logprob=list(logprobs),
        dp_spec_prefill_coordination_applied=False,
        is_extend_in_batch=False,
        can_run_decode_cuda_graph=True,
        can_run_dp_prefill_cuda_graph=True,
    )


class TestDPSpecPrefillCoordinationPlan(CustomTestCase):
    def setUp(self):
        self.plan = make_plan()

    def test_target_preserves_prefill_counts_and_scales_decode(self):
        expected = (
            [4096, 128, 0, 512, 28, 4, 76, 0],
            [1, 128, 0, 2, 28, 4, 76, 0],
        )
        self.assertEqual(self.plan.phase_counts("target"), expected)
        self.assertEqual(self.plan.phase_counts("draft_extend"), expected)

    def test_prefill_ranks_contribute_zero_draft_tokens(self):
        self.assertEqual(
            self.plan.phase_counts("draft"),
            ([0, 32, 0, 0, 7, 1, 19, 0],) * 2,
        )

    def test_uniform_or_idle_steps_are_not_heterogeneous(self):
        for counts, prefills in [
            ((0, 0), (False, False)),
            ((4096, 0), (True, False)),
            ((4096, 512), (True, True)),
            ((32, 8), (False, False)),
        ]:
            with self.subTest(counts=counts, prefills=prefills):
                self.assertFalse(
                    DPSpecPrefillCoordinationPlan(
                        list(counts), list(counts), torch.tensor(prefills), 1, 4
                    ).heterogeneous
                )
        self.assertTrue(self.plan.heterogeneous)

    def test_all_phases_disable_graphs_and_preserve_group_representation(self):
        for phase in ("draft", "target", "draft_extend"):
            tokens, logprobs = self.plan.phase_counts(phase)
            for rank in range(8):
                for local_only in (False, True):
                    with self.subTest(phase=phase, rank=rank, local_only=local_only):
                        batch = make_batch(
                            [self.plan.counts[rank]]
                            if local_only
                            else self.plan.counts,
                            [self.plan.logprob_counts[rank]]
                            if local_only
                            else self.plan.logprob_counts,
                        )
                        self.plan.apply(batch, phase, rank, local_only=local_only)
                        self.assertEqual(
                            batch.global_num_tokens,
                            [tokens[rank]] if local_only else tokens,
                        )
                        self.assertEqual(
                            batch.global_num_tokens_for_logprob,
                            [logprobs[rank]] if local_only else logprobs,
                        )
                        self.assertTrue(batch.dp_spec_prefill_coordination_applied)
                        self.assertTrue(batch.is_extend_in_batch)
                        self.assertFalse(batch.can_run_decode_cuda_graph)
                        self.assertFalse(batch.can_run_dp_prefill_cuda_graph)

    def test_local_draft_restores_target_counts(self):
        plan = DPSpecPrefillCoordinationPlan(
            [4096, 24], [1, 24], torch.tensor([1, 0]), 10, 32
        )
        batch = make_batch(plan.counts, plan.logprob_counts)
        plan.apply(batch, "draft", 1, local_only=True)
        self.assertEqual(batch.global_num_tokens, [240])
        plan.apply(batch, "target", 1, local_only=False)
        self.assertEqual(batch.global_num_tokens, [4096, 768])
        self.assertEqual(batch.global_num_tokens_for_logprob, [1, 768])
        plan.apply(batch, "draft_extend", 1, local_only=True)
        self.assertEqual(batch.global_num_tokens, [768])

    def test_invalid_phase_is_rejected(self):
        with self.assertRaises(ValueError):
            self.plan.phase_counts("unknown")

    def test_gather_resets_coordination_and_restores_pure_decode_flags(self):
        for local_only in (False, True):
            with self.subTest(local_only=local_only):
                batch = make_batch([128], [128])
                batch.dp_spec_prefill_coordination_applied = True
                info = SimpleNamespace(
                    num_tokens=32,
                    num_tokens_for_logprob=32,
                    global_num_tokens=[32, 7],
                    global_num_tokens_for_logprob=[32, 7],
                    tp0_info_cpu=torch.tensor([[32, 32, 1, 0], [7, 7, 1, 0]]),
                    is_extend_in_batch=False,
                    tbo_split_seq_index=None,
                    global_forward_mode=ForwardMode.DECODE,
                    can_run_decode_cuda_graph=True,
                    can_run_prefill_cuda_graph=False,
                    prefill_cuda_graph_max_prefix_len=0,
                )
                with envs.SGLANG_ENABLE_DP_SPEC_PREFILL_COORDINATION.override(True):
                    _update_gather_batch(batch, info, not local_only)
                self.assertFalse(batch.dp_spec_prefill_coordination_applied)
                self.assertEqual(
                    batch.global_num_tokens, [32] if local_only else [32, 7]
                )
                counts, logprobs, prefills = batch.dp_spec_prefill_coordination_metadata
                self.assertIs(counts, info.global_num_tokens)
                self.assertIs(logprobs, info.global_num_tokens_for_logprob)
                torch.testing.assert_close(prefills, info.tp0_info_cpu[:, 3])
                self.assertTrue(batch.can_run_decode_cuda_graph)
                self.assertFalse(batch.is_extend_in_batch)

    def test_forward_metadata_scales_only_raw_counts(self):
        for applied in (False, True):
            with self.subTest(applied=applied):
                batch = make_batch([4096, 128], [1, 128])
                batch.dp_spec_prefill_coordination_applied = applied
                forward = object.__new__(ForwardBatch)
                forward.spec_info = object()
                with patch(
                    "sglang.srt.speculative.spec_info.spec_scale_global_num_tokens",
                    return_value=([16384, 512], [4, 512]),
                ) as scale:
                    forward.init_mlp_sync_metadata(batch, "cpu")
                self.assertEqual(forward.dp_spec_prefill_coordination_applied, applied)
                self.assertEqual(scale.call_count, int(not applied))
                self.assertEqual(
                    forward.global_num_tokens_cpu,
                    [16384, 512] if not applied else [4096, 128],
                )

    def test_coordinated_padding_preserves_local_forward_modes(self):
        module = "sglang.srt.model_executor.forward_batch_info"
        for mode in (
            ForwardMode.DECODE,
            ForwardMode.TARGET_VERIFY,
            ForwardMode.DRAFT_EXTEND_V2,
            ForwardMode.IDLE,
        ):
            for hybrid in (False, True):
                with self.subTest(mode=mode, hybrid=hybrid):
                    idle = mode.is_idle()
                    width = 1 if mode.is_decode() or idle else 4
                    tokens = 0 if idle else 2 * width
                    counts = [tokens, 8] if idle else [0, tokens]
                    batch = ForwardBatch(
                        forward_mode=mode,
                        batch_size=0 if idle else 2,
                        input_ids=torch.arange(tokens),
                        req_pool_indices=torch.arange(0 if idle else 2),
                        seq_lens=torch.tensor([] if idle else [5, 6]),
                        out_cache_loc=torch.arange(tokens),
                        seq_lens_sum=0 if idle else 11,
                        is_extend_in_batch=True,
                        dp_spec_prefill_coordination_applied=True,
                        global_num_tokens_cpu=counts,
                        global_num_tokens_for_logprob_cpu=counts,
                        global_num_tokens_gpu=torch.tensor(counts),
                        spec_info=SimpleNamespace(
                            num_tokens_per_req=width,
                            is_draft_input=lambda: True,
                        ),
                    )
                    runner = MagicMock(enable_elastic_ep=False)
                    with (
                        patch(
                            f"{module}.get_parallel",
                            return_value=SimpleNamespace(attn_tp_size=1),
                        ),
                        patch(
                            f"{module}.get_exec",
                            return_value=SimpleNamespace(
                                graph=SimpleNamespace(
                                    cuda_graph_config=SimpleNamespace(
                                        prefill=SimpleNamespace(bs=[])
                                    )
                                )
                            ),
                        ),
                        patch.object(
                            DpPaddingMode,
                            "get_dp_padding_mode",
                            return_value=DpPaddingMode.MAX_LEN,
                        ),
                        patch(
                            f"{module}.dp_gather_slot", return_value=0 if idle else 1
                        ),
                        patch(f"{module}.set_dp_buffer_len_from_batch"),
                        patch(f"{module}.set_is_extend_in_batch"),
                        patch(
                            f"{module}.mambaish_config",
                            return_value=object() if hybrid else None,
                        ),
                        patch(f"{module}._is_cpu", True),
                        patch.object(ForwardBatch, "_pad_inputs_to_size"),
                        patch(
                            "sglang.srt.batch_overlap.two_batch_overlap.TboForwardBatchPreparer.prepare"
                        ),
                    ):
                        batch.prepare_mlp_sync_batch(runner)
                    self.assertEqual(batch.dp_padding_mode, DpPaddingMode.SUM_LEN)
                    self.assertEqual(batch.forward_mode, mode)
                    self.assertEqual(batch.global_num_tokens_cpu, counts)
                    self.assertEqual(batch.batch_size, 0 if idle else 2)


class TestDPSpecPrefillCoordinationWorker(CustomTestCase):
    def test_prefill_decode_and_idle_ranks_follow_the_same_phase_order(self):
        plan = make_plan()
        for rank in range(8):
            with self.subTest(rank=rank):
                events = []
                is_prefill = plan.prefills[rank]
                batch = make_batch(plan.counts, plan.logprob_counts)
                batch.forward_mode = (
                    ForwardMode.EXTEND
                    if is_prefill
                    else ForwardMode.DECODE
                    if plan.counts[rank]
                    else ForwardMode.IDLE
                )
                batch.decoding_reqs = None
                batch.req_to_token_pool = batch.token_to_kv_pool_allocator = None
                batch.tree_cache = batch.model_config = None
                batch.enable_overlap = True
                batch.spec_algorithm = SpeculativeAlgorithm.EAGLE
                batch.spec_info = (
                    None if is_prefill or not plan.counts[rank] else object()
                )
                batch.seq_lens = torch.tensor([12])
                idle = make_batch(plan.counts, plan.logprob_counts)
                idle.spec_info = None
                idle.prepare_for_idle = MagicMock()
                result = SimpleNamespace(
                    new_seq_lens=batch.seq_lens,
                    extra_keep_alive_refs=None,
                    logits_output=SimpleNamespace(
                        hidden_states=object(), mm_input_embeds=None
                    ),
                    next_token_ids=object(),
                )
                verify_input = object()

                def record(phase, current):
                    self.assertTrue(current.dp_spec_prefill_coordination_applied)
                    counts = plan.phase_counts(phase)[0]
                    if phase != "target" and worker._draft_worker.draft_owns_attention:
                        counts = [counts[rank]]
                    self.assertEqual(current.global_num_tokens, counts)
                    self.assertFalse(current.can_run_decode_cuda_graph)
                    self.assertFalse(current.can_run_dp_prefill_cuda_graph)
                    events.append(phase)

                def draft(current):
                    record("draft", current)
                    self.assertIs(current, idle if is_prefill else batch)
                    return verify_input

                def target(current, **kwargs):
                    record("target", current)
                    if not is_prefill:
                        self.assertIs(current.spec_info, verify_input)
                        self.assertEqual(kwargs["grammar_barrier"], "grammar")
                    return result

                def extend(current, *args):
                    record("draft_extend", current)
                    return object()

                def draft_context(group, *, owns_attention):
                    self.assertEqual(owns_attention, bool(rank % 2))
                    return contextlib.nullcontext()

                worker = object.__new__(EAGLEWorkerV2)
                worker.device = "cpu"
                worker.topk = 1
                worker.speculative_algorithm = SpeculativeAlgorithm.EAGLE
                worker._target_worker = SimpleNamespace(forward_batch_generation=target)
                worker._draft_worker = SimpleNamespace(
                    draft_runner=SimpleNamespace(tp_group=None),
                    draft_owns_attention=bool(rank % 2),
                    draft_tp_context=draft_context,
                    draft=draft,
                    _draft_extend_for_prefill=extend,
                    _draft_extend_for_decode=extend,
                )
                worker.verify = target
                with contextlib.ExitStack() as stack:
                    stack.enter_context(
                        patch(
                            f"{WORKER_MODULE}.get_parallel",
                            return_value=SimpleNamespace(attn_dp_rank=rank),
                        )
                    )
                    stack.enter_context(
                        patch(
                            f"{WORKER_MODULE}.ScheduleBatch.init_new", return_value=idle
                        )
                    )
                    stack.enter_context(
                        patch(
                            f"{WORKER_MODULE}.get_draft_recurrent_hidden_state_spec",
                            return_value=(16, torch.float32),
                        )
                    )
                    stack.enter_context(
                        patch(
                            f"{WORKER_MODULE}.EagleDraftInput.create_idle_input",
                            return_value=object(),
                        )
                    )
                    for name in (
                        "speculative_moe_backend_context",
                        "speculative_moe_a2a_backend_context",
                        "spec_stage_span",
                    ):
                        stack.enter_context(
                            patch(
                                f"{WORKER_MODULE}.{name}",
                                side_effect=lambda *args: contextlib.nullcontext(),
                            )
                        )
                    actual = worker._forward_dp_spec_prefill_coordination(
                        batch,
                        plan,
                        lambda value: events.append("publish"),
                        "grammar",
                        None,
                    )
                self.assertIs(actual, result)
                self.assertEqual(events, ["draft", "target", "publish", "draft_extend"])
                if is_prefill:
                    self.assertIn(idle, result.extra_keep_alive_refs)
                    idle.prepare_for_idle.assert_called_once()

    def test_local_mixed_batch_is_rejected_before_draft(self):
        worker = object.__new__(EAGLEWorkerV2)
        worker._draft_worker = SimpleNamespace(draft_owns_attention=False)
        batch = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            decoding_reqs=[object()],
            global_num_tokens=[1, 1],
        )
        with patch(
            f"{WORKER_MODULE}.get_parallel",
            return_value=SimpleNamespace(attn_dp_rank=0),
        ):
            with self.assertRaisesRegex(RuntimeError, "Local mixed"):
                worker._forward_dp_spec_prefill_coordination(
                    batch, make_plan(), None, None, None
                )

    def test_disabled_feature_retains_existing_dispatch_without_phase_metadata(self):
        worker = object.__new__(EAGLEWorkerV2)
        worker.enable_dp_spec_prefill_coordination = False
        worker._forward_prefill_batch = MagicMock(return_value=object())
        worker._forward_dp_spec_prefill_coordination = MagicMock()
        batch = SimpleNamespace(
            is_extend_in_batch=True,
            forward_mode=ForwardMode.DECODE,
        )
        result = worker.forward_batch_generation(batch)
        self.assertIs(result, worker._forward_prefill_batch.return_value)
        worker._forward_dp_spec_prefill_coordination.assert_not_called()

    def test_uniform_prefill_retains_existing_dispatch(self):
        worker = object.__new__(EAGLEWorkerV2)
        worker.enable_dp_spec_prefill_coordination = True
        worker.topk = 1
        worker.speculative_num_draft_tokens = 4
        worker._forward_prefill_batch = MagicMock(return_value=object())
        worker._forward_dp_spec_prefill_coordination = MagicMock()
        batch = SimpleNamespace(
            is_extend_in_batch=True,
            forward_mode=ForwardMode.EXTEND,
            dp_spec_prefill_coordination_metadata=(
                [4096, 0],
                [1, 0],
                torch.tensor([1, 0]),
            ),
        )
        result = worker.forward_batch_generation(batch)
        self.assertIs(result, worker._forward_prefill_batch.return_value)
        worker._forward_dp_spec_prefill_coordination.assert_not_called()


if __name__ == "__main__":
    unittest.main()
