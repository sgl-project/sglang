"""Stage execution must yield activations before committing accepted target KV."""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardMode,
    PPProxyTensors,
)
from sglang.srt.speculative.dflash_info import DFlashVerifyInput
from sglang.srt.speculative.dspark_components.dspark_draft import (
    DraftBlockResult,
    DraftProposal,
    make_next_draft_input,
)
from sglang.srt.speculative.dspark_components.dspark_planner import VerifyWindow
from sglang.srt.speculative.dspark_components.dspark_target_kv_inject import (
    TargetKVInjector,
)
from sglang.srt.speculative.dspark_components.dspark_verify import (
    AcceptOuts,
    TargetVerifyExecutor,
)
from sglang.srt.speculative.dspark_components.dspark_worker_v2 import (
    DSparkDecodeStep,
    DSparkWorkerV2,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

WORKER = "sglang.srt.speculative.dspark_components.dspark_worker_v2"
VERIFY = "sglang.srt.speculative.dspark_components.dspark_verify"


def make_fixture(*, final_stage, slot_offset=0):
    events = []
    prefixes = torch.tensor([5, 9])
    candidates = torch.tensor([[10, 11, 12, 13], [20, 21, 22, 23]])
    slots = torch.tensor([[7, 3, 1, 9], [8, 2, 6, 4]]) + slot_offset
    draft = make_next_draft_input(bonus_tokens=candidates[:, 0], new_seq_lens=prefixes)
    batch = SimpleNamespace(
        forward_mode=ForwardMode.DECODE,
        seq_lens=prefixes,
        seq_lens_cpu=prefixes.clone(),
        seq_lens_sum=14,
        spec_info=draft,
        req_pool_indices=torch.tensor([3, 1]) + slot_offset,
        reqs=[SimpleNamespace(rid="a"), SimpleNamespace(rid="b")],
        has_grammar=False,
        forward_iter=17,
        spec_verify_tier_num_tokens=8,
        global_num_tokens=None,
    )
    logits = (
        LogitsProcessorOutput(
            next_token_logits=torch.arange(256).float().reshape(8, 32),
            hidden_states=torch.ones(8, 3),
        )
        if final_stage
        else None
    )
    proxy = None if final_stage else PPProxyTensors({"hidden_states": slots.float()})
    output = GenerationBatchResult(
        logits_output=logits,
        can_run_cuda_graph=False,
        pp_hidden_states_proxy_tensors=proxy,
    )
    capture = SimpleNamespace(
        after_verify_forward=Mock(
            side_effect=lambda *a, **kw: events.append("capture-forward") or "frame"
        ),
        after_verify_accept=Mock(
            side_effect=lambda *a, **kw: events.append("capture-accept") or "ticket"
        ),
    )
    target = SimpleNamespace(
        training_capture=capture,
        forward_batch_generation=Mock(
            side_effect=lambda *a, **kw: events.append("forward") or output
        ),
    )
    injector = TargetKVInjector.__new__(TargetKVInjector)
    injector.inject_verify = Mock(side_effect=lambda **kw: events.append("kv-commit"))
    injector.ensure_context = Mock(side_effect=lambda *a: events.append("prefill-kv"))
    executor = TargetVerifyExecutor(
        target_worker=target,
        gamma=3,
        verify_num_draft_tokens=4,
        model_runner=None,
        kv_injector=injector,
    )
    executor._verify_backend_self_adds_seq_lens_cache = False
    acceptance = AcceptOuts(
        correct_len=torch.tensor([0, 2], dtype=torch.int32),
        bonus=torch.tensor([30, 31]),
        cap_trim_lens=torch.tensor([0, 1], dtype=torch.int32),
        commit_lens=torch.tensor([1, 3], dtype=torch.int32),
        new_seq_lens=torch.tensor([6, 12]),
        out_tokens=torch.tensor([[30, -999, -999, -999], [21, 22, 31, -999]]),
    )
    executor.accept_and_finalize = Mock(
        side_effect=lambda **kw: events.append("accept") or acceptance
    )
    worker = DSparkWorkerV2.__new__(DSparkWorkerV2)
    worker.device = "cpu"
    worker.verify_num_draft_tokens = 4
    worker._pending_prefill_step = None
    worker._target_kv_contract = object()
    worker._is_pd_prefill = False
    worker._capture_hidden_mode = CaptureHiddenMode.NULL
    worker._need_mamba_verify_commit = False
    worker._draft_is_moe = False
    worker._target_worker = target
    worker._kv_injector = injector
    worker._verify_executor = executor
    worker._observers = SimpleNamespace(
        segment=lambda *a: nullcontext(),
        observe_verify_step=Mock(side_effect=lambda **kw: events.append("observe")),
    )
    step = DSparkDecodeStep(
        batch=batch,
        draft_input=draft,
        prefix_lens=prefixes,
        verify_window=VerifyWindow(
            positions_2d=prefixes[:, None] + torch.arange(4),
            verify_cache_loc=slots.flatten(),
            verify_cache_loc_2d=slots,
        ),
        sampling_info=None,
        proposal=DraftProposal(
            draft_block_ids=candidates,
            draft_block=DraftBlockResult(
                draft_tokens=candidates[:, 1:],
                corrected_logits=None,
                greedy_mask=torch.ones(2, dtype=torch.bool),
                temperatures=torch.ones(2),
            ),
            draft_hidden=None,
        ),
        confidence=None,
        verify_token_budget=None,
        layout=None,
        run_compact=False,
        verify_ids_2d=candidates,
        grammar_tree=None,
        fold_eligible=False,
    )
    worker._pending_decode_step = step
    return worker, step, output, acceptance, events


class TestDSparkExecutionPhases(CustomTestCase):
    def setUp(self):
        mamba = patch(f"{WORKER}.prepare_mamba_track_for_verify")
        mamba.start()
        self.addCleanup(mamba.stop)

    def forward(self, worker, step, proxy=None, *, fail_prepare=False):
        original_lengths = step.batch.seq_lens_cpu

        def prepare(verify_input, batch, target):
            self.assertIs(target, worker.target_worker)
            self.assertEqual(batch.seq_lens_cpu.tolist(), [9, 13])
            self.assertEqual(batch.seq_lens_sum, 22)
            torch.testing.assert_close(
                batch.out_cache_loc, step.verify_window.verify_cache_loc
            )
            torch.testing.assert_close(
                verify_input.positions, step.verify_window.positions_2d.flatten()
            )
            if fail_prepare:
                raise RuntimeError("prepare failed")
            return SimpleNamespace(input_ids=verify_input.draft_token), None

        with patch.object(DFlashVerifyInput, "prepare_for_verify", new=prepare):
            try:
                return worker.forward_decode_stage(step, pp_proxy_tensors=proxy)
            finally:
                self.assertIs(step.batch.seq_lens_cpu, original_lengths)
                self.assertEqual(step.batch.seq_lens_sum, 14)

    def test_stage_handoff_precedes_acceptance_and_local_slot_commit(self):
        first, first_step, _, _, first_events = make_fixture(final_stage=False)
        last, last_step, _, _, last_events = make_fixture(
            final_stage=True, slot_offset=100
        )
        # Non-final stages have no logits even when sampling adjustments exist.
        first_step.sampling_info = object()
        with patch(f"{VERIFY}.apply_dflash_verify_logits_adjustments") as adjust:
            activation = self.forward(first, first_step).pp_proxy_tensors
            adjust.assert_not_called()
        self.assertEqual(first_events, ["forward", "capture-forward"])
        first._kv_injector.inject_verify.assert_not_called()
        with self.assertRaisesRegex(RuntimeError, "final stage"):
            first.accept_decode_step(first_step)
        self.forward(last, last_step, activation)
        self.assertIs(
            last.target_worker.forward_batch_generation.call_args.kwargs[
                "pp_proxy_tensors"
            ],
            activation,
        )
        acceptance = last.accept_decode_step(last_step)
        last._kv_injector.inject_verify.assert_not_called()
        results = []
        for worker, step, events in (
            (first, first_step, first_events),
            (last, last_step, last_events),
        ):
            results.append(
                worker.commit_decode_step(
                    step,
                    acceptance=acceptance,
                    on_publish=lambda lengths, events=events: events.append("publish"),
                )
            )
            args = worker._kv_injector.inject_verify.call_args.kwargs
            self.assertIs(args["batch"], step.batch)
            self.assertIs(args["verify_window"], step.verify_window)
            self.assertEqual(args["commit_lens"].tolist(), [1, 3])
            self.assertLess(events.index("capture-accept"), events.index("publish"))
            self.assertLess(events.index("publish"), events.index("kv-commit"))
            self.assertIsNone(worker._pending_decode_step)
        first._verify_executor.accept_and_finalize.assert_not_called()
        self.assertEqual(last_events.count("accept"), 1)
        self.assertNotIn("observe", first_events)
        for result in results:
            self.assertEqual(result.accept_lens.tolist(), [1, 3])
            self.assertEqual(result.block_accept_lens.tolist(), [1, 4])
            self.assertEqual(result.new_seq_lens.tolist(), [6, 12])
            self.assertEqual(result.next_draft_input.bonus_tokens.tolist(), [30, 31])
            self.assertEqual(result.training_capture, "ticket")
        torch.testing.assert_close(results[0].next_token_ids, results[1].next_token_ids)

    def test_agreed_proposal_rebuilds_verify_ids_and_grammar(self):
        worker, step, _, _, _ = make_fixture(final_stage=False)
        step.batch.has_grammar = True
        proposal = DraftProposal(
            draft_block_ids=step.proposal.draft_block_ids,
            draft_block=DraftBlockResult(
                draft_tokens=step.proposal.draft_block.draft_tokens + 7,
                corrected_logits=None,
                greedy_mask=step.proposal.draft_block.greedy_mask,
                temperatures=step.proposal.draft_block.temperatures,
            ),
            draft_hidden=None,
        )
        worker.set_decode_proposal(step, proposal)
        self.assertIs(step.proposal, proposal)
        expected = torch.cat(
            [proposal.draft_block_ids[:, :1], proposal.draft_block.draft_tokens], dim=1
        )
        torch.testing.assert_close(step.verify_ids_2d, expected)
        torch.testing.assert_close(step.grammar_tree.resolve()[2], expected)
        with self.assertRaisesRegex(RuntimeError, "unlaunched static"):
            worker.set_decode_proposal(step, proposal)

    def test_proposal_replacement_rejects_started_or_adaptive_steps(self):
        for name, value in (
            ("forward_started", True),
            ("run_compact", True),
            ("layout", object()),
            ("confidence", torch.ones(2)),
            ("fold_eligible", True),
        ):
            with self.subTest(name=name):
                worker, step, _, _, _ = make_fixture(final_stage=False)
                setattr(step, name, value)
                original = step.verify_ids_2d
                with self.assertRaisesRegex(RuntimeError, "unlaunched static"):
                    worker.set_decode_proposal(step, step.proposal)
                self.assertIs(step.verify_ids_2d, original)

    def test_logits_capture_precedes_adjustment_and_grammar_acceptance(self):
        worker, step, output, _, events = make_fixture(final_stage=True)
        step.sampling_info = object()
        raw = output.logits_output.next_token_logits.clone()
        observed = []
        worker.target_worker.training_capture.after_verify_forward.side_effect = (
            lambda batch, forward, logits, **kw: observed.append(
                logits.next_token_logits.clone()
            )
        )
        with patch(
            f"{VERIFY}.apply_dflash_verify_logits_adjustments",
            side_effect=lambda **kw: kw["next_token_logits"].add_(10),
        ) as adjust:
            self.forward(worker, step)
        adjust.assert_called_once()
        torch.testing.assert_close(observed[0], raw)
        step.batch.has_grammar = True
        barrier = object()
        mask = SimpleNamespace(apply=Mock(side_effect=lambda logits: logits.fill_(3)))
        with patch(f"{WORKER}.build_grammar_vocab_mask", return_value=mask) as build:
            acceptance = worker.accept_decode_step(step, grammar_barrier=barrier)
        self.assertIs(build.call_args.kwargs["barrier"], barrier)
        torch.testing.assert_close(
            worker._verify_executor.accept_and_finalize.call_args.kwargs[
                "target_logits"
            ],
            torch.full_like(raw, 3),
        )
        result = worker.commit_decode_step(step)
        self.assertIs(result.accept_lens, acceptance.commit_lens)
        self.assertIsNone(output.logits_output.hidden_states)
        self.assertEqual(events[-1], "observe")

    def test_stale_repeated_and_out_of_order_operations_are_rejected(self):
        worker, step, _, acceptance, _ = make_fixture(final_stage=True)
        _, foreign_step, _, _, _ = make_fixture(final_stage=True)
        with self.assertRaisesRegex(RuntimeError, "stale"):
            worker.forward_decode_stage(foreign_step)
        with self.assertRaisesRegex(RuntimeError, "target forward"):
            worker.commit_decode_step(step, acceptance=acceptance)
        for operation in (
            lambda: worker.prepare_decode_step(step.batch),
            lambda: worker.forward_prefill_stage(step.batch),
            lambda: worker.forward_batch_generation(step.batch),
        ):
            with self.assertRaisesRegex(RuntimeError, "pending step"):
                operation()
        self.forward(worker, step)
        with self.assertRaisesRegex(RuntimeError, "already launched target"):
            worker.forward_decode_stage(step)
        with self.assertRaisesRegex(RuntimeError, "final acceptance"):
            worker.commit_decode_step(step)
        worker.accept_decode_step(step)
        with self.assertRaisesRegex(RuntimeError, "already launched acceptance"):
            worker.accept_decode_step(step)
        _, _, _, foreign_acceptance, _ = make_fixture(final_stage=True)
        with self.assertRaisesRegex(RuntimeError, "replace local acceptance"):
            worker.commit_decode_step(step, acceptance=foreign_acceptance)
        worker.commit_decode_step(step)
        for operation in (
            worker.forward_decode_stage,
            worker.accept_decode_step,
            worker.commit_decode_step,
        ):
            with self.assertRaisesRegex(RuntimeError, "stale"):
                operation(step)

    def test_failed_forward_or_commit_prevents_buffer_reuse(self):
        for failure in ("prepare", "forward", "accept", "commit"):
            with self.subTest(failure=failure):
                worker, step, _, acceptance, _ = make_fixture(final_stage=True)
                if failure == "prepare":
                    with self.assertRaisesRegex(RuntimeError, "prepare failed"):
                        self.forward(worker, step, fail_prepare=True)
                elif failure == "forward":
                    worker.target_worker.forward_batch_generation.side_effect = (
                        RuntimeError("forward failed")
                    )
                    with self.assertRaisesRegex(RuntimeError, "forward failed"):
                        self.forward(worker, step)
                else:
                    self.forward(worker, step)
                    if failure == "accept":
                        worker._verify_executor.accept_and_finalize.side_effect = (
                            RuntimeError("accept failed")
                        )
                        with self.assertRaisesRegex(RuntimeError, "accept failed"):
                            worker.accept_decode_step(step)
                        with self.assertRaisesRegex(RuntimeError, "replace local"):
                            worker.commit_decode_step(step, acceptance=acceptance)
                    else:
                        worker.accept_decode_step(step)
                        worker._kv_injector.inject_verify.side_effect = RuntimeError(
                            "commit failed"
                        )
                        with self.assertRaisesRegex(RuntimeError, "commit failed"):
                            worker.commit_decode_step(step)
                        with self.assertRaisesRegex(RuntimeError, "committed"):
                            worker.commit_decode_step(step)
                with self.assertRaisesRegex(RuntimeError, "pending step"):
                    worker.forward_batch_generation(step.batch)
                self.assertIs(worker._pending_decode_step, step)

    def test_failed_decode_preparation_prevents_buffer_reuse(self):
        worker, step, _, _, _ = make_fixture(final_stage=True)
        worker._pending_decode_step = None
        with (
            patch(
                f"{WORKER}.torch.get_device_module",
                return_value=SimpleNamespace(current_stream=lambda: None),
            ),
            patch.object(
                torch.Tensor,
                "record_stream",
                side_effect=RuntimeError("prepare failed"),
            ),
            self.assertRaisesRegex(RuntimeError, "prepare failed"),
        ):
            worker.prepare_decode_step(step.batch)
        for operation in (
            worker.prepare_decode_step,
            worker.forward_prefill_stage,
            worker.forward_batch_generation,
        ):
            with self.assertRaisesRegex(RuntimeError, "pending step"):
                operation(step.batch)

    def test_prefill_waits_for_final_sample_and_owns_its_result(self):
        worker, step, output, _, events = make_fixture(final_stage=False)
        worker._pending_decode_step = None
        step.batch.forward_mode = ForwardMode.EXTEND
        proxy = PPProxyTensors({"hidden_states": torch.ones(2, 4)})
        result = worker.forward_prefill_stage(step.batch, pp_proxy_tensors=proxy)
        self.assertIs(result, output)
        self.assertIs(
            worker.target_worker.forward_batch_generation.call_args.kwargs[
                "pp_proxy_tensors"
            ],
            proxy,
        )
        worker._kv_injector.ensure_context.assert_not_called()
        with self.assertRaisesRegex(RuntimeError, "final stage's sample"):
            worker.commit_prefill_stage(step.batch, result)
        for batch, wrong_result in (
            (step.batch, GenerationBatchResult()),
            (SimpleNamespace(), result),
        ):
            with self.assertRaisesRegex(RuntimeError, "stale"):
                worker.commit_prefill_stage(batch, wrong_result)
        for operation in (worker.prepare_decode_step, worker.forward_prefill_stage):
            with self.assertRaisesRegex(RuntimeError, "pending step"):
                operation(step.batch)
        worker.commit_prefill_stage(
            step.batch,
            result,
            next_token_ids=torch.tensor([30, 31]),
            on_publish=lambda lengths: events.append("publish"),
        )
        self.assertEqual(events, ["forward", "publish", "prefill-kv"])
        self.assertEqual(result.next_draft_input.bonus_tokens.tolist(), [30, 31])
        self.assertEqual(result.new_seq_lens.tolist(), [5, 9])
        self.assertIsNone(worker._pending_prefill_step)
        with self.assertRaisesRegex(RuntimeError, "stale"):
            worker.commit_prefill_stage(step.batch, result)

    def test_failed_prefill_blocks_reuse_and_second_commit(self):
        for failure in ("forward", "commit"):
            with self.subTest(failure=failure):
                worker, step, output, _, _ = make_fixture(final_stage=False)
                worker._pending_decode_step = None
                if failure == "forward":
                    worker.target_worker.forward_batch_generation.side_effect = (
                        RuntimeError("prefill failed")
                    )
                    with self.assertRaisesRegex(RuntimeError, "prefill failed"):
                        worker.forward_prefill_stage(step.batch)
                else:
                    worker.forward_prefill_stage(step.batch)
                    worker._kv_injector.ensure_context.side_effect = RuntimeError(
                        "prefill commit failed"
                    )
                    with self.assertRaisesRegex(RuntimeError, "prefill commit failed"):
                        worker.commit_prefill_stage(
                            step.batch, output, next_token_ids=torch.tensor([30, 31])
                        )
                    with self.assertRaisesRegex(RuntimeError, "committed"):
                        worker.commit_prefill_stage(step.batch, output)
                with self.assertRaisesRegex(RuntimeError, "pending step"):
                    worker.forward_batch_generation(step.batch)


if __name__ == "__main__":
    unittest.main()
