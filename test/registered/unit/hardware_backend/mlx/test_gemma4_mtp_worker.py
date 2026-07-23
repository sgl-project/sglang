from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from sglang.test.ci.ci_register import register_mlx_ci

register_mlx_ci(est_time=3, suite="stage-a-unit-test-mlx")

_HAS_PROVIDER = (
    importlib.util.find_spec("mlx") is not None
    and importlib.util.find_spec("mlx_vlm.speculative.drafters.gemma4_assistant")
    is not None
)

if _HAS_PROVIDER:
    import mlx.core as mx
    import torch
    from registered.unit.hardware_backend.mlx.gemma4_test_utils import (
        build_runner,
        cache_logical_length,
        native_cache_snapshot,
        reference_tokens,
        tiny_gemma4,
        write_tiny_assistant_checkpoint,
    )

    from sglang.srt.hardware_backend.mlx.kv_cache.native_transaction import (
        clone_native_cache,
    )
    from sglang.srt.hardware_backend.mlx.model_runner import MlxModelRunner
    from sglang.srt.hardware_backend.mlx.speculative_worker import (
        MlxFrozenKVMTPDraftInput,
        MlxFrozenKVMTPWorker,
        MlxGemma4MTPProposer,
    )
    from sglang.srt.hardware_backend.mlx.tp_worker import MlxTpModelWorker
    from sglang.srt.mem_cache.allocation import assign_req_to_token_pool_func
    from sglang.srt.model_executor.forward_batch_info import ForwardMode


class _OracleProposer:
    def __init__(self, model, *, correct: bool):
        self.model = model
        self.correct = correct

    def propose_one(self, request_id, seed, cache):
        del request_id
        probe = clone_native_cache(cache)
        logits = self.model(mx.array([[seed.token_id]], dtype=mx.int32), cache=probe)
        token = int(mx.argmax(logits[:, -1, :], axis=-1).item())
        return token if self.correct else (token + 1) % self.model.args.vocab_size


@unittest.skipUnless(_HAS_PROVIDER, "requires mlx-vlm Gemma 4 assistant provider")
class TestMlxFrozenKVMTPWorker(unittest.TestCase):
    def test_target_loader_receives_pinned_revision(self):
        runner = MlxModelRunner.__new__(MlxModelRunner)
        runner.model_path = "target"
        runner.revision = "pinned-target-revision"
        runner.trust_remote_code = False
        runner._quantization = None
        fake_model = SimpleNamespace(parameters=lambda: ())
        with (
            mock.patch(
                "sglang.srt.hardware_backend.mlx.model_runner.mlx_lm_load",
                return_value=(fake_model, None, {}),
            ) as load,
            mock.patch("sglang.srt.hardware_backend.mlx.model_runner.mx.eval"),
        ):
            runner._load_model()
        self.assertEqual(load.call_args.kwargs["revision"], "pinned-target-revision")

    def test_cpu_stub_slot_assignment_uses_no_triton(self):
        req_to_token = torch.zeros((3, 8), dtype=torch.int32)
        assign_req_to_token_pool_func(
            req_pool_indices=torch.tensor([1, 2], dtype=torch.int32),
            req_to_token=req_to_token,
            start_offset=torch.tensor([2, 1], dtype=torch.int32),
            end_offset=torch.tensor([4, 4], dtype=torch.int32),
            out_cache_loc=torch.tensor([10, 11, 20, 21, 22], dtype=torch.int64),
            batch_size=2,
        )
        self.assertEqual(req_to_token[1, 2:4].tolist(), [10, 11])
        self.assertEqual(req_to_token[2, 1:4].tolist(), [20, 21, 22])

    def _make_worker(self, checkpoint: Path, *, correct: bool):
        model = tiny_gemma4()
        runner = build_runner(model)
        target_worker = MlxTpModelWorker.__new__(MlxTpModelWorker)
        target_worker._mlx_runner = runner
        target_worker._mlx_active_rids = set()
        target_worker._mlx_pool_initialized = True
        target_worker._model_runner = SimpleNamespace()
        server_args = SimpleNamespace(
            speculative_draft_model_path=str(checkpoint),
            speculative_draft_model_revision=None,
        )
        worker = MlxFrozenKVMTPWorker(
            server_args=server_args,
            gpu_id=0,
            ps=SimpleNamespace(),
            nccl_port=0,
            target_worker=target_worker,
        )
        worker._proposer = _OracleProposer(model, correct=correct)
        return model, runner, target_worker, worker

    @staticmethod
    def _request(prompt):
        return SimpleNamespace(
            rid="request",
            req_pool_idx=0,
            prefix_indices=torch.empty((0,), dtype=torch.long),
            get_fill_ids=lambda: list(prompt),
            mamba_last_track_seqlen=None,
        )

    def _prefill_batch(self, request, prompt):
        return SimpleNamespace(
            reqs=[request],
            forward_mode=ForwardMode.EXTEND,
            input_ids=torch.tensor(prompt, dtype=torch.long),
            out_cache_loc=torch.arange(len(prompt), dtype=torch.long),
            extend_lens=[len(prompt)],
            seq_lens=torch.tensor([len(prompt)], dtype=torch.int32),
            spec_info=None,
        )

    @staticmethod
    def _decode_batch(request, prefill_batch, draft_input):
        return SimpleNamespace(
            reqs=[request],
            forward_mode=ForwardMode.DECODE,
            seq_lens=prefill_batch.seq_lens.clone(),
            spec_info=draft_input,
        )

    def _run_round(self, *, correct: bool, prompt_len: int = 11):
        with tempfile.TemporaryDirectory() as temp:
            checkpoint = Path(temp)
            write_tiny_assistant_checkpoint(checkpoint)
            model, runner, target_worker, worker = self._make_worker(
                checkpoint, correct=correct
            )
            prompt = [1 + (index % 29) for index in range(prompt_len)]
            expected = reference_tokens(model, prompt, steps=3)
            request = self._request(prompt)
            prefill_batch = self._prefill_batch(request, prompt)
            prefill = worker.forward_batch_generation(prefill_batch)

            self.assertEqual(prefill.next_token_ids.tolist(), expected[:1])
            self.assertIsNone(prefill.accept_lens)
            self.assertIsInstance(prefill.next_draft_input, MlxFrozenKVMTPDraftInput)
            decode_batch = self._decode_batch(
                request, prefill_batch, prefill.next_draft_input
            )
            decode = worker.forward_batch_generation(decode_batch)

            if correct:
                self.assertEqual(decode.next_token_ids.tolist(), expected[1:3])
                self.assertEqual(decode.accept_lens.tolist(), [2])
                self.assertEqual(decode.num_correct_drafts, 1)
            else:
                self.assertEqual(decode.next_token_ids[0].item(), expected[1])
                self.assertEqual(decode.next_token_ids[1].item(), -1)
                self.assertEqual(decode.accept_lens.tolist(), [1])
                self.assertEqual(decode.num_correct_drafts, 0)
            self.assertEqual(
                decode.new_seq_lens.tolist(),
                [prompt_len + int(decode.accept_lens[0])],
            )
            self.assertEqual(decode.speculative_num_draft_tokens, 2)
            self.assertEqual(
                len(runner._req_token_ids[request.rid]),
                cache_logical_length(runner._req_caches[request.rid]) + 1,
            )
            self.assertGreater(worker.metrics.proposed_tokens, 0)
            self.assertEqual(worker.metrics.verified_tokens, 1)
            return decode

    def test_all_accept_and_all_reject_match_target_only(self):
        self._run_round(correct=True)
        self._run_round(correct=False)

    def test_worker_parity_past_rotation(self):
        self._run_round(correct=True, prompt_len=25)
        self._run_round(correct=False, prompt_len=25)

    def test_seedless_target_only_step_and_malformed_handoff(self):
        with tempfile.TemporaryDirectory() as temp:
            checkpoint = Path(temp)
            write_tiny_assistant_checkpoint(checkpoint)
            model, runner, _target_worker, worker = self._make_worker(
                checkpoint, correct=True
            )
            prompt = [1, 2, 3]
            expected = reference_tokens(model, prompt, steps=2)
            request = self._request(prompt)
            prefill_batch = self._prefill_batch(request, prompt)
            prefill = worker.forward_batch_generation(prefill_batch)
            draft_input = prefill.next_draft_input
            draft_input.draft_token_ids[0] = -1
            draft_input.valid_draft_counts[0] = 0
            decode = worker.forward_batch_generation(
                self._decode_batch(request, prefill_batch, draft_input)
            )
            self.assertEqual(decode.next_token_ids[0].item(), expected[1])
            self.assertEqual(decode.accept_lens.tolist(), [1])

            next_input = decode.next_draft_input
            next_input.draft_token_ids[0] = -1
            next_input.valid_draft_counts[0] = 1
            before = native_cache_snapshot(runner._req_caches[request.rid])
            with self.assertRaisesRegex(ValueError, "malformed"):
                worker.forward_batch_generation(
                    self._decode_batch(request, prefill_batch, next_input)
                )
            after = native_cache_snapshot(runner._req_caches[request.rid])
            self.assertEqual(
                [item["offset"] for item in before], [item["offset"] for item in after]
            )

    def test_target_exception_aborts_transaction(self):
        with tempfile.TemporaryDirectory() as temp:
            checkpoint = Path(temp)
            write_tiny_assistant_checkpoint(checkpoint)
            _model, runner, _target_worker, worker = self._make_worker(
                checkpoint, correct=True
            )
            prompt = [1, 2, 3, 4, 5, 6, 7, 8, 9]
            request = self._request(prompt)
            prefill_batch = self._prefill_batch(request, prompt)
            prefill = worker.forward_batch_generation(prefill_batch)
            before = native_cache_snapshot(runner._req_caches[request.rid])
            original_forward = runner._target_adapter.forward
            verify_calls = 0

            def fail_verify(*args, **kwargs):
                nonlocal verify_calls
                if kwargs.get("collect_hidden_states"):
                    verify_calls += 1
                    if verify_calls == 2:
                        raise RuntimeError("synthetic verify failure")
                return original_forward(*args, **kwargs)

            runner._target_adapter.forward = fail_verify
            with self.assertRaisesRegex(RuntimeError, "synthetic verify failure"):
                worker.forward_batch_generation(
                    self._decode_batch(request, prefill_batch, prefill.next_draft_input)
                )
            after = native_cache_snapshot(runner._req_caches[request.rid])
            self.assertEqual(
                [item["offset"] for item in before], [item["offset"] for item in after]
            )

    def test_lifecycle_delegation_clears_request_state_but_retains_weights(self):
        with tempfile.TemporaryDirectory() as temp:
            checkpoint = Path(temp)
            write_tiny_assistant_checkpoint(checkpoint)
            _model, runner, target_worker, worker = self._make_worker(
                checkpoint, correct=True
            )
            worker._proposer = MlxGemma4MTPProposer(worker._assistant_runtime)
            request = self._request([1, 2, 3])
            prefill = worker.forward_batch_generation(
                self._prefill_batch(request, [1, 2, 3])
            )
            fingerprint = worker._assistant_runtime.fingerprint
            load_count = worker._assistant_loader.load_count
            self.assertTrue(runner.has_request(request.rid))
            self.assertGreater(worker._assistant_runtime.request_binding_count, 0)
            active = worker.get_speculative_internal_state()
            self.assertEqual(active["implementation"], "mlx_gemma4_frozen_kv_mtp")
            self.assertGreater(active["proposed_tokens"], 0)
            self.assertEqual(active["native_request_count"], 1)
            self.assertGreater(active["assistant_request_binding_count"], 0)
            self.assertIsNone(worker.get_draft_kv_pool())

            worker.prepare_for_kv_cache_release(request)
            self.assertFalse(runner.has_request(request.rid))
            self.assertEqual(worker._assistant_runtime.request_binding_count, 0)
            self.assertEqual(worker._assistant_runtime.fingerprint, fingerprint)
            self.assertEqual(worker._assistant_loader.load_count, load_count)
            released = worker.get_speculative_internal_state()
            self.assertEqual(released["native_request_count"], 0)
            self.assertEqual(released["assistant_request_binding_count"], 0)
            self.assertEqual(released["assistant_load_count"], load_count)
            self.assertEqual(released["assistant_fingerprint"], fingerprint)

            worker.clear_cache_pool()
            self.assertEqual(runner._req_token_ids, {})
            self.assertEqual(target_worker._mlx_active_rids, set())
            success, message = worker.update_weights_from_disk(object())
            self.assertFalse(success)
            self.assertIn("assistant checkpoint", message)

    def test_draft_input_filter_and_merge_preserve_alignment(self):
        def make(rid, token):
            return MlxFrozenKVMTPDraftInput(
                request_ids=(rid,),
                draft_token_ids=torch.tensor([token]),
                valid_draft_counts=torch.tensor([1]),
                bonus_tokens=torch.tensor([token - 1]),
                new_seq_lens=torch.tensor([10]),
                accept_tokens=torch.tensor([token - 1, -1]),
                accept_lens=torch.tensor([1]),
            )

        left = make("a", 4)
        left.merge_batch(make("b", 8))
        self.assertEqual(left.request_ids, ("a", "b"))
        left.filter_batch(torch.tensor([1]), new_indices_cpu=[1])
        self.assertEqual(left.request_ids, ("b",))
        self.assertEqual(left.draft_token_ids.tolist(), [8])
        self.assertIsNone(left.hidden_states)


if __name__ == "__main__":
    unittest.main()
