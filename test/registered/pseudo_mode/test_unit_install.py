"""Unit tests for :func:`pseudo_mode.install.install_on_model_runner`."""

from __future__ import annotations

import unittest
from dataclasses import dataclass, field
from typing import Any, List, Optional
from unittest import mock

import torch

from sglang.srt.pseudo_mode.oracle import PseudoOracle
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2)


_VOCAB = 64
_EOS = 0
_DEVICE = torch.device("cpu")


@dataclass
class _StubPool:
    """Fake KV pool whose ``get_runners`` plug-in we control."""

    has_runners: bool = True


@dataclass
class _StubForwardMode:
    extend: bool = False
    decode: bool = True

    def is_extend(self, include_draft_extend_v2: bool = False) -> bool:
        return self.extend

    def is_decode(self) -> bool:
        return self.decode

    def is_mixed(self) -> bool:
        return False


@dataclass
class _StubForwardBatch:
    req_pool_indices: torch.Tensor
    forward_mode: _StubForwardMode = field(default_factory=_StubForwardMode)
    is_prefill_only: bool = False


@dataclass
class _StubReq:
    rid: str
    origin_input_ids: List[int]
    sampling_params: Any
    req_pool_idx: int = -1
    output_ids: List[int] = field(default_factory=list)
    fill_ids: List[int] = field(default_factory=list)
    is_retracted: bool = False
    _finished: bool = False

    def finished(self) -> bool:
        return self._finished


@dataclass
class _StubSamplingParams:
    max_new_tokens: int = 4


@dataclass
class _StubScheduleBatch:
    reqs: List[_StubReq]
    forward_mode: Optional[_StubForwardMode] = None


class _StubModelRunner:
    def __init__(self, *, has_runners: bool = True) -> None:
        self.token_to_kv_pool = _StubPool(has_runners=has_runners)
        self.calls: int = 0

    def sample(
        self, logits_output: Any, forward_batch: _StubForwardBatch
    ) -> torch.Tensor:
        self.calls += 1
        return torch.zeros(
            forward_batch.req_pool_indices.shape[0],
            dtype=torch.int64,
            device=_DEVICE,
        )


class _StubScheduler:
    def __init__(self) -> None:
        self.queued: List[_StubReq] = []
        self.processed_results: List[Any] = []

    def _add_request_to_queue(self, req: _StubReq, is_retracted: bool = False) -> None:
        if not is_retracted:
            self.queued.append(req)

    def process_batch_result(self, batch: _StubScheduleBatch, result: Any) -> None:
        self.processed_results.append((batch, result))


@dataclass
class _StubBatchPlan:
    """Tiny BatchPlan stand-in matching the fields the patch touches."""

    write_req_pool_indices: List[int]
    write_positions: List[int]
    write_req_entry_starts: List[int]
    write_req_entry_counts: List[int]
    num_write: int
    num_write_reqs: int
    expected_write_token_ids: Optional[List[int]] = None
    expected_write_positions: Optional[List[int]] = None


@dataclass
class _StubCanaryConfig:
    enabled: bool = True


def _fresh_oracle() -> PseudoOracle:
    return PseudoOracle(vocab_size=_VOCAB, eos_id=_EOS)


class TestInstallRequiresCanaryAttached(unittest.TestCase):
    """install_on_model_runner refuses to install without a canary attached."""

    def test_missing_canary_raises(self) -> None:
        from sglang.srt.pseudo_mode import install as install_mod

        mr = _StubModelRunner(has_runners=False)
        oracle = _fresh_oracle()
        with mock.patch.object(install_mod, "get_runners", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "canary"):
                install_mod.install_on_model_runner(
                    model_runner=mr, oracle=oracle, scheduler=None
                )


class TestPlanPatchFillsExpectedFields(unittest.TestCase):
    """After install, the canary plan_batch fn returns expected_* populated."""

    def test_plan_patch_populates_expected(self) -> None:
        from sglang.jit_kernel import kv_cache_canary_plan_ref as _canary_plan_ref
        from sglang.srt.kv_cache_canary import api as _canary_api
        from sglang.srt.pseudo_mode import install as install_mod

        oracle = _fresh_oracle()
        oracle.admit(req_id="r0", origin_input_ids=[10, 20, 30], max_new_tokens=4)
        oracle.register_chunk_commit(req_id="r0", chunk_size=3)
        oracle.register_req_pool_mapping(req_pool_idx=0, req_id="r0")
        oracle.commit_step(req_id="r0", output_token=42)

        # Reset patched-attr flags so re-running tests does not skip install.
        for attr in ("_pseudo_mode_plan_patched",):
            if hasattr(_canary_plan_ref, attr):
                delattr(_canary_plan_ref, attr)

        original_plan_fn = _canary_plan_ref.plan_batch_from_forward_batch
        try:
            stub_plan = _StubBatchPlan(
                write_req_pool_indices=[0],
                write_positions=[3],
                write_req_entry_starts=[0],
                write_req_entry_counts=[1],
                num_write=1,
                num_write_reqs=1,
            )

            def stub_plan_fn(*, forward_batch, config):
                return stub_plan

            _canary_plan_ref.plan_batch_from_forward_batch = stub_plan_fn
            _canary_api.plan_batch_from_forward_batch = stub_plan_fn
            install_mod._install_plan_patch(oracle=oracle)

            fb = _StubForwardBatch(
                req_pool_indices=torch.tensor([0], dtype=torch.int64)
            )
            with mock.patch(
                "sglang.srt.pseudo_mode.install.dataclasses.replace",
                wraps=lambda obj, **kw: _StubBatchPlan(
                    write_req_pool_indices=obj.write_req_pool_indices,
                    write_positions=obj.write_positions,
                    write_req_entry_starts=obj.write_req_entry_starts,
                    write_req_entry_counts=obj.write_req_entry_counts,
                    num_write=obj.num_write,
                    num_write_reqs=obj.num_write_reqs,
                    expected_write_token_ids=kw.get("expected_write_token_ids"),
                    expected_write_positions=kw.get("expected_write_positions"),
                ),
            ):
                patched = _canary_plan_ref.plan_batch_from_forward_batch
                result = patched(forward_batch=fb, config=_StubCanaryConfig())

            self.assertIsNotNone(result.expected_write_token_ids)
            self.assertIsNotNone(result.expected_write_positions)
            # decode mode, expected position = prefill_len + history - 1 = 3
            self.assertEqual(result.expected_write_positions, [3])
            # decode-step input = first committed output_token (42)
            self.assertEqual(result.expected_write_token_ids, [42])
        finally:
            _canary_plan_ref.plan_batch_from_forward_batch = original_plan_fn
            _canary_api.plan_batch_from_forward_batch = original_plan_fn
            for attr in ("_pseudo_mode_plan_patched",):
                if hasattr(_canary_plan_ref, attr):
                    delattr(_canary_plan_ref, attr)


class TestSchedulerAdmitHook(unittest.TestCase):
    """The admit hook fires on every _add_request_to_queue call."""

    def test_admit_routes_to_oracle(self) -> None:
        from sglang.srt.pseudo_mode import install as install_mod

        oracle = _fresh_oracle()
        scheduler = _StubScheduler()
        install_mod._install_scheduler_hooks(scheduler=scheduler, oracle=oracle)

        req = _StubReq(
            rid="hook-rid-0",
            origin_input_ids=[1, 2, 3],
            sampling_params=_StubSamplingParams(max_new_tokens=4),
        )
        scheduler._add_request_to_queue(req)

        self.assertTrue(oracle.has_req("hook-rid-0"))
        self.assertEqual(oracle.predict_input_token(req_id="hook-rid-0", position=0), 1)
        self.assertEqual(oracle.predict_input_token(req_id="hook-rid-0", position=2), 3)
        # Original behavior preserved: req is queued.
        self.assertIs(scheduler.queued[0], req)


class TestSchedulerCommitAndFinishHook(unittest.TestCase):
    """commit_step + finish fire from the post-step sync pass."""

    def test_commit_step_then_finish(self) -> None:
        from sglang.srt.pseudo_mode import install as install_mod

        oracle = _fresh_oracle()
        scheduler = _StubScheduler()
        install_mod._install_scheduler_hooks(scheduler=scheduler, oracle=oracle)

        req = _StubReq(
            rid="rfin",
            origin_input_ids=[7, 8, 9],
            sampling_params=_StubSamplingParams(max_new_tokens=2),
        )
        scheduler._add_request_to_queue(req)
        # Simulate prepare_for_extend: assign req_pool_idx and have
        # fill_ids reflect the consumed prompt.
        req.req_pool_idx = 5
        req.fill_ids = [7, 8, 9]
        oracle.register_req_pool_mapping(req_pool_idx=5, req_id="rfin")

        # Step 1: decode produces one output token.
        req.output_ids = [99]
        req.fill_ids = [7, 8, 9, 99]
        decode_mode = _StubForwardMode(extend=False, decode=True)
        batch = _StubScheduleBatch(reqs=[req], forward_mode=decode_mode)
        scheduler.process_batch_result(batch, result=None)
        # commit_step appended 99 → predict_input_token at position 3
        # (first decode slot) returns the committed output token.
        self.assertEqual(oracle.predict_input_token(req_id="rfin", position=3), 99)

        # Step 2: another output, then finish.
        req.output_ids = [99, 100]
        req.fill_ids = [7, 8, 9, 99, 100]
        req._finished = True
        scheduler.process_batch_result(batch, result=None)
        self.assertFalse(oracle.has_req("rfin"))


if __name__ == "__main__":
    unittest.main()
