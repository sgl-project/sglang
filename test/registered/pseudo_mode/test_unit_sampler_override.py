"""Unit tests for :func:`install_sampler_override`."""

from __future__ import annotations

import unittest
from dataclasses import dataclass, field
from typing import Any, List, Optional

import torch

from sglang.srt.pseudo_mode.oracle import PseudoOracle
from sglang.srt.pseudo_mode.sampler_override import install_sampler_override
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2)


_VOCAB = 64
_EOS = 0
_DEVICE = torch.device("cpu")


@dataclass
class _StubForwardMode:
    def is_decode(self) -> bool:
        return True


@dataclass
class _StubForwardBatch:
    req_pool_indices: torch.Tensor
    is_prefill_only: bool = False
    forward_mode: _StubForwardMode = field(default_factory=_StubForwardMode)


@dataclass
class _RealSampleResult:
    """Mimics the ``next_token_ids`` tensor returned by ModelRunner.sample."""

    tensor: torch.Tensor


class _StubModelRunner:
    """Tiny stand-in for ``ModelRunner`` exposing only ``sample``."""

    def __init__(self, *, real_tokens: torch.Tensor) -> None:
        self._real_tokens = real_tokens
        self.calls: int = 0

    def sample(
        self, logits_output: Any, forward_batch: _StubForwardBatch
    ) -> torch.Tensor:
        self.calls += 1
        # Return a fresh clone so .copy_() does not mutate the canonical
        # source tensor between consecutive calls.
        return self._real_tokens.clone()


def _make_oracle_with_reqs(reqs: List[str]) -> PseudoOracle:
    oracle = PseudoOracle(vocab_size=_VOCAB, eos_id=_EOS)
    for idx, rid in enumerate(reqs):
        oracle.admit(req_id=rid, origin_input_ids=[1, 2, 3], max_new_tokens=4)
        oracle.register_chunk_commit(req_id=rid, chunk_size=3)
        oracle.register_req_pool_mapping(req_pool_idx=idx, req_id=rid)
    return oracle


def _make_forward_batch(
    num_reqs: int, *, is_prefill_only: bool = False
) -> _StubForwardBatch:
    return _StubForwardBatch(
        req_pool_indices=torch.arange(num_reqs, dtype=torch.int64, device=_DEVICE),
        is_prefill_only=is_prefill_only,
    )


class TestSamplerOverridePassthrough(unittest.TestCase):
    """Without installing, ``model_runner.sample`` returns the real tokens."""

    def test_default_passthrough(self) -> None:
        real = torch.tensor([5, 6], dtype=torch.int64, device=_DEVICE)
        mr = _StubModelRunner(real_tokens=real)
        fb = _make_forward_batch(num_reqs=2)
        out = mr.sample(None, fb)
        self.assertTrue(torch.equal(out, real))


class TestSamplerOverrideForcesOracle(unittest.TestCase):
    """After install, the returned tokens equal the oracle prediction."""

    def test_overrides_to_oracle_values(self) -> None:
        real = torch.tensor([0, 0], dtype=torch.int64, device=_DEVICE)
        mr = _StubModelRunner(real_tokens=real)
        oracle = _make_oracle_with_reqs(["a", "b"])
        install_sampler_override(model_runner=mr, oracle=oracle)
        fb = _make_forward_batch(num_reqs=2)
        out = mr.sample(None, fb)
        expected = oracle.predict_next_tokens_for_active_batch(
            forward_batch=fb, device=_DEVICE
        )
        self.assertTrue(torch.equal(out, expected))
        # Real sample STILL ran (this is the whole point of the override).
        self.assertEqual(mr.calls, 1)


class TestSamplerOverrideIdempotent(unittest.TestCase):
    """Second install is a no-op (does not double-wrap)."""

    def test_double_install_no_op(self) -> None:
        real = torch.tensor([0, 0], dtype=torch.int64, device=_DEVICE)
        mr = _StubModelRunner(real_tokens=real)
        oracle = _make_oracle_with_reqs(["a", "b"])
        install_sampler_override(model_runner=mr, oracle=oracle)
        first_method = mr.sample
        install_sampler_override(model_runner=mr, oracle=oracle)
        self.assertIs(mr.sample, first_method)
        fb = _make_forward_batch(num_reqs=2)
        out = mr.sample(None, fb)
        # Sampling produces a tensor of length 2; mr.calls must equal 1
        # per outer call (not 2, which would mean we wrapped twice).
        self.assertEqual(mr.calls, 1)
        self.assertEqual(out.shape, (2,))


class TestSamplerOverridePrefillOnlySkipped(unittest.TestCase):
    """Prefill-only batches are pass-through (real sample returns dummy zeros)."""

    def test_prefill_only_does_not_overwrite(self) -> None:
        real = torch.tensor([0, 0, 0], dtype=torch.int64, device=_DEVICE)
        mr = _StubModelRunner(real_tokens=real)
        oracle = _make_oracle_with_reqs(["a"])
        install_sampler_override(model_runner=mr, oracle=oracle)
        fb = _make_forward_batch(num_reqs=3, is_prefill_only=True)
        out = mr.sample(None, fb)
        # Pass-through: equal to the real (zero) tensor, NOT the oracle
        # values which would be non-zero for any in-range step.
        self.assertTrue(torch.equal(out, real))


class TestOverlapClosurePicksUpOverride(unittest.TestCase):
    """Closure capturing ``model_runner.sample`` BEFORE install still routes
    through the patched method at call time. This mirrors tp_worker.py's
    ``delay_sample_func`` pattern (closure created in one step, invoked in
    the next). The closure must read ``model_runner.sample`` lazily —
    which sglang's closure already does, because it looks up
    ``self.model_runner.sample`` each call rather than caching the bound
    method."""

    def test_delay_closure_routes_through_patched_method(self) -> None:
        real = torch.tensor([0, 0], dtype=torch.int64, device=_DEVICE)
        mr = _StubModelRunner(real_tokens=real)
        oracle = _make_oracle_with_reqs(["a", "b"])
        fb = _make_forward_batch(num_reqs=2)

        captured: List[Optional[torch.Tensor]] = [None]

        def delay_sample_func() -> None:
            # Late-bind on ``mr.sample`` exactly like the sglang closure.
            captured[0] = mr.sample(None, fb)

        # Install AFTER closure construction.
        install_sampler_override(model_runner=mr, oracle=oracle)
        delay_sample_func()
        expected = oracle.predict_next_tokens_for_active_batch(
            forward_batch=fb, device=_DEVICE
        )
        self.assertTrue(torch.equal(captured[0], expected))


if __name__ == "__main__":
    unittest.main()
