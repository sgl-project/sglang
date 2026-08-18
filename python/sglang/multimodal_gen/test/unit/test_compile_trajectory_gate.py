# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the compile-plan promotion gate
(:mod:`sglang.multimodal_gen.runtime.utils.compile_trajectory_gate`).

Unlike a purely typed-contract test, this exercises a real
``torch.compile`` call against a toy stateful iterative module (standing in
for a denoising loop) on the local accelerator -- Apple Silicon MPS when
available, CPU otherwise -- and validates the promotion gate's pass/reject
decision against genuinely eager vs. compiled trajectories, not mocked
tensors.
"""

import pytest
import torch
import torch.nn as nn

from sglang.multimodal_gen.runtime.utils.compile_trajectory_gate import (
    CompiledPlanManifest,
    CompileGateError,
    CompileWorkloadSignature,
    TrajectoryGate,
    compute_tensor_metrics,
    run_trajectory_gate,
    select_validated_plan,
)


def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = _device()


class _ToyIterativeModel(nn.Module):
    """Stands in for a small stateful denoising step: hidden state carried
    across steps, one linear + nonlinearity per step."""

    def __init__(self, dim: int = 8):
        super().__init__()
        self.lin = nn.Linear(dim, dim)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.lin(hidden))


def _make_signature(**overrides) -> CompileWorkloadSignature:
    defaults = dict(
        model_revision="toy-rev-1",
        dtype="float32",
        backend="aot_eager",
        parallel_signature="tp1sp1",
        latent_shape_regime=(4, 8),
        num_inference_steps=4,
        cfg_mode="none",
        cache_mode="none",
        state_schema_version="v1",
    )
    defaults.update(overrides)
    return CompileWorkloadSignature(**defaults)


def _make_gate(
    checkpoints=("step_0", "terminal"), cosine_min=0.999, max_abs=1e-3
) -> TrajectoryGate:
    thresholds = {
        name: {"cosine_similarity": cosine_min, "max_abs": max_abs}
        for name in checkpoints
    }
    return TrajectoryGate(checkpoints=checkpoints, tensor_thresholds=thresholds)


def _run_toy_rollout(model: nn.Module, num_steps: int, dim: int = 8, seed: int = 0):
    """Runs a small stateful rollout and returns the per-step hidden states."""
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    hidden = torch.randn(dim, generator=gen, device=DEVICE)
    states = []
    for _ in range(num_steps):
        hidden = model(hidden)
        states.append(hidden)
    return states


class TestCompileWorkloadSignature:
    def test_digest_is_deterministic(self):
        sig_a = _make_signature()
        sig_b = _make_signature()
        assert sig_a.digest() == sig_b.digest()

    def test_digest_changes_with_any_field(self):
        base = _make_signature().digest()
        assert _make_signature(dtype="bfloat16").digest() != base
        assert _make_signature(num_inference_steps=8).digest() != base
        assert _make_signature(cache_mode="teacache").digest() != base


class TestTrajectoryGateValidation:
    def test_rejects_empty_checkpoints(self):
        with pytest.raises(ValueError):
            TrajectoryGate(checkpoints=(), tensor_thresholds={})

    def test_rejects_duplicate_checkpoints(self):
        with pytest.raises(ValueError):
            TrajectoryGate(
                checkpoints=("a", "a"),
                tensor_thresholds={"a": {"cosine_similarity": 0.9}},
            )

    def test_rejects_missing_threshold_entries(self):
        with pytest.raises(ValueError):
            TrajectoryGate(
                checkpoints=("a", "b"),
                tensor_thresholds={"a": {"cosine_similarity": 0.9}},
            )

    def test_output_metrics_defaults_to_empty_mapping(self):
        gate = _make_gate()
        assert gate.output_metrics == {}


class TestCompiledPlanManifest:
    def test_rejects_invalid_status(self):
        sig = _make_signature()
        with pytest.raises(ValueError):
            CompiledPlanManifest(
                signature=sig,
                regions=(),
                compile_options={},
                gate_digest=sig.digest(),
                status="maybe",
                checkpoint_metrics={},
            )

    def test_is_validated_property(self):
        sig = _make_signature()
        validated = CompiledPlanManifest(
            signature=sig,
            regions=(),
            compile_options={},
            gate_digest=sig.digest(),
            status="validated",
            checkpoint_metrics={},
        )
        rejected = CompiledPlanManifest(
            signature=sig,
            regions=(),
            compile_options={},
            gate_digest=sig.digest(),
            status="rejected",
            checkpoint_metrics={},
        )
        assert validated.is_validated is True
        assert rejected.is_validated is False


class TestComputeTensorMetrics:
    def test_identical_tensors_are_perfect(self):
        t = torch.randn(4, 8, device=DEVICE)
        metrics = compute_tensor_metrics(t, t.clone())
        assert metrics["cosine_similarity"] == pytest.approx(1.0, abs=1e-5)
        assert metrics["mae"] == pytest.approx(0.0, abs=1e-6)
        assert metrics["max_abs"] == pytest.approx(0.0, abs=1e-6)

    def test_shape_mismatch_raises(self):
        with pytest.raises(CompileGateError):
            compute_tensor_metrics(
                torch.randn(4, device=DEVICE), torch.randn(5, device=DEVICE)
            )

    def test_metrics_computed_on_device_tensors(self):
        t = torch.randn(4, 8, device=DEVICE)
        metrics = compute_tensor_metrics(t, t + 0.01)
        assert metrics["max_abs"] == pytest.approx(0.01, abs=1e-4)


class TestRunTrajectoryGateWithRealTorchCompile:
    """Exercises a real torch.compile call against a genuine eager reference,
    on DEVICE (MPS when available)."""

    def _build_models(self, dim=8, seed=123):
        torch.manual_seed(seed)
        eager_model = _ToyIterativeModel(dim=dim).to(DEVICE)
        compiled_model = torch.compile(
            eager_model, backend="aot_eager", fullgraph=False
        )
        return eager_model, compiled_model

    def test_matching_model_is_validated(self):
        eager_model, compiled_model = self._build_models()
        num_steps = 4
        gate = _make_gate(checkpoints=("step_0", "terminal"))
        signature = _make_signature(num_inference_steps=num_steps)

        # The candidate here calls the *compiled* wrapper around the same
        # weights, so the trajectories should match to numerical precision.
        eager_states = _run_toy_rollout(eager_model, num_steps)
        compiled_states = _run_toy_rollout(compiled_model, num_steps)

        checkpoint_schedule = [
            (
                i,
                (
                    "step_0"
                    if i == 0
                    else ("terminal" if i == num_steps - 1 else f"mid_{i}")
                ),
            )
            for i in range(num_steps)
        ]

        def reference_step_fn(step_index, checkpoint_name):
            return eager_states[step_index]

        def candidate_step_fn(step_index, checkpoint_name):
            return compiled_states[step_index]

        manifest = run_trajectory_gate(
            signature=signature,
            gate=gate,
            reference_step_fn=reference_step_fn,
            candidate_step_fn=candidate_step_fn,
            checkpoint_schedule=checkpoint_schedule,
            regions=("lin",),
            compile_options={"backend": "aot_eager"},
        )

        assert manifest.status == "validated"
        assert manifest.is_validated
        assert manifest.gate_digest == signature.digest()
        assert set(manifest.checkpoint_metrics) == {"step_0", "terminal"}
        for metrics in manifest.checkpoint_metrics.values():
            assert metrics["cosine_similarity"] > 0.999

    def test_diverging_candidate_is_rejected(self):
        eager_model, _ = self._build_models()
        num_steps = 4
        gate = _make_gate(
            checkpoints=("step_0", "terminal"), cosine_min=0.999, max_abs=1e-6
        )
        signature = _make_signature(num_inference_steps=num_steps)

        eager_states = _run_toy_rollout(eager_model, num_steps)
        # A deliberately "diverging" candidate: same shape, visibly different
        # values, standing in for a broken/miscompiled plan.
        diverging_states = [s + 0.5 for s in eager_states]

        checkpoint_schedule = [(0, "step_0"), (num_steps - 1, "terminal")]

        manifest = run_trajectory_gate(
            signature=signature,
            gate=gate,
            reference_step_fn=lambda i, name: eager_states[i],
            candidate_step_fn=lambda i, name: diverging_states[i],
            checkpoint_schedule=checkpoint_schedule,
        )

        assert manifest.status == "rejected"
        assert not manifest.is_validated

    def test_missing_checkpoint_in_schedule_raises(self):
        gate = _make_gate(checkpoints=("step_0", "terminal"))
        signature = _make_signature()
        t = torch.randn(8, device=DEVICE)

        with pytest.raises(CompileGateError):
            run_trajectory_gate(
                signature=signature,
                gate=gate,
                reference_step_fn=lambda i, name: t,
                candidate_step_fn=lambda i, name: t,
                # Never visits "terminal".
                checkpoint_schedule=[(0, "step_0")],
            )

    def test_step_fn_returning_none_for_declared_checkpoint_raises(self):
        gate = _make_gate(checkpoints=("step_0",))
        signature = _make_signature()

        with pytest.raises(CompileGateError):
            run_trajectory_gate(
                signature=signature,
                gate=gate,
                reference_step_fn=lambda i, name: None,
                candidate_step_fn=lambda i, name: torch.randn(8, device=DEVICE),
                checkpoint_schedule=[(0, "step_0")],
            )


class TestDecisionTraceMatching:
    """require_decision_trace_match must actually be enforced, not just
    declared -- these tests fail against a version of run_trajectory_gate
    that accepts the field but never reads it."""

    def _gate_with_trace_requirement(self):
        return TrajectoryGate(
            checkpoints=("step_0",),
            tensor_thresholds={"step_0": {"cosine_similarity": 0.0}},
            require_decision_trace_match=True,
        )

    def test_matching_traces_are_validated(self):
        gate = self._gate_with_trace_requirement()
        signature = _make_signature()
        t = torch.randn(8, device=DEVICE)

        manifest = run_trajectory_gate(
            signature=signature,
            gate=gate,
            reference_step_fn=lambda i, name: t,
            candidate_step_fn=lambda i, name: t,
            checkpoint_schedule=[(0, "step_0")],
            reference_decision_trace=["real", "reuse", "real"],
            candidate_decision_trace=["real", "reuse", "real"],
        )

        assert manifest.status == "validated"
        assert manifest.decision_trace_matched is True

    def test_mismatched_traces_are_rejected_even_if_tensors_match(self):
        gate = self._gate_with_trace_requirement()
        signature = _make_signature()
        t = torch.randn(8, device=DEVICE)

        manifest = run_trajectory_gate(
            signature=signature,
            gate=gate,
            reference_step_fn=lambda i, name: t,
            candidate_step_fn=lambda i, name: t,
            checkpoint_schedule=[(0, "step_0")],
            reference_decision_trace=["real", "reuse"],
            candidate_decision_trace=["real", "real"],
        )

        assert manifest.status == "rejected"
        assert manifest.decision_trace_matched is False

    def test_missing_traces_raise_when_required(self):
        gate = self._gate_with_trace_requirement()
        signature = _make_signature()
        t = torch.randn(8, device=DEVICE)

        with pytest.raises(CompileGateError):
            run_trajectory_gate(
                signature=signature,
                gate=gate,
                reference_step_fn=lambda i, name: t,
                candidate_step_fn=lambda i, name: t,
                checkpoint_schedule=[(0, "step_0")],
                # No decision traces supplied even though the gate requires them.
            )

    def test_decision_trace_matched_is_none_when_not_required(self):
        gate = _make_gate(checkpoints=("step_0",))
        signature = _make_signature()
        t = torch.randn(8, device=DEVICE)

        manifest = run_trajectory_gate(
            signature=signature,
            gate=gate,
            reference_step_fn=lambda i, name: t,
            candidate_step_fn=lambda i, name: t,
            checkpoint_schedule=[(0, "step_0")],
        )

        assert manifest.decision_trace_matched is None


class TestSelectValidatedPlan:
    def test_returns_manifest_matching_requested_signature(self):
        sig = _make_signature()
        validated = CompiledPlanManifest(
            signature=sig,
            regions=(),
            compile_options={},
            gate_digest=sig.digest(),
            status="validated",
            checkpoint_metrics={},
        )
        result = select_validated_plan([validated], sig)
        assert result is validated

    def test_falls_back_to_none_when_no_manifest_covers_signature(self):
        sig = _make_signature()
        other_sig = _make_signature(num_inference_steps=99)
        validated = CompiledPlanManifest(
            signature=sig,
            regions=(),
            compile_options={},
            gate_digest=sig.digest(),
            status="validated",
            checkpoint_metrics={},
        )
        assert select_validated_plan([validated], other_sig) is None

    def test_rejected_manifest_is_never_selected(self):
        sig = _make_signature()
        rejected = CompiledPlanManifest(
            signature=sig,
            regions=(),
            compile_options={},
            gate_digest=sig.digest(),
            status="rejected",
            checkpoint_metrics={},
        )
        assert select_validated_plan([rejected], sig) is None
