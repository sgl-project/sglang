# SPDX-License-Identifier: Apache-2.0
"""
Reproducible compile-plan promotion gate for stateful iterative models.

Connects SGLang Diffusion's regional ``torch.compile`` support
(:mod:`sglang.multimodal_gen.runtime.utils.torch_compile`) to a
full-trajectory validation gate, so "compile succeeded" can become "this
exact compile plan passed the declared end-to-end trajectory contract for
this workload signature."

Scope (phase 1, matching the RFC):
  - a data model for the workload a compiled plan is valid for
    (:class:`CompileWorkloadSignature`), the trajectory contract it must
    pass (:class:`TrajectoryGate`), and the resulting offline/CI artifact
    (:class:`CompiledPlanManifest`);
  - :func:`run_trajectory_gate`, which actually warms and runs both an
    eager reference and a compiled candidate through a caller-supplied
    stepping function, captures tensors at declared checkpoints, and scores
    them with the same tensor-parity metrics used by
    ``tools/compare_diffusion_trajectory_similarity.py`` (cosine
    similarity, MAE/MSE/RMSE, max-abs, L2);
  - :func:`select_validated_plan`, the runtime-side lookup: does an
    incoming workload signature match a validated manifest, or should the
    caller fall back to eager?

This module does not decide to enable compilation for any model; it only
gives the promotion decision a reproducible, testable shape. It has no
diffusion-specific dependencies (no ``imageio``, no model loading), so it
can be exercised with a toy ``torch.nn.Module`` and validated on CPU or the
local accelerator (e.g. Apple Silicon MPS) without a real diffusion model.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Callable, Dict, Mapping, Sequence, Tuple

import torch

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class CompileGateError(RuntimeError):
    """Raised when the compile-plan promotion contract is violated."""


@dataclass(frozen=True)
class CompileWorkloadSignature:
    """The exact workload regime a compiled plan was validated for."""

    model_revision: str
    dtype: str
    backend: str
    parallel_signature: str
    latent_shape_regime: Tuple[int, ...]
    num_inference_steps: int
    cfg_mode: str
    cache_mode: str
    state_schema_version: str

    def digest(self) -> str:
        payload = json.dumps(
            {
                "model_revision": self.model_revision,
                "dtype": self.dtype,
                "backend": self.backend,
                "parallel_signature": self.parallel_signature,
                "latent_shape_regime": list(self.latent_shape_regime),
                "num_inference_steps": self.num_inference_steps,
                "cfg_mode": self.cfg_mode,
                "cache_mode": self.cache_mode,
                "state_schema_version": self.state_schema_version,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class TrajectoryGate:
    """Declares the checkpoints and pass thresholds a compiled plan must clear.

    Attributes:
        checkpoints: Ordered names of the points captured during the
            rollout (e.g. ``("step_0", "step_mid", "terminal")``).
        tensor_thresholds: Per-checkpoint metric thresholds. For
            ``"cosine_similarity"`` the value is a *minimum* (higher is
            better); for every other metric name
            (``mae``/``mse``/``rmse``/``max_abs``/``l2``) it is a
            *maximum* (lower is better).
        output_metrics: Model-level output-quality thresholds. The
            framework does not compute these itself (quality validation is
            model-specific per the RFC); it only carries them through to
            the manifest for the caller's own scoring.
        require_decision_trace_match: If True, the candidate run's declared
            decision trace (e.g. cache/skip decisions) must exactly match
            the reference's, in addition to the tensor thresholds.
    """

    checkpoints: Tuple[str, ...]
    tensor_thresholds: Mapping[str, Mapping[str, float]]
    output_metrics: Mapping[str, float] = None  # type: ignore[assignment]
    require_decision_trace_match: bool = False

    def __post_init__(self) -> None:
        if not self.checkpoints:
            raise ValueError("checkpoints must be non-empty")
        if len(set(self.checkpoints)) != len(self.checkpoints):
            raise ValueError(f"checkpoints must be unique, got {self.checkpoints}")
        missing = set(self.checkpoints) - set(self.tensor_thresholds)
        if missing:
            raise ValueError(
                f"tensor_thresholds missing entries for checkpoints: {sorted(missing)}"
            )
        if self.output_metrics is None:
            object.__setattr__(self, "output_metrics", {})


@dataclass(frozen=True)
class CompiledPlanManifest:
    """Offline/CI artifact recording whether a compile plan was promoted."""

    signature: CompileWorkloadSignature
    regions: Tuple[str, ...]
    compile_options: Mapping[str, object]
    gate_digest: str
    status: str  # "validated" or "rejected"
    checkpoint_metrics: Mapping[str, Mapping[str, float]]
    decision_trace_matched: bool | None = None

    def __post_init__(self) -> None:
        if self.status not in ("validated", "rejected"):
            raise ValueError(
                f"status must be 'validated' or 'rejected', got {self.status!r}"
            )

    @property
    def is_validated(self) -> bool:
        return self.status == "validated"

    def to_dict(self) -> dict:
        """Machine-readable form for the offline/CI manifest artifact."""
        sig = self.signature
        return {
            "signature": {
                "model_revision": sig.model_revision,
                "dtype": sig.dtype,
                "backend": sig.backend,
                "parallel_signature": sig.parallel_signature,
                "latent_shape_regime": list(sig.latent_shape_regime),
                "num_inference_steps": sig.num_inference_steps,
                "cfg_mode": sig.cfg_mode,
                "cache_mode": sig.cache_mode,
                "state_schema_version": sig.state_schema_version,
            },
            "regions": list(self.regions),
            "compile_options": dict(self.compile_options),
            "gate_digest": self.gate_digest,
            "status": self.status,
            "checkpoint_metrics": {
                name: dict(metrics) for name, metrics in self.checkpoint_metrics.items()
            },
            "decision_trace_matched": self.decision_trace_matched,
        }

    @classmethod
    def from_dict(cls, data: dict) -> CompiledPlanManifest:
        sig_data = data["signature"]
        signature = CompileWorkloadSignature(
            model_revision=sig_data["model_revision"],
            dtype=sig_data["dtype"],
            backend=sig_data["backend"],
            parallel_signature=sig_data["parallel_signature"],
            latent_shape_regime=tuple(sig_data["latent_shape_regime"]),
            num_inference_steps=sig_data["num_inference_steps"],
            cfg_mode=sig_data["cfg_mode"],
            cache_mode=sig_data["cache_mode"],
            state_schema_version=sig_data["state_schema_version"],
        )
        return cls(
            signature=signature,
            regions=tuple(data["regions"]),
            compile_options=dict(data["compile_options"]),
            gate_digest=data["gate_digest"],
            status=data["status"],
            checkpoint_metrics={
                name: dict(metrics)
                for name, metrics in data["checkpoint_metrics"].items()
            },
            decision_trace_matched=data.get("decision_trace_matched"),
        )


def load_manifests(path: str) -> list:
    """Load a JSON file of CompiledPlanManifest records.

    The file is a JSON array of objects produced by
    ``CompiledPlanManifest.to_dict()``. Raises ``CompileGateError`` (not a
    bare JSON/OS error) if the file is missing or malformed, so a
    misconfigured ``--compile-trajectory-gate-manifest`` path fails clearly
    instead of silently falling back to eager for every request.
    """
    import json

    try:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, ValueError) as exc:
        raise CompileGateError(
            f"failed to load compile-trajectory-gate manifest {path!r}: {exc}"
        ) from exc

    if not isinstance(raw, list):
        raise CompileGateError(
            f"manifest file {path!r} must contain a JSON array, got {type(raw).__name__}"
        )

    return [CompiledPlanManifest.from_dict(entry) for entry in raw]


def _cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    flat_a, flat_b = a.reshape(-1), b.reshape(-1)
    norm_a, norm_b = (
        torch.linalg.vector_norm(flat_a).item(),
        torch.linalg.vector_norm(flat_b).item(),
    )
    if norm_a == 0.0 and norm_b == 0.0:
        return 1.0
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(torch.nn.functional.cosine_similarity(flat_a, flat_b, dim=0).item())


def compute_tensor_metrics(
    reference: torch.Tensor, candidate: torch.Tensor
) -> Dict[str, float]:
    """Tensor-parity metrics for one checkpoint.

    Mirrors the metric semantics used by
    ``tools/compare_diffusion_trajectory_similarity.py`` (cosine
    similarity, MAE/MSE/RMSE, max-abs, L2), reimplemented locally so this
    module has no dependency on that script's heavier import surface
    (``imageio``, CLI argument parsing).
    """
    ref = reference.detach().float()
    cand = candidate.detach().float()
    if ref.shape != cand.shape:
        raise CompileGateError(
            f"checkpoint shape mismatch: {tuple(ref.shape)} vs {tuple(cand.shape)}"
        )
    ref_cpu, cand_cpu = ref.cpu(), cand.cpu()
    diff = ref_cpu - cand_cpu
    mse = float(diff.square().mean().item())
    return {
        "cosine_similarity": _cosine_similarity(ref_cpu, cand_cpu),
        "mae": float(diff.abs().mean().item()),
        "mse": mse,
        "rmse": math.sqrt(mse),
        "max_abs": float(diff.abs().max().item()),
        "l2": float(torch.linalg.vector_norm(diff).item()),
    }


def _passes_thresholds(
    metrics: Mapping[str, float], thresholds: Mapping[str, float]
) -> bool:
    for metric_name, bound in thresholds.items():
        value = metrics[metric_name]
        if metric_name == "cosine_similarity":
            if value < bound:
                return False
        else:
            if value > bound:
                return False
    return True


# A stepper advances one rollout step and returns the tensor to record if
# `checkpoint_name` is one of the gate's declared checkpoints, else None.
StepFn = Callable[[int, str], "torch.Tensor | None"]


def run_trajectory_gate(
    *,
    signature: CompileWorkloadSignature,
    gate: TrajectoryGate,
    reference_step_fn: StepFn,
    candidate_step_fn: StepFn,
    checkpoint_schedule: Sequence[Tuple[int, str]],
    regions: Tuple[str, ...] = (),
    compile_options: Mapping[str, object] = None,  # type: ignore[assignment]
    reference_decision_trace: Sequence[object] = None,  # type: ignore[assignment]
    candidate_decision_trace: Sequence[object] = None,  # type: ignore[assignment]
) -> CompiledPlanManifest:
    """Warm and validate a compiled candidate against an eager reference.

    ``checkpoint_schedule`` maps each rollout step index to the checkpoint
    name it corresponds to (only entries whose name is in
    ``gate.checkpoints`` are captured/compared). Both step functions are
    called for every ``(step_index, checkpoint_name)`` pair in the
    schedule; this is the "complete warmup rollout" the RFC requires,
    rather than a single isolated forward.

    When ``gate.require_decision_trace_match`` is set, ``reference_decision_trace``
    and ``candidate_decision_trace`` (e.g. per-step cache/skip decisions) must
    both be supplied and must compare equal, in addition to the tensor
    thresholds, for the plan to be promoted.
    """
    if compile_options is None:
        compile_options = {}

    decision_trace_matched: bool | None = None
    if gate.require_decision_trace_match:
        if reference_decision_trace is None or candidate_decision_trace is None:
            raise CompileGateError(
                "gate.require_decision_trace_match is True but "
                "reference_decision_trace/candidate_decision_trace were not supplied"
            )
        decision_trace_matched = list(reference_decision_trace) == list(
            candidate_decision_trace
        )

    checkpoint_metrics: Dict[str, Dict[str, float]] = {}
    for step_index, checkpoint_name in checkpoint_schedule:
        ref_tensor = reference_step_fn(step_index, checkpoint_name)
        cand_tensor = candidate_step_fn(step_index, checkpoint_name)
        if checkpoint_name not in gate.checkpoints:
            continue
        if ref_tensor is None or cand_tensor is None:
            raise CompileGateError(
                f"checkpoint {checkpoint_name!r} is declared in the gate but the "
                "step function returned no tensor for it"
            )
        checkpoint_metrics[checkpoint_name] = compute_tensor_metrics(
            ref_tensor, cand_tensor
        )

    missing = set(gate.checkpoints) - set(checkpoint_metrics)
    if missing:
        raise CompileGateError(
            f"checkpoint_schedule never visited declared checkpoints: {sorted(missing)}"
        )

    status = "validated"
    for checkpoint_name, metrics in checkpoint_metrics.items():
        thresholds = gate.tensor_thresholds[checkpoint_name]
        if not _passes_thresholds(metrics, thresholds):
            status = "rejected"
            logger.info(
                "compile trajectory gate rejected at checkpoint %r: metrics=%s thresholds=%s",
                checkpoint_name,
                metrics,
                thresholds,
            )
            break

    if decision_trace_matched is False:
        status = "rejected"
        logger.info("compile trajectory gate rejected: decision trace mismatch")

    return CompiledPlanManifest(
        signature=signature,
        regions=regions,
        compile_options=dict(compile_options),
        gate_digest=signature.digest(),
        status=status,
        checkpoint_metrics=checkpoint_metrics,
        decision_trace_matched=decision_trace_matched,
    )


def select_validated_plan(
    manifests: Sequence[CompiledPlanManifest],
    requested: CompileWorkloadSignature,
) -> CompiledPlanManifest | None:
    """Return the validated manifest covering ``requested``, or None.

    The runtime does not rerun the reference rollout per request; it only
    checks whether the incoming workload signature is covered by a
    validated manifest and falls back to eager execution (by returning
    None) otherwise.
    """
    for manifest in manifests:
        if manifest.is_validated and manifest.signature == requested:
            return manifest
    return None
