"""Shared first-sight verification for bit-exact diffusion fast paths."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, TypeVar

import torch

T = TypeVar("T")
EqualFn = Callable[[Any, Any], bool]
DiagnosticHintFn = Callable[[], str | None]


def flashinfer_rmsnorm_diagnostic_hint() -> str:
    """Describe the live FlashInfer RMSNorm backend after an exactness miss.

    Keep the imports and metadata lookups inside this function: callers pass it
    as a callback, so none of this work runs on the verified steady-state path.
    """
    import importlib
    import importlib.metadata
    import os

    try:
        flashinfer_norm = importlib.import_module("flashinfer.norm")
        use_cuda_norm = getattr(flashinfer_norm, "_USE_CUDA_NORM", None)
    except Exception:
        backend = "unavailable"
    else:
        if use_cuda_norm is True:
            backend = "CUDA JIT"
        elif use_cuda_norm is False:
            backend = "CuTe DSL"
        else:
            backend = "legacy or unknown (no _USE_CUDA_NORM flag)"

    versions = []
    for package in (
        "flashinfer-python",
        "flashinfer-cubin",
        "flashinfer-jit-cache",
    ):
        try:
            package_version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            package_version = "not installed"
        except Exception:
            package_version = "unknown"
        versions.append(f"{package}={package_version}")

    env_backend = os.environ.get("FLASHINFER_USE_CUDA_NORM", "<unset>")
    return (
        "RMSNorm exactness can change when FlashInfer selects a different "
        f"reduction backend. Detected backend={backend}, "
        f"FLASHINFER_USE_CUDA_NORM={env_backend}, {', '.join(versions)}. "
        "Check that the FlashInfer packages are version-aligned and that the "
        "expected RMSNorm backend is selected"
    )


class BitExactFusionGate:
    """Track permanent disable + first-sight ``torch.equal`` verification.

    Two modes:

    * **once-for-all** (default): the first successful equal-check enables the
      fused path for every later call (GLM / Ernie).
    * **per-signature**: each distinct ``sig`` is verified independently
      (FLUX / Sana), matching aten LayerNorm dispatch that can vary by shape.

    The first-sight check runs the eager reference chain plus a host sync, so
    it must never happen inside ``torch.compile`` tracing or CUDA graph
    capture. Once-for-all callers get both guards from
    :meth:`can_attempt_once`; per-signature callers must keep their own
    compile/capture checks next to the signature lookup (see FLUX / Sana).
    """

    __slots__ = ("name", "disabled", "verified", "verified_sigs")

    def __init__(self, name: str, *, per_signature: bool = False) -> None:
        self.name = name
        self.disabled = False
        # This is a plain field because steady-state DiT blocks read it on
        # every invocation; a property descriptor is measurable at this scale.
        self.verified = False
        self.verified_sigs: set[Any] | None = set() if per_signature else None

    def is_verified(self, sig: Any = None) -> bool:
        if self.verified_sigs is not None:
            return sig in self.verified_sigs
        return self.verified

    def mark_verified(self, sig: Any = None) -> None:
        if self.verified_sigs is not None:
            assert sig is not None
            self.verified_sigs.add(sig)
        self.verified = True

    def disable(self) -> None:
        self.disabled = True

    def can_attempt_once(self) -> bool:
        """Once-for-all mode: may we launch the fused kernel right now?"""
        if self.disabled:
            return False
        if self.verified:
            return True
        # First-sight verify runs the eager reference chain and a host sync:
        # attempt neither inside compile tracing nor CUDA graph capture (the
        # sync would abort the capture; BCG then blocks the signature). Once
        # verified, the fused kernel runs alone and is compile/capture-safe.
        if torch.compiler.is_compiling():
            return False
        return not (
            torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
        )

    def on_exception(
        self,
        exc: BaseException,
        *,
        logger: logging.Logger | None = None,
        re_raise_if_compiling: bool = True,
    ) -> None:
        if re_raise_if_compiling and torch.compiler.is_compiling():
            raise exc
        if logger is not None:
            logger.warning_once(f"Disabling {self.name} fast path: {exc}")
        self.disable()

    def accept_or_fallback(
        self,
        out: T,
        ref: T,
        *,
        sig: Any = None,
        equal: EqualFn | None = None,
        logger: logging.Logger | None = None,
        mismatch_msg: str | None = None,
        diagnostic_hint: DiagnosticHintFn | None = None,
    ) -> T:
        """Return ``out`` when bit-exact; otherwise disable and return ``ref``."""
        if self.is_verified(sig):
            return out
        eq = equal or torch.equal
        if eq(out, ref):
            self.mark_verified(sig)
            return out
        if logger is not None:
            message = mismatch_msg or (
                f"{self.name} fast path is not bit-exact against this "
                "platform's reference dispatch; falling back to eager"
            )
            details = (
                "Correctness is preserved because the eager reference output "
                "is used. A platform-specific reference kernel or reduction-order "
                "change may have caused this fallback"
            )
            if diagnostic_hint is not None:
                try:
                    diagnostic_details = diagnostic_hint()
                except Exception:
                    diagnostic_details = None
                if diagnostic_details:
                    details = f"{details}. {diagnostic_details.rstrip('.')}"
            logger.warning_once(f"{message.rstrip('.')}. {details}.")
        self.disable()
        return ref


def tensors_equal(a: Any, b: Any) -> bool:
    """``torch.equal`` for a tensor or a sequence of tensors."""
    if isinstance(a, torch.Tensor):
        return torch.equal(a, b)
    return all(torch.equal(x, y) for x, y in zip(a, b, strict=True))
