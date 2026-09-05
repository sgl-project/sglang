"""OpAuto policy: reorder / filter kernel backends with sticky demotion."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable, List, Optional, Sequence

from sglang.kernels.opauto.persist import load_state_file, save_state_file
from sglang.kernels.opauto.state import BackendStatus, get_state

if TYPE_CHECKING:
    from sglang.kernels.spec import KernelBackend

logger = logging.getLogger(__name__)


def _env_enabled() -> bool:
    from sglang.srt.environ import envs

    return bool(envs.SGLANG_OPAUTO.get())


def _env_cold_skip_jit() -> bool:
    from sglang.srt.environ import envs

    return bool(envs.SGLANG_OPAUTO_COLD_SKIP_JIT.get())


def is_enabled() -> bool:
    return _env_enabled()


def enable_opauto(enabled: bool = True) -> None:
    """Process-local override used by tests (does not mutate the OS env)."""
    from sglang.srt.environ import envs

    envs.SGLANG_OPAUTO.set("1" if enabled else "0")


def set_cold_skip_jit(enabled: bool = True) -> None:
    from sglang.srt.environ import envs

    envs.SGLANG_OPAUTO_COLD_SKIP_JIT.set("1" if enabled else "0")


class OpAutoPolicy:
    """Pick among eligible backends; default = caller priority order."""

    def __init__(self) -> None:
        self._loaded = False

    def ensure_loaded(self) -> None:
        if self._loaded or not is_enabled():
            return
        data = load_state_file()
        entries = data.get("entries") if isinstance(data, dict) else None
        if isinstance(entries, dict):
            get_state().load_snapshot(entries)
        self._loaded = True

    def persist(self) -> None:
        if not is_enabled():
            return
        save_state_file({"entries": get_state().snapshot()})

    def pick(
        self,
        op_id: str,
        candidates: Sequence["KernelBackend"],
        *,
        shape_bucket: str = "*",
        priority: Optional[Sequence["KernelBackend"]] = None,
    ) -> List["KernelBackend"]:
        """Return filtered/reordered candidates.

        When OpAuto is disabled, returns ``list(candidates)`` unchanged
        (still respecting failed demotions only if previously marked — no-op).
        """
        if not candidates:
            return []
        if not is_enabled():
            return list(candidates)

        self.ensure_loaded()
        from sglang.kernels.spec import KernelBackend

        order = list(priority) if priority is not None else list(candidates)
        # Keep only candidates, stable by priority order.
        cand_set = set(candidates)
        ordered = [b for b in order if b in cand_set]
        for b in candidates:
            if b not in ordered:
                ordered.append(b)

        filtered: List[KernelBackend] = []
        for backend in ordered:
            name = backend.value if isinstance(backend, KernelBackend) else str(backend)
            if get_state().is_failed(op_id, name, shape_bucket=shape_bucket):
                continue
            if (
                backend is KernelBackend.JIT
                and _env_cold_skip_jit()
                and self._should_skip_cold_jit(op_id, shape_bucket=shape_bucket)
            ):
                continue
            filtered.append(backend)

        # If everything was filtered, fall back to original candidates minus
        # sticky failures only (never leave the caller with zero options).
        if not filtered:
            filtered = [
                b
                for b in ordered
                if not get_state().is_failed(
                    op_id,
                    b.value if isinstance(b, KernelBackend) else str(b),
                    shape_bucket=shape_bucket,
                )
            ] or list(candidates)
        return filtered

    def _should_skip_cold_jit(self, op_id: str, *, shape_bucket: str) -> bool:
        st = get_state().get(op_id, "jit", shape_bucket=shape_bucket)
        if st.status is BackendStatus.WARM:
            return False
        # Prefer skipping on pre-Ampere where JIT probes are known-bad.
        try:
            from sglang.kernels.jit.utils.arch import is_pre_ampere_cuda

            if is_pre_ampere_cuda():
                return True
        except Exception:
            pass
        return st.status is BackendStatus.COLD

    def demote(
        self,
        op_id: str,
        backend: str,
        *,
        shape_bucket: str = "*",
        reason: str = "",
        persist: bool = False,
    ) -> None:
        if not is_enabled():
            return
        get_state().mark_failed(
            op_id, backend, shape_bucket=shape_bucket, reason=reason
        )
        logger.info(
            "OpAuto demoted %s/%s (%s)", op_id, backend, reason or "unspecified"
        )
        if persist:
            self.persist()

    def mark_warm(
        self,
        op_id: str,
        backend: str,
        *,
        shape_bucket: str = "*",
        measured_us: Optional[float] = None,
    ) -> None:
        if not is_enabled():
            return
        get_state().mark_warm(
            op_id, backend, shape_bucket=shape_bucket, measured_us=measured_us
        )


_POLICY: Optional[OpAutoPolicy] = None


def get_policy() -> OpAutoPolicy:
    global _POLICY
    if _POLICY is None:
        _POLICY = OpAutoPolicy()
    return _POLICY


def should_skip_cold_jit(op_id: str, *, shape_bucket: str = "*") -> bool:
    """True when OpAuto wants to avoid a cold JIT probe for ``op_id``."""
    if not is_enabled() or not _env_cold_skip_jit():
        return False
    get_policy().ensure_loaded()
    return get_policy()._should_skip_cold_jit(op_id, shape_bucket=shape_bucket)


def can_use_or_demote(
    op_id: str,
    probe_fn: Callable[[], bool],
    *,
    backend: str = "jit",
    shape_bucket: str = "*",
) -> bool:
    """Run ``probe_fn``; sticky-demote ``backend`` on False/exception when enabled.

    When OpAuto is off, returns ``probe_fn()`` directly (no demotion).
    Pre-Ampere + cold-skip short-circuits before calling ``probe_fn``.
    """
    if not is_enabled():
        return probe_fn()

    policy = get_policy()
    policy.ensure_loaded()
    if get_state().is_failed(op_id, backend, shape_bucket=shape_bucket):
        return False
    if backend == "jit" and should_skip_cold_jit(op_id, shape_bucket=shape_bucket):
        policy.demote(
            op_id,
            backend,
            shape_bucket=shape_bucket,
            reason="cold_skip_jit",
        )
        return False
    try:
        ok = bool(probe_fn())
    except Exception as e:
        policy.demote(op_id, backend, shape_bucket=shape_bucket, reason=str(e))
        return False
    if ok:
        policy.mark_warm(op_id, backend, shape_bucket=shape_bucket)
    else:
        policy.demote(
            op_id, backend, shape_bucket=shape_bucket, reason="probe_returned_false"
        )
    return ok


def should_prefer_native_aot_fallback(
    op_id: str = "layernorm.rmsnorm", *, shape_bucket: str = "*"
) -> bool:
    """Prefer native over AOT rmsnorm when OpAuto says AOT is unsafe."""
    if not is_enabled():
        return False
    get_policy().ensure_loaded()
    if get_state().is_failed(op_id, "aot", shape_bucket=shape_bucket):
        return True
    if _env_cold_skip_jit():
        try:
            from sglang.kernels.jit.utils.arch import is_pre_ampere_cuda

            if is_pre_ampere_cuda():
                return True
        except Exception:
            pass
    return False


def status_payload() -> dict:
    """Compact status for consoles / eval cards."""
    enabled = is_enabled()
    return {
        "enabled": enabled,
        "cold_skip_jit": _env_cold_skip_jit() if enabled else False,
        "demoted": get_state().demoted_backends() if enabled else {},
        "state_path": str(
            __import__(
                "sglang.kernels.opauto.persist", fromlist=["default_state_path"]
            ).default_state_path()
        ),
    }
