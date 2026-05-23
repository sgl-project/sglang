from __future__ import annotations

from enum import IntEnum

import torch
import triton
import triton.language as tl

from sglang.srt.environ import envs


def _phase_repr(phase: int | IntEnum) -> str:
    if isinstance(phase, IntEnum):
        return f"{phase.name}({int(phase)})"
    return str(int(phase))


def _host_debug(msg: str) -> None:
    if envs.SGLANG_PHASE_CHECKER_DEBUG.get():
        print(msg, flush=True)


# debug=True so tl.device_assert below actually raises. Without it the assert
# is stripped at compile time and only tl.device_print fires (the assert is
# gated on the TRITON_DEBUG env var by default — see tl.device_assert docstring).
@triton.jit(debug=True)
def _phase_check_kernel(
    phase_ptr,
    enable_assert_ptr,
    EXPECT_PHASE: tl.constexpr,
    NEXT_PHASE: tl.constexpr,
    CALLER_TAG: tl.constexpr,
):
    cur = tl.load(phase_ptr)
    enable_assert = tl.load(enable_assert_ptr)
    if enable_assert != 0:
        if cur != EXPECT_PHASE:
            # constexpr values get baked into the prefix string at compile time;
            # only `cur` is runtime and shown as "(operand 0) <int>".
            tl.device_print(
                f"[SimplePhaseChecker FAIL] caller_tag={CALLER_TAG} "
                f"expect={EXPECT_PHASE} next={NEXT_PHASE} actual=",
                cur,
            )
        tl.device_assert(cur == EXPECT_PHASE, "SimplePhaseChecker: phase mismatch")
    tl.store(phase_ptr, NEXT_PHASE)


class SimplePhaseChecker:
    """GPU-side state machine for any int-keyed phase sequence.

    The check kernel is launched unconditionally on every ``update()`` call so
    it is safely captured into cuda graphs. Whether the kernel actually
    asserts on a phase mismatch is decided by a device-side flag
    (``_enable_assert_device``), which can be toggled without re-recording any
    graph. The flag is OFF at construction so init-time work (warmup, graph
    capture, piecewise compile) that legitimately violates the strict
    lifecycle does not raise. Call :meth:`enable_assert` once initialization
    is finished to turn it on.
    """

    def __init__(self, *, initial_phase: int | IntEnum, device: torch.device) -> None:
        self._initial_phase = int(initial_phase)
        self._phase = torch.tensor(
            self._initial_phase, dtype=torch.int32, device=device
        )
        self._enable_assert_device = torch.zeros(1, dtype=torch.int32, device=device)
        self._caller_tag_registry: dict[str, int] = {}
        _host_debug(
            f"[SimplePhaseChecker.__init__] device={device} "
            f"initial_phase={_phase_repr(initial_phase)} "
            f"enable_assert=OFF (call enable_assert() after init is done)"
        )

    def enable_assert(self) -> None:
        """Turn on the device-side assert and reset phase to ``initial_phase``.

        Called once after all init-time work (warmup, cuda graph capture,
        piecewise compile, etc.) so the post-init phase sequence starts from a
        known state regardless of what captured kernels left in the phase
        tensor during init."""
        self._phase.fill_(self._initial_phase)
        self._enable_assert_device.fill_(1)
        _host_debug(
            f"[SimplePhaseChecker.enable_assert] phase reset to "
            f"{self._initial_phase}, assert ENABLED"
        )

    def _resolve_caller_tag(self, caller_name: str) -> int:
        registry = self._caller_tag_registry
        if caller_name not in registry:
            registry[caller_name] = len(registry) + 1
            _host_debug(
                f"[SimplePhaseChecker] registered caller_tag "
                f"{registry[caller_name]} <- {caller_name!r}"
            )
        return registry[caller_name]

    def update(
        self,
        *,
        expect_phase: int | IntEnum,
        next_phase: int | IntEnum,
        caller_name: str = "",
    ) -> None:
        caller_tag = self._resolve_caller_tag(caller_name)
        _host_debug(
            f"[SimplePhaseChecker.update] caller={caller_name!r} "
            f"caller_tag={caller_tag} "
            f"expect={_phase_repr(expect_phase)} "
            f"next={_phase_repr(next_phase)} "
            f"capturing={torch.cuda.is_current_stream_capturing()}"
        )
        _phase_check_kernel[(1,)](
            self._phase,
            self._enable_assert_device,
            EXPECT_PHASE=int(expect_phase),
            NEXT_PHASE=int(next_phase),
            CALLER_TAG=caller_tag,
        )
