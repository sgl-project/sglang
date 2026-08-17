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
            # only `cur` is runtime.
            tl.device_print(
                f"[SimplePhaseChecker FAIL] caller_tag={CALLER_TAG} "
                f"expect={EXPECT_PHASE} next={NEXT_PHASE} actual=",
                cur,
            )
        tl.device_assert(cur == EXPECT_PHASE, "SimplePhaseChecker: phase mismatch")
    tl.store(phase_ptr, NEXT_PHASE)


class SimplePhaseChecker:
    """GPU-side state machine for any int-keyed phase sequence."""

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
        """Reset phase to initial_phase, then enable the device-side assert."""
        self._reset_to_idle()
        self._enable_assert_device.fill_(1)
        _host_debug(f"[SimplePhaseChecker.enable_assert] assert ENABLED")

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

    def _reset_to_idle(self) -> None:
        self._phase.fill_(self._initial_phase)
        _host_debug(
            f"[SimplePhaseChecker._reset_to_idle] phase reset to "
            f"{self._initial_phase}"
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
