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


@triton.jit
def _phase_check_kernel(
    phase_ptr,
    EXPECT_PHASE: tl.constexpr,
    NEXT_PHASE: tl.constexpr,
    CALLER_TAG: tl.constexpr,
):
    cur = tl.load(phase_ptr)
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
    """GPU-side state machine for any int-keyed phase sequence."""

    def __init__(self, *, initial_phase: int | IntEnum, device: torch.device) -> None:
        self._phase = torch.tensor(int(initial_phase), dtype=torch.int32, device=device)
        self._caller_tag_registry: dict[str, int] = {}
        _host_debug(
            f"[SimplePhaseChecker.__init__] device={device} "
            f"initial_phase={_phase_repr(initial_phase)}"
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
            EXPECT_PHASE=int(expect_phase),
            NEXT_PHASE=int(next_phase),
            CALLER_TAG=caller_tag,
        )
