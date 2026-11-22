"""Startup check that every rank of a megakernel server resolves the NVIDIA runtime
switches the same way.

The switches are SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION, every child that
follows it, and the opt-in pieces kept off it (nv_mega_runtime_opt_ins). Each
process resolves them from its own environment, so one rank launched with a
different environment would run different code around the same collectives; the
check turns that into a startup error that names the switch. It runs only on
servers started with --enable-nvidia-megakernel; switches set without that flag
are not cross-checked.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from sglang.srt.environ import nv_mega_runtime_opt_ins, nv_mega_runtime_switches

if TYPE_CHECKING:
    from sglang.srt.distributed.parallel_state import GroupCoordinator

logger = logging.getLogger(__name__)


def _agreed_switches():
    return nv_mega_runtime_switches() + nv_mega_runtime_opt_ins()


def resolved_runtime_switches() -> tuple[str, ...]:
    """Names of the switches that resolve on in this process, in registry order."""
    return tuple(field.name for field in _agreed_switches() if field.get())


def agree_runtime_switches(*, group: GroupCoordinator, megakernel: bool) -> None:
    """With megakernel (--enable-nvidia-megakernel), raise on every rank of group
    unless all its ranks resolve the same switches on, and log the agreed set once
    when it is not empty. Without it, return before any collective."""
    # The flag is a launch argument, identical on every rank, so either every rank
    # returns here or every rank enters the gather below.
    if not megakernel:
        return
    local = resolved_runtime_switches()
    gathered = group.all_gather_object(local) if group.world_size > 1 else [local]
    differences = _differences(gathered)
    if differences:
        raise RuntimeError(
            "NVIDIA runtime switches resolve differently across ranks; set each one "
            "the same way on every rank:\n" + "\n".join(differences)
        )
    if local and group.rank_in_group == 0:
        logger.info(
            "NVIDIA runtime switches on, agreed by %d ranks: %s",
            group.world_size,
            ", ".join(local),
        )


def _differences(gathered: list[tuple[str, ...]]) -> list[str]:
    order = {field.name: i for i, field in enumerate(_agreed_switches())}
    # A name this rank does not register sorts last, so the master is named first.
    names = sorted(set().union(*gathered), key=lambda n: (order.get(n, len(order)), n))
    lines = []
    for name in names:
        on = [rank for rank, names_on in enumerate(gathered) if name in names_on]
        if len(on) < len(gathered):
            off = [rank for rank in range(len(gathered)) if rank not in on]
            lines.append(f"  {name}: on at ranks {on}, off at ranks {off}")
    return lines
