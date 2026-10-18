# SPDX-License-Identifier: Apache-2.0
"""Startup checks for --enable-nvidia-megakernel on the resolved configuration."""

from __future__ import annotations

from typing import Any, Optional

from sglang.srt.megakernels.spec import MegakernelSpec


def device_isa() -> Optional[str]:
    """This process's GPU as "sm_<major><minor>", or None without a visible GPU."""
    raise NotImplementedError  # TODO(L98)


def deadline_s(watchdog_timeout: float) -> float:
    """The deadline of every cross-GPU wait, below the watchdog timeout."""
    raise NotImplementedError  # TODO(L98)


def megakernel_disable_reasons(
    *, cfg: Any, hf_config: Any, isa: Optional[str]
) -> tuple[Optional[MegakernelSpec], list[str]]:
    """The spec for this model and every reason it cannot serve this configuration."""
    raise NotImplementedError  # TODO(L98)


def handle_nvidia_megakernel(server_args: Any) -> None:
    """Resolution step: fail before the weights load when the flag cannot be served."""
    raise NotImplementedError  # TODO(L98)
