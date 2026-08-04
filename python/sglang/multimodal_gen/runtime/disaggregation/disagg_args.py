# SPDX-License-Identifier: Apache-2.0
"""Compatibility shim for disaggregated diffusion argument helpers."""

from __future__ import annotations

import argparse

from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.server_args.disagg import DisaggServerArgsMixin

# Keep the historical disagg_args import path working.
DISAGG_RESULT_PORT_OFFSETS = DisaggServerArgsMixin.DISAGG_RESULT_PORT_OFFSETS
DisaggArgsMixin = DisaggServerArgsMixin


def add_disagg_cli_args(parser: argparse.ArgumentParser) -> None:
    """Register disaggregated-diffusion CLI args through ServerArgs."""

    from sglang.multimodal_gen.runtime.server_args import ServerArgs

    ServerArgs.add_disagg_cli_args(parser)


def convert_disagg_role_string(kwargs: dict) -> None:
    """Convert ``disagg_role`` from string to ``RoleType`` enum in-place."""

    if "disagg_role" in kwargs and isinstance(kwargs["disagg_role"], str):
        kwargs["disagg_role"] = RoleType.from_string(kwargs["disagg_role"])
