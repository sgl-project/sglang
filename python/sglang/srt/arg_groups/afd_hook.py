# SPDX-License-Identifier: Apache-2.0
"""Validate AFD operator inputs through the normal resolution pipeline."""

from sglang.srt.afd.config import validate_afd_server_args
from sglang.srt.arg_groups.overrides import resolving_view


def handle_afd_config(server_args) -> None:
    validate_afd_server_args(resolving_view(server_args))
