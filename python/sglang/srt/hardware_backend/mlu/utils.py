# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING

from sglang.srt.arg_groups.overrides import declare_resolution

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs


def set_default_server_args(args: "ServerArgs") -> None:
    """Set default server arguments for the MLU backend."""
    # MLU does not support CustomAllReduce.
    declare_resolution(args, "set_default_server_args", disable_custom_all_reduce=True)
