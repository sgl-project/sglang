# SPDX-License-Identifier: Apache-2.0
from typing import Optional

from sglang.multimodal_gen.configs.sample.sampling_params import (
    validate_served_postprocess_model_path,
)
from sglang.multimodal_gen.runtime.server_args import get_global_server_args


def check_served_model_path(name: str, value: Optional[str]) -> None:
    """Apply --postprocess-model-paths where the weights are resolved.

    The HTTP layer rejects unlisted paths first. This repeats the check for a
    request that reaches a worker another way (scheduler socket, disaggregated
    role). Outside a server (CLI, Python API) no list is set and any path is
    accepted.
    """
    try:
        server_args = get_global_server_args()
    except ValueError:  # no server args in this process
        return
    allowed = getattr(server_args, "postprocess_model_paths", None)
    if allowed is not None:
        validate_served_postprocess_model_path(name, value, allowed)
