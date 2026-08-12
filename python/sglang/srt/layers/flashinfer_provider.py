"""Resolve the FlashInfer MNNVL CuTe DSL communication provider.

The stable FlashInfer surface is intentionally small: construct the backend-
specific workspace, then invoke the unified ``flashinfer.comm.allreduce_fusion``
function. Until the serving image contains that workspace implementation,
SGLang supplies the same two-part surface from a temporary copied provider.
"""

from __future__ import annotations

import inspect
import logging
from functools import lru_cache
from types import ModuleType
from typing import Any, Callable

import msgspec

logger = logging.getLogger(__name__)


_REQUIRED_WORKSPACE_PARAMETERS = {
    "tp_size",
    "tp_rank",
    "max_token_num",
    "hidden_dim",
    "dtype",
    "group",
    "top_k",
    "rms_eps",
    "routed_scaling_factor",
    "weight_bias",
    "include_shared_expert",
    "add_residual",
    "write_residual_output",
    "config",
}
_REQUIRED_ALLREDUCE_PARAMETERS = {
    "input",
    "workspace",
    "pattern",
    "launch_with_pdl",
    "residual_in",
    "residual_out",
    "norm_out",
    "rms_gamma",
    "rms_eps",
    "weight_bias",
    "expanded_idx_to_permuted_idx",
    "expert_scale_factor",
    "shared_expert_output",
}
_REQUIRED_SUPPORTS_PARAMETERS = {
    "tp_size",
    "num_tokens",
    "hidden_dim",
    "dtype",
}


class FlashInferMNNVLCuteDSLProvider(msgspec.Struct, frozen=True):
    workspace_type: type
    allreduce_fusion: Callable[..., Any]
    patterns: type
    default_config: Any


def _accepts_required_parameters(callable_object, required_parameters) -> bool:
    try:
        parameters = inspect.signature(callable_object).parameters
    except (TypeError, ValueError):
        return False
    return bool(
        required_parameters <= parameters.keys()
        or any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )
    )


def _make_provider(
    comm: ModuleType, workspace_type: type, *, default_config: Any
) -> FlashInferMNNVLCuteDSLProvider | None:
    if not all(
        hasattr(workspace_type, method)
        for method in ("is_buffer_size_sufficient", "destroy")
    ):
        return None
    if not _accepts_required_parameters(workspace_type, _REQUIRED_WORKSPACE_PARAMETERS):
        return None
    if not _accepts_required_parameters(
        workspace_type.is_buffer_size_sufficient, _REQUIRED_SUPPORTS_PARAMETERS
    ):
        return None
    allreduce_fusion = getattr(comm, "allreduce_fusion", None)
    if allreduce_fusion is None:
        return None
    if not _accepts_required_parameters(
        allreduce_fusion, _REQUIRED_ALLREDUCE_PARAMETERS
    ):
        return None
    patterns = getattr(comm, "AllReduceFusionPattern", None)
    if patterns is None or not all(
        hasattr(patterns, name)
        for name in (
            "kARResidualRMSNorm",
            "kMoEFinalizeARResidualRMSNorm",
        )
    ):
        return None
    return FlashInferMNNVLCuteDSLProvider(
        workspace_type=workspace_type,
        allreduce_fusion=allreduce_fusion,
        patterns=patterns,
        default_config=default_config,
    )


@lru_cache(maxsize=1)
def get_flashinfer_comm_provider() -> FlashInferMNNVLCuteDSLProvider:
    """Return the upstream provider, or the API-compatible copied fallback."""
    import flashinfer.comm as upstream_comm

    try:
        from flashinfer.comm.mnnvl_cutedsl_ar import (
            MNNVLCuteDSLAllReduceFusionWorkspace as upstream_workspace_type,
        )
        from flashinfer.comm.mnnvl_cutedsl import (
            DEFAULT_CONFIG as upstream_default_config,
        )
    except ImportError as error:
        logger.debug("Upstream FlashInfer MNNVL CuTe DSL import failed: %s", error)
    else:
        provider = _make_provider(
            upstream_comm,
            upstream_workspace_type,
            default_config=upstream_default_config,
        )
        if provider is not None:
            logger.info("Using upstream FlashInfer MNNVL CuTe DSL fusion backend")
            return provider
        logger.debug(
            "Installed FlashInfer contains an incompatible MNNVL CuTe DSL API; "
            "using SGLang's copied provider"
        )

    try:
        from sglang.srt.layers.flashinfer_fallback import comm as fallback_comm
        from sglang.srt.layers.flashinfer_fallback.comm.mnnvl_cutedsl_ar import (
            MNNVLCuteDSLAllReduceFusionWorkspace as fallback_workspace_type,
        )
        from sglang.srt.layers.flashinfer_fallback.comm.mnnvl_cutedsl import (
            DEFAULT_CONFIG as fallback_default_config,
        )
    except ImportError as error:
        raise RuntimeError(
            "MNNVL CuTe DSL fusion requires either a FlashInfer release with "
            "the backend or SGLang's copied provider dependencies, including "
            "nvidia-cutlass-dsl and cuda-python"
        ) from error

    provider = _make_provider(
        fallback_comm,
        fallback_workspace_type,
        default_config=fallback_default_config,
    )
    if provider is None:
        raise RuntimeError(
            "SGLang's copied FlashInfer MNNVL CuTe DSL provider has an "
            "incompatible API"
        )
    logger.warning(
        "Installed FlashInfer does not provide the stable MNNVL CuTe DSL API; "
        "using SGLang's copied provider"
    )
    return provider
