# SPDX-License-Identifier: Apache-2.0
"""SGLang's binding to the AMD monokernel plugin behind --enable-amd-monokernel,
plus the MSA one, which --enable-nvidia-megakernel routes here.

The shared contract -- startup validation, graph-size alignment, weight
observation and the forward install -- lives in ``sglang.srt.kernel_plugins.
mono``, keyed only by the (backend, flag) selected below; this module holds
what is AMD-specific."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Sequence

import torch

from sglang.kernels.plugin_kernels.amd.spec import (
    BACKEND,
    FLAG,
    StepMetadata,
    spec_registered,
)
from sglang.kernels.plugin_kernels.nvidia.spec import FLAG as NVIDIA_FLAG
from sglang.srt.kernel_plugins.mono import shape_key

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

# The MSA plugin serves NVIDIA Blackwell through the monokernel contract, so
# --enable-nvidia-megakernel routes the architectures it registers here.
MSA_BACKEND = "msa"


def msa_serves(architectures: Optional[Sequence[str]]) -> bool:
    """Whether the MSA plugin, not sgl-nvidia-mega, serves these architectures."""
    return spec_registered(backend=MSA_BACKEND, architectures=architectures)


def selected_monokernel(cfg: Any) -> Optional[tuple[str, str]]:
    """The (backend, flag) an AMD-family monokernel may serve here."""
    if cfg.enable_amd_monokernel:
        return BACKEND, FLAG
    if cfg.enable_nvidia_megakernel:
        return MSA_BACKEND, NVIDIA_FLAG
    return None


def device_isa() -> Optional[str]:
    if not torch.cuda.is_available() or not torch.version.hip:
        return None
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    return props.gcnArchName.split(":")[0]


def monokernel_serves(backend: str, architectures: Optional[Sequence[str]]) -> bool:
    """Whether the selected backend claims this model at all."""
    return backend != MSA_BACKEND or msa_serves(architectures)


def is_metadata_unread(
    model: torch.nn.Module, forward_batch: ForwardBatch, metadata: StepMetadata
) -> bool:
    """Whether the monokernel forward that runs this step never reads the metadata."""
    served, unread = model.__dict__.get("_monokernel_unread_metadata", ((), ()))
    return metadata in unread and shape_key(forward_batch) in served
