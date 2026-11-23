# SPDX-License-Identifier: Apache-2.0
"""Restricted pickle decoding for the HTTP frontend -> diffusion head hop.

This is an internal transport, not an authenticated public API. Only the server's
configured sampling contract is added to the shared tensor/container allowlist.
"""

import io
from dataclasses import is_dataclass
from enum import Enum
from typing import get_args, get_type_hints

import msgspec

from sglang.multimodal_gen.configs.sample.cosmos3_multiview import MultiviewViewInput
from sglang.multimodal_gen.configs.sample.sampling_params import (
    SamplingParams,
    SkipSoftmaxParams,
)
from sglang.multimodal_gen.configs.sample.spectrum import SpectrumParams
from sglang.multimodal_gen.configs.sample.teacache import TeaCacheParams
from sglang.multimodal_gen.configs.sample.wan import (
    _wan_1_3b_coefficients,
    _wan_14b_coefficients,
)
from sglang.multimodal_gen.runtime.distributed.ipc_cuda import CudaIpcRef
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.utils.perf_logger import (
    MemorySnapshot,
    RequestMetrics,
)
from sglang.srt.observability.trace import TraceNullContext, TraceReqContext
from sglang.srt.utils.serialization import SafeUnpickler


def request_pickle_globals(sampling_params_cls: type) -> dict[tuple[str, str], object]:
    """Build once from trusted server types, never from names in a payload.

    Follow only data annotations on sampling params, including out-of-tree model
    contracts. Fields typed Any do not authorize arbitrary classes or callables.
    """
    allowed = {
        (obj.__module__, obj.__qualname__): obj
        for obj in (
            Req,
            CudaIpcRef,
            TraceNullContext,
            TraceReqContext,
            RequestMetrics,
            MemorySnapshot,
            MultiviewViewInput,
            SkipSoftmaxParams,
            # Existing Wan defaults carry these two pure coefficient selectors.
            _wan_1_3b_coefficients,
            _wan_14b_coefficients,
        )
    }
    pending = [SamplingParams, sampling_params_cls, TeaCacheParams, SpectrumParams]
    seen = set()
    while pending:
        annotation = pending.pop()
        if not isinstance(annotation, type):
            pending.extend(get_args(annotation))
            continue
        if annotation in seen:
            continue
        seen.add(annotation)
        if is_dataclass(annotation) or issubclass(annotation, msgspec.Struct):
            allowed[(annotation.__module__, annotation.__qualname__)] = annotation
            pending.extend(get_type_hints(annotation).values())
        elif issubclass(annotation, Enum):
            allowed[(annotation.__module__, annotation.__qualname__)] = annotation
    return allowed


class RequestUnpickler(SafeUnpickler):
    # Array/image reconstruction only; do not allow module prefixes or file refs.
    ALLOWED_GLOBALS = SafeUnpickler.ALLOWED_GLOBALS | {
        ("numpy", "ndarray"),
        ("numpy", "dtype"),
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy.core.multiarray", "scalar"),
        ("numpy.core.numeric", "_frombuffer"),
        ("numpy._core.multiarray", "_reconstruct"),
        ("numpy._core.multiarray", "scalar"),
        ("numpy._core.numeric", "_frombuffer"),
        ("PIL.Image", "Image"),
        ("torch._C", "Generator"),
    }

    def __init__(self, data: bytes, allowed_globals: dict[tuple[str, str], object]):
        super().__init__(io.BytesIO(data))
        self._request_globals = allowed_globals

    def find_class(self, module, name):
        key = (module, name)
        if key in self._request_globals:
            return self._request_globals[key]
        return super().find_class(module, name)
