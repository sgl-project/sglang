from __future__ import annotations

from typing import (
    Any,
    Mapping,
    MutableMapping,
    Optional,
    Protocol,
    runtime_checkable,
)

import torch

from sglang.srt.sampling.sampling_observer import (
    DeviceAuxiliaryOutput,
    SamplingObserver,
)


@runtime_checkable
class PipelineParallelAuxiliaryOutput(Protocol):
    def to_pp_tensors(self) -> Mapping[str, torch.Tensor]: ...


@runtime_checkable
class PipelineParallelSamplingObserver(Protocol):
    def from_pp_tensors(
        self, tensors: Mapping[str, torch.Tensor]
    ) -> DeviceAuxiliaryOutput: ...


_OUTPUT_PREFIX = "__sampling_observer_output__."


def add_auxiliary_output_to_pp_tensors(
    tensors: MutableMapping[str, Any],
    output: Optional[DeviceAuxiliaryOutput],
) -> None:
    if output is None:
        return
    if not isinstance(output, PipelineParallelAuxiliaryOutput):
        raise RuntimeError(
            "auxiliary output does not support pipeline-parallel transport"
        )

    output_tensors = output.to_pp_tensors()
    if not output_tensors:
        raise RuntimeError("auxiliary PP output must contain at least one tensor")

    for name, tensor in output_tensors.items():
        if not isinstance(name, str) or not name:
            raise RuntimeError("auxiliary PP tensor names must be non-empty strings")
        if not torch.is_tensor(tensor):
            raise RuntimeError(f"auxiliary PP output {name!r} is not a tensor")
        key = f"{_OUTPUT_PREFIX}{name}"
        if key in tensors:
            raise RuntimeError(f"duplicate auxiliary PP tensor {name!r}")
        tensors[key] = tensor


def pop_auxiliary_output_from_pp_tensors(
    tensors: MutableMapping[str, Any],
    observer: Optional[SamplingObserver],
) -> Optional[DeviceAuxiliaryOutput]:
    output_tensors = {
        key.removeprefix(_OUTPUT_PREFIX): value
        for key, value in tensors.items()
        if key.startswith(_OUTPUT_PREFIX)
    }
    if not output_tensors:
        return None
    if observer is None:
        raise RuntimeError("received auxiliary PP output without a sampling observer")
    if not isinstance(observer, PipelineParallelSamplingObserver):
        raise RuntimeError(
            "sampling observer does not support pipeline-parallel transport"
        )
    if any(not torch.is_tensor(tensor) for tensor in output_tensors.values()):
        raise RuntimeError("received a non-tensor auxiliary PP output")

    output = observer.from_pp_tensors(output_tensors)
    if output is None:
        raise RuntimeError("sampling observer did not reconstruct its PP output")
    for name in output_tensors:
        del tensors[f"{_OUTPUT_PREFIX}{name}"]
    return output
