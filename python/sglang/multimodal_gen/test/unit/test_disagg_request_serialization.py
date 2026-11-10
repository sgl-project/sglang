# SPDX-License-Identifier: Apache-2.0
"""The diffusion head accepts its data contract, not arbitrary Python globals."""

import importlib
import io
import pickle
import pkgutil
from dataclasses import dataclass

import numpy as np
import pytest
import torch
from PIL import Image

from sglang.multimodal_gen.configs import sample
from sglang.multimodal_gen.configs.sample.action import ActionSamplingParams
from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.runtime.disaggregation.transport.request import (
    RequestUnpickler,
    request_pickle_globals,
)
from sglang.multimodal_gen.runtime.distributed.ipc_cuda import CudaIpcRef
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.utils.perf_logger import RequestMetrics
from sglang.srt.observability.trace import TraceReqContext
from sglang.srt.utils.serialization import safe_pickle_loads


def _sampling_classes():
    # A newly added model must not silently break this shared transport.
    for info in pkgutil.iter_modules(sample.__path__, sample.__name__ + "."):
        module = importlib.import_module(info.name)
        for cls in vars(module).values():
            if (
                isinstance(cls, type)
                and cls.__module__ == module.__name__
                and issubclass(cls, (SamplingParams, ActionSamplingParams))
            ):
                yield cls


@pytest.mark.parametrize(
    "params_cls", list(_sampling_classes()), ids=lambda c: c.__name__
)
def test_model_request_roundtrip(params_cls):
    params = params_cls()
    req = Req(sampling_params=params, request_id="roundtrip", prompt="test")
    payload = pickle.dumps([req])
    restored = RequestUnpickler(payload, request_pickle_globals(params_cls)).load()[0]
    assert type(restored.sampling_params) is params_cls
    assert restored.request_id == req.request_id
    assert vars(restored.sampling_params) == vars(params)
    assert type(restored.trace_ctx) is type(req.trace_ctx)


@dataclass
class _NestedParams:
    value: int = 7


@dataclass
class _PluginSamplingParams(SamplingParams):
    nested: _NestedParams | None = None


def test_server_owned_plugin_contract():
    req = Req(sampling_params=_PluginSamplingParams(nested=_NestedParams()))
    payload = pickle.dumps([req])
    restored = RequestUnpickler(
        payload, request_pickle_globals(_PluginSamplingParams)
    ).load()[0]
    assert restored.sampling_params.nested == _NestedParams()
    # Loading this contract does not widen another endpoint or the SRT loader.
    for load in (
        lambda p: RequestUnpickler(p, request_pickle_globals(SamplingParams)).load(),
        safe_pickle_loads,
    ):
        with pytest.raises(RuntimeError, match="Blocked unsafe global"):
            load(payload)


@pytest.mark.parametrize("protocol", [4, 5])
def test_request_media_and_metadata(protocol):
    tensor = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    array = np.arange(12, dtype=np.float32).reshape(3, 4)
    image = Image.new("RGB", (4, 3), "red")
    ipc = CudaIpcRef(
        "float32",
        (3, 4),
        (4, 1),
        0,
        0,
        b"handle",
        48,
        0,
        b"counter",
        0,
        b"event",
        False,
    )
    req = Req(
        request_id="media",
        prompt_embeds=[tensor],
        condition_image=image,
        condition_inputs={"array": array, "scalar": np.float32(1), "ipc": ipc},
        generator=torch.Generator().manual_seed(42),
        metrics=RequestMetrics(request_id="media"),
        trace_ctx=TraceReqContext("media", trace_level=0),
    )
    restored = RequestUnpickler(
        pickle.dumps([req], protocol=protocol), request_pickle_globals(SamplingParams)
    ).load()[0]
    torch.testing.assert_close(restored.prompt_embeds[0], tensor)
    np.testing.assert_array_equal(restored.condition_inputs["array"], array)
    assert restored.condition_inputs["scalar"] == np.float32(1)
    assert restored.condition_inputs["ipc"] == ipc
    assert restored.condition_image.tobytes() == image.tobytes()
    assert restored.generator.initial_seed() == 42
    assert restored.metrics.request_id == "media"
    assert not restored.trace_ctx.tracing_enable


class _UnsafePayload:
    def __reduce__(self):
        return eval, ("1 + 1",)


class _NestedTorchPayload:
    def __reduce__(self):
        buffer = io.BytesIO()
        torch.save(_UnsafePayload(), buffer)
        return torch.storage._load_from_bytes, (buffer.getvalue(),)


@pytest.mark.parametrize("obj", [_UnsafePayload(), _NestedTorchPayload()])
def test_reject_unsafe_globals_including_nested_torch_load(obj):
    with pytest.raises(
        (RuntimeError, pickle.UnpicklingError), match="[Bb]locked|Weights only"
    ):
        RequestUnpickler(
            pickle.dumps(obj), request_pickle_globals(SamplingParams)
        ).load()


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
