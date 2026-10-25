"""Image responses carry the bytes encoded on the output rank instead of reading a file back."""

import asyncio
import base64
import copy
import os
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import AsyncMock

import imageio
import numpy as np
import pytest
import torch
import zmq
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from sglang.multimodal_gen.configs.sample.sampling_params import (
    DataType,
    SamplingParams,
)
from sglang.multimodal_gen.runtime.disaggregation.transport.codec import unpack_tensors
from sglang.multimodal_gen.runtime.entrypoints.openai import image_api
from sglang.multimodal_gen.runtime.entrypoints.openai import utils as openai_utils
from sglang.multimodal_gen.runtime.entrypoints.utils import (
    MaterializedOutput,
    save_materialized_output,
    save_outputs,
)
from sglang.multimodal_gen.runtime.managers.gpu_worker import GPUWorker
from sglang.multimodal_gen.runtime.managers.scheduler import Scheduler
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import (
    OutputBatch,
    Req,
)
from sglang.multimodal_gen.runtime.scheduler_client import async_scheduler_client


def _frame(channels: int) -> np.ndarray:
    shape = (17, 23) if channels == 0 else (17, 23, channels)
    return np.random.default_rng(channels).integers(0, 256, shape, dtype=np.uint8)


def _decode(data: bytes) -> np.ndarray:
    with Image.open(BytesIO(data)) as image:
        return np.asarray(image)


def _uint8_hwc(output: torch.Tensor) -> np.ndarray:
    return (output * 255).clamp(0, 255).to(torch.uint8).permute(1, 2, 0).numpy()


def _encode(frame: np.ndarray, path: str, **kwargs) -> bytes:
    return save_materialized_output(
        MaterializedOutput(sample=None, frames=[frame]),
        DataType.IMAGE,
        path,
        return_image_bytes=True,
        **kwargs,
    )


# channels 0 is a 2-D grayscale frame, 1 a single-channel HWC frame
@pytest.mark.parametrize("channels", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("output_compression", [None, 75, 100])
def test_returned_png_holds_the_source_pixels_and_writes_nothing(
    tmp_path, channels, output_compression
):
    frame = _frame(channels)

    data = _encode(
        frame,
        str(tmp_path / "out.png"),
        save_output=False,
        output_compression=output_compression,
    )

    np.testing.assert_array_equal(_decode(data), frame.reshape(17, 23, -1).squeeze())
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("extension", ["png", "jpg", "webp"])
def test_saved_file_holds_the_returned_bytes(tmp_path, extension):
    path = tmp_path / f"out.{extension}"

    data = _encode(_frame(3), str(path))

    assert path.read_bytes() == data


@pytest.mark.parametrize("extension", ["jpg", "webp"])
@pytest.mark.parametrize("output_compression", [None, 90])
def test_lossy_formats_keep_the_file_writer_bytes(
    tmp_path, extension, output_compression
):
    frame = _frame(3)
    reference = tmp_path / f"reference.{extension}"
    imageio.imwrite(
        reference,
        frame,
        quality=75 if output_compression is None else output_compression,
    )

    data = _encode(
        frame,
        str(tmp_path / f"out.{extension}"),
        save_output=False,
        output_compression=output_compression,
    )

    assert data == reference.read_bytes()


def test_save_outputs_returns_one_image_per_output(tmp_path):
    outputs = [torch.rand(3, 8, 8), torch.rand(3, 8, 8)]
    image_bytes = []

    paths = save_outputs(
        outputs,
        DataType.IMAGE,
        1,
        False,
        lambda idx: str(tmp_path / f"out_{idx}.png"),
        image_bytes_out=image_bytes,
    )

    assert paths == [str(tmp_path / "out_0.png"), str(tmp_path / "out_1.png")]
    for output, data in zip(outputs, image_bytes, strict=True):
        np.testing.assert_array_equal(_decode(data), _uint8_hwc(output))
    assert list(tmp_path.iterdir()) == []


def _image_req(tmp_path, *, save_output: bool, return_image_bytes: bool) -> Req:
    return Req(
        sampling_params=SamplingParams(
            data_type=DataType.IMAGE,
            save_output=save_output,
            return_image_bytes=return_image_bytes,
            output_path=str(tmp_path),
            output_file_name="out.png",
        )
    )


@pytest.mark.parametrize(
    ("save_output", "return_image_bytes", "encodes"),
    [
        (True, False, True),
        (False, True, True),
        (True, True, True),
        (False, False, False),
    ],
)
def test_output_rank_encodes_when_saving_or_returning_bytes(
    tmp_path, save_output, return_image_bytes, encodes
):
    req = _image_req(
        tmp_path, save_output=save_output, return_image_bytes=return_image_bytes
    )

    assert GPUWorker._encodes_on_output_rank(req) is encodes


@pytest.mark.parametrize("save_output", [False, True])
def test_worker_returns_image_bytes_and_writes_only_when_saving(tmp_path, save_output):
    req = _image_req(tmp_path, save_output=save_output, return_image_bytes=True)
    output = torch.rand(3, 8, 8)
    batch = OutputBatch(output=[output])

    GPUWorker._save_output_paths(SimpleNamespace(is_output_rank=True), req, batch)

    path = tmp_path / "out.png"
    assert batch.output_file_paths == [str(path)]
    (data,) = batch.output_image_bytes
    np.testing.assert_array_equal(_decode(data), _uint8_hwc(output))
    assert path.exists() == save_output
    if save_output:
        assert path.read_bytes() == data


def test_worker_without_image_bytes_request_returns_none(tmp_path):
    req = _image_req(tmp_path, save_output=True, return_image_bytes=False)
    batch = OutputBatch(output=[torch.rand(3, 8, 8)])

    GPUWorker._save_output_paths(SimpleNamespace(is_output_rank=True), req, batch)

    assert batch.output_image_bytes is None
    assert (tmp_path / "out.png").exists()


def test_dynamic_batch_split_keeps_each_requests_image_bytes():
    scheduler = object.__new__(Scheduler)
    reqs = [Req(sampling_params=SamplingParams()) for _ in range(2)]

    outputs = scheduler._split_batched_output(
        OutputBatch(
            output_file_paths=["a.png", "b.png"], output_image_bytes=[b"a", b"b"]
        ),
        reqs,
    )

    assert [output.output_image_bytes for output in outputs] == [[b"a"], [b"b"]]


@pytest.mark.parametrize("sequential", [False, True])
@pytest.mark.parametrize("output_count", [1, 2])
@pytest.mark.parametrize("persistent", [False, True])
def test_disagg_returns_pixels_for_head_encoding(
    monkeypatch, tmp_path, sequential, output_count, persistent
):
    req = _image_req(tmp_path, save_output=persistent, return_image_bytes=True)
    req.num_outputs_per_prompt = output_count
    req.prior_token_id = torch.arange(output_count).reshape(-1, 1)
    head_req = copy.deepcopy(req)
    pixels = torch.rand(output_count, 3, 8, 8)
    worker = object.__new__(GPUWorker)
    worker.is_output_rank = True

    def compute(reqs):
        (request,) = reqs
        output = OutputBatch(output=pixels[request.prior_token_id.flatten()])
        worker._materialize_output_transport(
            output, request, lambda batch: worker._save_output_paths(request, batch)
        )
        return output

    # Stub only the model computation, not materialization or the wire format.
    monkeypatch.setattr(worker, "execute_forward", compute)
    monkeypatch.setattr(
        worker,
        "execute_forward_sequentially",
        lambda reqs: (compute([r]) for r in reqs),
    )
    scheduler = object.__new__(Scheduler)
    scheduler.worker = worker
    scheduler._disagg_metrics = None
    scheduler.server_args = SimpleNamespace(
        pipeline_config=SimpleNamespace(
            supports_sequential_multi_output_inference=lambda: sequential
        )
    )
    with zmq.Context() as context:
        with context.socket(zmq.PAIR) as sender, context.socket(zmq.PAIR) as receiver:
            sender.bind("inproc://image-result")
            receiver.connect("inproc://image-result")
            receiver.setsockopt(zmq.RCVTIMEO, 1000)
            scheduler._pool_result_push = sender
            scheduler._execute_glm_distributed_denoiser_request(req, "image-request")
            tensors, scalars = unpack_tensors(receiver.recv_multipart(), device="cpu")

    assert scalars["request_id"] == "image-request"
    assert "error" not in scalars
    torch.testing.assert_close(tensors["output"], pixels, rtol=0, atol=0)
    assert list(tmp_path.iterdir()) == []

    monkeypatch.setattr(
        openai_utils,
        "get_global_server_args",
        lambda: SimpleNamespace(batching_max_size=1),
    )
    client = SimpleNamespace(forward=AsyncMock(return_value=OutputBatch(**tensors)))
    paths, result = asyncio.run(openai_utils.process_generation_batch(client, head_req))
    assert len(paths) == len(result.output_image_bytes) == output_count
    for path, data, output in zip(
        paths, result.output_image_bytes, pixels, strict=True
    ):
        np.testing.assert_array_equal(_decode(data), _uint8_hwc(output))
        assert os.path.exists(path) == persistent
        if persistent:
            with open(path, "rb") as file:
                assert file.read() == data


@pytest.mark.parametrize("endpoint", ["generations", "edits"])
@pytest.mark.parametrize("persistent", [False, True])
@pytest.mark.parametrize("error", [None, "output save failed"])
def test_b64_response_publishes_only_successful_outputs(
    monkeypatch, tmp_path, endpoint, persistent, error
):
    output_path = str(tmp_path / "outputs") if persistent else None
    monkeypatch.setattr(
        image_api,
        "get_global_server_args",
        lambda: SimpleNamespace(output_path=output_path, input_save_path=None),
    )
    monkeypatch.setattr(
        image_api, "resolve_sampling_params_cls", lambda _: SamplingParams
    )
    monkeypatch.setattr(image_api, "maybe_enhance_prompt", AsyncMock(return_value="p"))
    monkeypatch.setattr(image_api.cloud_storage, "is_enabled", lambda: False)
    captured = {}

    def build(request_id, **kwargs):
        captured.update(kwargs)
        return SamplingParams()

    monkeypatch.setattr(image_api, "_build_request_sampling_params", build)
    monkeypatch.setattr(
        image_api, "prepare_request", lambda **_: Req(sampling_params=SamplingParams())
    )
    monkeypatch.setattr(
        openai_utils,
        "get_global_server_args",
        lambda: SimpleNamespace(batching_max_size=1),
    )
    png = BytesIO()
    Image.fromarray(_frame(3)).save(png, format="PNG")
    png = png.getvalue()

    async def generate(_batches):
        # no file exists at the returned path; the response must not need one
        path = os.path.join(captured["output_path"], captured["output_file_name"])
        return OutputBatch(
            output_file_paths=[path], output_image_bytes=[png], error=error
        )

    monkeypatch.setattr(async_scheduler_client, "forward", generate)
    app = FastAPI()
    app.include_router(image_api.router)
    with TestClient(app, raise_server_exceptions=False) as client:
        if endpoint == "generations":
            response = client.post(
                "/v1/images/generations",
                json={"prompt": "p", "response_format": "b64_json"},
            )
        else:
            response = client.post(
                "/v1/images/edits",
                data={"prompt": "p", "response_format": "b64_json"},
                files={"image": ("reference.png", png, "image/png")},
            )

    if error:
        assert response.status_code == 500, response.text
        assert "b64_json" not in response.text
        return
    assert response.status_code == 200, response.text
    (item,) = response.json()["data"]
    assert base64.b64decode(item["b64_json"]) == png
    assert captured["return_image_bytes"] is True
    assert captured["save_output"] is persistent
