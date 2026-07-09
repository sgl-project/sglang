# SPDX-License-Identifier: Apache-2.0

import asyncio
import io

from starlette.datastructures import UploadFile as StarletteUploadFile

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.runtime.entrypoints.openai.utils import (
    _parse_size_or_raise,
    _save_upload_to_path,
    _validate_positive_int,
    process_generation_batch,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import (
    OutputBatch,
    Req,
)


class _FakeSchedulerClient:
    def __init__(self, result):
        self.result = result
        self.forward_arg = None

    async def forward(self, arg):
        self.forward_arg = arg
        return self.result


def test_save_upload_to_path_accepts_starlette_upload_file(tmp_path):
    upload = StarletteUploadFile(
        io.BytesIO(b"image-bytes"),
        filename="input.png",
    )
    target_path = tmp_path / "input.png"

    saved_path = asyncio.run(_save_upload_to_path(upload, str(target_path)))

    assert saved_path == str(target_path)
    assert target_path.read_bytes() == b"image-bytes"


def test_parse_size_or_raise_accepts_positive_size():
    assert _parse_size_or_raise("512x768") == (512, 768)


def test_parse_size_or_raise_rejects_malformed_size():
    try:
        _parse_size_or_raise("not-a-size")
    except Exception as exc:
        assert exc.status_code == 400
        assert "positive WIDTHxHEIGHT" in exc.detail
    else:
        raise AssertionError("expected bad request")


def test_parse_size_or_raise_rejects_non_positive_size():
    try:
        _parse_size_or_raise("0x512")
    except Exception as exc:
        assert exc.status_code == 400
        assert "positive WIDTHxHEIGHT" in exc.detail
    else:
        raise AssertionError("expected bad request")


def test_validate_positive_int_rejects_non_positive_sampling_fields():
    try:
        _validate_positive_int({"num_frames": 0}, "num_frames")
    except Exception as exc:
        assert exc.status_code == 400
        assert "num_frames must be positive" in exc.detail
    else:
        raise AssertionError("expected bad request")


def test_process_generation_batch_expands_multi_output_request(tmp_path):
    num_outputs = 2
    req = Req(
        sampling_params=SamplingParams(
            request_id="rid",
            prompt="draw a cube",
            output_path=str(tmp_path),
            output_file_name="image.png",
            num_outputs_per_prompt=num_outputs,
            seed=42,
        )
    )
    result = OutputBatch(
        output_file_paths=[
            str(tmp_path / f"image_{idx}.png") for idx in range(num_outputs)
        ]
    )
    scheduler_client = _FakeSchedulerClient(result)

    paths, returned_result = asyncio.run(
        process_generation_batch(scheduler_client, req)
    )

    assert returned_result is result
    assert paths == result.output_file_paths
    assert len(scheduler_client.forward_arg) == 1
    expanded_group = scheduler_client.forward_arg[0]
    assert len(expanded_group) == num_outputs
    assert [item.seed for item in expanded_group] == [42, 43]
    assert [item.num_outputs_per_prompt for item in expanded_group] == [1, 1]
    assert [item.output_file_name for item in expanded_group] == [
        f"image_{idx}.png" for idx in range(num_outputs)
    ]
