# SPDX-License-Identifier: Apache-2.0

import asyncio
import base64
import io
import json
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image
from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import (
    ImageGenerationsRequest,
)
from sglang.srt.entrypoints.openai.serving_sensenova_u1_images import (
    _clear_u1_prefix_cache_for_test,
    _parse_u1_image_size,
    _single_seed,
    _u1_next_t_index,
    _u1_tensor_bytes_to_png,
    _validate_u1_image_request,
    serve_sensenova_u1_image_edit,
    serve_sensenova_u1_image_generation,
)
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.srt.model_executor.model_runner_components.cuda_graph_setup import (
    capture_prefill_graph_for_model,
)
from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import (
    PrefillCudaGraphRunner,
)
from starlette.datastructures import UploadFile


class _FakeTokenizer:
    eos_token_id = 151645

    def __call__(self, text, add_special_tokens=False):
        del add_special_tokens
        return {"input_ids": list(range(len(text.split())))}

    @staticmethod
    def convert_tokens_to_ids(token):
        return {
            "<img>": 151668,
            "<IMG_CONTEXT>": 151669,
        }[token]

    @staticmethod
    def convert_ids_to_tokens(token_id):
        assert token_id == 151645
        return "<|endoftext|>"


class _FakeTokenizerManager:
    def __init__(self):
        self.model_config = SimpleNamespace(
            hf_config=SimpleNamespace(model_type="neo_chat")
        )
        self.tokenizer = _FakeTokenizer()
        self.requests = []
        self.raw_requests = []
        self.aborted_rids = []

    async def generate_request(self, request, raw_request):
        self.requests.append(request)
        self.raw_requests.append(raw_request)
        custom_params = request.sampling_params.get("custom_params") or {}
        if "sensenova_u1_flow" not in custom_params:
            prompt_ids = (
                request.input_ids if request.input_ids is not None else list(range(20))
            )
            yield {
                "meta_info": {"prompt_tokens": len(prompt_ids)},
                "prompt_token_ids": prompt_ids,
            }
            return

        flow = custom_params["sensenova_u1_flow"]
        tensor = np.zeros(
            (1, 3, flow["height"], flow["width"]),
            dtype=np.float16,
        )
        yield {
            "meta_info": {
                "prompt_tokens": flow["image_start"] + flow["image_tokens"],
                "cached_tokens": flow["image_start"],
                "sensenova_u1_flow_image_b64": [
                    base64.b64encode(tensor.tobytes()).decode("ascii")
                ],
                "sensenova_u1_flow_image_shape": [list(tensor.shape)],
            }
        }

    def abort_request(self, rid):
        self.aborted_rids.append(rid)


class _FakeRawRequest:
    def __init__(self, disconnected=False):
        self.disconnected = disconnected

    async def is_disconnected(self):
        return self.disconnected


class _TrackingBytesIO(io.BytesIO):
    def __init__(self, value):
        super().__init__(value)
        self.bytes_read = 0

    def read(self, size=-1):
        chunk = super().read(size)
        self.bytes_read += len(chunk)
        return chunk


class _BlockingTokenizerManager(_FakeTokenizerManager):
    def __init__(self, block_stage):
        super().__init__()
        self.block_stage = block_stage
        self.started = asyncio.Event()

    async def generate_request(self, request, raw_request):
        custom_params = request.sampling_params.get("custom_params") or {}
        stage = "flow" if "sensenova_u1_flow" in custom_params else "prefix"
        if stage == self.block_stage:
            self.requests.append(request)
            self.raw_requests.append(raw_request)
            self.started.set()
            await asyncio.Future()
            return

        async for result in super().generate_request(request, raw_request):
            yield result


class _FailingTokenizerManager(_FakeTokenizerManager):
    def __init__(self, *, disconnected):
        super().__init__()
        self.disconnected = disconnected

    async def generate_request(self, request, raw_request):
        self.requests.append(request)
        self.raw_requests.append(raw_request)
        raw_request.disconnected = self.disconnected
        raise ValueError("stage failed")
        yield


def _raw_request():
    return _FakeRawRequest()


def test_u1_image_size_parsing_prefers_explicit_dimensions() -> None:
    request = ImageGenerationsRequest(
        prompt="test",
        size="1024x1024",
        width=64,
        height=96,
    )

    assert _parse_u1_image_size(request) == (64, 96)


def test_u1_tensor_bytes_encode_png() -> None:
    tensor = np.zeros((1, 3, 32, 64), dtype=np.float16)
    png = _u1_tensor_bytes_to_png(tensor.tobytes(), list(tensor.shape))

    image = Image.open(io.BytesIO(png))
    assert image.size == (64, 32)
    assert np.asarray(image).mean() == 128


def test_u1_next_t_index_collapses_input_image_span() -> None:
    assert (
        _u1_next_t_index(
            [10, 151668, 151669, 151669, 11],
            image_start_token_id=151668,
            image_context_token_id=151669,
        )
        == 4
    )


def test_u1_image_generation_uses_native_flow_contract() -> None:
    _clear_u1_prefix_cache_for_test()
    manager = _FakeTokenizerManager()
    raw_request = _raw_request()
    response = asyncio.run(
        serve_sensenova_u1_image_generation(
            manager,
            ImageGenerationsRequest(
                prompt="a red square",
                size="64x64",
                response_format="b64_json",
                seed=7,
                num_inference_steps=2,
            ),
            raw_request=raw_request,
        )
    )

    assert len(manager.requests) == 2
    assert manager.raw_requests == [raw_request, raw_request]
    flow_request = manager.requests[1]
    flow = flow_request.sampling_params["custom_params"]["sensenova_u1_flow"]
    assert flow["image_tokens"] == 4
    assert response.usage.prompt_tokens_details.cached_tokens == flow["image_start"]
    png = base64.b64decode(response.data[0].b64_json)
    assert Image.open(io.BytesIO(png)).size == (64, 64)


def test_u1_image_edit_reuses_multimodal_flow_contract() -> None:
    _clear_u1_prefix_cache_for_test()
    manager = _FakeTokenizerManager()
    raw_request = _raw_request()
    response = asyncio.run(
        serve_sensenova_u1_image_edit(
            manager,
            ImageGenerationsRequest(
                prompt="paint it as a watercolor",
                size="64x64",
                response_format="b64_json",
                seed=7,
                num_inference_steps=2,
            ),
            image_data=[b"fake-image"],
            raw_request=raw_request,
        )
    )

    assert len(manager.requests) == 2
    assert manager.raw_requests == [raw_request, raw_request]
    assert manager.requests[0].image_data == [b"fake-image"]
    assert manager.requests[1].image_data == [b"fake-image"]
    assert manager.requests[1].text.endswith("<|endoftext|>" * 4)
    assert response.usage.prompt_tokens_details.cached_tokens == 20


def test_u1_image_generation_skips_warmed_prefix_prime() -> None:
    _clear_u1_prefix_cache_for_test()
    manager = _FakeTokenizerManager()
    request = ImageGenerationsRequest(
        prompt="a red square",
        size="64x64",
        response_format="b64_json",
        seed=7,
        num_inference_steps=2,
    )
    raw_request = _raw_request()

    asyncio.run(
        serve_sensenova_u1_image_generation(
            manager,
            request,
            raw_request=raw_request,
        )
    )
    asyncio.run(
        serve_sensenova_u1_image_generation(
            manager,
            request,
            raw_request=raw_request,
        )
    )

    assert len(manager.requests) == 3
    assert manager.requests[0].sampling_params.get("custom_params") is None
    assert all(
        "sensenova_u1_flow"
        in generated_request.sampling_params.get("custom_params", {})
        for generated_request in manager.requests[1:]
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("n", 0, "supports n=1"),
        ("n", -1, "supports n=1"),
        ("num_inference_steps", 0, "num_steps must be positive"),
        ("num_inference_steps", -1, "num_steps must be positive"),
        ("num_inference_steps", 65, "num_steps exceeds"),
        ("flow_shift", 0.0, "flow_shift must be a positive finite number"),
        ("flow_shift", -1.0, "flow_shift must be a positive finite number"),
        ("flow_shift", float("nan"), "flow_shift must be a positive finite number"),
        ("flow_shift", float("inf"), "flow_shift must be a positive finite number"),
        ("guidance_scale", 0.0, "supports guidance_scale=1"),
        ("guidance_scale", float("nan"), "supports guidance_scale=1"),
    ],
)
def test_u1_image_request_rejects_invalid_numeric_values(
    field,
    value,
    message,
) -> None:
    request = ImageGenerationsRequest(prompt="test", **{field: value})

    with pytest.raises(ValueError, match=message):
        _validate_u1_image_request(request)


@pytest.mark.parametrize("seed", [-1, 2**63])
def test_u1_image_request_rejects_out_of_range_seed(seed) -> None:
    with pytest.raises(ValueError, match=r"seed must be in \[0, 2\*\*63\)"):
        _single_seed(seed)


def test_u1_image_request_preserves_explicit_valid_zero_seed() -> None:
    assert _single_seed(0) == 0


def test_u1_image_generation_aborts_prefix_on_cancellation() -> None:
    async def scenario():
        _clear_u1_prefix_cache_for_test()
        manager = _BlockingTokenizerManager("prefix")
        task = asyncio.create_task(
            serve_sensenova_u1_image_generation(
                manager,
                ImageGenerationsRequest(
                    prompt="a red square",
                    size="64x64",
                    response_format="b64_json",
                ),
                raw_request=_raw_request(),
            )
        )
        await asyncio.wait_for(manager.started.wait(), timeout=1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert len(manager.aborted_rids) == 1
        assert manager.aborted_rids[0].endswith("-prefix")

    asyncio.run(scenario())


def test_u1_image_generation_aborts_flow_on_cancellation() -> None:
    async def scenario():
        _clear_u1_prefix_cache_for_test()
        manager = _BlockingTokenizerManager("flow")
        task = asyncio.create_task(
            serve_sensenova_u1_image_generation(
                manager,
                ImageGenerationsRequest(
                    prompt="a red square",
                    size="64x64",
                    response_format="b64_json",
                ),
                raw_request=_raw_request(),
            )
        )
        await asyncio.wait_for(manager.started.wait(), timeout=1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert len(manager.aborted_rids) == 1
        assert manager.aborted_rids[0].endswith("-flow")

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("disconnected", "error_type"),
    [
        (False, ValueError),
        (True, asyncio.CancelledError),
    ],
)
def test_u1_image_generation_aborts_active_rid_on_stage_failure(
    disconnected,
    error_type,
) -> None:
    async def scenario():
        _clear_u1_prefix_cache_for_test()
        manager = _FailingTokenizerManager(disconnected=disconnected)
        with pytest.raises(error_type):
            await serve_sensenova_u1_image_generation(
                manager,
                ImageGenerationsRequest(
                    prompt="a red square",
                    size="64x64",
                    response_format="b64_json",
                ),
                raw_request=_raw_request(),
            )
        assert len(manager.aborted_rids) == 1
        assert manager.aborted_rids[0].endswith("-prefix")

    asyncio.run(scenario())


def test_u1_image_edit_rejects_oversized_multipart_and_closes_upload() -> None:
    async def scenario():
        from sglang.srt.entrypoints import http_server

        source = _TrackingBytesIO(b"x" * 17)
        upload = UploadFile(
            source,
            filename="oversized.png",
        )
        manager = SimpleNamespace(
            server_args=SimpleNamespace(media_url_max_file_size_mb=0)
        )
        global_state = SimpleNamespace(tokenizer_manager=manager)
        with patch.object(http_server, "_global_state", global_state):
            response = await http_server.sensenova_u1_image_edits(
                request=_raw_request(),
                image=upload,
                prompt="test",
                n=1,
                response_format="b64_json",
                size="64x64",
                output_format="png",
                seed=0,
                guidance_scale=1,
                num_inference_steps=2,
                mask=None,
            )
        return response, upload, source

    response, upload, source = asyncio.run(scenario())

    assert response.status_code == 400
    assert (
        "image upload exceeds the maximum 0 bytes"
        in json.loads(response.body)["error"]["message"]
    )
    assert source.bytes_read == 1
    assert upload.file.closed


def test_u1_flow_selects_matching_prefill_cuda_graph_variant() -> None:
    forward_batch = SimpleNamespace(
        sampling_info=SimpleNamespace(
            custom_params=[
                {
                    "__sglang_prefill_cuda_graph_variant": "sensenova_u1_flow",
                    "sensenova_u1_flow": {
                        "image_start": 264,
                        "image_tokens": 64,
                    },
                }
            ]
        ),
        global_num_tokens_cpu=None,
        batch_size=1,
        input_ids=np.zeros(64, dtype=np.int64),
        input_embeds=None,
        replace_embeds=None,
        extend_seq_lens_cpu=[64],
        extend_prefix_lens_cpu=[264],
        forward_mode=SimpleNamespace(is_target_verify=lambda: False),
        capture_hidden_mode=None,
        return_logprob=False,
    )
    runner = object.__new__(PrefillCudaGraphRunner)
    runner.model_runner = SimpleNamespace(
        prefill_cuda_graph_variant="sensenova_u1_flow",
        server_args=SimpleNamespace(enable_lora=False),
    )
    runner.enable_lora = False
    runner.enable_cp_v2_bcg_capture = False
    runner._capture_lora = False
    runner._capture_chunked_prefix = False
    runner._has_inactive_dp_rank = lambda _: False
    runner.can_replay_locally = lambda **_: True
    runner.capture_num_tokens = [64]

    assert runner.can_run_graph(forward_batch)

    forward_batch.extend_prefix_lens_cpu = [263]
    assert not runner.can_run_graph(forward_batch)

    forward_batch.extend_prefix_lens_cpu = [264]
    forward_batch.extend_seq_lens_cpu = [32]
    forward_batch.input_ids = np.zeros(32, dtype=np.int64)
    assert not runner.can_run_graph(forward_batch)


def test_u1_prefill_graph_trim_preserves_customized_info() -> None:
    runner = object.__new__(PrefillCudaGraphRunner)
    runner.raw_bs = 1
    runner.raw_num_tokens = 64
    runner._is_full_backend = False
    runner.model_runner = SimpleNamespace(
        spec_algorithm=SimpleNamespace(is_speculative=lambda: False)
    )
    customized_info = {
        "sensenova_u1_flow_image_b64": ["encoded-image"],
    }

    trimmed = runner._trim_logits_output(
        LogitsProcessorOutput(
            next_token_logits=np.zeros((64, 4)),
            customized_info=customized_info,
        )
    )

    assert trimmed.customized_info is customized_info


def test_u1_prefill_capture_uses_model_declared_variant() -> None:
    text_model = SimpleNamespace(
        prefill_cuda_graph_capture_variant="sensenova_u1_flow",
        prefill_cuda_graph_capture_flag=("force_mot_gen_for_prefill_graph_capture"),
        force_mot_gen_for_prefill_graph_capture=False,
    )
    model_runner = SimpleNamespace(
        model=SimpleNamespace(
            language_model=SimpleNamespace(model=text_model),
        ),
        server_args=SimpleNamespace(
            cuda_graph_config=SimpleNamespace(
                prefill=SimpleNamespace(backend=Backend.BREAKABLE),
            )
        ),
    )
    capture_result = object()

    def fake_capture_prefill_graph(**kwargs):
        assert kwargs["model_runner"] is model_runner
        assert text_model.force_mot_gen_for_prefill_graph_capture
        assert model_runner.prefill_cuda_graph_variant == "sensenova_u1_flow"
        return capture_result

    with patch(
        "sglang.srt.model_executor.model_runner_components.cuda_graph_setup."
        "capture_prefill_graph",
        side_effect=fake_capture_prefill_graph,
    ):
        assert (
            capture_prefill_graph_for_model(
                model_runner=model_runner,
                eager_runner=object(),
            )
            is capture_result
        )

    assert not text_model.force_mot_gen_for_prefill_graph_capture
