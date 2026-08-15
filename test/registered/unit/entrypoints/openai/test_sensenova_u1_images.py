# SPDX-License-Identifier: Apache-2.0

import asyncio
import base64
import io
from types import SimpleNamespace

import numpy as np
from PIL import Image
from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import (
    ImageGenerationsRequest,
)
from sglang.srt.entrypoints.openai.serving_sensenova_u1_images import (
    _clear_u1_prefix_cache_for_test,
    _parse_u1_image_size,
    _u1_next_t_index,
    _u1_tensor_bytes_to_png,
    serve_sensenova_u1_image_edit,
    serve_sensenova_u1_image_generation,
)
from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import (
    PrefillCudaGraphRunner,
)


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

    async def generate_request(self, request, raw_request):
        del raw_request
        self.requests.append(request)
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
        )
    )

    assert len(manager.requests) == 2
    flow_request = manager.requests[1]
    flow = flow_request.sampling_params["custom_params"]["sensenova_u1_flow"]
    assert flow["image_tokens"] == 4
    assert response.usage.prompt_tokens_details.cached_tokens == flow["image_start"]
    png = base64.b64decode(response.data[0].b64_json)
    assert Image.open(io.BytesIO(png)).size == (64, 64)


def test_u1_image_edit_reuses_multimodal_flow_contract() -> None:
    _clear_u1_prefix_cache_for_test()
    manager = _FakeTokenizerManager()
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
        )
    )

    assert len(manager.requests) == 2
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

    asyncio.run(serve_sensenova_u1_image_generation(manager, request))
    asyncio.run(serve_sensenova_u1_image_generation(manager, request))

    assert len(manager.requests) == 3
    assert manager.requests[0].sampling_params.get("custom_params") is None
    assert all(
        "sensenova_u1_flow"
        in generated_request.sampling_params.get("custom_params", {})
        for generated_request in manager.requests[1:]
    )


def test_u1_flow_disables_prefill_cuda_graph_replay() -> None:
    forward_batch = SimpleNamespace(
        sampling_info=SimpleNamespace(
            custom_params=[{"__sglang_disable_prefill_cuda_graph": True}]
        )
    )

    assert not PrefillCudaGraphRunner.can_run_graph(object(), forward_batch)
