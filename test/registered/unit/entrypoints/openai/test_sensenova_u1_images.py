# SPDX-License-Identifier: Apache-2.0

import asyncio
import base64
import io
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image
from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import (
    ImageGenerationsRequest,
)
from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.srt.entrypoints.openai.serving_sensenova_u1_images import (
    _clear_u1_prefix_cache_for_test,
    _parse_u1_image_size,
    _u1_next_t_index,
    _u1_tensor_bytes_to_png,
    serve_sensenova_u1_image_edit,
    serve_sensenova_u1_image_generation,
)
from sglang.srt.model_executor.model_runner_components.cuda_graph_setup import (
    capture_prefill_graph_for_model,
)
from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import (
    PrefillCudaGraphRunner,
)
from sglang.srt.layers.logits_processor import LogitsProcessorOutput


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
        prefill_cuda_graph_capture_flag=(
            "force_mot_gen_for_prefill_graph_capture"
        ),
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
