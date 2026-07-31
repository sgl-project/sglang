# SPDX-License-Identifier: Apache-2.0

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from starlette.requests import Request

from sglang.multimodal_gen.configs.pipeline_configs.bagel import (
    BagelThinkingPipelineConfig,
)
from sglang.multimodal_gen.configs.sample.bagel import BagelThinkingSamplingParams
from sglang.multimodal_gen.runtime.entrypoints.cli.generate import (
    _resolve_cli_sampling_params_cls,
)
from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import DiffGenerator
from sglang.multimodal_gen.runtime.entrypoints.openai.image_api import (
    _build_image_response_kwargs,
    generations,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import (
    ImageGenerationsRequest,
)
from sglang.multimodal_gen.runtime.managers.gpu_worker import GPUWorker
from sglang.multimodal_gen.runtime.models.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch, Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.decoding import DecodingStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.bagel import (
    GEN_THINK_SYSTEM_PROMPT,
    BagelThinkingBeforeDenoisingStage,
)
from sglang.multimodal_gen.runtime.warmup_request_builder import build_warmup_reqs


class _CapturedSampling(Exception):
    pass


class _Tokenizer:
    token_ids = {
        "<|im_start|>": 151644,
        "<|im_end|>": 151645,
        "<|vision_start|>": 151652,
        "<|vision_end|>": 151653,
    }
    unk_token_id = 0
    unk_token = "<unk>"

    def __init__(self) -> None:
        self.encoded_texts: list[str] = []

    def convert_tokens_to_ids(self, token: str) -> int:
        return self.token_ids.get(token, self.unk_token_id)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        assert not add_special_tokens
        if text in self.token_ids:
            return [self.token_ids[text]]
        self.encoded_texts.append(text)
        return [17, 23]

    def decode(self, token_ids: list[int], skip_special_tokens: bool = False) -> str:
        assert not skip_special_tokens
        assert token_ids[0] == self.token_ids["<|im_start|>"]
        return "<|im_start|><think>plan</think>"


class _Transformer:
    def __init__(self, config: BagelThinkingPipelineConfig) -> None:
        self.config = config.dit_config
        self.system_ids: torch.Tensor | None = None
        self.user_ids: torch.Tensor | None = None
        self.generated_kwargs: dict = {}
        self.thought_ids: torch.Tensor | None = None

    def prepare_thinking_prefixes(
        self, system_ids: torch.Tensor, user_ids: torch.Tensor
    ) -> tuple[object, object]:
        self.system_ids = system_ids.clone()
        self.user_ids = user_ids.clone()
        return object(), object()

    def generate_text(self, _prefix, **kwargs) -> torch.Tensor:
        self.generated_kwargs = kwargs
        return torch.tensor([151644, 17, 23])

    def build_thinking_context(
        self, _system_prefix, _user_prefix, thought_ids: torch.Tensor, **kwargs
    ) -> SimpleNamespace:
        self.thought_ids = thought_ids.clone()
        return SimpleNamespace(is_thinking=True, has_three_way_cfg=True, **kwargs)


def _server_args() -> SimpleNamespace:
    return SimpleNamespace(pipeline_config=BagelThinkingPipelineConfig())


def _batch(seed: int = 11, max_think_tokens: int = 7) -> Req:
    params = BagelThinkingSamplingParams(
        prompt="draw a blue fox",
        height=32,
        width=32,
        seed=seed,
        num_inference_steps=4,
        max_think_tokens=max_think_tokens,
        save_output=False,
    )
    batch = Req(sampling_params=params)
    batch.generator = [torch.Generator("cpu").manual_seed(seed)]
    batch.seeds = [seed]
    return batch


def _stage() -> BagelThinkingBeforeDenoisingStage:
    args = _server_args()
    stage = BagelThinkingBeforeDenoisingStage(
        transformer=_Transformer(args.pipeline_config),
        tokenizer=_Tokenizer(),
        scheduler=FlowMatchEulerDiscreteScheduler(shift=1.0),
    )
    stage.server_args = args
    stage._registered_stage_name = "bagel_thinking_before_denoising_stage"
    return stage


def test_thinking_stage_rewraps_clean_plan_and_preserves_noise_stream() -> None:
    stage = _stage()
    batch = _batch()
    expected_generator = torch.Generator("cpu").manual_seed(11)
    expected_latents = torch.randn(
        4, 64, generator=expected_generator, dtype=torch.float32
    )

    with patch(
        "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages."
        "bagel.get_local_torch_device",
        return_value=torch.device("cpu"),
    ):
        output = stage.forward(batch, stage.server_args)

    assert stage.tokenizer.encoded_texts == [
        GEN_THINK_SYSTEM_PROMPT,
        "draw a blue fox",
        "<think>plan</think>",
    ]
    assert GEN_THINK_SYSTEM_PROMPT == (
        "You should first think about the planning process in the mind and then "
        "generate the image.\n"
        "The planning process is enclosed within <think> </think> tags, i.e. "
        "<think> planning process here </think> image here"
    )
    assert stage.transformer.generated_kwargs == {
        "bos_token_id": 151644,
        "eos_token_id": 151645,
        "max_length": 7,
        "do_sample": False,
        "temperature": 0.3,
        "seed": 11,
    }
    assert stage.transformer.thought_ids is not None
    assert stage.transformer.thought_ids.tolist() == [151644, 17, 23, 151645]
    assert output.extra["bagel_context"].is_thinking
    assert output.extra["revised_prompt"] == ("draw a blue fox\n<think>plan</think>")
    torch.testing.assert_close(output.latents.cpu(), expected_latents)


def test_thinking_taylorseer_keeps_plan_and_noise_request_local() -> None:
    baseline_stage = _stage()
    accelerated_stage = _stage()
    baseline = _batch()
    accelerated = _batch()
    accelerated.enable_taylorseer = True

    with patch(
        "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages."
        "bagel.get_local_torch_device",
        return_value=torch.device("cpu"),
    ):
        baseline_stage.forward(baseline, baseline_stage.server_args)
        accelerated_stage.forward(accelerated, accelerated_stage.server_args)

    assert accelerated.extra["revised_prompt"] == baseline.extra["revised_prompt"]
    torch.testing.assert_close(accelerated.latents, baseline.latents, rtol=0, atol=0)
    taylorseer = accelerated.extra["bagel_taylorseer_context"]
    assert taylorseer.secondary_unconditional is not None
    assert taylorseer.conditional is not taylorseer.unconditional


def test_thinking_warmup_caps_only_effective_decode_length() -> None:
    stage = _stage()
    batch = _batch(max_think_tokens=1000)
    batch.is_warmup = True

    with patch(
        "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages."
        "bagel.get_local_torch_device",
        return_value=torch.device("cpu"),
    ):
        stage.forward(batch, stage.server_args)

    assert stage.transformer.generated_kwargs["max_length"] == 2
    assert batch.max_think_tokens == 1000


def test_thinking_stage_declares_only_transformer_prefill() -> None:
    stage = _stage()

    uses = stage.component_uses(stage.server_args)

    assert [(use.component_name, use.phase) for use in uses] == [
        ("transformer", "prefill")
    ]


def test_revised_prompt_reaches_openai_and_offline_results(tmp_path) -> None:
    output_path = tmp_path / "image.png"
    output_path.write_bytes(b"image")
    revised_prompt = "draw a blue fox\n<think>plan</think>"
    output_batch = OutputBatch(revised_prompts=[revised_prompt])

    response = _build_image_response_kwargs(
        [str(output_path)],
        "b64_json",
        "draw a blue fox",
        "request-id",
        output_batch,
        b64_list=["aW1hZ2U="],
    )
    result_common = DiffGenerator._result_common(
        _batch(), output_batch, generation_time=1.0, output_index=0
    )

    assert response["data"][0].revised_prompt == revised_prompt
    assert result_common["prompt"] == "draw a blue fox"
    assert result_common["revised_prompt"] == revised_prompt


def test_expanded_output_merge_preserves_missing_revised_prompt_position() -> None:
    merged = GPUWorker._merge_expanded_output_batches(
        [
            OutputBatch(output=torch.zeros(1, 1)),
            OutputBatch(
                output=torch.ones(1, 1),
                revised_prompts=["prompt-b\n<think>plan-b</think>"],
            ),
        ]
    )

    assert merged.revised_prompts == [None, "prompt-b\n<think>plan-b</think>"]


def test_standard_decoding_copies_request_revised_prompt() -> None:
    config = BagelThinkingPipelineConfig(vae_precision="fp32")
    args = SimpleNamespace(
        pipeline_config=config,
        disable_autocast=True,
        enable_torch_compile=False,
        model_loaded={"vae": True},
        model_paths={},
    )
    stage = DecodingStage(torch.nn.Identity())
    stage.server_args = args
    batch = _batch()
    batch.latents = torch.zeros(1, 16, 4, 4)
    batch.extra["revised_prompt"] = "draw a blue fox\n<think>plan</think>"

    with patch.object(stage, "decode", return_value=torch.zeros(1, 3, 32, 32)):
        output = stage.forward(batch, args)

    assert output.revised_prompts == ["draw a blue fox\n<think>plan</think>"]


def test_http_generation_forwards_thinking_controls() -> None:
    request = ImageGenerationsRequest(
        prompt="draw a blue fox",
        response_format="b64_json",
        max_think_tokens=12,
        think_do_sample=True,
        think_temperature=0.8,
    )
    raw_request = Request({"type": "http", "headers": []})

    with (
        patch(
            "sglang.multimodal_gen.runtime.entrypoints.openai.image_api."
            "get_global_server_args",
            return_value=SimpleNamespace(model_path="", output_path=None),
        ),
        patch(
            "sglang.multimodal_gen.runtime.entrypoints.openai.image_api."
            "build_sampling_params",
            side_effect=_CapturedSampling,
        ) as sampling_mock,
    ):
        with pytest.raises(_CapturedSampling):
            asyncio.run(generations(request, raw_request))

    kwargs = sampling_mock.call_args.kwargs
    assert kwargs["max_think_tokens"] == 12
    assert kwargs["think_do_sample"] is True
    assert kwargs["think_temperature"] == 0.8


def test_cli_and_warmup_resolve_explicit_thinking_pipeline() -> None:
    args = SimpleNamespace(
        pipeline_class_name="BagelThinkingPipeline",
        pipeline_config=BagelThinkingPipelineConfig(),
        model_path="unused",
        model_id=None,
        backend="sglang",
        warmup_steps=1,
        enable_cfg_parallel=False,
        enable_torch_compile=False,
    )

    sampling_params_cls = _resolve_cli_sampling_params_cls(args)
    requests = build_warmup_reqs(
        args,
        warmup_resolutions=None,
        server_based_warmup=True,
    )

    assert sampling_params_cls is BagelThinkingSamplingParams
    assert len(requests) == 1
    assert isinstance(requests[0].sampling_params, BagelThinkingSamplingParams)
    assert requests[0].is_warmup
    assert requests[0].max_think_tokens == 1000
