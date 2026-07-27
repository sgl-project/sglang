# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from PIL import Image

from sglang.multimodal_gen.configs.pipeline_configs.bagel import (
    BagelEditPipelineConfig,
)
from sglang.multimodal_gen.configs.sample.bagel import BagelEditSamplingParams
from sglang.multimodal_gen.runtime.models.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.bagel import (
    BagelEditBeforeDenoisingStage,
    BagelEditInputValidationStage,
)
from sglang.multimodal_gen.runtime.warmup_request_builder import build_warmup_reqs


class _Tokenizer:
    token_ids = {
        "<|im_start|>": 151644,
        "<|im_end|>": 151645,
        "<|vision_start|>": 151652,
        "<|vision_end|>": 151653,
    }
    unk_token_id = 0
    unk_token = "<unk>"

    def convert_tokens_to_ids(self, token: str) -> int:
        return self.token_ids.get(token, self.unk_token_id)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        if text in self.token_ids:
            return [self.token_ids[text]]
        return [17, 23]


class _VAE:
    def __init__(self) -> None:
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.seeds: list[int] = []

    @property
    def device(self) -> torch.device:
        return self.anchor.device

    @property
    def dtype(self) -> torch.dtype:
        return self.anchor.dtype

    def encode(
        self, pixels: torch.Tensor, *, generator: torch.Generator
    ) -> torch.Tensor:
        assert pixels.shape == (1, 3, 32, 32)
        self.seeds.append(generator.initial_seed())
        return torch.zeros(1, 16, 4, 4)


class _ImageEncoder:
    def __init__(self) -> None:
        self.images: list[tuple[int, int]] = []

    def encode_image(self, image: Image.Image) -> torch.Tensor:
        self.images.append(image.size)
        return torch.zeros(4, 3584)


class _Transformer:
    def __init__(self, config: BagelEditPipelineConfig) -> None:
        self.config = config.dit_config
        self.calls: list[SimpleNamespace] = []

    def build_editing_context(
        self,
        vae_patches,
        vae_position_ids,
        vision_embeddings,
        text_input_ids,
        **kwargs,
    ):
        context = SimpleNamespace(
            is_editing=True,
            has_three_way_cfg=True,
            vae_shape=tuple(vae_patches.shape),
            vae_position_ids=vae_position_ids.clone(),
            vision_shape=tuple(vision_embeddings.shape),
            text_input_ids=text_input_ids.clone(),
            **kwargs,
        )
        self.calls.append(context)
        return context


def _server_args() -> SimpleNamespace:
    return SimpleNamespace(pipeline_config=BagelEditPipelineConfig())


def _batch(seed: int = 11) -> Req:
    params = BagelEditSamplingParams(
        prompt="turn the background blue",
        image_path=["input.png"],
        height=32,
        width=32,
        seed=seed,
        num_inference_steps=4,
        save_output=False,
    )
    batch = Req(sampling_params=params)
    batch.condition_image = [Image.new("RGB", (32, 32), color="white")]
    batch.generator = [torch.Generator("cpu").manual_seed(seed)]
    batch.seeds = [seed]
    return batch


def _stage() -> BagelEditBeforeDenoisingStage:
    args = _server_args()
    stage = BagelEditBeforeDenoisingStage(
        transformer=_Transformer(args.pipeline_config),
        vae=_VAE(),
        image_encoder=_ImageEncoder(),
        tokenizer=_Tokenizer(),
        scheduler=FlowMatchEulerDiscreteScheduler(shift=1.0),
    )
    stage.server_args = args
    stage._registered_stage_name = "bagel_edit_before_denoising_stage"
    return stage


def test_editing_stage_builds_three_way_context_and_request_rng() -> None:
    stage = _stage()
    batch = _batch()

    with patch(
        "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages."
        "bagel.get_local_torch_device",
        return_value=torch.device("cpu"),
    ):
        output = stage.forward(batch, stage.server_args)

    context = output.extra["bagel_context"]
    assert context.is_editing
    assert context.vae_shape == (4, 64)
    assert context.vision_shape == (4, 3584)
    assert context.vae_position_ids.tolist() == [0, 1, 64, 65]
    assert stage.vae.seeds == [11]
    assert stage.image_encoder.images == [(32, 32)]
    assert output.latents.shape == (4, 64)
    assert output.generator[0].initial_seed() == 11


def test_editing_taylorseer_preserves_conditioning_rng() -> None:
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

    assert baseline_stage.vae.seeds == accelerated_stage.vae.seeds == [11]
    torch.testing.assert_close(accelerated.latents, baseline.latents, rtol=0, atol=0)
    assert (
        accelerated.extra["bagel_taylorseer_context"].secondary_unconditional
        is not None
    )


def test_editing_stage_rejects_multiple_images_before_encoding() -> None:
    stage = _stage()
    batch = _batch()
    batch.condition_image.append(Image.new("RGB", (32, 32)))

    with pytest.raises(ValueError, match="exactly one input image"):
        stage.forward(batch, stage.server_args)


def test_editing_stage_declares_all_component_phases() -> None:
    stage = _stage()
    uses = stage.component_uses(stage.server_args)
    assert [(use.component_name, use.phase) for use in uses] == [
        ("vae", "encode"),
        ("image_encoder", "encode"),
        ("transformer", "prefill"),
    ]


def test_editing_component_lifecycle_matches_declared_order() -> None:
    stage = _stage()
    events: list[tuple[str, str]] = []

    class ResidencyManager:
        server_args = stage.server_args
        state = SimpleNamespace(stage_name="bagel_edit_before_denoising_stage")

        @contextmanager
        def use_component(self, use, module):
            events.append(("start", use.component_name))
            yield module
            events.append(("end", use.component_name))

        def finish_active_use(self) -> None:
            pass

    stage._component_residency_manager = ResidencyManager()
    with patch(
        "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages."
        "bagel.get_local_torch_device",
        return_value=torch.device("cpu"),
    ):
        stage.forward(_batch(), stage.server_args)

    assert events == [
        ("start", "vae"),
        ("end", "vae"),
        ("start", "image_encoder"),
        ("end", "image_encoder"),
        ("start", "transformer"),
        ("end", "transformer"),
    ]


def _validation_batch(
    image_paths: list[str],
    *,
    width: int = 1024,
    height: int = 1024,
    explicit_fields: list[str] | None = None,
) -> Req:
    params = BagelEditSamplingParams(
        prompt="edit",
        image_path=image_paths,
        width=width,
        height=height,
        seed=7,
        num_inference_steps=4,
        save_output=False,
    )
    batch = Req(sampling_params=params)
    batch.extra["explicit_fields"] = explicit_fields or []
    return batch


def test_editing_input_validation_uses_official_size_when_unspecified(tmp_path) -> None:
    image_path = tmp_path / "source.png"
    Image.new("RGB", (400, 800), color="white").save(image_path)
    batch = _validation_batch([str(image_path)])
    args = SimpleNamespace(
        pipeline_config=BagelEditPipelineConfig(), enable_cfg_parallel=False
    )

    output = BagelEditInputValidationStage().forward(batch, args)

    assert (output.width, output.height) == (512, 1024)
    assert output.original_condition_image_size == (400, 800)
    assert len(output.condition_image) == 1
    assert output.condition_image[0].size == (512, 1024)
    assert output.generator[0].device.type == "cpu"


@pytest.mark.parametrize(
    ("width", "height", "matches"),
    [(512, 1024, True), (768, 768, False)],
)
def test_editing_input_validation_checks_explicit_size(
    tmp_path, width: int, height: int, matches: bool
) -> None:
    image_path = tmp_path / "source.png"
    Image.new("RGB", (400, 800), color="white").save(image_path)
    batch = _validation_batch(
        [str(image_path)],
        width=width,
        height=height,
        explicit_fields=["width", "height"],
    )
    args = SimpleNamespace(
        pipeline_config=BagelEditPipelineConfig(), enable_cfg_parallel=False
    )
    stage = BagelEditInputValidationStage()

    if matches:
        output = stage.forward(batch, args)
        assert (output.width, output.height) == (512, 1024)
    else:
        with pytest.raises(ValueError, match="size must match"):
            stage.forward(batch, args)


def test_editing_input_validation_rejects_multiple_paths(tmp_path) -> None:
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    Image.new("RGB", (32, 32)).save(first)
    Image.new("RGB", (32, 32)).save(second)
    batch = _validation_batch([str(first), str(second)])
    args = SimpleNamespace(
        pipeline_config=BagelEditPipelineConfig(), enable_cfg_parallel=False
    )

    with pytest.raises(ValueError, match="exactly one input image"):
        BagelEditInputValidationStage().forward(batch, args)


def test_editing_warmup_preserves_pipeline_sampling_and_image_path() -> None:
    defaults = BagelEditSamplingParams(width=512, height=512)
    args = SimpleNamespace(
        pipeline_config=BagelEditPipelineConfig(),
        pipeline_class_name="BagelEditPipeline",
        warmup_steps=1,
        enable_cfg_parallel=False,
        enable_torch_compile=False,
        backend="sglang",
    )

    with patch(
        "sglang.multimodal_gen.runtime.warmup_request_builder."
        "get_model_sampling_defaults",
        return_value=defaults,
    ):
        requests = build_warmup_reqs(
            args,
            warmup_resolutions=None,
            warmup_input_path="/tmp/warmup.png",
            server_based_warmup=True,
        )

    assert len(requests) == 1
    assert isinstance(requests[0].sampling_params, BagelEditSamplingParams)
    assert requests[0].image_path == ["/tmp/warmup.png"]
