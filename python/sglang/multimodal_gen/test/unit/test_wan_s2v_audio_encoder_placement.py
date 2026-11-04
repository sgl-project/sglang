# SPDX-License-Identifier: Apache-2.0
"""Wan2.2-S2V before-denoising stage: residency, output rate, remote-input downloads."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from transformers import BatchEncoding

from sglang.multimodal_gen.configs.models.encoders.base import BaseEncoderOutput
from sglang.multimodal_gen.configs.pipeline_configs.wan import Wan2_2_S2V_14B_Config
from sglang.multimodal_gen.runtime.loader.component_loaders import (
    audio_encoder_loader as audio_loader_module,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.component_manager import (
    ComponentResidencyManager,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.component_residency import (
    COMPONENT_OFFLOAD,
    LAYERWISE_OFFLOAD,
    RESIDENT,
    SNAPSHOT_OFFLOAD,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages import (
    wan_s2v as stage_module,
)


def _stage():
    stage = object.__new__(stage_module.WanS2VBeforeDenoisingStage)
    stage._component_residency_manager = None
    stage._registered_stage_name = None
    return stage


class _AudioEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))

    def encode_audio(self, audio_path, *, infer_frames, fps, dtype, m):
        return torch.zeros(1), 1


def test_every_request_stages_the_audio_encoder_through_the_residency_manager():
    """An offloaded audio encoder must be brought to the device by the shared residency
    manager before every request's encode; request 2 once ran wav2vec2 on the host."""
    encoder = _AudioEncoder()
    stage = _stage()
    stage.transformer = SimpleNamespace(
        fps=16, param_dtype=torch.bfloat16, audio_sample_m=0
    )
    stage.audio_encoder = encoder
    args = SimpleNamespace(
        pipeline_config=Wan2_2_S2V_14B_Config(),
        component_precisions={},
        enable_layerwise_nvtx_marker=False,
        explicit_residency_mode=lambda _: None,
    )
    pipeline = SimpleNamespace(
        modules={"audio_encoder": encoder},
        _stage_name_mapping={"s2v": stage},
        component_residency_strategies={},
    )
    manager = ComponentResidencyManager(pipeline, args)
    strategy = Mock()
    strategy.prefetch_for_use.return_value = False
    manager.strategy_for = Mock(return_value=strategy)
    stage.set_component_residency_manager(manager)
    batch = SimpleNamespace(is_warmup=False)
    for _ in range(2):
        manager.begin_request([stage], batch, args)
        manager.before_stage(stage, 0, batch, args)
        manager.begin_stage()
        stage._encode_audio("talk.wav", infer_frames=81)
        manager.end_stage()
        manager.finish_request()
    prepared = [call.args[0] for call in strategy.prepare_for_use.call_args_list]
    assert prepared == [encoder, encoder]


@pytest.mark.parametrize(
    "explicit_mode, effective_mode, rejected",
    [
        (RESIDENT, RESIDENT, False),
        (COMPONENT_OFFLOAD, COMPONENT_OFFLOAD, False),
        (SNAPSHOT_OFFLOAD, SNAPSHOT_OFFLOAD, False),
        (LAYERWISE_OFFLOAD, LAYERWISE_OFFLOAD, True),
        # --layerwise-offload-components all: not aimed at the audio encoder
        (None, LAYERWISE_OFFLOAD, False),
    ],
)
def test_only_an_explicit_layerwise_audio_encoder_is_rejected(
    explicit_mode, effective_mode, rejected
):
    """wav2vec2 has no layerwise hooks, so --component-residency audio_encoder=
    layerwise-offload must fail at startup; a layerwise "all" selection must not."""
    server_args = SimpleNamespace(
        canonical_residency_mode=lambda name: (
            explicit_mode if name == "audio_encoder" else None
        ),
        residency_mode=lambda name: (
            effective_mode if name == "audio_encoder" else RESIDENT
        ),
    )
    if rejected:
        with pytest.raises(ValueError, match="audio encoder"):
            Wan2_2_S2V_14B_Config().validate_server_args(server_args)
    else:
        Wan2_2_S2V_14B_Config().validate_server_args(server_args)


def test_audio_loader_records_no_layerwise_support_before_placing_the_encoder():
    """A layerwise "all" selection resolves the audio encoder through its recorded
    capability; recorded after placement, the encoder would start on the host."""
    events = []
    server_args = SimpleNamespace(
        record_component_layerwise_capability=lambda name, *, supported: events.append(
            ("capability", name, supported)
        ),
        should_start_component_on_cpu=lambda name: (
            events.append(("place", name)) or False
        ),
        pipeline_config=SimpleNamespace(
            audio_encoder_config=Mock(), audio_encoder_precision="fp32"
        ),
    )
    with (
        patch.object(
            audio_loader_module,
            "get_diffusers_component_config",
            return_value={"_class_name": "WanS2VAudioEncoder"},
        ),
        patch.object(
            audio_loader_module.ModelRegistry,
            "resolve_model_cls",
            return_value=(Mock(), None),
        ),
    ):
        audio_loader_module.AudioEncoderLoader().load_customized(
            "/unused", server_args, "audio_encoder"
        )
    assert events == [
        ("capability", "audio_encoder", False),
        ("place", "audio_encoder"),
    ]


class _StopAfterFps(Exception):
    pass


def _stop(*args, **kwargs):
    raise _StopAfterFps


def test_stage_saves_at_the_checkpoint_rate_not_the_requested_one(tmp_path):
    """A served request defaults fps to 24 while S2V paces audio at the checkpoint's 16;
    saved at 24, the video ends at 3.2 s of a 5 s clip and freezes on its last frame."""
    image, audio = tmp_path / "ref.png", tmp_path / "talk.wav"
    image.touch()
    audio.touch()
    stage = object.__new__(stage_module.WanS2VBeforeDenoisingStage)
    # Stops forward right after the fps is resolved, before any model work.
    stage.transformer = SimpleNamespace(
        supports_standard_denoising=True, fps=16, _normalize_infer_frames=_stop
    )
    batch = SimpleNamespace(
        image_path=str(image),
        audio_path=str(audio),
        pose_video_path=None,
        num_clip=None,
        num_frames=81,
        fps=24,
    )
    with (
        patch.object(stage, "_generate_seed_and_generator"),
        pytest.raises(_StopAfterFps),
    ):
        stage.forward(batch, server_args=SimpleNamespace())
    assert batch.fps == 16


class _Response:
    def __init__(self, body):
        self._body = body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        if isinstance(self._body, Exception):
            raise self._body
        return self._body


def test_retry_after_failed_download_gets_the_whole_file(tmp_path):
    """The download cache returns any file already at its path, so a download that times out
    mid-body must not leave one there for the retry to pick up."""
    stage = object.__new__(stage_module.WanS2VBeforeDenoisingStage)
    responses = iter(
        [_Response(TimeoutError("read timed out")), _Response(b"RIFF-complete")]
    )
    with (
        patch.object(stage_module.tempfile, "gettempdir", return_value=str(tmp_path)),
        patch.object(stage_module, "urlopen", lambda *a, **kw: next(responses)),
    ):
        with pytest.raises(TimeoutError):
            stage._download_remote_input("https://example.com/talk.wav", "audio_path")
        path = stage._download_remote_input(
            "https://example.com/talk.wav", "audio_path"
        )
    with open(path, "rb") as f:
        assert f.read() == b"RIFF-complete"


class _T5Stub(torch.nn.Module):
    """Returns a fixed last_hidden_state for the real Wan postprocess to pad and mask."""

    def __init__(self, hidden):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.hidden = hidden

    def forward(self, input_ids, attention_mask, output_hidden_states):
        return BaseEncoderOutput(
            last_hidden_state=self.hidden, attention_mask=attention_mask
        )


def test_text_prompt_encodes_through_the_pipelines_postprocess():
    """The stage runs the pipeline's own text postprocess, which may return a
    TextConditioningOutput; it must still hand out the valid-token embeddings."""
    valid, max_len, dim = 5, 512, 8
    hidden = torch.randn(1, max_len, dim)
    mask = torch.zeros(1, max_len, dtype=torch.long)
    mask[:, :valid] = 1
    stage = _stage()
    stage.text_encoder = _T5Stub(hidden)
    stage.tokenizer = lambda prompt, **kwargs: BatchEncoding(
        {
            "input_ids": torch.ones(1, max_len, dtype=torch.long),
            "attention_mask": mask,
        }
    )
    server_args = SimpleNamespace(pipeline_config=Wan2_2_S2V_14B_Config())
    with patch.object(
        stage_module, "get_local_torch_device", return_value=torch.device("cpu")
    ):
        embeds = stage._encode_text_prompt("a cat", server_args=server_args)
    assert torch.equal(embeds, hidden[:, :valid])
