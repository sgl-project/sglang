# SPDX-License-Identifier: Apache-2.0
"""Image components run only inside their declared residency uses.

The image release has no residency plan (``supports_auto_residency=False``):
the text encoder is loaded host-offloaded and only a component use moves it.
A residency-manager double marks a module resident only inside
``use_component``; the component doubles fail when called outside it, as the
real Qwen3VL does on GPU when its weights are still on the host.
"""

from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from sglang.multimodal_gen.runtime.managers.memory_managers.component_manager import (
    ComponentUse,
)
from sglang.multimodal_gen.runtime.pipelines.minimax_h3_pipeline import (
    MiniMaxH3ImagePipeline,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3 import (
    keyframe_encoding,
    presentation,
    reference_encoding,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.constants import (
    MINIMAX_H3_REFERENCE_IMAGE_ROWS_EXTRA_KEY as REFS,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.constants import (
    MINIMAX_H3_TEXT_EMBEDDINGS_EXTRA_KEY as TEXT,
)
from sglang.multimodal_gen.test.unit.test_minimax_h3_image_pipeline import (
    RecordingVAE,
    ZeroVelocityModel,
    make_req,
)
from sglang.multimodal_gen.test.unit.test_minimax_h3_image_pipeline import (
    server as server,
)

# The image stages' component uses, in request order. Only the decode use
# declares a target dtype: video_vae resolves to vae_precision="fp32", so that
# move is a no-op cast on the fp32 VAE double.
RELEASE_USES = [
    ComponentUse(
        "",
        "text_encoder",
        preferred_ready_after_request=True,
        start_at_stage_entry=False,
    ),
    ComponentUse("", "video_vae"),
    ComponentUse("", "transformer"),
    ComponentUse("", "video_vae", target_dtype=torch.float32),
]


class Manager:
    def __init__(self, server_args):
        self.server_args = server_args
        self.state = SimpleNamespace(stage_name=None)
        self.uses = []
        self.finished_unused = []

    @contextmanager
    def use_component(self, use, module):
        self.uses.append(replace(use, stage_name=""))
        module.resident = True
        try:
            yield module
        finally:
            module.resident = False

    def finish_active_use(self):
        pass

    def finish_unused_component(self, use, module):
        assert use.component_name == "text_encoder"
        assert not getattr(module, "resident", False)
        self.finished_unused.append(use.component_name)


def _resident(module, name):
    if not getattr(module, "resident", False):
        raise AssertionError(f"{name} called outside its declared component use")


class Encoder(torch.nn.Module):
    model = SimpleNamespace(get_rope_index=None)

    def encode_ids(self, ids, **kwargs):
        _resident(self, "text_encoder")
        return torch.zeros(ids.shape[0], 5120, dtype=torch.bfloat16)


class DiT(ZeroVelocityModel):
    def forward(self, **kwargs):
        _resident(self, "transformer")
        return super().forward(**kwargs)


class VAE(torch.nn.Module):
    parallel_tiling = False

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.decoder = RecordingVAE()

    def decode_image_raw(self, x):
        _resident(self, "video_vae decode")
        return self.decoder.decode_image_raw(x)


def _image_stages(server):
    processor = SimpleNamespace(
        image_processor=lambda images, return_tensors: {
            "pixel_values": torch.zeros(len(images) * 4, 1536),
            "image_grid_thw": torch.tensor([[1, 2, 2]] * len(images)),
        }
    )
    processor.image_processor.merge_size = 2
    modules = {
        "text_encoder": Encoder(),
        "tokenizer": object(),
        "processor": processor,
        "video_vae": VAE(),
        "transformer": DiT(),
    }
    pipeline = object.__new__(MiniMaxH3ImagePipeline)
    pipeline.get_module = modules.get
    stages = []
    pipeline.add_stage = stages.append
    pipeline.create_pipeline_stages(server)
    return pipeline, stages, modules


@pytest.mark.parametrize("task", ["t2i", "edit"])
def test_every_component_runs_inside_its_declared_use(server, monkeypatch, task):
    # The component doubles use CPU tensors, including on CUDA test hosts.
    monkeypatch.setattr(
        PipelineStage, "device", property(lambda self: torch.device("cpu"))
    )
    monkeypatch.setattr(
        presentation,
        "minimax_h3_image_presentation",
        lambda tokenizer, **kwargs: (
            torch.arange(4),
            torch.ones(4, dtype=torch.long),
            torch.zeros(3, 1, 4, dtype=torch.long),
        ),
    )
    monkeypatch.setattr(
        reference_encoding,
        "minimax_h3_prepared_reference_image",
        lambda batch, plan: {
            "images": [
                {"image": Image.new("RGB", (64, 32)), "condition_index": index}
                for index in range(2)
            ]
        },
    )

    def encode_rows(vae, image, arch, crop_to_even, generator):
        _resident(vae, "video_vae encode")
        assert vae.weight.dtype == torch.float32
        return torch.full((2, 96), 0.5)

    monkeypatch.setattr(
        keyframe_encoding, "minimax_h3_encode_keyframe_cond_rows", encode_rows
    )
    pipeline, stages, modules = _image_stages(server)
    manager = Manager(server)
    req = make_req(server, task)
    req.extra.pop(TEXT)
    req.extra.pop(REFS, None)
    for stage in stages:
        stage.set_component_residency_manager(manager)
        output = stage.forward(req, server)
        req = req if stage is stages[-1] else output
    assert set(req.extra[TEXT]) == {"positive"}
    # One forward per fused head of the 6-step release.
    assert len(modules["transformer"].calls) == 6
    assert output.output.shape == (1, 3, 32, 64)
    expected = list(RELEASE_USES)
    if task == "edit":
        assert len(req.extra[REFS]["images"]) == 2
    else:
        assert REFS not in req.extra
        expected.pop(1)  # t2i has no reference to encode
    assert manager.uses == expected
    assert manager.finished_unused == ["text_encoder"]
    assert not any(getattr(module, "resident", False) for module in modules.values())
