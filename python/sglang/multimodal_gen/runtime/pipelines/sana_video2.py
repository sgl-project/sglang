# SPDX-License-Identifier: Apache-2.0
"""Native SANA-Video 2.0 loading and T2V/TI2V pipeline."""

from pathlib import Path
from typing import ClassVar

import torch
import yaml

from sglang.multimodal_gen.runtime.loader.component_loaders.component_loader import (
    ComponentLoader,
)
from sglang.multimodal_gen.runtime.models.dits.sana_video2 import (
    SanaVideo2Transformer3DModel,
)
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages import InputValidationStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.sana_video2 import (
    SanaVideo2DenoisingStage,
    SanaVideo2LatentPreparationStage,
    SanaVideo2TextEncodingStage,
)
from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import maybe_download_model
from sglang.multimodal_gen.runtime.utils.precision import resolve_component_precision


class SanaVideo2TransformerLoader(ComponentLoader):
    expected_library = "diffusers"

    def component_load_precision(self, server_args, component_name):
        return server_args.component_precisions.get(component_name)

    def load_customized(self, component_model_path, server_args, component_name):
        root = Path(component_model_path)
        with (root / "config.yaml").open() as file:
            source = yaml.safe_load(file)
        checkpoint = torch.load(
            root / "checkpoints/SANA_Video_2.0_5B_720p.pth",
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )["state_dict"]
        config = server_args.pipeline_config.dit_config
        model_fields = config.arch_config.__dataclass_fields__
        values = {
            name: value
            for section in (source["model"], source["text_encoder"])
            for name, value in section.items()
            if name in model_fields
        }
        if "pos_embed" in checkpoint:
            values["input_size"] = int(checkpoint["pos_embed"].shape[1] ** 0.5)
        config.update_model_arch(values)
        dtype = resolve_component_precision(server_args, component_name)
        with torch.device("meta"):
            model = SanaVideo2Transformer3DModel(config)
        model.load_state_dict(checkpoint, strict=True, assign=True)
        model = model.to(dtype=dtype)
        model.post_load_weights()
        server_args.model_paths[component_name] = component_model_path
        return model.eval().requires_grad_(False)


class SanaVideo2Pipeline(ComposedPipelineBase):
    pipeline_name = "SanaVideo2Pipeline"
    is_video_pipeline = True
    _required_config_modules: ClassVar[list[str]] = [
        "text_encoder",
        "tokenizer",
        "vae",
        "transformer",
    ]
    component_loaders: ClassVar[dict[str, type[ComponentLoader]]] = {
        "transformer": SanaVideo2TransformerLoader
    }

    def _load_config(self):
        self.model_path = maybe_download_model(
            self.model_path,
            revision=self.server_args.revision,
            allow_patterns=["config.yaml", "checkpoints/SANA_Video_2.0_5B_720p.pth"],
        )
        with (Path(self.model_path) / "config.yaml").open() as file:
            source = yaml.safe_load(file)
        self.server_args.pipeline_config.prompt_instruction = "\n".join(
            source["text_encoder"]["chi_prompt"]
        )
        return {
            "_class_name": self.pipeline_name,
            "_diffusers_version": "0.37.0",
            "text_encoder": ["transformers", "Gemma2Model"],
            "tokenizer": ["transformers", "AutoTokenizer"],
            "vae": ["diffusers", "AutoencoderKLLTX2Video"],
            "transformer": ["diffusers", "SanaVideo2Transformer3DModel"],
        }

    def _resolve_component_path(self, server_args, module_name, load_module_name):
        path = server_args.component_paths.get(module_name)
        if path is None:
            if module_name == "transformer":
                return self.model_path
            if module_name == "vae":
                root = maybe_download_model(
                    "Efficient-Large-Model/LTX-2.3-Diffusers", allow_patterns=["vae/**"]
                )
                return str(Path(root) / "vae")
            path = server_args.component_paths.get(
                "text_encoder", "Efficient-Large-Model/gemma-2-2b-it"
            )
        path = maybe_download_model(path)
        if module_name == "vae" and (Path(path) / "vae/config.json").is_file():
            path = str(Path(path) / "vae")
        return path

    def create_pipeline_stages(self, server_args):
        self.add_stage(InputValidationStage())
        self.add_stage(
            SanaVideo2TextEncodingStage(
                [self.get_module("text_encoder")],
                [self.get_module("tokenizer")],
                server_args.pipeline_config.prompt_instruction,
            ),
            "prompt_encoding_stage_primary",
        )
        self.add_stage(SanaVideo2LatentPreparationStage(self.get_module("vae")))
        self.add_stage(SanaVideo2DenoisingStage(self.get_module("transformer")))
        self.add_standard_decoding_stage()


EntryClass = SanaVideo2Pipeline
