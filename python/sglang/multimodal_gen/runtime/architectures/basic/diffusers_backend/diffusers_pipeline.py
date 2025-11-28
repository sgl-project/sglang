# SPDX-License-Identifier: Apache-2.0
import argparse
import warnings
from io import BytesIO
from typing import Any

import numpy as np
import requests
import torch
import torchvision.transforms as T
from PIL import Image

from sglang.multimodal_gen.configs.pipeline_configs.base import PipelineConfig
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import ComposedPipelineBase
from sglang.multimodal_gen.runtime.pipelines_core.executors.pipeline_executor import PipelineExecutor
from sglang.multimodal_gen.runtime.pipelines_core.executors.sync_executor import SyncExecutor
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages import PipelineStage
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import maybe_download_model
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class DiffusersExecutionStage(PipelineStage):
    def __init__(self, diffusers_pipe: Any):
        super().__init__()
        self.diffusers_pipe = diffusers_pipe

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        kwargs = self._build_pipeline_kwargs(batch, server_args)
        if "output_type" not in kwargs:
            kwargs["output_type"] = "pt"
        with torch.no_grad(), warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            try:
                output = self.diffusers_pipe(**kwargs)
            except TypeError as e:
                if "output_type" in str(e):
                    kwargs.pop("output_type", None)
                    output = self.diffusers_pipe(**kwargs)
                else:
                    raise
        batch.output = self._extract_output(output)
        if batch.output is not None:
            batch.output = self._postprocess_output(batch.output)
        return batch

    def _extract_output(self, output: Any) -> torch.Tensor | None:
        for attr in ["images", "frames", "video", "sample"]:
            if not hasattr(output, attr):
                continue
            data = getattr(output, attr)
            if data is None:
                continue
            result = self._convert_to_tensor(data)
            if result is not None:
                logger.info("Extracted output from '%s': shape=%s", attr, result.shape)
                return result
        logger.warning("Could not extract output from pipeline result")
        return None

    def _convert_to_tensor(self, data: Any) -> torch.Tensor | None:
        if isinstance(data, torch.Tensor):
            return data
        if isinstance(data, np.ndarray):
            t = torch.from_numpy(data).float()
            if t.max() > 1.0:
                t = t / 255.0
            if t.ndim == 4:
                t = t.permute(0, 3, 1, 2)
            elif t.ndim == 5:
                t = t.permute(0, 4, 1, 2, 3)
            return t
        if hasattr(data, "mode"):
            return T.ToTensor()(data)
        if isinstance(data, list) and len(data) > 0:
            return self._convert_list_to_tensor(data)
        return None

    def _convert_list_to_tensor(self, data: list) -> torch.Tensor | None:
        first = data[0]
        if isinstance(first, list) and len(first) > 0:
            data, first = first, first[0]
        if hasattr(first, "mode"):
            ts = [T.ToTensor()(img) for img in data]
            s = torch.stack(ts)
            return s.permute(1, 0, 2, 3) if len(ts) > 1 else s[0]
        if isinstance(first, torch.Tensor):
            s = torch.stack(data)
            return s.permute(1, 0, 2, 3) if len(data) > 1 else s[0]
        if isinstance(first, np.ndarray):
            ts = [torch.from_numpy(a).float() for a in data]
            if ts[0].max() > 1.0:
                ts = [t / 255.0 for t in ts]
            if ts[0].ndim == 3:
                ts = [t.permute(2, 0, 1) for t in ts]
            s = torch.stack(ts)
            return s.permute(1, 0, 2, 3) if len(data) > 1 else s[0]
        return None

    def _postprocess_output(self, output: torch.Tensor) -> torch.Tensor:
        output = output.cpu().float()
        if torch.isnan(output).any() or torch.isinf(output).any():
            output = torch.nan_to_num(output, nan=0.5, posinf=1.0, neginf=0.0)
        mn, mx = output.min().item(), output.max().item()
        if mn < -0.5 or mx > 1.5:
            output = (output + 1) / 2
        output = output.clamp(0, 1)
        output = self._fix_output_shape(output)
        logger.info("Final output tensor shape: %s", output.shape)
        return output

    def _fix_output_shape(self, output: torch.Tensor) -> torch.Tensor:
        if output.dim() == 5:
            return output.permute(0, 2, 1, 3, 4)
        if output.dim() == 4:
            if output.shape[0] == 1 or output.shape[1] in [1, 3, 4]:
                return output
            return output.unsqueeze(0).permute(0, 2, 1, 3, 4)
        if output.dim() == 3:
            c, h, w = output.shape
            if c > 4 and w <= 4:
                output = output.permute(2, 0, 1)
            if output.shape[0] == 1:
                output = output.repeat(3, 1, 1)
            return output.unsqueeze(0)
        if output.dim() == 2:
            return output.unsqueeze(0).repeat(3, 1, 1).unsqueeze(0)
        return output

    def _build_pipeline_kwargs(self, batch: Req, server_args: ServerArgs) -> dict:
        kwargs = {}
        if batch.prompt is not None:
            kwargs["prompt"] = batch.prompt
        if batch.negative_prompt:
            kwargs["negative_prompt"] = batch.negative_prompt
        if batch.num_inference_steps is not None:
            kwargs["num_inference_steps"] = batch.num_inference_steps
        if batch.guidance_scale is not None:
            kwargs["guidance_scale"] = batch.guidance_scale
        if batch.height is not None:
            kwargs["height"] = batch.height
        if batch.width is not None:
            kwargs["width"] = batch.width
        if batch.num_frames is not None and batch.num_frames > 1:
            kwargs["num_frames"] = batch.num_frames
        if batch.generator is not None:
            kwargs["generator"] = batch.generator
        elif batch.seed is not None:
            dev = self._get_pipeline_device()
            kwargs["generator"] = torch.Generator(device=dev).manual_seed(batch.seed)
        img = self._load_input_image(batch)
        if img is not None:
            kwargs["image"] = img
        if batch.num_outputs_per_prompt > 1:
            kwargs["num_images_per_prompt"] = batch.num_outputs_per_prompt
        if batch.extra:
            dk = batch.extra.get("diffusers_kwargs", {})
            if dk:
                kwargs.update(dk)
        return kwargs

    def _get_pipeline_device(self) -> str:
        for attr in ["unet", "transformer", "vae"]:
            c = getattr(self.diffusers_pipe, attr, None)
            if c is not None:
                try:
                    return next(c.parameters()).device
                except StopIteration:
                    pass
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _load_input_image(self, batch: Req) -> Image.Image | None:
        if batch.pil_image is not None:
            return batch.pil_image
        if not batch.image_path:
            return None
        try:
            if batch.image_path.startswith(("http://", "https://")):
                r = requests.get(batch.image_path, timeout=30)
                r.raise_for_status()
                return Image.open(BytesIO(r.content)).convert("RGB")
            return Image.open(batch.image_path).convert("RGB")
        except Exception as e:
            logger.error("Failed to load image from %s: %s", batch.image_path, e)
            return None


class DiffusersPipeline(ComposedPipelineBase):
    pipeline_name = "DiffusersPipeline"
    is_video_pipeline = False
    _required_config_modules: list[str] = []

    def __init__(self, model_path: str, server_args: ServerArgs, required_config_modules=None, loaded_modules=None, executor=None):
        self.server_args = server_args
        self.model_path = model_path
        self._stages = []
        self._stage_name_mapping = {}
        self.modules = {}
        self.post_init_called = False
        self.executor = executor or SyncExecutor(server_args=server_args)
        logger.info("Loading diffusers pipeline from %s", model_path)
        self.diffusers_pipe = self._load_diffusers_pipeline(model_path, server_args)
        self._detect_pipeline_type()

    def _load_diffusers_pipeline(self, model_path: str, server_args: ServerArgs) -> Any:
        from diffusers import DiffusionPipeline
        model_path = maybe_download_model(model_path)
        self.model_path = model_path
        dtype = self._get_dtype(server_args)
        logger.info("Loading diffusers pipeline with dtype=%s", dtype)
        try:
            pipe = DiffusionPipeline.from_pretrained(model_path, torch_dtype=dtype, trust_remote_code=server_args.trust_remote_code, revision=server_args.revision)
        except Exception as e:
            logger.warning("Failed with dtype=%s, falling back to float32: %s", dtype, e)
            pipe = DiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float32, trust_remote_code=server_args.trust_remote_code, revision=server_args.revision)
        if torch.cuda.is_available():
            pipe = pipe.to("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            pipe = pipe.to("mps")
        logger.info("Loaded diffusers pipeline: %s", pipe.__class__.__name__)
        return pipe

    def _get_dtype(self, server_args: ServerArgs) -> torch.dtype:
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        if hasattr(server_args, "pipeline_config") and server_args.pipeline_config:
            dp = getattr(server_args.pipeline_config, "dit_precision", None)
            if dp == "fp16":
                dtype = torch.float16
            elif dp == "bf16":
                dtype = torch.bfloat16
            elif dp == "fp32":
                dtype = torch.float32
        return dtype

    def _detect_pipeline_type(self):
        cn = self.diffusers_pipe.__class__.__name__.lower()
        vi = ["video", "animat", "cogvideo", "wan", "hunyuan"]
        self.is_video_pipeline = any(i in cn for i in vi)
        logger.info("Detected pipeline type: %s", "video" if self.is_video_pipeline else "image")

    def load_modules(self, server_args, loaded_modules=None):
        return {"diffusers_pipeline": self.diffusers_pipe}

    def create_pipeline_stages(self, server_args):
        self.add_stage("diffusers_execution", DiffusersExecutionStage(self.diffusers_pipe))

    def initialize_pipeline(self, server_args):
        pass

    def post_init(self):
        if self.post_init_called:
            return
        self.post_init_called = True
        self.initialize_pipeline(self.server_args)
        self.create_pipeline_stages(self.server_args)

    def add_stage(self, stage_name, stage):
        self._stages.append(stage)
        self._stage_name_mapping[stage_name] = stage
        setattr(self, stage_name, stage)

    @property
    def stages(self):
        return self._stages

    @torch.no_grad()
    def forward(self, batch, server_args):
        if not self.post_init_called:
            self.post_init()
        return self.executor.execute(self.stages, batch, server_args)

    @classmethod
    def from_pretrained(cls, model_path, device=None, torch_dtype=None, pipeline_config=None, args=None, required_config_modules=None, loaded_modules=None, **kwargs):
        kwargs["model_path"] = model_path
        server_args = ServerArgs.from_kwargs(**kwargs)
        pipe = cls(model_path, server_args, required_config_modules=required_config_modules, loaded_modules=loaded_modules)
        pipe.post_init()
        return pipe

    def get_module(self, module_name, default_value=None):
        if module_name == "diffusers_pipeline":
            return self.diffusers_pipe
        return self.modules.get(module_name, default_value)


EntryClass = DiffusersPipeline
