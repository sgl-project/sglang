# SPDX-License-Identifier: Apache-2.0
"""NVIDIA OmniDreams pipeline (autoregressive video world model).

The checkpoint (``single_view/2b_res720p_30fps_i2v_hdmap_distilled.pt``) is a
flat, DiT-only ``.pt`` (570 keys, bf16) -- not a diffusers layout. This pipeline
therefore overrides ``_load_config`` (fabricates a model_index-like dict) and
``load_modules`` (loads the flat DiT directly, mirroring the Hunyuan3D
precedent). The VAE (Wan 2.1) and text encoder (Cosmos-Reason1-7B / Qwen2.5-VL)
are loaded alongside it.

Stage layout (Hybrid monolithic, autoregressive):
``BeforeDenoising -> OmniDreamsDenoising (AR rollout) -> standard DecodingStage``.
The denoising stage concatenates the AR chunks into ``batch.latents``; the
standard single-pass VAE decode lets the Wan VAE's causal temporal feature cache
flow across chunk boundaries, giving correct continuity and the FlashDreams
frame counts (chunk0 -> 1+(len_t-1)*4, each later chunk -> len_t*4).
"""

from __future__ import annotations

import glob
import os
from itertools import chain
from typing import Any

import torch
import torch.nn as nn

from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.loader.fsdp_load import (
    load_model_from_full_model_state_dict,
    set_default_torch_dtype,
)
from sglang.multimodal_gen.runtime.loader.utils import get_param_names_mapping
from sglang.multimodal_gen.runtime.models.dits.omnidreams import OmniDreamsDiT
from sglang.multimodal_gen.runtime.models.schedulers.scheduling_omnidreams_flow_match import (  # noqa: E501
    OmniDreamsFlowMatchScheduler,
)
from sglang.multimodal_gen.runtime.models.vaes.wanvae import AutoencoderKLWan
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.omnidreams import (  # noqa: E501
    OmniDreamsBeforeDenoisingStage,
    OmniDreamsDenoisingStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.utils import PRECISION_TO_TYPE

logger = init_logger(__name__)

# Default in-repo location of the distilled single-view checkpoint.
_DEFAULT_CKPT_RELPATH = "single_view/2b_res720p_30fps_i2v_hdmap_distilled.pt"

# Cosmos-Reason1-7B text encoder (Qwen2.5-VL), pinned for numerical parity with
# FlashDreams (flashdreams/infra/encoder/text/cosmos_reason1.py).
_TEXT_ENCODER_ID = "nvidia/Cosmos-Reason1-7B"
_TEXT_ENCODER_REVISION = "3210bec0495fdc7a8d3dbb8d58da5711eab4b423"

# Wan 2.1 VAE weights subdirectory candidates (relative to the model path).
_VAE_RELDIRS = ("vae", "wan_vae", "Wan2.1_VAE")


class OmniDreamsPipeline(ComposedPipelineBase):
    pipeline_name = "OmniDreamsPipeline"
    is_video_pipeline = True
    _required_config_modules = [
        "transformer",
        "vae",
        "text_encoder",
        "tokenizer",
        "scheduler",
    ]

    def _load_config(self) -> dict[str, Any]:
        return {
            "_class_name": self.pipeline_name,
            "_diffusers_version": "0.0.0",
            "transformer": ["sglang", "OmniDreamsDiT"],
            "vae": ["diffusers", "AutoencoderKLWan"],
            "text_encoder": ["transformers", "AutoModel"],
            "tokenizer": ["transformers", "AutoProcessor"],
            "scheduler": ["sglang", "OmniDreamsFlowMatchScheduler"],
        }

    # ----- path resolution -------------------------------------------------- #
    @staticmethod
    def _resolve_ckpt_path(model_path: str) -> str:
        """Locate the flat DiT ``.pt`` from a directory, repo, or direct file."""
        if os.path.isfile(model_path):
            return model_path
        candidate = os.path.join(model_path, _DEFAULT_CKPT_RELPATH)
        if os.path.isfile(candidate):
            return candidate
        matches = sorted(
            glob.glob(os.path.join(model_path, "**", "*.pt"), recursive=True)
        )
        if matches:
            return matches[0]
        raise FileNotFoundError(
            f"OmniDreams checkpoint (.pt) not found under {model_path}"
        )

    @staticmethod
    def _resolve_vae_path(model_path: str) -> str:
        """Locate the Wan 2.1 VAE weights (a flat ``.pth`` or a weights dir).

        Looks for a ``vae``-like subdirectory first, then any ``*VAE*.pth`` /
        ``*vae*.safetensors`` under the model path. The caller (GPU bring-up)
        may also point ``model_path`` directly at a Wan VAE directory.
        """
        base = model_path if os.path.isdir(model_path) else os.path.dirname(model_path)
        for sub in _VAE_RELDIRS:
            cand = os.path.join(base, sub)
            if os.path.isdir(cand) or os.path.isfile(cand):
                return cand
        for pattern in ("**/*vae*.safetensors", "**/*VAE*.safetensors"):
            matches = sorted(glob.glob(os.path.join(base, pattern), recursive=True))
            if matches:
                return matches[0]
        raise FileNotFoundError(
            f"Diffusers-format Wan 2.1 VAE not found under {base}. Place a "
            f"diffusers Wan VAE (*.safetensors + config.json) under a 'vae/' "
            f"subdirectory or pass its path explicitly."
        )

    # ----- component loaders ------------------------------------------------ #
    @classmethod
    def _load_flat_dit(
        cls,
        dit_config: Any,
        ckpt_path: str,
        device: torch.device,
        dtype: torch.dtype,
    ) -> nn.Module:
        """Instantiate OmniDreamsDiT on meta, load the flat .pt, then fuse weights.

        The flat checkpoint keys equal the submodule names, so the param mapping
        is the identity. The custom loader bypasses the generic loader's
        post-load hook, so ``post_load_weights`` (72->68 padding-mask fuse +
        last-layer shuffle fuse) is invoked explicitly here.
        """
        with set_default_torch_dtype(dtype), torch.device("meta"):
            model = OmniDreamsDiT(config=dit_config, hf_config={})

        weights = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        weight_iterator = ((k, v) for k, v in weights.items())
        mapping_fn = get_param_names_mapping(model.param_names_mapping)

        load_model_from_full_model_state_dict(
            model,
            weight_iterator,
            device,
            dtype,
            strict=True,
            param_names_mapping=mapping_fn,
        )

        model.post_load_weights()

        for name, p in chain(model.named_parameters(), model.named_buffers()):
            if p.is_meta:
                raise RuntimeError(f"Unexpected param/buffer {name} on meta device.")
            if isinstance(p, nn.Parameter):
                p.requires_grad = False

        return model.eval()

    @classmethod
    def _load_wan_vae(
        cls,
        vae_config: Any,
        vae_path: str,
        device: torch.device,
        dtype: torch.dtype,
    ) -> nn.Module:
        """Build the SGLang Wan 2.1 VAE and load diffusers-format weights.

        SGLang's ``AutoencoderKLWan`` uses the diffusers WanVAE key naming, so
        the VAE must be supplied in **diffusers format**: a ``vae/`` directory
        with ``*.safetensors`` (+ ``config.json``) as exported by diffusers.

        The original lightx2v flat ``Wan2.1_VAE.pth`` uses a different
        (original-Wan) key naming and is intentionally **not** remapped here —
        converting between the two schemes is exactly what diffusers'
        ``convert_wan_to_diffusers`` already does. Point this at a diffusers Wan
        VAE (any ``Wan2.1-*-Diffusers/vae``) instead of the flat ``.pth``.
        """
        with set_default_torch_dtype(dtype):
            vae = AutoencoderKLWan(vae_config)

        state = cls._read_vae_state_dict(vae_path)

        try:
            vae.load_state_dict(state, strict=True)
        except RuntimeError as exc:
            raise RuntimeError(
                f"Failed to load Wan VAE weights from '{vae_path}' into the "
                "diffusers-format AutoencoderKLWan. Supply the VAE in diffusers "
                "format (a 'vae/' directory with *.safetensors + config.json). "
                "The flat lightx2v 'Wan2.1_VAE.pth' uses original-Wan key names; "
                "convert it with diffusers' convert_wan_to_diffusers first.\n"
                f"Underlying error: {exc}"
            ) from exc
        return vae.to(device=device, dtype=dtype).eval()

    @staticmethod
    def _read_vae_state_dict(vae_path: str) -> dict[str, torch.Tensor]:
        """Read a VAE state dict from a diffusers ``*.safetensors`` dir/file."""
        if os.path.isdir(vae_path):
            files = sorted(glob.glob(os.path.join(vae_path, "*.safetensors")))
            if not files:
                raise FileNotFoundError(
                    f"No *.safetensors found under '{vae_path}'. Supply a "
                    "diffusers-format Wan VAE directory."
                )
            from safetensors.torch import load_file as safetensors_load_file

            state: dict[str, torch.Tensor] = {}
            for f in files:
                state.update(safetensors_load_file(f))
        elif vae_path.endswith(".safetensors"):
            from safetensors.torch import load_file as safetensors_load_file

            state = safetensors_load_file(vae_path)
        else:
            state = torch.load(vae_path, map_location="cpu", weights_only=True)

        # Some checkpoints nest the state dict under "model"/"state_dict".
        if "state_dict" in state and isinstance(state["state_dict"], dict):
            state = state["state_dict"]
        return state

    @staticmethod
    def _resolve_text_encoder_src(model_path: str) -> tuple[str, str | None]:
        """Resolve the Cosmos-Reason1-7B source.

        Prefers a local ``<model_path>/text_encoder`` directory (offline /
        mirrored deployments), returning ``(local_dir, None)``; otherwise falls
        back to the pinned HF id + revision.
        """
        if os.path.isdir(model_path):
            local = os.path.join(model_path, "text_encoder")
            if os.path.isfile(os.path.join(local, "config.json")):
                return local, None
        return _TEXT_ENCODER_ID, _TEXT_ENCODER_REVISION

    @classmethod
    def _load_text_encoder(
        cls, model_path: str, device: torch.device
    ) -> tuple[Any, Any]:
        """Load Cosmos-Reason1-7B (Qwen2.5-VL) + processor.

        Uses a local ``<model_path>/text_encoder`` dir when present, else the
        pinned HF id + revision (see :meth:`_resolve_text_encoder_src`).
        """
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        src, revision = cls._resolve_text_encoder_src(model_path)
        logger.info("OmniDreams: loading text encoder from %s", src)
        processor = AutoProcessor.from_pretrained(src, revision=revision)
        text_encoder = (
            Qwen2_5_VLForConditionalGeneration.from_pretrained(
                src,
                revision=revision,
                torch_dtype=torch.bfloat16,  # canonical kwarg; avoids fp32 load
            )
            .eval()
            .requires_grad_(False)
            .to(device)
        )
        return text_encoder, processor

    def load_modules(
        self,
        server_args: ServerArgs,
        loaded_modules: dict[str, torch.nn.Module] | None = None,
    ) -> dict[str, Any]:
        pipeline_config = server_args.pipeline_config
        device = get_local_torch_device()
        dit_dtype = PRECISION_TO_TYPE[pipeline_config.dit_precision]
        vae_dtype = PRECISION_TO_TYPE[pipeline_config.vae_precision]

        # Honor the CPU-offload flags at load time: each flagged component is
        # staged on CPU so the three heavy weights (the 2B DiT, the 7B text
        # encoder, and the Wan VAE) never need to co-reside on the GPU while
        # loading. The ComponentResidencyManager then brings each one to the GPU
        # only around its use-site (declared in the stages' ``component_uses``)
        # and releases it afterwards. Without this the custom loaders push all
        # three straight to the GPU, so a small-VRAM card OOMs while loading the
        # text encoder even though --text-encoder-cpu-offload was requested.
        cpu_device = torch.device("cpu")

        def _load_device(offload_flag: object) -> torch.device:
            offload = bool(offload_flag) and not server_args.use_fsdp_inference
            return cpu_device if offload else device

        dit_device = _load_device(server_args.dit_cpu_offload)
        vae_device = _load_device(server_args.vae_cpu_offload)
        text_encoder_device = _load_device(server_args.text_encoder_cpu_offload)

        model_path = server_args.model_path
        ckpt_path = self._resolve_ckpt_path(model_path)
        logger.info("OmniDreams: loading flat DiT from %s", ckpt_path)
        transformer = self._load_flat_dit(
            pipeline_config.dit_config, ckpt_path, dit_device, dit_dtype
        )

        vae_path = self._resolve_vae_path(model_path)
        logger.info("OmniDreams: loading Wan 2.1 VAE from %s", vae_path)
        vae = self._load_wan_vae(
            pipeline_config.vae_config, vae_path, vae_device, vae_dtype
        )

        text_encoder, tokenizer = self._load_text_encoder(
            model_path, text_encoder_device
        )

        scheduler = OmniDreamsFlowMatchScheduler(
            num_inference_steps=len(pipeline_config.denoising_timesteps),
            denoising_timesteps=tuple(pipeline_config.denoising_timesteps),
            shift=(
                pipeline_config.flow_shift
                if pipeline_config.flow_shift is not None
                else 5.0
            ),
            sigma_min=pipeline_config.sigma_min,
            device=device,
        )

        # Phase 6: populate memory budgets (GiB, approximate) for the
        # ComponentResidencyManager offload scheduler. Exact values are
        # TODO(gpu): measure on the target GPU with torch.cuda.memory_stats().
        self.memory_usages = {
            "transformer": 4.0,  # ~2B params in bf16 ≈ 4 GiB
            "text_encoder": 14.0,  # Cosmos-Reason1-7B ≈ 14 GiB
            "vae": 1.0,  # Wan 2.1 VAE ≈ 1 GiB
        }

        return {
            "transformer": transformer,
            "vae": vae,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "scheduler": scheduler,
        }

    def create_pipeline_stages(self, server_args: ServerArgs):
        config = server_args.pipeline_config
        self.add_stage(
            stage_name="omnidreams_before_denoising",
            stage=OmniDreamsBeforeDenoisingStage(
                transformer=self.get_module("transformer"),
                scheduler=self.get_module("scheduler"),
                text_encoder=self.get_module("text_encoder"),
                tokenizer=self.get_module("tokenizer"),
                vae=self.get_module("vae"),
                config=config,
            ),
        )
        self.add_stage(
            stage_name="omnidreams_denoising",
            stage=OmniDreamsDenoisingStage(
                transformer=self.get_module("transformer"),
                scheduler=self.get_module("scheduler"),
                vae=self.get_module("vae"),
            ),
        )
        # Standard single-pass VAE decode. The denoising stage concatenates the
        # AR chunks into batch.latents; the Wan VAE's causal temporal feature
        # cache flows frame-to-frame within one decode() call, giving correct
        # cross-chunk continuity and the FlashDreams frame counts (chunk0 ->
        # 1+(len_t-1)*4, each later chunk -> len_t*4).
        self.add_standard_decoding_stage(stage_name="omnidreams_decoding")


EntryClass = OmniDreamsPipeline
