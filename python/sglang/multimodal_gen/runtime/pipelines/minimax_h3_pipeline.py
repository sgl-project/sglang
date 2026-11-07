# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
import shutil
from dataclasses import replace

import torch

from sglang.multimodal_gen import envs
from sglang.multimodal_gen.configs.pipeline_configs.minimax_h3 import (
    MiniMaxH3PipelineConfig,
)
from sglang.multimodal_gen.configs.sample.minimax_h3 import MiniMaxH3SamplingParams
from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.lora.pipeline import LoRAPipeline
from sglang.multimodal_gen.runtime.pipelines_core.stages import InputValidationStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3 import (
    MiniMaxH3AudioEncodingStage,
    MiniMaxH3DecodingStage,
    MiniMaxH3DenoisingStage,
    MiniMaxH3LatentPreparationStage,
    MiniMaxH3TextEncodingStage,
    MiniMaxH3TimestepPreparationStage,
    MiniMaxH3VisualEncodingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.release_metadata import (
    MiniMaxH3PartitionAdmissionStage,
    MiniMaxH3ReleaseMetadata,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.task_profiles import (
    MINIMAX_H3_TASK_PARTITIONS,
)
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.server_args import ServerArgs


class MiniMaxH3Pipeline(LoRAPipeline, ComposedPipelineBase):
    pipeline_name = "MiniMaxH3Pipeline"
    default_model_subfolder = "FL2VA"
    is_video_pipeline = True
    pipeline_config_cls = MiniMaxH3PipelineConfig
    sampling_params_cls = MiniMaxH3SamplingParams
    _required_config_modules = [
        "processor",
        "text_encoder",
        "tokenizer",
        "video_vae",
        "audio_vae",
        # scheduler intentionally absent: model_index carries scheduler=null;
        # per-modality sigma schedules are generated in TimestepPreparation
        # from the task profile, and the loop scheduler math lives in
        # minimax_h3/denoise_loop.py (stages accept scheduler=None).
        "transformer",
    ]

    # Releases that ship only add-ons for the MiniMax-H3 base: the model path is
    # the release repo, of which only ``release_files`` (at ``release_revision``)
    # are fetched, and the weights load from ``base_model``'s partition at
    # ``base_revision``. None loads the model path as a full H3 checkpoint.
    base_model: str | None = None
    base_revision: str | None = None
    release_revision: str | None = None
    release_files: tuple[str, ...] = ()
    # the release's LoRA adapter, relative to the release dir
    release_lora_file: str | None = None

    def __init__(self, *args, **kwargs):
        # TODO: Enable this check on ROCm after adding ffmpeg to the AMD Docker
        # image and CI dependency installer.
        if not current_platform.is_rocm():
            missing_media_tools = [
                executable
                for executable in ("ffmpeg", "ffprobe")
                if shutil.which(executable) is None
            ]
            if missing_media_tools:
                raise RuntimeError(
                    "MiniMax H3 requires ffmpeg and ffprobe for media processing "
                    "and validated output delivery; missing executables: "
                    f"{', '.join(missing_media_tools)}. Install the ffmpeg system "
                    "package before starting SGLang."
                )
        super().__init__(*args, **kwargs)

    @staticmethod
    def model_subfolder_for_variant(variant: str) -> str:
        if not isinstance(variant, str) or not variant.strip():
            raise ValueError("MiniMax H3 model variant must be a non-empty string")
        normalized = variant.strip().lower()
        subfolders = {
            "fl2va": "FL2VA",
            "ref2va": "Ref2VA",
            "hybrid": "Ref2VA",
        }
        try:
            return subfolders[normalized]
        except KeyError as exc:
            raise ValueError(
                f"unsupported MiniMax H3 model variant {variant!r}; "
                f"supported: {sorted(subfolders)!r}"
            ) from exc

    def _load_config(self):
        model_variant = self.server_args.model_variant
        if model_variant is not None:
            semantic_subfolder = self.model_subfolder_for_variant(model_variant)
            if model_variant.strip().lower() == "hybrid" and not (
                self.server_args.component_weights_paths.get("transformer")
                or self.server_args.transformer_weights_path
            ):
                raise ValueError(
                    "MiniMax H3 --model-variant hybrid requires explicit merged "
                    "weights via --component-weights-paths.transformer"
                )
            explicit_subfolder = self.server_args.model_subfolder
            if (
                explicit_subfolder is not None
                and explicit_subfolder.strip().lower() != semantic_subfolder.lower()
            ):
                raise ValueError(
                    "MiniMax H3 --model-variant and --model-subfolder select "
                    f"different weight partitions: variant={model_variant!r} maps to "
                    f"{semantic_subfolder!r}, model_subfolder="
                    f"{explicit_subfolder!r}"
                )
            self.server_args.model_subfolder = semantic_subfolder
        model_index = (
            super()._load_config()
            if self.base_model is None
            else self._load_base_partition_config()
        )
        self.release_metadata = MiniMaxH3ReleaseMetadata.from_model_index(model_index)
        if (
            model_variant is not None
            and self.release_metadata.partition != semantic_subfolder.lower()
        ):
            raise ValueError(
                "MiniMax H3 loaded checkpoint partition does not match "
                f"--model-variant {model_variant!r}"
            )
        if model_variant is not None and model_variant.strip().lower() == "hybrid":
            # merged checkpoints share the native graph across all three tasks
            # keep the base partition contract strict unless explicitly selected
            self.release_metadata = replace(
                self.release_metadata,
                partition="hybrid",
                tasks=tuple(MINIMAX_H3_TASK_PARTITIONS),
            )
        self._apply_release_lora_adapter()
        return model_index

    def _load_base_partition_config(self) -> dict:
        """Fetch the release's add-ons and load the base model's partition."""
        from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import (
            maybe_download_model,
            verify_model_config_and_directory,
        )

        self.release_path = maybe_download_model(
            self.model_path,
            allow_patterns=list(self.release_files),
            revision=self.server_args.revision or self.release_revision,
        )
        missing = [
            name
            for name in self.release_files
            if not os.path.isfile(os.path.join(self.release_path, name))
        ]
        if missing:
            raise ValueError(f"{self.pipeline_name} release is missing {missing}")
        subfolder = self.server_args.model_subfolder or self.default_model_subfolder
        base_root = maybe_download_model(
            self.base_model,
            allow_patterns=[f"{subfolder}/**"],
            revision=self.base_revision,
        )
        self.model_path = os.path.join(base_root, subfolder)
        return verify_model_config_and_directory(self.model_path)

    def _release_lora_adapter_path(self) -> str | None:
        if self.release_lora_file is not None:
            return os.path.join(self.release_path, self.release_lora_file)
        adapter = self.release_metadata.lora_adapter
        return None if adapter is None else os.path.join(self.model_path, adapter)

    def _apply_release_lora_adapter(self) -> None:
        """Load a release's bundled LoRA adapter as the startup adapter."""
        adapter_path = self._release_lora_adapter_path()
        if adapter_path is None:
            return
        if not os.path.isfile(adapter_path):
            raise ValueError(
                f"{self.pipeline_name} release adapter not found: {adapter_path}"
            )
        if self.server_args.lora_path not in (None, adapter_path):
            raise ValueError(
                f"{self.pipeline_name} applies its own release adapter "
                f"({os.path.basename(adapter_path)}); --lora-path cannot replace it"
            )
        # LoRAPipeline reads server_args.lora_path once the modules are loaded
        self.server_args.lora_path = adapter_path

    def validate_disagg_role(self, role: RoleType) -> None:
        if role != RoleType.MONOLITHIC:
            raise ValueError(
                "MiniMaxH3Pipeline only supports monolithic deployment; "
                f"disaggregation role {role.value!r} is not supported"
            )

    def initialize_pipeline(self, server_args: ServerArgs) -> None:
        self.pdd_heads = self._resolve_pdd_heads()
        transformer = self.get_module("transformer")
        if (
            self.pdd_heads
            and transformer is not None
            and transformer.final_layer._pdd_heads is None
        ):
            transformer.final_layer.load_pdd_fused_heads(self.pdd_heads)

    def _resolve_pdd_heads(self) -> str | None:
        """The env override, else a PDD release's own fused heads."""
        if envs.SGLANG_DIFFUSION_MINIMAX_H3_PDD_HEADS:
            return envs.SGLANG_DIFFUSION_MINIMAX_H3_PDD_HEADS
        release_metadata = getattr(self, "release_metadata", None)
        if release_metadata is None or release_metadata.pdd_fused_heads is None:
            return None
        return os.path.join(self.model_path, release_metadata.pdd_fused_heads)

    def create_pipeline_stages(self, server_args: ServerArgs) -> None:
        # Per-model sigma override from model_index.json; contract tests
        # construct the pipeline without model_path, hence the guard.
        release_metadata = getattr(self, "release_metadata", None)
        sigma_shift_scales = (
            release_metadata.sigma_shift_scales
            if release_metadata is not None
            else None
        )
        self.add_stage(InputValidationStage())
        if release_metadata is not None:
            self.add_stage(MiniMaxH3PartitionAdmissionStage(release_metadata))
        self.add_stage(
            MiniMaxH3TextEncodingStage(
                text_encoder=self.get_module("text_encoder"),
                tokenizer=self.get_module("tokenizer"),
                processor=self.get_module("processor"),
            )
        )
        self.add_stage(
            MiniMaxH3VisualEncodingStage(
                video_vae=self.get_module("video_vae"),
                vae_arch_config=server_args.pipeline_config.vae_config.arch_config,
            )
        )
        self.add_stage(
            MiniMaxH3AudioEncodingStage(
                audio_vae=self.get_module("audio_vae"),
                vae_arch_config=server_args.pipeline_config.audio_vae_config.arch_config,
            )
        )
        self.add_source_stages(server_args)
        self.add_stage(MiniMaxH3LatentPreparationStage())
        self.add_stage(
            MiniMaxH3TimestepPreparationStage(
                sigma_shift_scales=sigma_shift_scales,
                dmd_denoising_steps=(
                    release_metadata.dmd_denoising_steps
                    if release_metadata is not None
                    else None
                ),
                pdd_heads=getattr(self, "pdd_heads", None),
            )
        )
        self.add_stage(
            MiniMaxH3DenoisingStage(
                transformer=self.get_module("transformer"),
                pipeline=self,
                stochastic_sampling=(
                    release_metadata is not None
                    and release_metadata.stochastic_sampling
                ),
            )
        )
        self.add_stage(
            MiniMaxH3DecodingStage(
                video_vae=self.get_module("video_vae"),
                audio_vae=self.get_module("audio_vae"),
            )
        )

    def add_source_stages(self, server_args: ServerArgs) -> None:
        """Stages that prepare the denoise state ahead of latent preparation."""

    def create_comfyui_stages(self, server_args: ServerArgs) -> None:
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.stages.comfyui_step import (
            MiniMaxH3ComfyUIStepStage,
        )

        self.add_stage(
            MiniMaxH3ComfyUIStepStage(transformer=self.get_module("transformer"))
        )


class FastH3Pipeline(MiniMaxH3Pipeline):
    """FastH3: 8-step DMD2-distilled MiniMax-H3 (V2, t2va only), and the
    FastH3 Omni Ref2VA PDD student.

    The flat single-partition repo is materialized into the base-H3 layout by
    its registered model overlay (see runtime/utils/model_overlay.py), so every stage, loader,
    and admission path below is exactly the MiniMax-H3 one. There is no
    FL2VA/Ref2VA partition layout to default into.
    """

    pipeline_name = "FastH3Pipeline"
    default_model_subfolder = None


class PikaH3FlexPipeline(MiniMaxH3Pipeline):
    """Pika H3 Flex: MiniMax-H3 plus Pika's six-step LoRA adapters.

    The release's model overlay links the pinned MiniMax-H3 base into the
    base-H3 layout (root and ``FL2VA/`` for t2va/fl2va, ``Ref2VA/`` for
    ref2va) with the matching adapter next to each partition. Its
    ``_minimax_h3`` block carries the sampling contract (knots
    1000/750/500/250/100/40, video shift 12, audio shift 3, re-noising) and
    the adapter, which loads at startup.
    """

    pipeline_name = "PikaH3FlexPipeline"


class OpenH3RegeneratePipeline(MiniMaxH3Pipeline):
    """Open-H3-Regenerate-2K: MiniMax-H3 refining a draft at up to 2560x1440.

    NVlabs/Sana's open replacement for MiniMax's H3-Regenerate-2K stage. From
    the release repo only the merged refiner LoRA (the startup adapter), the
    LMS Sharp timestep-curve AdaLN branches and the author latent upscaler are
    fetched; the weights are MiniMax-H3's FL2VA partition at the revision the
    release pins. A request draft is encoded, upscaled, re-noised and refined
    in 3 Euler steps; see ``MiniMaxH3RegenerateSourceStage``.
    """

    pipeline_name = "OpenH3RegeneratePipeline"
    base_model = "MiniMaxAI/MiniMax-H3"
    base_revision = "6818f6c32d12b210915e44ad56a4228c2608f160"
    release_revision = "5809c33730c18b6b1c827b40974855a4288641b3"
    release_lora_file = "lora/open_h3_regenerate_2k_lora.safetensors"
    upscaler_file = (
        "latent_upsampler/minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors"
    )
    curve_branches_file = "lora/sharp_curve_branches.safetensors"
    release_config_file = "lora/lora_config.json"
    release_files = (
        release_lora_file,
        curve_branches_file,
        release_config_file,
        upscaler_file,
    )

    def initialize_pipeline(self, server_args: ServerArgs) -> None:
        from safetensors.torch import load_file

        from sglang.multimodal_gen.runtime.models.upsampler.minimax_h3_latent_upscaler import (
            MiniMaxH3LatentUpscaler,
        )

        super().initialize_pipeline(server_args)
        self.get_module("transformer").load_adaln_curve_branches(
            os.path.join(self.release_path, self.curve_branches_file),
            self._adaln_curve_table_path(),
        )
        upscaler = MiniMaxH3LatentUpscaler.from_state_dict(
            load_file(os.path.join(self.release_path, self.upscaler_file))
        )
        device = current_platform.get_local_torch_device()
        self.latent_upscaler = upscaler.to(device, torch.bfloat16).eval()

    def _adaln_curve_table_path(self) -> str:
        """The 1025x8 curve table the release pins, cached on first use.

        The release does not redistribute it; it is one tensor of a Comfy-Org
        MiniMax-H3 checkpoint, read from the local hub cache when that file is
        there and by HTTP byte range otherwise, then checked against the
        release config's sha256.
        """
        import hashlib
        import json

        with open(os.path.join(self.release_path, self.release_config_file)) as f:
            table = json.load(f)["curve_branches"]["table"]
        path = os.path.join(
            envs.SGLANG_DIFFUSION_CACHE_ROOT,
            "minimax_h3",
            f"adaln_t_table-{table['sha256'][:16]}.bin",
        )
        if os.path.isfile(path):
            return path
        data = _read_hub_safetensors_tensor(
            table["repo"], table["revision"], table["file"], table["tensor"]
        )
        if hashlib.sha256(data).hexdigest() != table["sha256"]:
            raise ValueError(
                "adaln_t_table does not match the sha256 the release config pins"
            )
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.{os.getpid()}.tmp"
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
        return path

    def add_source_stages(self, server_args: ServerArgs) -> None:
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.stages.regenerate_source import (
            MiniMaxH3RegenerateSourceStage,
        )

        self.add_stage(
            MiniMaxH3RegenerateSourceStage(
                video_vae=self.get_module("video_vae"),
                audio_vae=self.get_module("audio_vae"),
                upscaler=self.latent_upscaler,
                pipeline_config=server_args.pipeline_config,
            )
        )


def _read_hub_safetensors_tensor(
    repo: str, revision: str, path: str, name: str
) -> bytes:
    """One fp32 tensor's raw bytes from a hub safetensors file.

    Uses the local hub cache when the file is there; otherwise reads only the
    header and the tensor by HTTP byte range.
    """
    import json
    import struct

    from huggingface_hub import HfFileSystem, try_to_load_from_cache

    cached = try_to_load_from_cache(repo, path, revision=revision)
    if isinstance(cached, str):
        from safetensors import safe_open

        with safe_open(cached, "pt") as f:
            return f.get_tensor(name).to(torch.float32).contiguous().numpy().tobytes()
    with HfFileSystem().open(f"{repo}@{revision}/{path}", "rb", block_size=0) as f:
        (header_len,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(header_len))
        begin, end = header[name]["data_offsets"]
        f.seek(8 + header_len + begin)
        return f.read(end - begin)


EntryClass = [
    MiniMaxH3Pipeline,
    FastH3Pipeline,
    PikaH3FlexPipeline,
    OpenH3RegeneratePipeline,
]
