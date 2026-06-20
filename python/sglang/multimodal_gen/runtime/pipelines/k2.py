"""Krea-2 (K2) text-to-image pipeline.

K2 ships a single MMDiT safetensors (no diffusers ``model_index.json``); the
text encoder (Qwen3-VL-4B) and the autoencoder (Qwen-Image VAE) are pulled from
their own Hugging Face repos. ``load_modules`` assembles the three sources, then
the Hybrid stage chain runs: Krea2BeforeDenoisingStage -> DenoisingStage ->
DecodingStage.
"""

import os
from typing import Any

import torch
from transformers import (
    AutoModel,
    AutoTokenizer,
    Qwen2TokenizerFast,
    Qwen3VLForConditionalGeneration,
)

from sglang.multimodal_gen.configs.models.dits.k2 import K2DitConfig
from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.loader.component_loaders.vae_loader import VAELoader
from sglang.multimodal_gen.runtime.loader.fsdp_load import (
    load_model_from_full_model_state_dict,
)
from sglang.multimodal_gen.runtime.loader.utils import (
    _list_safetensors_files,
    get_param_names_mapping,
    set_default_torch_dtype,
)
from sglang.multimodal_gen.runtime.loader.weight_utils import (
    safetensors_weights_iterator,
)
from sglang.multimodal_gen.runtime.models.registry import ModelRegistry
from sglang.multimodal_gen.runtime.models.schedulers.scheduling_k2_flow import (
    K2FlowMatchScheduler,
)
from sglang.multimodal_gen.runtime.pipelines_core import LoRAPipeline
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages import DenoisingStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.k2 import (
    Krea2BeforeDenoisingStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import maybe_download_model
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.utils.precision import resolve_precision

logger = init_logger(__name__)

TEXT_ENCODER_REPO = "Qwen/Qwen3-VL-4B-Instruct"
VAE_REPO = "Qwen/Qwen-Image"
_TEXT_MAX_LENGTH = 512


class K2Pipeline(LoRAPipeline, ComposedPipelineBase):
    pipeline_name = "K2Pipeline"

    from sglang.multimodal_gen.configs.pipeline_configs.k2 import K2PipelineConfig
    from sglang.multimodal_gen.configs.sample.k2 import K2SamplingParams

    pipeline_config_cls = K2PipelineConfig
    sampling_params_cls = K2SamplingParams

    _required_config_modules = [
        "text_encoder",
        "tokenizer",
        "processor",
        "vae",
        "transformer",
        "scheduler",
    ]

    def initialize_pipeline(self, server_args: ServerArgs):
        self.modules["scheduler"] = K2FlowMatchScheduler()
        vae_config = server_args.pipeline_config.vae_config
        if hasattr(vae_config, "post_init"):
            vae_config.post_init()

    def _resolve_dit_weights(self) -> list[str]:
        """Resolve the K2 MMDiT to a list of safetensors shard paths.

        Handles a single ``.safetensors`` file, a local directory, and a Hugging
        Face repo id (downloaded first). Within a directory it prefers the named
        ``turbo``/``raw`` checkpoints, then discovers single or sharded weights in
        a diffusers-style ``transformer/`` subfolder, then at the root.
        """
        path = self.model_path
        if os.path.isfile(path) and path.endswith(".safetensors"):
            return [path]
        if not os.path.isdir(path):
            # HF repo id (or any non-local path): resolve to a local snapshot.
            path = maybe_download_model(path)
        # Prefer the distilled turbo checkpoint, then the raw one, by name.
        for name in ("turbo.safetensors", "raw.safetensors"):
            candidate = os.path.join(path, name)
            if os.path.isfile(candidate):
                return [candidate]
        # Otherwise discover all safetensors (single or sharded, via the index) in
        # a diffusers-style transformer/ subfolder, then at the root.
        for sub in ("transformer", "."):
            files = _list_safetensors_files(os.path.join(path, sub))
            if files:
                return files
        raise FileNotFoundError(
            f"No K2 MMDiT safetensors found at {self.model_path} (looked for "
            "turbo.safetensors / raw.safetensors, a transformer/ subfolder, and "
            "*.safetensors at the root)."
        )

    def _resolve_repo_dir(self) -> str | None:
        """Local snapshot directory of the model (for its bundled aux subfolders), or
        ``None`` for a bare single-file checkpoint."""
        p = self.model_path
        if os.path.isfile(p):
            return None
        if not os.path.isdir(p):
            p = maybe_download_model(p)
        return p if os.path.isdir(p) else None

    def _dit_load_error_hint(
        self, dit_weights: list[str], model, exc: Exception
    ) -> str:
        """Build an actionable message when the strict DiT load fails.

        Surfaces the checkpoint vs. model parameter naming so a renamed public
        checkpoint can be fixed with a ``param_names_mapping`` entry.
        """
        ckpt_keys: list[str] = []
        try:
            from safetensors import safe_open

            for f in dit_weights:
                with safe_open(f, framework="pt") as sf:
                    ckpt_keys.extend(sf.keys())
        except Exception:
            pass
        model_keys = list(model.state_dict().keys())
        only_ckpt = sorted(set(ckpt_keys) - set(model_keys))[:8]
        only_model = sorted(set(model_keys) - set(ckpt_keys))[:8]
        return (
            f"Failed to load the K2 MMDiT from {dit_weights}: {exc}\n"
            f"Checkpoint has {len(ckpt_keys)} tensors; the model expects "
            f"{len(model_keys)} params. If the published checkpoint renamed "
            "parameters, add a param_names_mapping in K2ArchConfig.\n"
            f"  checkpoint-only keys (sample): {only_ckpt}\n"
            f"  model-only keys (sample): {only_model}"
        )

    def _load_transformer(self, server_args: ServerArgs):
        dit_weights = self._resolve_dit_weights()
        logger.info("Loading K2 MMDiT from %s", dit_weights)

        dit_config = server_args.pipeline_config.dit_config
        if not isinstance(dit_config, K2DitConfig):
            dit_config = K2DitConfig()
            server_args.pipeline_config.dit_config = dit_config

        model_cls, _ = ModelRegistry.resolve_model_cls("K2Transformer2DModel")
        default_dtype = resolve_precision(
            server_args, "dit", precision_attr="dit_precision"
        )
        server_args.model_paths["transformer"] = os.path.dirname(dit_weights[0]) or "."

        with set_default_torch_dtype(default_dtype), torch.device("meta"):
            model = model_cls(config=dit_config, hf_config={})

        try:
            load_model_from_full_model_state_dict(
                model,
                safetensors_weights_iterator(dit_weights),
                get_local_torch_device(),
                default_dtype,
                strict=True,
                cpu_offload=server_args.dit_cpu_offload,
                param_names_mapping=get_param_names_mapping(
                    dit_config.arch_config.param_names_mapping
                ),
            )
        except Exception as exc:
            raise RuntimeError(
                self._dit_load_error_hint(dit_weights, model, exc)
            ) from exc
        for n, p in model.named_parameters():
            p.requires_grad = False
        return model

    def load_modules(
        self,
        server_args: ServerArgs,
        loaded_modules: dict[str, torch.nn.Module] | None = None,
    ) -> dict[str, Any]:
        transformer = self._load_transformer(server_args)

        te_dtype = torch.bfloat16
        # The released repo bundles text_encoder/ tokenizer/ vae/ subfolders; use them
        # so the model is self-contained, falling back to the standalone Qwen repos for
        # a bare single-file checkpoint.
        repo = self._resolve_repo_dir()

        def _subdir(name: str) -> str | None:
            d = os.path.join(repo, name) if repo else None
            return d if d and os.path.isdir(d) else None

        te_dir, tok_dir, vae_dir = (
            _subdir("text_encoder"),
            _subdir("tokenizer"),
            _subdir("vae"),
        )

        te_src = te_dir or TEXT_ENCODER_REPO
        logger.info("Loading text encoder from %s", te_src)
        # Loaded on CPU; the component residency manager moves it to the GPU only for
        # text encoding and offloads it during the denoise loop (frees ~8GB) when
        # --text-encoder-cpu-offload is set, also avoiding DiT co-residence on a 32GB card.
        if te_dir is not None:
            # Bundled encoder: load the architecture declared in its config (Qwen3VLModel).
            text_encoder = AutoModel.from_pretrained(te_dir, torch_dtype=te_dtype)
        else:
            text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
                TEXT_ENCODER_REPO, torch_dtype=te_dtype
            )
        text_encoder = text_encoder.eval().requires_grad_(False)
        # K2 is text-only: drop the unused vision tower to shrink the encoder and its
        # CPU<->GPU page. (It may sit on the encoder directly or under .model.)
        for owner in (text_encoder, getattr(text_encoder, "model", None)):
            if owner is not None and getattr(owner, "visual", None) is not None:
                del owner.visual
                break

        tok_src = tok_dir or TEXT_ENCODER_REPO
        tokenizer = AutoTokenizer.from_pretrained(tok_src, max_length=_TEXT_MAX_LENGTH)
        processor = Qwen2TokenizerFast.from_pretrained(
            tok_src, max_length=_TEXT_MAX_LENGTH
        )

        vae_path = vae_dir or os.path.join(maybe_download_model(VAE_REPO), "vae")
        logger.info("Loading VAE from %s", vae_path)
        vae = VAELoader().load_customized(vae_path, server_args, "vae")

        return {
            "transformer": transformer,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "processor": processor,
            "vae": vae,
            "scheduler": self.modules.get("scheduler"),
        }

    def create_pipeline_stages(self, server_args: ServerArgs):
        self.add_stage(
            Krea2BeforeDenoisingStage(
                vae=self.get_module("vae"),
                text_encoder=self.get_module("text_encoder"),
                tokenizer=self.get_module("tokenizer"),
                processor=self.get_module("processor"),
                transformer=self.get_module("transformer"),
                scheduler=self.get_module("scheduler"),
            ),
            "k2_before_denoising_stage",
        )
        self.add_stage(
            DenoisingStage(
                transformer=self.get_module("transformer"),
                scheduler=self.get_module("scheduler"),
            ),
        )
        self.add_standard_decoding_stage()


EntryClass = [K2Pipeline]
