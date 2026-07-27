# Copyright 2025 ByteDance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Native pipelines for BAGEL text-to-image generation and image editing.

Source: https://github.com/ByteDance-Seed/Bagel/blob/a2fa77dd8caeefc41e6607ae0ec17408d3f4ee9f/inferencer.py
"""

import json
import os
import re
import shutil
import tempfile
from collections.abc import Callable
from fnmatch import fnmatch
from typing import Any

import torch
from transformers import AutoTokenizer

from sglang.multimodal_gen import envs
from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.loader.utils import (
    get_memory_usage_of_component,
    set_default_torch_dtype,
)
from sglang.multimodal_gen.runtime.loader.weight_utils import (
    safetensors_weights_iterator,
)
from sglang.multimodal_gen.runtime.models.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.bagel import (
    BagelBeforeDenoisingStage,
    BagelEditBeforeDenoisingStage,
    BagelEditInputValidationStage,
    BagelInputValidationStage,
    validate_bagel_special_tokens,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import snapshot_download
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.utils.precision import resolve_precision

logger = init_logger(__name__)

_REQUIRED_CHECKPOINT_FILES = (
    "config.json",
    "llm_config.json",
    "ema.safetensors",
    "ae.safetensors",
)
_DOWNLOAD_PATTERNS = [
    *_REQUIRED_CHECKPOINT_FILES,
    "tokenizer*",
    "vocab*",
    "merges.txt",
    "*.model",
    "special_tokens_map.json",
    "added_tokens.json",
]
_TOKENIZER_ASSET_PATTERNS = tuple(_DOWNLOAD_PATTERNS[len(_REQUIRED_CHECKPOINT_FILES) :])
_LEGACY_CONFIG_PATTERN = re.compile(
    r'\{\s*"name"\s*:\s*\[\s*"BAGEL-7B-MoT"\s*\]\s*,?\s*\}\s*'
)
_EXPECTED_LLM_CONFIG = {
    "model_type": "qwen2",
    "hidden_size": 3584,
    "intermediate_size": 18944,
    "num_hidden_layers": 28,
    "num_attention_heads": 28,
    "num_key_value_heads": 4,
    "vocab_size": 152064,
}


class BagelPipeline(ComposedPipelineBase):
    """Load and execute BAGEL as a request-stateless T2I pipeline."""

    pipeline_name = "BagelPipeline"

    from sglang.multimodal_gen.configs.pipeline_configs.bagel import BagelPipelineConfig
    from sglang.multimodal_gen.configs.sample.bagel import BagelSamplingParams

    pipeline_config_cls = BagelPipelineConfig
    sampling_params_cls = BagelSamplingParams

    _required_config_modules = ["transformer", "vae", "tokenizer", "scheduler"]

    def validate_disagg_role(self, role: RoleType) -> None:
        """Reject disaggregated execution until BAGEL context transfer is defined."""
        if role != RoleType.MONOLITHIC:
            raise ValueError("BAGEL T2I supports monolithic serving only")

    @staticmethod
    def _validate_runtime_capabilities(server_args: ServerArgs) -> None:
        """Fail before model resolution when an unsupported runtime mode is set."""
        unsupported: list[str] = []
        if getattr(server_args, "enable_cfg_parallel", False):
            unsupported.append("CFG parallel")
        if int(getattr(server_args, "tp_size", 1) or 1) != 1:
            unsupported.append("TP")
        if int(getattr(server_args, "sp_degree", 1) or 1) != 1:
            unsupported.append("SP")
        if int(getattr(server_args, "ulysses_degree", 1) or 1) != 1:
            unsupported.append("Ulysses SP")
        if int(getattr(server_args, "ring_degree", 1) or 1) != 1:
            unsupported.append("Ring SP")
        if getattr(server_args, "use_fsdp_inference", False):
            unsupported.append("FSDP inference")
        if getattr(server_args, "enable_torch_compile", False):
            unsupported.append("torch.compile")
        if getattr(server_args, "dit_layerwise_offload", False) or getattr(
            server_args, "layerwise_offload_components", None
        ):
            unsupported.append("layerwise offload")
        if getattr(server_args, "dit_cpu_offload", False):
            unsupported.append("DiT CPU offload")
        if getattr(server_args, "vae_cpu_offload", False):
            unsupported.append("VAE CPU offload")
        if getattr(server_args, "cache_dit_config", None) is not None or bool(
            envs.SGLANG_CACHE_DIT_ENABLED
        ):
            unsupported.append("Cache-DiT")
        if getattr(server_args, "quantization", None) is not None:
            unsupported.append("quantization")
        if getattr(server_args, "lora_path", None) is not None:
            unsupported.append("LoRA")
        if getattr(server_args, "comfyui_mode", False):
            unsupported.append("ComfyUI mode")
        pipeline_config = server_args.pipeline_config
        if getattr(pipeline_config, "dit_precision", None) != "bf16":
            unsupported.append("non-BF16 DiT precision")
        if getattr(pipeline_config, "vae_precision", None) != "bf16":
            unsupported.append("non-BF16 VAE precision")
        if (
            getattr(
                getattr(pipeline_config, "image_encoder_config", None),
                "prefix",
                None,
            )
            == "vit_model"
            and pipeline_config.image_encoder_precision != "bf16"
        ):
            unsupported.append("non-BF16 image encoder precision")
        if unsupported:
            raise ValueError(
                "BAGEL pipeline does not support: "
                + ", ".join(sorted(set(unsupported)))
            )

    @staticmethod
    def _validate_checkpoint(path: str) -> dict[str, Any]:
        """Validate the strict non-Diffusers checkpoint marker set and architecture."""
        missing = [
            name
            for name in _REQUIRED_CHECKPOINT_FILES
            if not os.path.isfile(os.path.join(path, name))
        ]
        if missing:
            raise FileNotFoundError(
                f"BAGEL checkpoint {path!r} is missing required files: {missing}"
            )
        llm_config_path = os.path.join(path, "llm_config.json")
        try:
            with open(llm_config_path, encoding="utf-8") as config_file:
                llm_config = json.load(config_file)
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(
                f"Invalid BAGEL llm_config.json at {llm_config_path}: {error}"
            ) from error
        mismatches = {
            key: (expected, llm_config.get(key))
            for key, expected in _EXPECTED_LLM_CONFIG.items()
            if llm_config.get(key) != expected
        }
        if mismatches:
            raise ValueError(
                "BAGEL llm_config.json architecture mismatch: "
                + ", ".join(
                    f"{key}=expected {expected!r}, got {actual!r}"
                    for key, (expected, actual) in mismatches.items()
                )
            )

        config_path = os.path.join(path, "config.json")
        try:
            with open(config_path, encoding="utf-8") as config_file:
                raw_config = config_file.read()
            config = json.loads(raw_config)
        except json.JSONDecodeError as error:
            # Early local BAGEL snapshots shipped a name-only config with a
            # trailing comma. Accept it only after the exact LLM architecture
            # above has been verified; official Hub snapshots remain strict.
            if _LEGACY_CONFIG_PATTERN.fullmatch(raw_config) is None:
                raise ValueError(
                    f"Invalid BAGEL config.json at {config_path}: {error}"
                ) from error
            logger.warning(
                "Using legacy BAGEL config fallback for %s: %s",
                config_path,
                error,
            )
            return {
                "model_type": "bagel",
                "architectures": ["BagelForConditionalGeneration"],
                "llm_config": llm_config,
            }
        except OSError as error:
            raise ValueError(f"Unable to read BAGEL config.json: {error}") from error

        architecture_markers = [
            str(item).lower() for item in config.get("architectures", [])
        ]
        model_type = str(config.get("model_type", "")).lower()
        legacy_names = config.get("name", [])
        if isinstance(legacy_names, str):
            legacy_names = [legacy_names]
        has_bagel_marker = (
            "bagel" in model_type
            or any("bagel" in marker for marker in architecture_markers)
            or any("bagel" in str(name).lower() for name in legacy_names)
        )
        if not has_bagel_marker:
            raise ValueError(
                "Checkpoint config.json does not identify a BAGEL architecture"
            )
        # The validated sidecar is authoritative. Do not allow a stale or
        # conflicting embedded config to bypass architecture validation in
        # later tokenizer/model construction.
        config["llm_config"] = llm_config
        return config

    @staticmethod
    def _load_tokenizer(checkpoint_path: str, checkpoint_config: dict[str, Any]) -> Any:
        """Load tokenizer assets without re-reading BAGEL's root config.

        Early local checkpoints contain a trailing-comma ``config.json`` and
        current Transformers may inspect that file even when
        ``tokenizer_config.json`` names ``Qwen2Tokenizer``. Stage only the
        tokenizer files plus the already-validated Qwen2 config in a temporary
        directory; never rewrite the user's checkpoint.

        Args:
            checkpoint_path: Validated BAGEL checkpoint directory.
            checkpoint_config: Parsed BAGEL config containing ``llm_config``.

        Returns:
            Loaded Hugging Face tokenizer.

        Raises:
            FileNotFoundError: If no tokenizer assets are present.
            OSError: If Transformers cannot construct the tokenizer.
        """
        llm_config = checkpoint_config["llm_config"]
        asset_names = sorted(
            name
            for name in os.listdir(checkpoint_path)
            if os.path.isfile(os.path.join(checkpoint_path, name))
            and any(fnmatch(name, pattern) for pattern in _TOKENIZER_ASSET_PATTERNS)
        )
        if not asset_names:
            raise FileNotFoundError(
                f"BAGEL checkpoint {checkpoint_path!r} has no tokenizer assets"
            )

        with tempfile.TemporaryDirectory(prefix="sglang-bagel-tokenizer-") as temp_dir:
            for name in asset_names:
                shutil.copy2(
                    os.path.join(checkpoint_path, name),
                    os.path.join(temp_dir, name),
                )
            with open(
                os.path.join(temp_dir, "config.json"), "w", encoding="utf-8"
            ) as file:
                json.dump(llm_config, file)
            return AutoTokenizer.from_pretrained(
                temp_dir,
                local_files_only=True,
                trust_remote_code=False,
            )

    def _resolve_checkpoint(
        self, server_args: ServerArgs
    ) -> tuple[str, dict[str, Any]]:
        model_path = str(self.model_path)
        if os.path.isdir(model_path):
            resolved = os.path.realpath(model_path)
        elif os.path.exists(model_path):
            raise ValueError(
                "BAGEL model_path must be a directory or Hugging Face repo ID"
            )
        else:
            resolved = snapshot_download(
                repo_id=model_path,
                revision=server_args.revision,
                allow_patterns=_DOWNLOAD_PATTERNS,
            )
        return resolved, self._validate_checkpoint(resolved)

    @staticmethod
    def _stream_weights(
        module,
        weight_file: str,
        component_name: str,
        *,
        key_filter: Callable[[str], bool] | None = None,
    ) -> None:
        """Stream one safetensors file through a component's strict loader."""
        try:
            loaded = module.load_weights(
                safetensors_weights_iterator(
                    [weight_file], to_cpu=True, key_filter=key_filter
                )
            )
        except (OSError, RuntimeError) as error:
            message = str(error).lower()
            if "mmap" in message or "cannot allocate memory" in message:
                raise RuntimeError(
                    f"Unable to mmap {component_name} weights at {weight_file}. "
                    "BAGEL uses a large safetensors checkpoint; run on a host with "
                    "sufficient virtual-address and mmap capacity."
                ) from error
            raise
        logger.info("Loaded %d BAGEL %s tensors", len(loaded), component_name)

    @staticmethod
    def _validate_special_tokens(modules: dict[str, Any]) -> None:
        tokenizer = modules.get("tokenizer")
        if tokenizer is None:
            return
        validate_bagel_special_tokens(tokenizer)

    def load_modules(
        self,
        server_args: ServerArgs,
        loaded_modules: dict[str, torch.nn.Module] | None = None,
    ) -> dict[str, Any]:
        """Load only missing modules, resolving the checkpoint only when needed.

        Args:
            server_args: Runtime and precision configuration.
            loaded_modules: Optional injected modules, primarily for tests.

        Returns:
            All modules required by the selected BAGEL pipeline.

        Raises:
            ValueError: If runtime capabilities or checkpoint architecture are invalid.
            FileNotFoundError: If the checkpoint marker set is incomplete.
            RuntimeError: If the large checkpoint cannot be memory-mapped.
        """
        self._validate_runtime_capabilities(server_args)
        modules: dict[str, Any] = dict(loaded_modules or {})

        if "scheduler" not in modules:
            modules["scheduler"] = FlowMatchEulerDiscreteScheduler(shift=1.0)

        weight_backed_components = {"transformer", "vae", "tokenizer"}
        if "image_encoder" in self._required_config_modules:
            weight_backed_components.add("image_encoder")
        weight_backed_missing = weight_backed_components - modules.keys()
        checkpoint_path: str | None = None
        checkpoint_config: dict[str, Any] | None = None
        if weight_backed_missing:
            checkpoint_path, checkpoint_config = self._resolve_checkpoint(server_args)
            self.model_path = checkpoint_path
            for component_name in weight_backed_components:
                server_args.model_paths[component_name] = checkpoint_path

        device = get_local_torch_device()
        if "transformer" not in modules:
            assert checkpoint_path is not None and checkpoint_config is not None
            from sglang.multimodal_gen.runtime.models.dits.bagel_transformer import (
                BagelTransformer,
            )

            dtype = resolve_precision(
                server_args, "dit", precision_attr="dit_precision"
            )
            attention_backend = getattr(server_args, "attention_backend", None)
            if hasattr(server_args, "resolve_component_attention_backend"):
                component_backend, _ = server_args.resolve_component_attention_backend(
                    "transformer", "dit"
                )
                if component_backend is not None:
                    attention_backend = component_backend
            with set_default_torch_dtype(dtype), torch.device("meta"):
                transformer = BagelTransformer(
                    server_args.pipeline_config.dit_config,
                    hf_config=checkpoint_config,
                    attention_backend=attention_backend,
                )
            self._stream_weights(
                transformer,
                os.path.join(checkpoint_path, "ema.safetensors"),
                "transformer",
                key_filter=lambda name: not name.startswith(
                    (
                        "connector.",
                        "vit_model.",
                        "vit_pos_embed.",
                        "language_model.lm_head.",
                    )
                ),
            )
            modules["transformer"] = transformer.to(device=device, dtype=dtype).eval()

        if (
            "image_encoder" in weight_backed_components
            and "image_encoder" not in modules
        ):
            assert checkpoint_path is not None
            from sglang.multimodal_gen.runtime.models.encoders.bagel_vit import (
                BagelImageEncoder,
            )

            dtype = resolve_precision(
                server_args,
                "image_encoder",
                precision_attr="image_encoder_precision",
            )
            with set_default_torch_dtype(dtype), torch.device("meta"):
                image_encoder = BagelImageEncoder(
                    server_args.pipeline_config.image_encoder_config
                )
            self._stream_weights(
                image_encoder,
                os.path.join(checkpoint_path, "ema.safetensors"),
                "image encoder",
                key_filter=lambda name: name.startswith(
                    ("connector.", "vit_model.", "vit_pos_embed.")
                ),
            )
            modules["image_encoder"] = image_encoder.to(
                device=device, dtype=dtype
            ).eval()

        if "vae" not in modules:
            assert checkpoint_path is not None
            from sglang.multimodal_gen.runtime.models.vaes.bagel_vae import BagelVAE

            dtype = resolve_precision(
                server_args, "vae", precision_attr="vae_precision"
            )
            with set_default_torch_dtype(dtype), torch.device("meta"):
                vae = BagelVAE(server_args.pipeline_config.vae_config)
            self._stream_weights(
                vae,
                os.path.join(checkpoint_path, "ae.safetensors"),
                "VAE",
                key_filter=lambda name: (
                    name.startswith("encoder.")
                    and server_args.pipeline_config.vae_config.load_encoder
                )
                or (
                    name.startswith("decoder.")
                    and server_args.pipeline_config.vae_config.load_decoder
                )
                or name.startswith("reg."),
            )
            modules["vae"] = vae.to(device=device, dtype=dtype).eval()

        if "tokenizer" not in modules:
            assert checkpoint_path is not None and checkpoint_config is not None
            modules["tokenizer"] = self._load_tokenizer(
                checkpoint_path, checkpoint_config
            )

        missing_modules = set(self._required_config_modules) - modules.keys()
        if missing_modules:
            raise RuntimeError(
                f"BAGEL loader did not create modules: {sorted(missing_modules)}"
            )

        self._validate_special_tokens(modules)
        if not hasattr(self, "memory_usages"):
            self.memory_usages = {}
        for component_name in self._required_config_modules:
            memory_usage = get_memory_usage_of_component(modules[component_name])
            self.memory_usages[component_name] = memory_usage or 0.0
            server_args.model_loaded[component_name] = True
        return modules

    def initialize_pipeline(self, server_args: ServerArgs) -> None:
        """Validate runtime gates and initialize decoder-only VAE metadata."""
        self._validate_runtime_capabilities(server_args)
        vae_config = server_args.pipeline_config.vae_config
        if hasattr(vae_config, "post_init"):
            vae_config.post_init()

    def create_pipeline_stages(self, server_args: ServerArgs) -> None:
        """Build validation -> prefill -> standard denoise -> standard decode."""
        self.add_stage(BagelInputValidationStage(), "input_validation_stage")
        self.add_stage(
            BagelBeforeDenoisingStage(
                transformer=self.get_module("transformer"),
                tokenizer=self.get_module("tokenizer"),
                scheduler=self.get_module("scheduler"),
            ),
            "bagel_before_denoising_stage",
        )
        self.add_standard_denoising_stage(vae_key=None)
        self.add_standard_decoding_stage()


class BagelEditPipeline(BagelPipeline):
    """Load and execute BAGEL's explicit single-image Editing path."""

    pipeline_name = "BagelEditPipeline"

    from sglang.multimodal_gen.configs.pipeline_configs.bagel import (
        BagelEditPipelineConfig,
    )
    from sglang.multimodal_gen.configs.sample.bagel import BagelEditSamplingParams

    pipeline_config_cls = BagelEditPipelineConfig
    sampling_params_cls = BagelEditSamplingParams
    _required_config_modules = [
        "transformer",
        "vae",
        "image_encoder",
        "tokenizer",
        "scheduler",
    ]

    def create_pipeline_stages(self, server_args: ServerArgs) -> None:
        """Build validation -> image prefill -> standard denoise/decode."""
        self.add_stage(BagelEditInputValidationStage(), "input_validation_stage")
        self.add_stage(
            BagelEditBeforeDenoisingStage(
                transformer=self.get_module("transformer"),
                vae=self.get_module("vae"),
                image_encoder=self.get_module("image_encoder"),
                tokenizer=self.get_module("tokenizer"),
                scheduler=self.get_module("scheduler"),
            ),
            "bagel_edit_before_denoising_stage",
        )
        self.add_standard_denoising_stage(vae_key=None)
        self.add_standard_decoding_stage()


EntryClass = [BagelPipeline, BagelEditPipeline]
