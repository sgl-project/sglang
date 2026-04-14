# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field
from typing import Any

from sglang.multimodal_gen.configs.models.base import ArchConfig, ModelConfig
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum


@dataclass
class AdapterArchConfig(ArchConfig):
    hidden_size: int = 0
    num_attention_heads: int = 0
    num_channels_latents: int = 0


@dataclass
class AdapterConfig(ModelConfig[AdapterArchConfig]):
    arch_config: AdapterArchConfig = field(default_factory=AdapterArchConfig)
    _internal_config_fields = (
        "_fsdp_shard_conditions",
        "_compile_conditions",
        "_supported_attention_backends",
        "param_names_mapping",
        "reverse_param_names_mapping",
        "exclude_lora_layers",
        "boundary_ratio",
    )

    # sglang-diffusion Adapter-specific parameters
    prefix: str = ""
    _fsdp_shard_conditions: list = field(default_factory=list)
    _compile_conditions: list = field(default_factory=list)
    param_names_mapping: dict = field(default_factory=dict)
    reverse_param_names_mapping: dict = field(default_factory=dict)
    _supported_attention_backends: set[AttentionBackendEnum] = field(
        default_factory=lambda: {
            AttentionBackendEnum.SLIDING_TILE_ATTN,
            AttentionBackendEnum.SAGE_ATTN,
            AttentionBackendEnum.FA,
            AttentionBackendEnum.AITER,
            AttentionBackendEnum.AITER_SAGE,
            AttentionBackendEnum.TORCH_SDPA,
            AttentionBackendEnum.VIDEO_SPARSE_ATTN,
            AttentionBackendEnum.VMOBA_ATTN,
            AttentionBackendEnum.SAGE_ATTN_3,
        }
    )
    exclude_lora_layers: list[str] = field(default_factory=list)
    boundary_ratio: float | None = None

    def refresh_model_config(self) -> None:
        if hasattr(self.arch_config, "_fsdp_shard_conditions"):
            self._fsdp_shard_conditions = list(self.arch_config._fsdp_shard_conditions)
        if hasattr(self.arch_config, "_compile_conditions"):
            self._compile_conditions = list(self.arch_config._compile_conditions)
        if hasattr(self.arch_config, "param_names_mapping"):
            self.param_names_mapping = dict(self.arch_config.param_names_mapping)
        if hasattr(self.arch_config, "reverse_param_names_mapping"):
            self.reverse_param_names_mapping = dict(
                self.arch_config.reverse_param_names_mapping
            )
        if hasattr(self.arch_config, "_supported_attention_backends"):
            self._supported_attention_backends = set(
                self.arch_config._supported_attention_backends
            )
        if hasattr(self.arch_config, "exclude_lora_layers"):
            self.exclude_lora_layers = list(self.arch_config.exclude_lora_layers)
        if hasattr(self.arch_config, "boundary_ratio"):
            self.boundary_ratio = self.arch_config.boundary_ratio
        if not self._compile_conditions:
            self._compile_conditions = self._fsdp_shard_conditions.copy()

    @staticmethod
    def add_cli_args(parser: Any, prefix: str = "dit-config") -> Any:
        """Add CLI arguments for AdapterConfig fields"""
        parser.add_argument(
            f"--{prefix}.prefix",
            type=str,
            dest=f"{prefix.replace('-', '_')}.prefix",
            default=AdapterConfig.prefix,
            help="Prefix for the Adapter",
        )

        return parser
