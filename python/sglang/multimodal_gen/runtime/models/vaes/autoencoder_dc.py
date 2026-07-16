# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable

import torch
from torch import nn

from sglang.multimodal_gen.configs.models.vaes.sana import SanaVAEConfig
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_decode_parallel_rank,
    get_decode_parallel_world_size,
)
from sglang.multimodal_gen.runtime.layers.parallel_conv import (
    gather_and_trim_height,
    split_height_for_parallel_decode,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload import (
    LayerwiseOffloadableModuleMixin,
)
from sglang.multimodal_gen.runtime.models.vaes.common import (
    can_install_spatial_shard_parallel_decode,
)
from sglang.multimodal_gen.runtime.models.vaes.parallel.diffusers_spatial import (
    enable_diffusers_decoder_spatial_parallel,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class AutoencoderDC(nn.Module, LayerwiseOffloadableModuleMixin):
    """Deep Compression Autoencoder wrapper with 32x spatial compression."""

    layerwise_offload_dit_group_enabled = False
    layer_names = ["_inner_model.encoder.down_blocks", "_inner_model.decoder.up_blocks"]

    def __init__(self, config: SanaVAEConfig = None, **kwargs):
        super().__init__()
        self._config = config
        self._inner_model = None
        self._loaded_state_dict: dict[str, torch.Tensor] = {}
        self._spatial_parallel_decode_enabled = False

    def _ensure_inner_model(self, state_dict: dict[str, torch.Tensor] | None = None):
        if self._inner_model is not None:
            return

        from diffusers import AutoencoderDC as DiffusersAutoencoderDC

        device = "cpu"
        state_to_load = (
            state_dict if state_dict is not None else self._loaded_state_dict
        )
        if state_to_load:
            first_tensor = next(iter(state_to_load.values()))
            device = first_tensor.device
        hf_config = {}
        if self._config is not None:
            arch = self._config.arch_config
            for key, value in vars(arch).items():
                if key == "extra_attrs" and isinstance(value, dict):
                    for ek, ev in value.items():
                        hf_config[ek] = ev
                elif not key.startswith("_") and not callable(value):
                    hf_config[key] = value

        self._inner_model = DiffusersAutoencoderDC.from_config(hf_config)

        if state_to_load:
            missing, unexpected = self._inner_model.load_state_dict(
                state_to_load, strict=False
            )
            if missing:
                logger.warning(
                    "AutoencoderDC missing keys when loading: %d keys", len(missing)
                )
                if len(missing) > 10:
                    logger.debug("First 10 missing keys: %s", list(missing)[:10])
                else:
                    logger.debug("Missing keys: %s", list(missing))
            if unexpected:
                logger.debug(
                    "AutoencoderDC unexpected keys when loading: %d keys",
                    len(unexpected),
                )
            if state_dict is None:
                self._loaded_state_dict.clear()

        self._inner_model = self._inner_model.to(device)
        if can_install_spatial_shard_parallel_decode(self._config):
            enable_diffusers_decoder_spatial_parallel(self._inner_model.decoder)
            self._spatial_parallel_decode_enabled = True

    @property
    def config(self):
        if self._inner_model is not None:
            return self._inner_model.config
        return self._config

    @property
    def dtype(self):
        if self._inner_model is not None:
            return next(self._inner_model.parameters()).dtype
        return torch.float32

    @property
    def device(self):
        if self._inner_model is not None:
            return next(self._inner_model.parameters()).device
        return torch.device("cpu")

    def encode(self, x: torch.Tensor, **kwargs):
        self._ensure_inner_model()
        return self._inner_model.encode(x, **kwargs)

    def decode(self, z: torch.Tensor, **kwargs):
        self._ensure_inner_model()
        z = z.to(dtype=self.dtype)
        if not self._spatial_parallel_decode_enabled:
            return self._inner_model.decode(z, **kwargs)

        expected_height = (
            z.shape[-2] * self._config.arch_config.spatial_compression_ratio
        )
        z, expected_height = split_height_for_parallel_decode(
            z,
            expected_height=expected_height,
            world_size=get_decode_parallel_world_size(),
            rank=get_decode_parallel_rank(),
        )
        decoded = self._inner_model.decode(z, **kwargs)
        if isinstance(decoded, tuple):
            sample = gather_and_trim_height(decoded[0], expected_height)
            return (sample, *decoded[1:])
        sample = gather_and_trim_height(decoded.sample, expected_height)
        return decoded.__class__(sample=sample)

    def forward(self, x: torch.Tensor, **kwargs):
        self._ensure_inner_model()
        return self._inner_model(x, **kwargs)

    def load_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        strict: bool = True,
        assign: bool = False,
    ):
        """Intercept load_state_dict to route weights into the inner diffusers model."""
        self._ensure_inner_model(state_dict=state_dict)

    def state_dict(self, *args, **kwargs) -> dict[str, torch.Tensor]:
        self._ensure_inner_model()
        return self._inner_model.state_dict(*args, **kwargs)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Buffer weights for deferred loading. The inner model is built lazily."""
        loaded_params: set[str] = set()
        for name, weight in weights:
            self._loaded_state_dict[name] = weight
            loaded_params.add(name)
        return loaded_params

    def to(self, *args, **kwargs):
        if self._inner_model is not None:
            self._inner_model = self._inner_model.to(*args, **kwargs)
        return super().to(*args, **kwargs)


EntryClass = AutoencoderDC
