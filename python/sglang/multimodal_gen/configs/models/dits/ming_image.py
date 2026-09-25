# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models.dits.zimage import (
    ZImageArchConfig,
    ZImageDitConfig,
)


@dataclass
class MingImageArchConfig(ZImageArchConfig):
    axes_lens: tuple[int, ...] = (20480, 512, 512)
    alignment_padding_mode: str | None = None
    multi_frame_output: bool | None = None


@dataclass
class MingImageDitConfig(ZImageDitConfig):
    arch_config: MingImageArchConfig = field(default_factory=MingImageArchConfig)
    prefix: str = "ming_image"

    def update_model_arch(self, config):
        config = dict(config)
        config["num_layers"] = config.pop("n_layers", self.num_layers)
        config["num_attention_heads"] = config.pop("n_heads", self.num_attention_heads)
        if type(config.get("multi_frame_output")) is not bool or (
            config.get("alignment_padding_mode"),
            config.get("multi_frame_output"),
        ) not in (
            ("zero_masked", False),
            ("learned", True),
        ):
            raise ValueError(
                "Ming-Image requires an explicit, valid checkpoint padding/output contract"
            )
        super().update_model_arch(config)
