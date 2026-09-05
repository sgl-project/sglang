# SPDX-License-Identifier: Apache-2.0

from sglang.multimodal_gen.runtime.loader.component_loaders.component_loader import (
    PlainStateDictComponentLoader,
)


class SoundTokenizerLoader(PlainStateDictComponentLoader):
    component_names = ["sound_tokenizer"]
    default_precision_attr = "vae_precision"

    def build_model_config(self, config, component_name):
        return {"config": config}

    def validate_checkpoint_keys(self, missing, unexpected, component_name):
        # the native tokenizer is decoder-only; encoder weights are unused
        unexpected = [name for name in unexpected if not name.startswith("encoder.")]
        super().validate_checkpoint_keys(missing, unexpected, component_name)
