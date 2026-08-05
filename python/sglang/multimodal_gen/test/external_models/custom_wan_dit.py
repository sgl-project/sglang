from typing import Any

from sglang.multimodal_gen.runtime.models.dits.wanvideo import (
    WanTransformer3DModel as OriginalWanTransformer3DModel,
)


class WanTransformer3DModel(OriginalWanTransformer3DModel):
    """Custom WAN DiT model for testing external model replacement."""

    _is_custom = True

    def __init__(
        self,
        config,
        hf_config: dict[str, Any] | None = None,
        quant_config=None,
    ) -> None:
        super().__init__(config, hf_config=hf_config, quant_config=quant_config)
        print(f"[CustomDiT] Initialized custom WanTransformer3DModel")


EntryClass = WanTransformer3DModel
