from sglang.multimodal_gen.configs.models.bridges.mova_dual_tower import (
    MOVADualTowerConfig,
)
from sglang.multimodal_gen.runtime.loader.component_loaders.component_loader import (
    PlainStateDictComponentLoader,
)


class BridgeLoader(PlainStateDictComponentLoader):
    """Loader for MOVA dual tower bridge with FSDP support."""

    supports_fsdp_loading = True

    component_names = ["dual_tower_bridge"]
    config_classes = {"dual_tower_bridge": MOVADualTowerConfig}
