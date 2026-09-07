from sglang.multimodal_gen.configs.models.adapter.ltx_2_connector import (
    LTX2ConnectorConfig,
)
from sglang.multimodal_gen.configs.models.adapter.ltx_2_duration_head import (
    LTX2DurationHeadConfig,
)
from sglang.multimodal_gen.runtime.loader.component_loaders.component_loader import (
    PlainStateDictComponentLoader,
)


class AdapterLoader(PlainStateDictComponentLoader):
    component_names = ["connectors", "duration_head"]

    config_classes = {
        "connectors": LTX2ConnectorConfig,
        "duration_head": LTX2DurationHeadConfig,
    }
