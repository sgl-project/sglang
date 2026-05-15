from sglang.multimodal_gen.runtime.pipelines.wan_pipeline import (
    WanPipeline as OriginalWanPipeline,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs


class WanPipeline(OriginalWanPipeline):
    """Custom WAN pipeline that overrides the built-in implementation."""

    pipeline_name = "WanPipeline"

    def initialize_pipeline(self, server_args: ServerArgs):
        print("[CustomPipeline] WanPipeline.initialize_pipeline")
        super().initialize_pipeline(server_args)

    def create_pipeline_stages(self, server_args: ServerArgs) -> None:
        print("[CustomPipeline] WanPipeline.create_pipeline_stages")
        super().create_pipeline_stages(server_args)


EntryClass = WanPipeline
