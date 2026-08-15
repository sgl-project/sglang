from sglang.multimodal_gen.runtime.pipelines.flux_2 import Flux2Pipeline


class Flux2KleinPipeline(Flux2Pipeline):
    pipeline_name = "Flux2KleinPipeline"


EntryClass = Flux2KleinPipeline
