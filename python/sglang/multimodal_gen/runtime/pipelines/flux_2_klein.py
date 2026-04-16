from sglang.multimodal_gen.runtime.pipelines.flux_2 import Flux2Pipeline
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.flux_2_klein_kvcache import (
    Flux2KleinKVCacheDenoisingStage,
)
from sglang.multimodal_gen.runtime.server_args import get_global_server_args


class Flux2KleinPipeline(Flux2Pipeline):
    pipeline_name = "Flux2KleinPipeline"

    def add_standard_denoising_stage(
        self,
        transformer_key: str = "transformer",
        transformer_2_key: str | None = "transformer_2",
        scheduler_key: str = "scheduler",
        vae_key: str | None = "vae",
    ):
        """Use KV-cache denoising stage when pipeline config has use_kv_cache=True."""
        if not getattr(get_global_server_args().pipeline_config, "use_kv_cache", False):
            return super().add_standard_denoising_stage(
                transformer_key, transformer_2_key, scheduler_key, vae_key
            )

        kwargs = {
            "transformer": self.get_module(transformer_key),
            "scheduler": self.get_module(scheduler_key),
        }

        if transformer_2_key:
            transformer_2 = self.get_module(transformer_2_key, None)
            if transformer_2 is not None:
                kwargs["transformer_2"] = transformer_2

        if vae_key:
            vae = self.get_module(vae_key, None)
            if vae is not None:
                kwargs["vae"] = vae
                kwargs["pipeline"] = self

        return self.add_stage(Flux2KleinKVCacheDenoisingStage(**kwargs))


EntryClass = Flux2KleinPipeline
