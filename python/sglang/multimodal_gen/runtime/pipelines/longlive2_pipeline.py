# SPDX-License-Identifier: Apache-2.0
from sglang.multimodal_gen.configs.pipeline_configs.longlive2 import LongLive2T2VConfig
from sglang.multimodal_gen.configs.sample.longlive2 import LongLive2SamplingParams
from sglang.multimodal_gen.runtime.models.schedulers.scheduling_flow_unipc_multistep import (
    FlowUniPCMultistepScheduler,
)
from sglang.multimodal_gen.runtime.pipelines.wan_causal_dmd_pipeline import (
    WanCausalDMDPipeline,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages import InputValidationStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.longlive2 import (
    LongLive2CausalDenoisingStage,
    LongLive2ImageVAEEncodingStage,
    LongLive2LatentPreparationStage,
    LongLive2TextEncodingStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs


class LongLive2Pipeline(WanCausalDMDPipeline):
    pipeline_name = "LongLive2Pipeline"
    pipeline_config_cls = LongLive2T2VConfig
    sampling_params_cls = LongLive2SamplingParams
    def initialize_pipeline(self, server_args: ServerArgs):
        self.modules["scheduler"] = FlowUniPCMultistepScheduler(
            num_train_timesteps=1000,
            shift=1,
            use_dynamic_shifting=False,
        )

    def create_pipeline_stages(self, server_args: ServerArgs) -> None:
        self.add_stage(InputValidationStage())
        self.add_stage(
            LongLive2TextEncodingStage(
                text_encoders=[self.get_module("text_encoder")],
                tokenizers=[self.get_module("tokenizer")],
            )
        )
        self.add_stage(
            LongLive2ImageVAEEncodingStage(
                vae=self.get_module("vae"),
                component_name="vae",
            )
        )
        self.add_stage(
            LongLive2LatentPreparationStage(
                scheduler=self.get_module("scheduler"),
                transformer=self.get_module("transformer"),
            )
        )
        self.add_stage(
            LongLive2CausalDenoisingStage(
                transformer=self.get_module("transformer"),
                scheduler=self.get_module("scheduler"),
            ),
        )
        self.add_standard_decoding_stage()

EntryClass = LongLive2Pipeline
