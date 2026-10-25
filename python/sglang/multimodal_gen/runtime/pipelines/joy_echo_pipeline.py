# SPDX-License-Identifier: Apache-2.0
from sglang.multimodal_gen.configs.pipeline_configs.joy_echo import (
    JoyEchoPipelineConfig,
)
from sglang.multimodal_gen.runtime.pipelines.ltx_2_pipeline import (
    _add_ltx2_front_stages,
    _BaseLTX2Pipeline,
    prepare_ltx2_mu,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.image_encoding import (
    LTX2ImageEncodingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.joy_echo import (
    JoyEchoAVDecodingStage,
    JoyEchoDMDDenoisingStage,
    JoyEchoMemoryBankFetchStage,
    JoyEchoMultishotSetupStage,
    JoyEchoSigmaPreparationStage,
    PairedAudioVideoMemoryBank,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.ltx_2 import (
    LTX2AVLatentPreparationStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs


class JoyEchoPipeline(_BaseLTX2Pipeline):
    pipeline_name = "JoyEchoPipeline"
    is_video_pipeline = True

    def __init__(self, *args, **kwargs):
        self._memory_bank: PairedAudioVideoMemoryBank | None = None
        self.multishot_index: int = 0
        self._multishot_session_id: str | None = None
        super().__init__(*args, **kwargs)

    def _get_or_create_memory_bank(
        self, config: JoyEchoPipelineConfig
    ) -> PairedAudioVideoMemoryBank:
        if self._memory_bank is None:
            self._memory_bank = PairedAudioVideoMemoryBank(
                max_size=int(config.memory_max_size),
                num_fix_frames=int(config.memory_num_fix_frames),
            )
        return self._memory_bank

    def reset_memory_bank(self) -> None:
        if self._memory_bank is not None:
            self._memory_bank.memory.clear()

    def create_pipeline_stages(self, server_args: ServerArgs):
        config = server_args.pipeline_config
        if not isinstance(config, JoyEchoPipelineConfig):
            raise TypeError(
                f"JoyEchoPipeline requires JoyEchoPipelineConfig, got {type(config)}"
            )

        memory_bank = self._get_or_create_memory_bank(config)
        self.add_stage(JoyEchoMultishotSetupStage(pipeline=self))
        _add_ltx2_front_stages(self)
        self.add_stage(JoyEchoSigmaPreparationStage())
        self.add_standard_timestep_preparation_stage(
            prepare_extra_kwargs=[prepare_ltx2_mu]
        )
        self.add_stages(
            [
                LTX2AVLatentPreparationStage(
                    scheduler=self.get_module("scheduler"),
                    transformer=self.get_module("transformer"),
                    audio_vae=self.get_module("audio_vae"),
                ),
                LTX2ImageEncodingStage(
                    vae=self.get_module("vae"),
                ),
                JoyEchoMemoryBankFetchStage(
                    memory_bank=memory_bank,
                    vae=self.get_module("vae"),
                ),
                JoyEchoDMDDenoisingStage(
                    transformer=self.get_module("transformer"),
                    scheduler=self.get_module("scheduler"),
                    vae=self.get_module("vae"),
                    audio_vae=self.get_module("audio_vae"),
                    sampler_name="euler",
                    pipeline=self,
                ),
                JoyEchoAVDecodingStage(
                    vae=self.get_module("vae"),
                    audio_vae=self.get_module("audio_vae"),
                    vocoder=self.get_module("vocoder"),
                    memory_bank=memory_bank,
                    pipeline=self,
                ),
            ]
        )


EntryClass = JoyEchoPipeline
