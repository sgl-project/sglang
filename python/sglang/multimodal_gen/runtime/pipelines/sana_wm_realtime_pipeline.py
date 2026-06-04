# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from sglang.multimodal_gen.configs.pipeline_configs.sana_wm import (
    SanaWMRealtimeConfig,
)
from sglang.multimodal_gen.runtime.pipelines.sana_wm_pipeline import (
    SanaWMTwoStagePipeline,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages import (
    RealtimeInputValidationStage,
    RealtimeTextEncodingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.sana_wm import (
    SanaWMRealtimeStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.sana_wm.refiner import (
    default_sana_wm_refiner_dtype,
    sana_wm_skip_refiner_enabled,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import maybe_download_model

DEFAULT_SANA_WM_TEXT_ENCODER = "Efficient-Large-Model/gemma-2-2b-it"


class SanaWMRealtimePipeline(SanaWMTwoStagePipeline):
    """SANA-WM realtime interactive pipeline.

    Extends the two-stage pipeline so it inherits the LTX-2 refiner sub-module
    loading (``transformer_2`` / ``connectors`` / ``text_encoder_2`` /
    ``tokenizer_2``). The realtime stage drives OUR incremental stage-1 session +
    chunked refiner runner per user action, so we build a streaming refiner stage
    here purely as the carrier of those loaded refiner modules and hand it to
    ``SanaWMRealtimeStage`` (rather than adding it to the stage list).
    """

    pipeline_name = "SanaWMRealtimePipeline"
    is_video_pipeline = True
    # Must be the realtime config so has_realtime_model_adapter() matches and
    # the /v1/realtime_video WS router mounts (registry keys on SanaWMRealtimeConfig).
    pipeline_config_cls = SanaWMRealtimeConfig

    def _resolve_component_path(
        self, server_args: ServerArgs, module_name: str, load_module_name: str
    ) -> str:
        if (
            module_name in {"text_encoder", "tokenizer"}
            and module_name not in server_args.component_paths
        ):
            return maybe_download_model(DEFAULT_SANA_WM_TEXT_ENCODER)
        return super()._resolve_component_path(
            server_args,
            module_name,
            load_module_name,
        )

    def _build_realtime_refiner_stage(self, server_args: ServerArgs):
        """Construct OUR chunked streaming refiner stage from the loaded refiner
        modules, mirroring ``SanaWMTwoStagePipeline._maybe_add_refiner_stage``.
        Returns ``None`` when the refiner is disabled / not loaded."""
        if sana_wm_skip_refiner_enabled():
            return None
        if self.get_module("transformer_2") is None:
            return None
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.sana_wm.streaming_refiner import (
            SanaWMStreamingRefinerStage,
        )

        pc = server_args.pipeline_config
        return SanaWMStreamingRefinerStage(
            transformer=self.get_module("transformer_2"),
            connectors=self.get_module("connectors"),
            text_encoder=self.get_module("text_encoder_2"),
            tokenizer=self.get_module("tokenizer_2"),
            dtype=default_sana_wm_refiner_dtype(server_args),
            block_size=int(getattr(pc, "refiner_block_size", 3)),
            kv_max_frames=int(getattr(pc, "refiner_kv_max_frames", 11)),
            sink_size=int(getattr(pc, "sink_size", 1)),
            seed=int(getattr(pc, "refiner_seed", 42)),
        )

    def create_pipeline_stages(self, server_args: ServerArgs):
        self.add_stage(RealtimeInputValidationStage())
        self.add_stage(
            RealtimeTextEncodingStage(
                text_encoders=[self.get_module("text_encoder")],
                tokenizers=[self.get_module("tokenizer")],
            )
        )
        self.add_stage(
            SanaWMRealtimeStage(
                transformer=self.get_module("transformer"),
                vae=self.get_module("vae"),
                model_path=self.model_path,
                refiner_stage=self._build_realtime_refiner_stage(server_args),
            )
        )


EntryClass = SanaWMRealtimePipeline
