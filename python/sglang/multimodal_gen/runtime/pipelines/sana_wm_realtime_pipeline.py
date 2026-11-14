# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from sglang.multimodal_gen.configs.pipeline_configs.sana_wm import (
    SanaWMRealtimeConfig,
)
from sglang.multimodal_gen.runtime.pipelines.sana_wm_pipeline import (
    SanaWMTwoStagePipeline,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.sana_wm import (
    SanaWMTextEncodingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.sana_wm.realtime_chain import (
    SanaWMCameraCondStage,
    SanaWMCausalDecodeChainStage,
    SanaWMChunkedRefinerChainStage,
    SanaWMCondFrameEncodeStage,
    SanaWMRealtimeLatentPrepStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.sana_wm.refiner import (
    default_sana_wm_refiner_dtype,
    sana_wm_skip_refiner_enabled,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.sana_wm.streaming import (
    SanaWMStreamingDenoisingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.sana_wm.streaming_refiner import (
    SanaWMStreamingRefinerStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.realtime import (
    RealtimeInputValidationStage,
    RealtimeTextEncodingStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import maybe_download_model

DEFAULT_SANA_WM_TEXT_ENCODER = "Efficient-Large-Model/gemma-2-2b-it"


class SanaWMRealtimeTextEncodingStage(
    RealtimeTextEncodingStage, SanaWMTextEncodingStage
):
    """Realtime text encoding using SANA-WM prompt processing.

    MRO contract: ``RealtimeTextEncodingStage.forward`` (per-session cache) calls
    ``super().forward``, which must resolve to ``SanaWMTextEncodingStage.forward``.
    This preserves the chi prompt prefix and official prompt window used by the
    batch path.
    """


class SanaWMRealtimePipeline(SanaWMTwoStagePipeline):
    """SANA-WM realtime interactive pipeline.

    Extends the two-stage pipeline to inherit refiner sub-module loading (``transformer_2`` /
    ``connectors`` / ``text_encoder_2`` / ``tokenizer_2``). The streaming refiner stage built
    here is purely a carrier of those modules handed to ``SanaWMRealtimeStage``, not added to
    the stage list (the realtime stage drives the incremental stage-1 session + chunked refiner
    runner per user action).
    """

    pipeline_name = "SanaWMRealtimePipeline"
    is_video_pipeline = True
    # Must be the realtime config so get_realtime_model_adapter() resolves the
    # SANA-WM adapter (the realtime registry keys on SanaWMRealtimeConfig).
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
        """Build the chunked streaming refiner carrier when refiner modules exist."""
        if sana_wm_skip_refiner_enabled():
            return None
        if self.get_module("transformer_2") is None:
            return None

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
        refiner_stage = self._build_realtime_refiner_stage(server_args)
        common = dict(
            transformer=self.get_module("transformer"),
            vae=self.get_module("vae"),
            model_path=self.model_path,
        )
        self.add_stage(RealtimeInputValidationStage())
        self.add_stage(
            SanaWMRealtimeTextEncodingStage(
                text_encoders=[self.get_module("text_encoder")],
                tokenizers=[self.get_module("tokenizer")],
            )
        )
        self.add_stage(SanaWMCondFrameEncodeStage(**common))
        self.add_stage(
            SanaWMRealtimeLatentPrepStage(
                use_refiner=refiner_stage is not None, **common
            )
        )
        self.add_stage(SanaWMCameraCondStage(**common))
        self.add_stage(
            SanaWMStreamingDenoisingStage(
                transformer=self.get_module("transformer"), keep_resident=True
            )
        )
        if refiner_stage is not None:
            self.add_stage(
                SanaWMChunkedRefinerChainStage(refiner_stage=refiner_stage, **common)
            )
        self.add_stage(SanaWMCausalDecodeChainStage(**common))


EntryClass = SanaWMRealtimePipeline
