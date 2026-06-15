"""MiMo-V2-ASR model.

Reuses the LM scaffold of ``MiMoForCausalLM`` and adds audio encoder
components via ``AudioEncoderMixin``. The encoder modules are attached as
top-level attributes (no ``audio_encoder.`` prefix) so the checkpoint
state_dict aligns 1:1 with ``self.named_parameters()``.
"""

import logging
from typing import Any, Iterable, List, Optional, Tuple

import torch

from sglang.srt.managers.mm_utils import (
    MultiModalityDataPaddingPatternMultimodalTokens,
    general_mm_embed_routine,
)
from sglang.srt.managers.schedule_batch import MultimodalInputs
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models import mimo_audio as _mimo_audio_module
from sglang.srt.models.mimo import MiMoForCausalLM
from sglang.srt.models.mimo_audio import AudioEncoderMixin, MiMoAudioEncoderConfig

logger = logging.getLogger(__name__)


def _maybe_override_audio_attn_for_blackwell() -> None:
    """Swap mimo_audio.flash_attn_varlen_func to upstream FA2 on GPUs that
    sgl-kernel's FA3 doesn't support.

    sgl-kernel FA3 only covers sm80/86/89/90 — on Blackwell consumer cards
    (sm_120 / RTX 50xx) its varlen kernel raises NotImplementedError. ASR is
    small enough to be deployed on those GPUs, so when FA3 isn't supported
    we replace the module-level reference with upstream flash-attn (FA2),
    which works on sm_120. No-op on supported GPUs (FA3 stays).

    MiMo-V2 (the heavy multimodal model) is only deployed on H100/A100, so
    this override never triggers in its hot path.
    """
    try:
        from sgl_kernel.flash_attn import is_fa3_supported
    except ImportError:
        return
    if is_fa3_supported():
        return
    try:
        from flash_attn import flash_attn_varlen_func
    except ImportError as e:
        raise RuntimeError(
            "MiMo-V2-ASR audio encoder needs upstream flash-attn on this GPU "
            "(sgl-kernel FA3 doesn't support sm_120). Install with "
            "`pip install flash-attn --no-build-isolation`."
        ) from e
    _mimo_audio_module.flash_attn_varlen_func = flash_attn_varlen_func


MiMoV2ASRConfig = Any

# Top-level audio sub-module name prefixes (after AUDIO_WEIGHT_REMAP). Loaded
# directly by default_weight_loader because the LM branch's qkv/gate-up fused
# stacked-params mapping doesn't apply to the vanilla HF Qwen2Model used
# inside the audio encoder.
_AUDIO_NAME_PREFIXES: Tuple[str, ...] = (
    "projection.",
    "input_local_transformer.",
    "speech_embeddings.",
)

# Training-only weights present in checkpoint but not used at inference.
# Checked AFTER the audio-prefix load path so substring matching here is
# safe: legitimate audio weights (``input_local_transformer.*``) are
# already consumed by ``_AUDIO_NAME_PREFIXES`` above.
_SKIP_NAME_SUBSTRINGS: Tuple[str, ...] = (
    "hidden_states_downcast",
    "local_transformer",
)


class MiMoV2ASRForCausalLM(MiMoForCausalLM, AudioEncoderMixin):
    def __init__(
        self,
        config: MiMoV2ASRConfig,
        quant_config=None,
        prefix: str = "",
    ) -> None:
        _maybe_override_audio_attn_for_blackwell()
        super().__init__(config, quant_config=quant_config, prefix=prefix)
        self.build_audio_encoder(MiMoAudioEncoderConfig(**config.audio_config))

    def pad_input_ids(self, input_ids: List[int], mm_inputs: MultimodalInputs):
        pattern = MultiModalityDataPaddingPatternMultimodalTokens()
        return pattern.pad_input_tokens(input_ids, mm_inputs)

    def get_input_embeddings(self):
        if getattr(self.config, "encoder_only", False):
            return None
        return self.model.embed_tokens

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        get_embedding: bool = False,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
        if getattr(self.config, "encoder_only", False):
            raise NotImplementedError(
                "forward() is not supported in encoder_only mode. "
                "Use get_audio_feature() instead."
            )

        hidden_states = general_mm_embed_routine(
            input_ids=input_ids,
            forward_batch=forward_batch,
            language_model=self.model,
            multimodal_model=self,
            positions=positions,
            pp_proxy_tensors=pp_proxy_tensors,
        )

        if not get_embedding:
            return self.logits_processor(
                input_ids, hidden_states, self.lm_head, forward_batch
            )
        return self.pooler(hidden_states, forward_batch)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        params_dict = dict(self.named_parameters())
        deferred: List[Tuple[str, torch.Tensor]] = []

        for name, loaded_weight in weights:
            if name.startswith("audio_encoder."):
                name = name[len("audio_encoder.") :]
            name = self.remap_audio_weight_name(name)

            if name.startswith(_AUDIO_NAME_PREFIXES):
                if name not in params_dict:
                    logger.warning(
                        f"Audio param {name} not found in params_dict, skipping"
                    )
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                if name.startswith("speech_embeddings."):
                    weight_loader(param, loaded_weight[: param.shape[0], :])
                else:
                    weight_loader(param, loaded_weight)
                continue

            if any(s in name for s in _SKIP_NAME_SUBSTRINGS):
                continue

            deferred.append((name, loaded_weight))

        super().load_weights(iter(deferred))


EntryClass = MiMoV2ASRForCausalLM
