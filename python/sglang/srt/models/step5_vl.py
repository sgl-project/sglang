"""Step5 VL model entry point: Step3.7-compatible wrapper with the Step5 backbone."""

from collections.abc import Iterable

from sglang.srt.lora.utils import RoutedExpertLoRATargets
from sglang.srt.models.step3p7 import Step3p7ForConditionalGeneration


class Step5ForConditionalGeneration(
    Step3p7ForConditionalGeneration, RoutedExpertLoRATargets
):
    """Open-source Step5 VL checkpoint (model_type "step5_vl").

    Same backbone and perception-encoder vision tower as Step3.7: reuses its
    model wrapper, multimodal processor and weight loading (including the
    Step5 text_config unwrap and BF16 vision checks).
    """

    kv_cache_dtype_full_attention_only = True

    def get_hidden_dim(self, module_name: str, layer_idx: int) -> tuple[int, int]:
        return self.language_model.get_hidden_dim(module_name, layer_idx)

    def lora_targets_routed_experts(self, target_modules: Iterable[str]) -> bool:
        return self.language_model.lora_targets_routed_experts(target_modules)


EntryClass = Step5ForConditionalGeneration
