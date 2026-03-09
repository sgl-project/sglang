import pytest
import torch

from sglang.multimodal_gen.test.server.accuracy_config import (
    ComponentType,
    get_skip_reason,
    get_threshold,
    should_skip_component,
)
from sglang.multimodal_gen.test.server.accuracy_utils import (
    build_deterministic_text_encoder_inputs,
    extract_output_tensor,
    resolve_text_encoder_forward_module,
)
from sglang.multimodal_gen.test.server.component_accuracy import AccuracyEngine
from sglang.multimodal_gen.test.server.testcase_configs import TWO_GPU_CASES_A


@pytest.mark.parametrize("case", TWO_GPU_CASES_A, ids=lambda x: x.id)
class TestAccuracy2GPU_A:
    """2-GPU Component Accuracy Suite (Set A)."""

    def test_vae_accuracy(self, case):
        if should_skip_component(case, ComponentType.VAE):
            pytest.skip(get_skip_reason(case, ComponentType.VAE))
        AccuracyEngine.clear_memory()
        sgl, ref, device = AccuracyEngine.load_component_pair(
            case, ComponentType.VAE, "diffusers", 2
        )

        sgl_out, ref_out = AccuracyEngine.run_component_pair_native(
            case,
            ComponentType.VAE,
            sgl,
            ref,
            device,
        )

        AccuracyEngine.check_accuracy(
            sgl_out,
            ref_out,
            f"{case.id}_vae",
            get_threshold(case.id, ComponentType.VAE),
        )
        del sgl, ref
        AccuracyEngine.clear_memory()

    def test_transformer_accuracy(self, case):
        if should_skip_component(case, ComponentType.TRANSFORMER):
            pytest.skip(get_skip_reason(case, ComponentType.TRANSFORMER))
        AccuracyEngine.clear_memory()
        sgl, ref, device = AccuracyEngine.load_component_pair(
            case, ComponentType.TRANSFORMER, "diffusers", 2
        )

        sgl_out, ref_out = AccuracyEngine.run_component_pair_native(
            case,
            ComponentType.TRANSFORMER,
            sgl,
            ref,
            device,
        )

        AccuracyEngine.check_accuracy(
            sgl_out,
            ref_out,
            f"{case.id}_transformer",
            get_threshold(case.id, ComponentType.TRANSFORMER),
        )
        del sgl, ref
        AccuracyEngine.clear_memory()

    def test_encoder_accuracy(self, case):
        if should_skip_component(case, ComponentType.TEXT_ENCODER):
            pytest.skip(get_skip_reason(case, ComponentType.TEXT_ENCODER))
        AccuracyEngine.clear_memory()
        sgl, ref, device = AccuracyEngine.load_component_pair(
            case, ComponentType.TEXT_ENCODER, "transformers", 2
        )
        input_ids, attention_mask = build_deterministic_text_encoder_inputs(
            ref.config, device
        )

        with torch.no_grad():
            ref_model = resolve_text_encoder_forward_module(ref)
            sgl_model = resolve_text_encoder_forward_module(sgl)

            sgl_out = sgl_model(
                input_ids, attention_mask=attention_mask, output_hidden_states=True
            )
            ref_out = ref_model(
                input_ids, attention_mask=attention_mask, output_hidden_states=True
            )

            s_t = extract_output_tensor(sgl_out)
            r_t = extract_output_tensor(ref_out)
            AccuracyEngine.check_accuracy(
                s_t,
                r_t,
                f"{case.id}_encoder",
                get_threshold(case.id, ComponentType.TEXT_ENCODER),
            )

        del sgl, ref
        AccuracyEngine.clear_memory()
