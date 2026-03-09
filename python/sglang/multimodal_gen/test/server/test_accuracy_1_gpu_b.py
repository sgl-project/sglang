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
from sglang.multimodal_gen.test.server.testcase_configs import ONE_GPU_CASES_B


@pytest.mark.parametrize("case", ONE_GPU_CASES_B, ids=lambda x: x.id)
class TestAccuracy1GPU_B:
    """1-GPU Component Accuracy Suite (Set B)."""

    def test_vae_accuracy(self, case):
        if should_skip_component(case, ComponentType.VAE):
            pytest.skip(get_skip_reason(case, ComponentType.VAE))
        AccuracyEngine.clear_memory()

        sgl, ref, device = AccuracyEngine.load_component_pair(
            case, ComponentType.VAE, "diffusers", 1
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
            case, ComponentType.TRANSFORMER, "diffusers", 1
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
        if "wan" not in case.id:
            pytest.skip("Skipping encoder test for this variant in Set B")

        sgl, ref, device = AccuracyEngine.load_component_pair(
            case, ComponentType.TEXT_ENCODER, "transformers", 1
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

            AccuracyEngine.check_accuracy(
                extract_output_tensor(sgl_out),
                extract_output_tensor(ref_out),
                f"{case.id}_encoder",
                get_threshold(case.id, ComponentType.TEXT_ENCODER),
            )

        del sgl, ref
        AccuracyEngine.clear_memory()
