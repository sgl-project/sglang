import pytest
import torch

from sglang.multimodal_gen.test.server.accuracy_config import (
    ComponentType,
    get_skip_reason,
    get_threshold,
    should_skip_component,
)
from sglang.multimodal_gen.test.server.accuracy_utils import extract_output_tensor
from sglang.multimodal_gen.test.server.component_accuracy import AccuracyEngine
from sglang.multimodal_gen.test.server.testcase_configs import ONE_GPU_CASES_A


@pytest.mark.parametrize("case", ONE_GPU_CASES_A, ids=lambda x: x.id)
class TestAccuracy1GPU_A:
    """1-GPU Component Accuracy Suite (Set A)."""

    def test_vae_accuracy(self, case):
        if should_skip_component(case, ComponentType.VAE):
            pytest.skip(get_skip_reason(case, ComponentType.VAE))
        AccuracyEngine.clear_memory()
        sgl, ref, device, adapter = AccuracyEngine.load_component_pair(
            case, ComponentType.VAE, "diffusers", 1
        )

        with torch.no_grad():
            assert adapter is not None
            inputs = adapter.generate_inputs(case, sgl, device, ref)
            sgl_out = adapter.run_sglang(sgl, inputs)
            ref_out = adapter.run_reference(ref, inputs)

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
        sgl, ref, device, adapter = AccuracyEngine.load_component_pair(
            case, ComponentType.TRANSFORMER, "diffusers", 1
        )

        with torch.no_grad():
            assert (
                adapter is not None
            ), "Transformer test requires an adapter for reliable inputs"
            inputs = adapter.generate_inputs(case, sgl, device, ref)
            sgl_out = adapter.run_sglang(sgl, inputs)
            ref_out = adapter.run_reference(ref, inputs)

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
        sgl, ref, device, _ = AccuracyEngine.load_component_pair(
            case, ComponentType.TEXT_ENCODER, "transformers", 1
        )

        # Vocab size resolution
        vocab_size = getattr(ref.config, "vocab_size", 32000)
        if hasattr(ref.config, "text_config"):
            vocab_size = getattr(ref.config.text_config, "vocab_size", vocab_size)

        # Use safe clamped range to avoid out-of-bounds or special tokens that trigger assertions
        torch.manual_seed(42)
        input_ids = torch.randint(
            100, min(vocab_size, 30000), (1, 32), device="cpu", dtype=torch.long
        ).to(device)

        with torch.no_grad():
            sgl_out = sgl(input_ids, output_hidden_states=True)
            ref_out = ref(input_ids, output_hidden_states=True)

            AccuracyEngine.check_accuracy(
                extract_output_tensor(sgl_out),
                extract_output_tensor(ref_out),
                f"{case.id}_encoder",
                get_threshold(case.id, ComponentType.TEXT_ENCODER),
            )

        del sgl, ref
        AccuracyEngine.clear_memory()
