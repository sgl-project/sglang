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
from sglang.multimodal_gen.test.server.testcase_configs import TWO_GPU_CASES_B


@pytest.mark.parametrize("case", TWO_GPU_CASES_B, ids=lambda x: x.id)
class TestAccuracy2GPU_B:
    """2-GPU Component Accuracy Suite (Set B)."""

    def test_vae_accuracy(self, case):
        if should_skip_component(case, ComponentType.VAE):
            pytest.skip(get_skip_reason(case, ComponentType.VAE))
        AccuracyEngine.clear_memory()
        sgl, ref, device, adapter = AccuracyEngine.load_component_pair(
            case, ComponentType.VAE, "diffusers", 2
        )

        with torch.no_grad():
            assert (
                adapter is not None
            ), "VAE test requires an adapter for reliable input generation"
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
            case, ComponentType.TRANSFORMER, "diffusers", 2
        )

        with torch.no_grad():
            if adapter:
                inputs = adapter.generate_inputs(case, sgl, device, ref)
                sgl_out = adapter.run_sglang(sgl, inputs)
                ref_out = adapter.run_reference(ref, inputs)
            else:
                kwargs = AccuracyEngine.get_forward_inputs(case, ref, device)
                ref_out = ref(**kwargs)
                if not isinstance(ref_out, torch.Tensor):
                    ref_out = ref_out.sample
                sgl_out = sgl(**kwargs)
                if not isinstance(sgl_out, torch.Tensor):
                    sgl_out = sgl_out.sample

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
        if "wan" not in case.id.lower():
            pytest.skip("Skipping encoder test for this variant in Set B")

        sgl, ref, device, _ = AccuracyEngine.load_component_pair(
            case, ComponentType.TEXT_ENCODER, "transformers", 2
        )

        vocab_size = getattr(ref.config, "vocab_size", 32000)
        if hasattr(ref.config, "text_config"):
            vocab_size = getattr(ref.config.text_config, "vocab_size", vocab_size)

        # Mandatory Deterministic Seeding for Tensor Parallel (TP) rank consistency
        torch.manual_seed(42)
        # Generate on CPU to ensure all ranks get the exact same IDs
        input_ids = torch.randint(
            100, min(vocab_size, 30000), (1, 32), device="cpu", dtype=torch.long
        ).to(device)
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            ref_model = ref.get_encoder() if hasattr(ref, "get_encoder") else ref
            sgl_model = sgl.get_encoder() if hasattr(sgl, "get_encoder") else sgl

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
