import unittest

import requests
import torch
from PIL import Image
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor
from vllm.distributed import init_distributed_environment, initialize_model_parallel

from sglang.srt.models.molmo import MolmoVisionBackbone, MolmoVisionBackboneConfig
from sglang.test.runners import HFRunner, SRTRunner

PROMPT = "Describe this image."

MODELS = [
    ("allenai/Molmo-7B-D-0924", "bfloat16", 1e-5),
]

CONVS = [
    (
        "What's in this image?",
        Image.open(
            requests.get("https://picsum.photos/id/237/536/354", stream=True).raw
        ),
    )
]


class TestMolmo(unittest.TestCase):
    def assert_srt_vision_backbone_and_hf_vision_backbone_close(self, text, images):
        config = AutoConfig.from_pretrained(
            "allenai/Molmo-7B-D-0924", trust_remote_code=True
        )
        processor = AutoProcessor.from_pretrained(
            "allenai/Molmo-7B-D-0924", trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            "allenai/Molmo-7B-D-0924", trust_remote_code=True, torch_dtype=torch.float32
        ).to("cuda")

        init_distributed_environment(
            backend="nccl",
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method="tcp://127.0.0.1:29500",
        )
        initialize_model_parallel(tensor_model_parallel_size=1)
        # Create a new MolmoVisionBackbone with the same config
        molmo_vision_backbone = MolmoVisionBackbone(config, MolmoVisionBackboneConfig())

        # Prepare dictionaries to store weights
        vision_backbone_weights = {}
        image_projector_weights = {}

        # Extract vision backbone weights from the HuggingFace model
        for name, param in model.model.vision_backbone.named_parameters():
            if "image_projector" in name:
                if "w1" in name:
                    image_projector_weights["gate_proj"] = param
                elif "w3" in name:
                    image_projector_weights["up_proj"] = param
                elif "w2" in name:
                    image_projector_weights["down_proj"] = param
                else:
                    raise ValueError(f"Unexpected projector weight: {name}")
            else:
                vision_backbone_weights[name] = param

        # Load weights into our custom MolmoVisionBackbone
        molmo_vision_backbone.load_state_dict(vision_backbone_weights, strict=False)
        molmo_vision_backbone.to(device="cuda", dtype=torch.float32)

        for k, v in molmo_vision_backbone.state_dict().items():
            if k in model.model.vision_backbone.state_dict():
                assert torch.allclose(
                    v, model.model.vision_backbone.state_dict()[k], atol=1e-5
                ), f"Weight {k} is not close"
            else:
                print(f"Missing {k}")

        # Load image projector weights
        gate_up_proj_weight = torch.cat(
            [image_projector_weights["gate_proj"], image_projector_weights["up_proj"]],
            dim=0,
        )
        molmo_vision_backbone.image_projector.gate_up_proj.weight.data.copy_(
            gate_up_proj_weight
        )
        molmo_vision_backbone.image_projector.down_proj.weight.data.copy_(
            image_projector_weights["down_proj"]
        )

        # Ensure both models are in the same mode (eval or train)
        model.model.vision_backbone.eval()
        molmo_vision_backbone.eval()

        # Test forward pass
        inputs = processor.process(images=images, text=text, return_tensors="pt")
        with torch.no_grad():
            images = inputs["images"].unsqueeze(0).to("cuda")
            image_masks = inputs["image_masks"].unsqueeze(0).to("cuda")
            hf_outputs = model.model.vision_backbone(
                images=images, image_masks=image_masks
            )
            srt_outputs = molmo_vision_backbone(images=images, image_masks=image_masks)

        # Compare outputs
        assert torch.allclose(
            hf_outputs[0], srt_outputs[0], atol=1e-5
        ), f"image_features are not all close: {hf_outputs[0].shape} {srt_outputs[0].shape}, hf_outputs[0]: {hf_outputs[0]}, srt_outputs[0]: {srt_outputs[0]}"
        assert torch.allclose(
            hf_outputs[1], srt_outputs[1], atol=1e-5
        ), f"cls_embed are not all close: {hf_outputs[1].shape} {srt_outputs[1].shape}, hf_outputs[1]: {hf_outputs[1]}, srt_outputs[1]: {srt_outputs[1]}"

    def test_molmo_d(self):
        for text, images in CONVS:
            for model_path, torch_dtype, tolerance in MODELS:
                self.assert_srt_vision_backbone_and_hf_vision_backbone_close(
                    text, images
                )


if __name__ == "__main__":
    unittest.main()
