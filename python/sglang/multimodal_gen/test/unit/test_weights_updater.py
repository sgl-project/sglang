"""Unit tests for the diffusion weight-reload path (issue #31924).

Exercises WeightsUpdater.update_weights_from_disk end to end against fake modules:
  - buffers (BatchNorm running stats) are loaded alongside parameters
  - TextEncoder modules are routed to their own load_weights() with UNMAPPED names

The DTensor guard is not covered here: constructing a real DTensor needs a device
mesh and distributed init. It is covered by the GPU offload test.

CPU-only; no model loading, no distributed init.
"""

import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

from sglang.multimodal_gen.configs.models.encoders.base import TextEncoderConfig
from sglang.multimodal_gen.runtime.models.encoders.base import TextEncoder
from sglang.multimodal_gen.runtime.post_training.weights_updater import WeightsUpdater


def _write_module_weights(root: Path, module_name: str, tensors: dict) -> None:
    """Lay out <root>/<module_name>/model.safetensors, as update_weights_from_disk expects."""
    module_dir = root / module_name
    module_dir.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(module_dir / "model.safetensors"))


class FakePipeline:
    """Minimal stand-in: WeightsUpdater only needs .modules, .model_path, get_module()."""

    def __init__(self, modules: dict):
        self.modules = modules
        self.model_path = "unused"

    def get_module(self, name):
        return self.modules.get(name)


class BufferModule(torch.nn.Module):
    """Plain nn.Module -> takes the generic path. affine=False mirrors the FLUX.2 VAE,
    so the BatchNorm contributes 3 buffers and 0 parameters."""

    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(4, 2)
        self.bn = torch.nn.BatchNorm1d(2, affine=False, track_running_stats=True)


class FakeTextEncoder(TextEncoder):
    """Records the names it is handed, then loads them the way a real encoder does."""

    def __init__(self):
        super().__init__(TextEncoderConfig())
        self.fc = torch.nn.Linear(4, 2)
        self.received_names: list[str] = []

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def load_weights(self, weights):
        params = dict(self.named_parameters())
        loaded = set()
        for name, tensor in weights:
            self.received_names.append(name)
            if name.startswith("model."):
                name = name[len("model.") :]
            if name in params:
                params[name].data.copy_(tensor)
                loaded.add(name)
        return loaded


class TestUpdateWeightsFromDisk(unittest.TestCase):
    def test_buffers_are_loaded_alongside_parameters(self):
        module = BufferModule()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_module_weights(
                root,
                "vae",
                {
                    "fc.weight": torch.ones(2, 4),
                    "fc.bias": torch.full((2,), 0.5),
                    "bn.running_mean": torch.tensor([1.0, 2.0]),
                    "bn.running_var": torch.tensor([0.5, 0.5]),
                    "bn.num_batches_tracked": torch.tensor(7),
                },
            )
            updater = WeightsUpdater(FakePipeline({"vae": module}))
            success, message = updater.update_weights_from_disk(str(root))

        self.assertTrue(success)
        self.assertIn("vae: 5", message)
        torch.testing.assert_close(module.bn.running_mean, torch.tensor([1.0, 2.0]))
        self.assertEqual(module.bn.num_batches_tracked, 7)

    def test_text_encoder_receives_unmapped_names(self):
        module = FakeTextEncoder()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_module_weights(
                root,
                "text_encoder",
                {
                    "model.fc.weight": torch.ones(2, 4),  # "model." prefix, as on disk
                    "model.fc.bias": torch.full((2,), 0.5),
                },
            )
            updater = WeightsUpdater(FakePipeline({"text_encoder": module}))
            success, message = updater.update_weights_from_disk(str(root))
        self.assertTrue(success)
        self.assertIn("text_encoder: 2", message)
        torch.testing.assert_close(module.fc.bias, torch.full((2,), 0.5))
        torch.testing.assert_close(module.fc.weight, torch.ones(2, 4))
        self.assertEqual(
            sorted(module.received_names), ["model.fc.bias", "model.fc.weight"]
        )


if __name__ == "__main__":
    unittest.main()
