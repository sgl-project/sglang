import tempfile
import unittest

import torch
from torch import nn

from sglang.srt.utils.disk_offloader import DiskOffloader
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")

HIDDEN_SIZE = 512
NUM_LAYERS = 5
LAYER_BYTES = 2 * HIDDEN_SIZE * HIDDEN_SIZE * 4


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.up = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE, bias=False)
        self.down = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE, bias=False)
        self.norm = nn.LayerNorm(HIDDEN_SIZE)

    def forward(self, hidden_states):
        return self.norm(hidden_states + self.down(torch.relu(self.up(hidden_states))))


def _run(layers, hidden_states):
    for layer in layers:
        hidden_states = layer(hidden_states)
    return hidden_states


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestDiskOffloader(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.storage_dir = tempfile.mkdtemp()
        self.reference = nn.ModuleList(_Block() for _ in range(NUM_LAYERS)).cuda()
        self.inputs = torch.randn(3, HIDDEN_SIZE, device="cuda")

    def _build_offloaded(self, host_cache_bytes, prefetch_step):
        offloader = DiskOffloader(
            group_size=1,
            num_in_group=1,
            prefetch_step=prefetch_step,
            storage_dir=self.storage_dir,
            host_cache_bytes=host_cache_bytes,
        )
        layers = offloader.wrap_modules(_Block().cuda() for _ in range(NUM_LAYERS))
        for layer, reference_layer in zip(layers, self.reference):
            for name, param in layer.named_parameters():
                param.data.copy_(reference_layer.get_parameter(name).data)
        return offloader, layers

    def test_streams_identical_weights_across_passes(self):
        expected = _run(self.reference, self.inputs)
        for host_cache_bytes, prefetch_step in [(0, 1), (2 * LAYER_BYTES, 2)]:
            with self.subTest(host_cache_bytes=host_cache_bytes):
                offloader, layers = self._build_offloaded(
                    host_cache_bytes, prefetch_step
                )
                offloader.post_init()
                self.assertEqual(layers[0].up.weight.device.type, "meta")
                self.assertEqual(layers[0].norm.weight.device.type, "cuda")
                for _ in range(3):
                    torch.testing.assert_close(_run(layers, self.inputs), expected)

    def test_keeps_weights_replaced_after_loading(self):
        offloader, layers = self._build_offloaded(host_cache_bytes=0, prefetch_step=1)
        for layer in layers:
            transposed = layer.up.weight.data.t().contiguous().t()
            layer.up.weight.data = transposed * 2
            layer.down.weight = nn.Parameter(layer.down.weight.data.clone())
        for reference_layer in self.reference:
            reference_layer.up.weight.data *= 2
        offloader.post_init()
        torch.testing.assert_close(
            _run(layers, self.inputs), _run(self.reference, self.inputs)
        )

    def test_place_on_host_moves_parameter_to_cpu(self):
        offloader = DiskOffloader(
            group_size=1,
            num_in_group=1,
            prefetch_step=1,
            storage_dir=self.storage_dir,
            host_cache_bytes=0,
        )
        weight = torch.randn(1000, 64)
        param = nn.Parameter(torch.empty(1000, 64, device="cuda"), requires_grad=False)
        self.assertTrue(offloader.place_on_host(param))
        param.data.copy_(weight)
        self.assertEqual(param.device.type, "cpu")
        torch.testing.assert_close(param.data, weight)


if __name__ == "__main__":
    unittest.main()
