"""BLAKE3 reference vectors, structured collision regressions and stream use."""

import unittest

import blake3
import numpy as np
import torch

from sglang.kernels.ops.memory.gpu_tensor_hash import gpu_tensor_hash
from sglang.kernels.ops.memory.gpu_tensor_hash_blake3 import (
    gpu_tensor_blake3,
    gpu_tensors_blake3,
)
from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.multimodal.cache.digest import structured_digest
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=45, stage="base-b-kernel-unit", runner_config="1-gpu-large")

# Two equal salt deltas exchange contributions in the retired additive hash.
PAIRS = (
    (77110, 81718, 0x369AF600),
    (353907, 359029, 0x10F9A5A6),
    (721955, 728491, 0xF138DF38),
    (906686, 908226, 0xA017474C),
    (1050801, 1053391, 0x0162D14E),
)


def _reference(tensor):
    raw = tensor.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy()
    return blake3.blake3(memoryview(raw)).digest()


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestGpuTensorBlake3(CustomTestCase):
    def test_reference_boundaries_and_unaligned_views(self):
        sizes = sorted(
            {0, 1, 2, 3, 4, 7, 31}
            | {
                n + d
                for n in (64, 1024, 32768, 65536, 131072, 262144, 16777216)
                for d in (-1, 0, 1)
            }
        )
        for size in sizes:
            for offset in range(4):
                with self.subTest(size=size, offset=offset):
                    x = (
                        torch.arange(size + offset, device="cuda")
                        .remainder(251)
                        .byte()[offset:]
                    )
                    self.assertEqual(gpu_tensor_blake3(x), _reference(x))

    def test_fifteen_structured_collisions(self):
        for i, j, delta in PAIRS:
            for dtype in (torch.uint8, torch.bfloat16, torch.float32):
                with self.subTest(positions=(i, j), dtype=dtype):
                    a = np.zeros(max(3 * 384 * 384 // 4, j + 1), dtype=np.uint32)
                    b = a.copy()
                    b[i], b[j] = delta, delta
                    x = torch.from_numpy(a.view(np.uint8)).view(dtype).cuda()
                    y = torch.from_numpy(b.view(np.uint8)).view(dtype).cuda()
                    self.assertNotEqual(gpu_tensor_blake3(x), gpu_tensor_blake3(y))
                    self.assertNotEqual(gpu_tensor_hash(x), gpu_tensor_hash(y))
                    first = MultimodalDataItem(modality=Modality.IMAGE, feature=x)
                    second = MultimodalDataItem(modality=Modality.IMAGE, feature=y)
                    self.assertNotEqual(first.cache_key, second.cache_key)
                    self.assertNotEqual(first.padding_values(), second.padding_values())

    def test_zero_padding_is_not_input_length(self):
        for size in (1, 2, 3, 63, 1023, 1025):
            x = torch.zeros(size, device="cuda", dtype=torch.uint8)
            x[0] = 42
            y = torch.cat((x, torch.zeros(4, device=x.device, dtype=x.dtype)))
            self.assertNotEqual(gpu_tensor_blake3(x), gpu_tensor_blake3(y))

    def test_device_layout_batch_and_bit_patterns(self):
        for dtype in (torch.uint8, torch.bfloat16, torch.float32):
            x = torch.arange(4096, device="cuda").remainder(251).byte().view(dtype)
            tensors = [x, x[::2], x.reshape(32, -1).T]
            self.assertEqual(
                gpu_tensors_blake3(tensors), [_reference(t) for t in tensors]
            )
            self.assertEqual(
                structured_digest(tensors),
                structured_digest([t.cpu() for t in tensors]),
            )
        # Every BF16 representation, including NaNs, signed zero and infinities.
        x = torch.arange(65536, device="cuda").to(torch.int16).view(torch.bfloat16)
        self.assertEqual(gpu_tensor_blake3(x), _reference(x))

    def test_stream_order_and_reused_storage(self):
        streams = [torch.cuda.Stream(), torch.cuda.Stream()]
        for i in range(32):
            with torch.cuda.stream(streams[i % 2]):
                x = torch.full((131073,), i, dtype=torch.uint8, device="cuda")
                before = gpu_tensor_blake3(x)
                self.assertEqual(before, _reference(x))
                x[-1] ^= 1
                self.assertNotEqual(before, gpu_tensor_blake3(x))

    def test_lazy_views_hash_logical_values(self):
        x = torch.arange(8, device="cuda", dtype=torch.float32)
        values = (torch._neg_view(x), torch.complex(x, x + 1).conj())
        for value in values:
            with self.subTest(dtype=value.dtype):
                resolved = value.resolve_neg().resolve_conj()
                self.assertEqual(gpu_tensor_blake3(value), _reference(resolved))
                self.assertEqual(
                    structured_digest(value), structured_digest(value.cpu())
                )


if __name__ == "__main__":
    unittest.main()
