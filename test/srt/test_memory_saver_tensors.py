import gc
import unittest

import torch
import triton
import triton.language as tl
from sglang.srt.memory_saver_tensors import DisposableTensor, LazyTensor
from sglang.srt.memory_saver_tensors import WrapperTensor

_DEVICE = torch.device("cuda:0")


class TestMemorySaverTensors(unittest.TestCase):
    def test_disposable_tensor_memory(self):
        memory_before = _check_memory()

        x = DisposableTensor(torch.zeros((1_000_000_000,), device=_DEVICE, dtype=torch.float32))
        assert _check_memory() - memory_before >= 1_000_000_000 * 4

        x.dispose()
        assert _check_memory() - memory_before <= 1000

        print(f"{type(x)}")  # ensure there are still ref to it

    def test_lazy_tensor_memory(self):
        memory_before = _check_memory()

        x = LazyTensor((1_000_000_000,), device=_DEVICE, dtype=torch.float32)
        assert _check_memory() - memory_before <= 1000

        x += 1
        assert _check_memory() - memory_before >= 1_000_000_000 * 4

        del x
        assert _check_memory() - memory_before <= 1000

    def test_disposable_tensor_operations(self):
        x = DisposableTensor(torch.tensor([3.0, 4.0, 5.0], device=_DEVICE))
        self._common_test_operations(x)

    def test_lazy_tensor_operations(self):
        x = LazyTensor((3,), device=_DEVICE)
        x[0] = 3.0
        x[1:3] = torch.tensor([4.0, 5.0], device=_DEVICE)
        self._common_test_operations(x)

    def _common_test_operations(self, x: torch.Tensor):
        print(f"common_test {type(x)=} {x=}")
        self.assertEqual(torch.max(x).item(), 5.0)
        self.assertTrue(torch.allclose(x + torch.tensor([2.0, 2.0, 2.0], device=_DEVICE),
                                       torch.tensor([5.0, 6.0, 7.0], device=_DEVICE)))
        self.assertTrue(torch.allclose(_add_by_triton(x, torch.tensor([2.0, 2.0, 2.0], device=_DEVICE)),
                                       torch.tensor([5.0, 6.0, 7.0], device=_DEVICE)))
        self.assertEqual(torch.empty_like(x).shape, (3,))
        self.assertTrue(torch.allclose(torch.full_like(x, 42), torch.tensor([42.0, 42.0, 42.0], device=_DEVICE)))
        self.assertTrue(x.is_contiguous())
        self.assertEqual(x.numel(), 3)
        self.assertEqual(x.shape, (3,))


def _add_by_triton(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)
    assert x.device == _DEVICE and y.device == _DEVICE and output.device == _DEVICE
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    _add_kernel[grid](WrapperTensor.unwrap(x), WrapperTensor.unwrap(y), output, n_elements, BLOCK_SIZE=1024)
    return output


@triton.jit
def _add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


def _check_memory():
    gc.collect()
    torch.cuda.empty_cache()
    return torch.cuda.memory_allocated()


if __name__ == "__main__":
    unittest.main()
