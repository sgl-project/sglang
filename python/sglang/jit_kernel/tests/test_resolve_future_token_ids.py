import pytest
import torch

from sglang.jit_kernel.resolve_future_token_ids import resolve_future_token_ids_cuda


def _reference_resolve(input_ids, future_map):
    """Reference implementation using plain torch."""
    result = input_ids.clone()
    result[:] = torch.where(
        result < 0,
        future_map[torch.clamp(-result, min=0)],
        result,
    )
    return result


@pytest.mark.parametrize("size", [1, 2, 127, 128, 255, 256, 1024, 4097])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
class TestResolveFutureTokenIds:
    def test_all_negative(self, size: int, dtype: torch.dtype) -> None:
        map_size = 8192
        future_map = torch.randint(0, 50000, (map_size,), dtype=dtype, device="cuda")
        # Negative indices in range [-map_size+1, -1]
        input_ids = -torch.randint(1, map_size, (size,), dtype=dtype, device="cuda")

        expected = _reference_resolve(input_ids, future_map)
        resolve_future_token_ids_cuda(input_ids, future_map)
        assert torch.equal(input_ids, expected)

    def test_all_non_negative(self, size: int, dtype: torch.dtype) -> None:
        map_size = 16
        future_map = torch.randint(0, 50000, (map_size,), dtype=dtype, device="cuda")
        input_ids = torch.randint(0, 50000, (size,), dtype=dtype, device="cuda")

        expected = input_ids.clone()
        resolve_future_token_ids_cuda(input_ids, future_map)
        assert torch.equal(input_ids, expected)

    def test_mixed(self, size: int, dtype: torch.dtype) -> None:
        map_size = 8192
        future_map = torch.randint(0, 50000, (map_size,), dtype=dtype, device="cuda")
        # Mix of negative and non-negative
        input_ids = torch.randint(
            -map_size + 1, 50000, (size,), dtype=dtype, device="cuda"
        )

        expected = _reference_resolve(input_ids, future_map)
        resolve_future_token_ids_cuda(input_ids, future_map)
        assert torch.equal(input_ids, expected)

    def test_zeros(self, size: int, dtype: torch.dtype) -> None:
        map_size = 16
        future_map = torch.randint(0, 50000, (map_size,), dtype=dtype, device="cuda")
        input_ids = torch.zeros(size, dtype=dtype, device="cuda")

        expected = input_ids.clone()
        resolve_future_token_ids_cuda(input_ids, future_map)
        assert torch.equal(input_ids, expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
