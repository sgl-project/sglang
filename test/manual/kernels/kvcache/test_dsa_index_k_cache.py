"""
Correctness tests for DSA Indexer K/S Buffer Access with Fused Triton Kernels.

This test verifies that GetKAndS.triton() produces identical results to a
torch gather of the K and S regions of the paged buffer.
"""

import pytest
import torch

from sglang.kernels.ops.kvcache.dsa_index_k_cache import GetKAndS


class MockDSATokenToKVPool:
    """Mock pool object that mimics DSATokenToKVPool for testing."""

    def __init__(
        self,
        page_size: int = 64,
        index_head_dim: int = 128,
        quant_block_size: int = 128,
        device: str = "cuda",
    ):
        self.index_page_size = page_size
        self.index_head_dim = index_head_dim
        self.quant_block_size = quant_block_size
        self.device = device


def create_test_buffer(
    num_pages: int,
    page_size: int = 64,
    index_head_dim: int = 128,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Create a test buffer mimicking the K/S buffer structure.

    Buffer layout per page:
    - First page_size * index_head_dim bytes: K data (fp8, stored as uint8)
    - Next page_size * 4 bytes: S data (fp32 scales, stored as uint8)

    Args:
        num_pages: Number of pages to allocate
        page_size: Tokens per page (typically 64)
        index_head_dim: Dimension of K vectors (typically 128)
        device: Device to allocate on

    Returns:
        Buffer of shape (num_pages, page_size * index_head_dim + page_size * 4)
    """
    buf_numel_per_page = page_size * index_head_dim + page_size * 4
    buf = torch.randint(
        0, 256, (num_pages, buf_numel_per_page), dtype=torch.uint8, device=device
    )
    return buf


def torch_get_k(pool, buf, seq_len: int, page_indices: torch.Tensor):
    """Gather (seq_len, index_head_dim) uint8 K bytes from the paged buffer."""
    num_k_bytes_per_page = pool.index_page_size * pool.index_head_dim
    flat_indices = (page_indices * buf.shape[1])[:, None] + torch.arange(
        num_k_bytes_per_page, dtype=torch.int32, device=buf.device
    )[None, :]
    flat_indices = flat_indices.flatten()[: seq_len * pool.index_head_dim]
    return buf.flatten()[flat_indices].view(-1, pool.index_head_dim)


def torch_get_s(pool, buf, seq_len: int, page_indices: torch.Tensor):
    """Gather (seq_len, 4) uint8 fp32-scale bytes from the paged buffer."""
    s_offset_in_page = pool.index_page_size * pool.index_head_dim
    num_s_bytes_per_page = buf.shape[1] - s_offset_in_page
    num_s_bytes_per_token = pool.index_head_dim // pool.quant_block_size * 4
    flat_indices = (
        (page_indices * buf.shape[1])[:, None]
        + torch.arange(num_s_bytes_per_page, dtype=torch.int32, device=buf.device)[
            None, :
        ]
        + s_offset_in_page
    )
    flat_indices = flat_indices.flatten()[: seq_len * num_s_bytes_per_token]
    return buf.flatten()[flat_indices].view(-1, 4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestGetKAndS:
    """Test cases for GetKAndS.triton() correctness."""

    @pytest.mark.parametrize("num_pages", [1, 2, 4, 8, 16])
    @pytest.mark.parametrize("seq_len", [64, 128, 256, 512, 1024])
    @pytest.mark.parametrize("page_size", [64])
    @pytest.mark.parametrize("index_head_dim", [128])
    def test_get_k_and_s_correctness(
        self, num_pages, seq_len, page_size, index_head_dim
    ):
        """Test GetKAndS.triton() produces same output as the torch gathers."""
        device = torch.device("cuda")

        # Ensure seq_len doesn't exceed available pages
        max_seq_len = num_pages * page_size
        seq_len = min(seq_len, max_seq_len)
        seq_len_tensor = torch.tensor([seq_len], dtype=torch.int64, device=device)

        # Create mock pool
        pool = MockDSATokenToKVPool(
            page_size=page_size, index_head_dim=index_head_dim, device=device
        )

        # Create test buffer
        buf = create_test_buffer(
            num_pages=num_pages,
            page_size=page_size,
            index_head_dim=index_head_dim,
            device=device,
        )

        # Create page indices
        num_pages_needed = (seq_len + page_size - 1) // page_size
        page_indices = torch.randint(
            0, num_pages, (num_pages_needed,), dtype=torch.int32, device=device
        )
        page_indices_ = page_indices.unsqueeze(0)

        # Run baseline: separate torch gathers
        k_torch = torch_get_k(pool, buf, seq_len, page_indices)
        s_torch = torch_get_s(pool, buf, seq_len, page_indices)

        # Run fused Triton implementation
        k_triton, s_triton = GetKAndS.triton(
            pool, buf, page_indices_, seq_len_tensor, seq_len, seq_len
        )

        # Verify shapes
        assert k_torch.shape == (seq_len, index_head_dim)
        assert s_torch.shape == (seq_len, 4)
        assert k_triton.shape == (seq_len, index_head_dim)
        assert s_triton.shape == (seq_len, 4)

        # Verify dtypes
        assert k_torch.dtype == torch.uint8
        assert s_torch.dtype == torch.uint8
        assert k_triton.dtype == torch.uint8
        assert s_triton.dtype == torch.uint8

        # Compare K results
        torch.testing.assert_close(
            k_triton, k_torch, rtol=0, atol=0, msg="GetKAndS K outputs differ"
        )

        # Compare S results
        torch.testing.assert_close(
            s_triton, s_torch, rtol=0, atol=0, msg="GetKAndS S outputs differ"
        )

    def test_get_k_and_s_sequential_pages(self):
        """Test GetKAndS with sequential page indices."""
        device = torch.device("cuda")
        page_size = 64
        index_head_dim = 128
        num_pages = 10
        seq_len = 320  # 5 pages
        seq_len_tensor = torch.tensor([seq_len], dtype=torch.int64, device=device)

        pool = MockDSATokenToKVPool(
            page_size=page_size, index_head_dim=index_head_dim, device=device
        )
        buf = create_test_buffer(num_pages, page_size, index_head_dim, device)

        # Sequential page indices [0, 1, 2, 3, 4]
        page_indices = torch.arange(5, dtype=torch.int32, device=device)
        page_indices_ = page_indices.unsqueeze(0)

        # Baseline
        k_torch = torch_get_k(pool, buf, seq_len, page_indices)
        s_torch = torch_get_s(pool, buf, seq_len, page_indices)

        # Fused
        k_triton, s_triton = GetKAndS.triton(
            pool, buf, page_indices_, seq_len_tensor, seq_len, seq_len
        )

        torch.testing.assert_close(k_triton, k_torch, rtol=0, atol=0)
        torch.testing.assert_close(s_triton, s_torch, rtol=0, atol=0)

    def test_get_k_and_s_repeated_pages(self):
        """Test GetKAndS with repeated page indices."""
        device = torch.device("cuda")
        page_size = 64
        index_head_dim = 128
        num_pages = 5
        seq_len = 192  # 3 pages
        seq_len_tensor = torch.tensor([seq_len], dtype=torch.int64, device=device)

        pool = MockDSATokenToKVPool(
            page_size=page_size, index_head_dim=index_head_dim, device=device
        )
        buf = create_test_buffer(num_pages, page_size, index_head_dim, device)

        # Repeated page indices [2, 2, 2]
        page_indices = torch.full((3,), 2, dtype=torch.int32, device=device)
        page_indices_ = page_indices.unsqueeze(0)

        # Baseline
        k_torch = torch_get_k(pool, buf, seq_len, page_indices)
        s_torch = torch_get_s(pool, buf, seq_len, page_indices)

        # Fused
        k_triton, s_triton = GetKAndS.triton(
            pool, buf, page_indices_, seq_len_tensor, seq_len, seq_len
        )

        torch.testing.assert_close(k_triton, k_torch, rtol=0, atol=0)
        torch.testing.assert_close(s_triton, s_torch, rtol=0, atol=0)

    def test_get_k_and_s_partial_page(self):
        """Test GetKAndS when seq_len is not a multiple of page_size."""
        device = torch.device("cuda")
        page_size = 64
        index_head_dim = 128
        num_pages = 5
        seq_len = 100  # Not a multiple of 64
        seq_len_tensor = torch.tensor([seq_len], dtype=torch.int64, device=device)

        pool = MockDSATokenToKVPool(
            page_size=page_size, index_head_dim=index_head_dim, device=device
        )
        buf = create_test_buffer(num_pages, page_size, index_head_dim, device)

        num_pages_needed = (seq_len + page_size - 1) // page_size
        page_indices = torch.arange(num_pages_needed, dtype=torch.int32, device=device)
        page_indices_ = page_indices.unsqueeze(0)

        # Baseline
        k_torch = torch_get_k(pool, buf, seq_len, page_indices)
        s_torch = torch_get_s(pool, buf, seq_len, page_indices)

        # Fused
        k_triton, s_triton = GetKAndS.triton(
            pool, buf, page_indices_, seq_len_tensor, seq_len, seq_len
        )

        # Should handle partial pages correctly
        torch.testing.assert_close(k_triton, k_torch, rtol=0, atol=0)
        torch.testing.assert_close(s_triton, s_torch, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_token(self):
        """Test with seq_len=1 (single token)."""
        device = torch.device("cuda")
        page_size = 64
        index_head_dim = 128
        num_pages = 2
        seq_len = 1
        seq_len_tensor = torch.tensor([seq_len], dtype=torch.int64, device=device)

        pool = MockDSATokenToKVPool(
            page_size=page_size, index_head_dim=index_head_dim, device=device
        )
        buf = create_test_buffer(num_pages, page_size, index_head_dim, device)
        page_indices = torch.tensor([0], dtype=torch.int32, device=device)
        page_indices_ = page_indices.unsqueeze(0)

        k_torch = torch_get_k(pool, buf, seq_len, page_indices)
        s_torch = torch_get_s(pool, buf, seq_len, page_indices)
        k_triton, s_triton = GetKAndS.triton(
            pool, buf, page_indices_, seq_len_tensor, seq_len, seq_len
        )
        torch.testing.assert_close(k_triton, k_torch, rtol=0, atol=0)
        torch.testing.assert_close(s_triton, s_torch, rtol=0, atol=0)

    def test_exact_page_boundary(self):
        """Test when seq_len exactly matches page boundaries."""
        device = torch.device("cuda")
        page_size = 64
        index_head_dim = 128
        num_pages = 5
        seq_len = 192  # Exactly 3 pages
        seq_len_tensor = torch.tensor([seq_len], dtype=torch.int64, device=device)

        pool = MockDSATokenToKVPool(
            page_size=page_size, index_head_dim=index_head_dim, device=device
        )
        buf = create_test_buffer(num_pages, page_size, index_head_dim, device)
        page_indices = torch.arange(3, dtype=torch.int32, device=device)
        page_indices_ = page_indices.unsqueeze(0)

        k_torch = torch_get_k(pool, buf, seq_len, page_indices)
        s_torch = torch_get_s(pool, buf, seq_len, page_indices)
        k_triton, s_triton = GetKAndS.triton(
            pool, buf, page_indices_, seq_len_tensor, seq_len, seq_len
        )
        torch.testing.assert_close(k_triton, k_torch, rtol=0, atol=0)
        torch.testing.assert_close(s_triton, s_torch, rtol=0, atol=0)

    def test_large_seq_len(self):
        """Test with large sequence length."""
        device = torch.device("cuda")
        page_size = 64
        index_head_dim = 128
        num_pages = 100
        seq_len = 4096  # 64 pages
        seq_len_tensor = torch.tensor([seq_len], dtype=torch.int64, device=device)

        pool = MockDSATokenToKVPool(
            page_size=page_size, index_head_dim=index_head_dim, device=device
        )
        buf = create_test_buffer(num_pages, page_size, index_head_dim, device)

        num_pages_needed = (seq_len + page_size - 1) // page_size
        page_indices = torch.randint(
            0, num_pages, (num_pages_needed,), dtype=torch.int32, device=device
        )
        page_indices_ = page_indices.unsqueeze(0)

        k_torch = torch_get_k(pool, buf, seq_len, page_indices)
        s_torch = torch_get_s(pool, buf, seq_len, page_indices)
        k_triton, s_triton = GetKAndS.triton(
            pool, buf, page_indices_, seq_len_tensor, seq_len, seq_len
        )
        torch.testing.assert_close(k_triton, k_torch, rtol=0, atol=0)
        torch.testing.assert_close(s_triton, s_torch, rtol=0, atol=0)


def print_test_summary():
    """Print a summary message about the test suite."""
    print("\n" + "=" * 80)
    print("DSA Indexer K/S Buffer Accessor Correctness Tests")
    print("=" * 80)
    print("Testing GetKAndS.triton() against separate torch K/S gathers")
    print("=" * 80)
    print()


if __name__ == "__main__":
    # Run tests manually
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping tests.")
        exit(0)

    print_test_summary()

    # Run a few sample tests
    print("Running sample correctness tests...\n")

    # Test GetKAndS
    print("Testing GetKAndS SeqLen=256...")
    test_get_k_and_s = TestGetKAndS()
    test_get_k_and_s.test_get_k_and_s_correctness(
        num_pages=4, seq_len=256, page_size=64, index_head_dim=128
    )
    test_get_k_and_s.test_get_k_and_s_sequential_pages()
    test_get_k_and_s.test_get_k_and_s_partial_page()
    print("✓ GetKAndS SeqLen=256 tests passed\n")

    print("Testing GetKAndS SeqLen=128K...")
    test_get_k_and_s = TestGetKAndS()
    test_get_k_and_s.test_get_k_and_s_correctness(
        num_pages=2048, seq_len=131072, page_size=64, index_head_dim=128
    )
    test_get_k_and_s.test_get_k_and_s_sequential_pages()
    test_get_k_and_s.test_get_k_and_s_partial_page()
    print("✓ GetKAndS SeqLen=128K tests passed\n")

    # Test edge cases
    print("Testing edge cases...")
    test_edge = TestEdgeCases()
    test_edge.test_single_token()
    test_edge.test_exact_page_boundary()
    test_edge.test_large_seq_len()
    print("✓ Edge case tests passed\n")

    print("=" * 80)
    print("All correctness tests passed successfully!")
    print("=" * 80)
