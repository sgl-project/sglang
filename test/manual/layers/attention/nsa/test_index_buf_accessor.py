"""
Correctness tests for NSA Indexer K/S Buffer Access with Fused Triton Kernels.

This test verifies that the optimized Triton implementations (GetK, GetS, GetKAndS)
produce identical results to the torch_fast baseline implementations.

Test coverage:
- GetK.triton() vs GetK.torch_fast()
- GetS.triton() vs GetS.torch_fast()
- GetKAndS.triton() vs separate GetK.torch_fast() + GetS.torch_fast()
"""

import pytest
import torch

from sglang.srt.layers.attention.nsa.index_buf_accessor import GetK, GetKAndS, GetS


class MockNSATokenToKVPool:
    """Mock pool object that mimics NSATokenToKVPool for testing."""

    def __init__(
        self,
        page_size: int = 64,
        index_head_dim: int = 128,
        quant_block_size: int = 128,
        device: str = "cuda",
    ):
        self.page_size = page_size
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestGetK:
    """Test cases for GetK.triton() correctness."""

    @pytest.mark.parametrize("num_pages", [1, 2, 4, 8, 16])
    @pytest.mark.parametrize("seq_len", [64, 128, 256, 512, 1024])
    @pytest.mark.parametrize("page_size", [64])
    @pytest.mark.parametrize("index_head_dim", [128])
    def test_getk_correctness(self, num_pages, seq_len, page_size, index_head_dim):
        """Test GetK.triton() produces same output as GetK.torch_fast()."""
        device = torch.device("cuda")

        # Ensure seq_len doesn't exceed available pages
        max_seq_len = num_pages * page_size
        seq_len = min(seq_len, max_seq_len)

        # Create mock pool
        pool = MockNSATokenToKVPool(
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

        # Run both implementations
        output_torch = GetK.torch_fast(pool, buf, seq_len, page_indices)
        output_triton = GetK.triton(pool, buf, seq_len, page_indices)

        # Verify shapes
        assert output_torch.shape == (seq_len, index_head_dim)
        assert output_triton.shape == (seq_len, index_head_dim)
        assert output_torch.dtype == torch.uint8
        assert output_triton.dtype == torch.uint8

        # Compare results (should be exact match)
        torch.testing.assert_close(
            output_triton, output_torch, rtol=0, atol=0, msg="GetK outputs differ"
        )

    def test_getk_sequential_pages(self):
        """Test GetK with sequential page indices."""
        device = torch.device("cuda")
        page_size = 64
        index_head_dim = 128
        num_pages = 10
        seq_len = 320  # 5 pages

        pool = MockNSATokenToKVPool(
            page_size=page_size, index_head_dim=index_head_dim, device=device
        )
        buf = create_test_buffer(num_pages, page_size, index_head_dim, device)

        # Sequential page indices [0, 1, 2, 3, 4]
        page_indices = torch.arange(5, dtype=torch.int32, device=device)

        output_torch = GetK.torch_fast(pool, buf, seq_len, page_indices)
        output_triton = GetK.triton(pool, buf, seq_len, page_indices)

        torch.testing.assert_close(output_triton, output_torch, rtol=0, atol=0)

    def test_getk_repeated_pages(self):
        """Test GetK with repeated page indices."""
        device = torch.device("cuda")
        page_size = 64
        index_head_dim = 128
        num_pages = 5
        seq_len = 192  # 3 pages

        pool = MockNSATokenToKVPool(
            page_size=page_size, index_head_dim=index_head_dim, device=device
        )
        buf = create_test_buffer(num_pages, page_size, index_head_dim, device)

        # Repeated page indices [2, 2, 2]
        page_indices = torch.full((3,), 2, dtype=torch.int32, device=device)

        output_torch = GetK.torch_fast(pool, buf, seq_len, page_indices)
        output_triton = GetK.triton(pool, buf, seq_len, page_indices)

        torch.testing.assert_close(output_triton, output_torch, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestGetS:
    """Test cases for GetS.triton() correctness."""

    @pytest.mark.parametrize("num_pages", [1, 2, 4, 8, 16])
    @pytest.mark.parametrize("seq_len", [64, 128, 256, 512, 1024])
    @pytest.mark.parametrize("page_size", [64])
    @pytest.mark.parametrize("index_head_dim", [128])
    def test_gets_correctness(self, num_pages, seq_len, page_size, index_head_dim):
        """Test GetS.triton() produces same output as GetS.torch_fast()."""
        device = torch.device("cuda")

        # Ensure seq_len doesn't exceed available pages
        max_seq_len = num_pages * page_size
        seq_len = min(seq_len, max_seq_len)

        # Create mock pool
        pool = MockNSATokenToKVPool(
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

        # Run both implementations
        output_torch = GetS.torch_fast(pool, buf, seq_len, page_indices)
        output_triton = GetS.triton(pool, buf, seq_len, page_indices)

        # Verify shapes
        assert output_torch.shape == (seq_len, 4)
        assert output_triton.shape == (seq_len, 4)
        assert output_torch.dtype == torch.uint8
        assert output_triton.dtype == torch.uint8

        # Compare results (should be exact match)
        torch.testing.assert_close(
            output_triton, output_torch, rtol=0, atol=0, msg="GetS outputs differ"
        )

    def test_gets_sequential_pages(self):
        """Test GetS with sequential page indices."""
        device = torch.device("cuda")
        page_size = 64
        index_head_dim = 128
        num_pages = 10
        seq_len = 320  # 5 pages

        pool = MockNSATokenToKVPool(
            page_size=page_size, index_head_dim=index_head_dim, device=device
        )
        buf = create_test_buffer(num_pages, page_size, index_head_dim, device)

        # Sequential page indices [0, 1, 2, 3, 4]
        page_indices = torch.arange(5, dtype=torch.int32, device=device)

        output_torch = GetS.torch_fast(pool, buf, seq_len, page_indices)
        output_triton = GetS.triton(pool, buf, seq_len, page_indices)

        torch.testing.assert_close(output_triton, output_torch, rtol=0, atol=0)

    def test_gets_repeated_pages(self):
        """Test GetS with repeated page indices."""
        device = torch.device("cuda")
        page_size = 64
        index_head_dim = 128
        num_pages = 5
        seq_len = 192  # 3 pages

        pool = MockNSATokenToKVPool(
            page_size=page_size, index_head_dim=index_head_dim, device=device
        )
        buf = create_test_buffer(num_pages, page_size, index_head_dim, device)

        # Repeated page indices [2, 2, 2]
        page_indices = torch.full((3,), 2, dtype=torch.int32, device=device)

        output_torch = GetS.torch_fast(pool, buf, seq_len, page_indices)
        output_triton = GetS.triton(pool, buf, seq_len, page_indices)

        torch.testing.assert_close(output_triton, output_torch, rtol=0, atol=0)


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
        """Test GetKAndS.triton() produces same output as separate torch_fast calls."""
        device = torch.device("cuda")

        # Ensure seq_len doesn't exceed available pages
        max_seq_len = num_pages * page_size
        seq_len = min(seq_len, max_seq_len)

        # Create mock pool
        pool = MockNSATokenToKVPool(
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

        # Run baseline: separate torch_fast calls
        k_torch = GetK.torch_fast(pool, buf, seq_len, page_indices)
        s_torch = GetS.torch_fast(pool, buf, seq_len, page_indices)

        # Run fused Triton implementation
        k_triton, s_triton = GetKAndS.triton(pool, buf, seq_len, page_indices)

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

        pool = MockNSATokenToKVPool(
            page_size=page_size, index_head_dim=index_head_dim, device=device
        )
        buf = create_test_buffer(num_pages, page_size, index_head_dim, device)

        # Sequential page indices [0, 1, 2, 3, 4]
        page_indices = torch.arange(5, dtype=torch.int32, device=device)

        # Baseline
        k_torch = GetK.torch_fast(pool, buf, seq_len, page_indices)
        s_torch = GetS.torch_fast(pool, buf, seq_len, page_indices)

        # Fused
        k_triton, s_triton = GetKAndS.triton(pool, buf, seq_len, page_indices)

        torch.testing.assert_close(k_triton, k_torch, rtol=0, atol=0)
        torch.testing.assert_close(s_triton, s_torch, rtol=0, atol=0)

    def test_get_k_and_s_repeated_pages(self):
        """Test GetKAndS with repeated page indices."""
        device = torch.device("cuda")
        page_size = 64
        index_head_dim = 128
        num_pages = 5
        seq_len = 192  # 3 pages

        pool = MockNSATokenToKVPool(
            page_size=page_size, index_head_dim=index_head_dim, device=device
        )
        buf = create_test_buffer(num_pages, page_size, index_head_dim, device)

        # Repeated page indices [2, 2, 2]
        page_indices = torch.full((3,), 2, dtype=torch.int32, device=device)

        # Baseline
        k_torch = GetK.torch_fast(pool, buf, seq_len, page_indices)
        s_torch = GetS.torch_fast(pool, buf, seq_len, page_indices)

        # Fused
        k_triton, s_triton = GetKAndS.triton(pool, buf, seq_len, page_indices)

        torch.testing.assert_close(k_triton, k_torch, rtol=0, atol=0)
        torch.testing.assert_close(s_triton, s_torch, rtol=0, atol=0)

    def test_get_k_and_s_partial_page(self):
        """Test GetKAndS when seq_len is not a multiple of page_size."""
        device = torch.device("cuda")
        page_size = 64
        index_head_dim = 128
        num_pages = 5
        seq_len = 100  # Not a multiple of 64

        pool = MockNSATokenToKVPool(
            page_size=page_size, index_head_dim=index_head_dim, device=device
        )
        buf = create_test_buffer(num_pages, page_size, index_head_dim, device)

        num_pages_needed = (seq_len + page_size - 1) // page_size
        page_indices = torch.arange(num_pages_needed, dtype=torch.int32, device=device)

        # Baseline
        k_torch = GetK.torch_fast(pool, buf, seq_len, page_indices)
        s_torch = GetS.torch_fast(pool, buf, seq_len, page_indices)

        # Fused
        k_triton, s_triton = GetKAndS.triton(pool, buf, seq_len, page_indices)

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

        pool = MockNSATokenToKVPool(
            page_size=page_size, index_head_dim=index_head_dim, device=device
        )
        buf = create_test_buffer(num_pages, page_size, index_head_dim, device)
        page_indices = torch.tensor([0], dtype=torch.int32, device=device)

        # Test GetK
        k_torch = GetK.torch_fast(pool, buf, seq_len, page_indices)
        k_triton = GetK.triton(pool, buf, seq_len, page_indices)
        torch.testing.assert_close(k_triton, k_torch, rtol=0, atol=0)

        # Test GetS
        s_torch = GetS.torch_fast(pool, buf, seq_len, page_indices)
        s_triton = GetS.triton(pool, buf, seq_len, page_indices)
        torch.testing.assert_close(s_triton, s_torch, rtol=0, atol=0)

        # Test GetKAndS
        k_triton2, s_triton2 = GetKAndS.triton(pool, buf, seq_len, page_indices)
        torch.testing.assert_close(k_triton2, k_torch, rtol=0, atol=0)
        torch.testing.assert_close(s_triton2, s_torch, rtol=0, atol=0)

    def test_exact_page_boundary(self):
        """Test when seq_len exactly matches page boundaries."""
        device = torch.device("cuda")
        page_size = 64
        index_head_dim = 128
        num_pages = 5
        seq_len = 192  # Exactly 3 pages

        pool = MockNSATokenToKVPool(
            page_size=page_size, index_head_dim=index_head_dim, device=device
        )
        buf = create_test_buffer(num_pages, page_size, index_head_dim, device)
        page_indices = torch.arange(3, dtype=torch.int32, device=device)

        # Test GetK
        k_torch = GetK.torch_fast(pool, buf, seq_len, page_indices)
        k_triton = GetK.triton(pool, buf, seq_len, page_indices)
        torch.testing.assert_close(k_triton, k_torch, rtol=0, atol=0)

        # Test GetS
        s_torch = GetS.torch_fast(pool, buf, seq_len, page_indices)
        s_triton = GetS.triton(pool, buf, seq_len, page_indices)
        torch.testing.assert_close(s_triton, s_torch, rtol=0, atol=0)

        # Test GetKAndS
        k_triton2, s_triton2 = GetKAndS.triton(pool, buf, seq_len, page_indices)
        torch.testing.assert_close(k_triton2, k_torch, rtol=0, atol=0)
        torch.testing.assert_close(s_triton2, s_torch, rtol=0, atol=0)

    def test_large_seq_len(self):
        """Test with large sequence length."""
        device = torch.device("cuda")
        page_size = 64
        index_head_dim = 128
        num_pages = 100
        seq_len = 4096  # 64 pages

        pool = MockNSATokenToKVPool(
            page_size=page_size, index_head_dim=index_head_dim, device=device
        )
        buf = create_test_buffer(num_pages, page_size, index_head_dim, device)

        num_pages_needed = (seq_len + page_size - 1) // page_size
        page_indices = torch.randint(
            0, num_pages, (num_pages_needed,), dtype=torch.int32, device=device
        )

        # Test GetK
        k_torch = GetK.torch_fast(pool, buf, seq_len, page_indices)
        k_triton = GetK.triton(pool, buf, seq_len, page_indices)
        torch.testing.assert_close(k_triton, k_torch, rtol=0, atol=0)

        # Test GetS
        s_torch = GetS.torch_fast(pool, buf, seq_len, page_indices)
        s_triton = GetS.triton(pool, buf, seq_len, page_indices)
        torch.testing.assert_close(s_triton, s_torch, rtol=0, atol=0)

        # Test GetKAndS
        k_triton2, s_triton2 = GetKAndS.triton(pool, buf, seq_len, page_indices)
        torch.testing.assert_close(k_triton2, k_torch, rtol=0, atol=0)
        torch.testing.assert_close(s_triton2, s_torch, rtol=0, atol=0)


def print_test_summary():
    """Print a summary message about the test suite."""
    print("\n" + "=" * 80)
    print("NSA Indexer K/S Buffer Accessor Correctness Tests")
    print("=" * 80)
    print("Testing Triton implementations against torch_fast baseline:")
    print("  - GetK.triton() vs GetK.torch_fast()")
    print("  - GetS.triton() vs GetS.torch_fast()")
    print("  - GetKAndS.triton() vs separate GetK/GetS torch_fast() calls")
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

    # Test GetK
    print("Testing GetK...")
    test_getk = TestGetK()
    test_getk.test_getk_correctness(
        num_pages=4, seq_len=256, page_size=64, index_head_dim=128
    )
    test_getk.test_getk_sequential_pages()
    print("✓ GetK tests passed\n")

    # Test GetS
    print("Testing GetS...")
    test_gets = TestGetS()
    test_gets.test_gets_correctness(
        num_pages=4, seq_len=256, page_size=64, index_head_dim=128
    )
    test_gets.test_gets_sequential_pages()
    print("✓ GetS tests passed\n")

    # Test GetKAndS
    print("Testing GetKAndS...")
    test_get_k_and_s = TestGetKAndS()
    test_get_k_and_s.test_get_k_and_s_correctness(
        num_pages=4, seq_len=256, page_size=64, index_head_dim=128
    )
    test_get_k_and_s.test_get_k_and_s_sequential_pages()
    test_get_k_and_s.test_get_k_and_s_partial_page()
    print("✓ GetKAndS tests passed\n")

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
