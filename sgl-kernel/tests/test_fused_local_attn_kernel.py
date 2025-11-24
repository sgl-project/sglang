import torch
import numpy as np
from typing import Tuple
import sys


def cdiv(x: int, y: int) -> int:
    """Ceiling division."""
    return (x + y - 1) // y


def make_local_attention_virtual_batches(
    attn_chunk_size: int,
    query_start_loc_np: np.ndarray,
    seq_lens_np: np.ndarray,
    block_table: torch.Tensor,
    page_size: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, torch.Tensor]:
    """
    Python reference implementation.
    """
    max_seq_len = seq_lens_np.max()
    effective_chunk_size = min(attn_chunk_size, max_seq_len)
    effective_chunk_size = (effective_chunk_size // page_size) * page_size
    if effective_chunk_size < page_size:
        effective_chunk_size = page_size
    attn_chunk_size = effective_chunk_size

    q_seqlens = query_start_loc_np[1:] - query_start_loc_np[:-1]
    actual_batch_size = seq_lens_np.shape[0]

    q_tokens_in_first_block = np.minimum(
        attn_chunk_size - ((seq_lens_np - q_seqlens) % attn_chunk_size), q_seqlens
    ).astype(np.int32)
    tokens_in_last_block = attn_chunk_size + (seq_lens_np % -attn_chunk_size)
    local_blocks = 1 + cdiv(q_seqlens - q_tokens_in_first_block, attn_chunk_size)

    cu_num_blocks = np.cumsum(local_blocks)
    virtual_batches = cu_num_blocks[-1]
    block_offsets = np.repeat(cu_num_blocks - local_blocks, local_blocks)
    arange = np.arange(virtual_batches, dtype=np.int32) - block_offsets
    rarange = np.repeat(local_blocks, local_blocks) - arange - 1
    
    seqlens_q_local = np.repeat(q_seqlens - q_tokens_in_first_block, local_blocks)
    seqlens_q_local[arange == 0] = q_tokens_in_first_block
    seqlens_q_local[arange > 0] = np.minimum(
        seqlens_q_local - attn_chunk_size * (arange - 1), attn_chunk_size
    )[arange > 0]

    cu_seqlens_q_local = np.pad(np.cumsum(seqlens_q_local), (1, 0)).astype(np.int32)

    seqlens_k_local = np.full(cu_num_blocks[-1], attn_chunk_size, dtype=np.int32)
    seqlens_k_local[cu_num_blocks - 1] = tokens_in_last_block

    k_seqstarts_absolute = np.repeat(seq_lens_np, local_blocks) - (
        rarange * attn_chunk_size + np.repeat(tokens_in_last_block, local_blocks)
    )
    block_starts = k_seqstarts_absolute // page_size

    assert attn_chunk_size % page_size == 0
    pages_per_local_batch = attn_chunk_size // page_size

    block_indices = np.broadcast_to(
        np.arange(pages_per_local_batch, dtype=np.int32),
        (virtual_batches, pages_per_local_batch),
    ) + np.expand_dims(block_starts, axis=1)
    block_indices = block_indices.flatten().clip(max=block_table.shape[1] - 1)
    batch_indices = np.repeat(
        np.arange(actual_batch_size, dtype=np.int32),
        local_blocks * pages_per_local_batch,
    )
    block_table_local = block_table[batch_indices, block_indices].view(
        virtual_batches, -1
    )

    return seqlens_q_local, cu_seqlens_q_local, seqlens_k_local, block_table_local


def generate_test_case(
    batch_size: int,
    attn_chunk_size: int,
    page_size: int,
    device: str = "cuda",
    seed: int = None
) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, int, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate random test inputs."""
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    # Generate random sequence lengths - ALL must be int32!
    min_seq_len = page_size * 2
    max_seq_len = attn_chunk_size * 3
    seq_lens = torch.randint(min_seq_len, max_seq_len + 1, (batch_size,), dtype=torch.int32, device='cpu')
    
    # Generate random query lengths (decode: usually 1, prefill: could be larger)
    # For testing, we'll use small query lengths
    q_lens = torch.randint(1, min(attn_chunk_size, 10) + 1, (batch_size,), dtype=torch.int32, device='cpu')
    q_lens = torch.minimum(q_lens, seq_lens).to(torch.int32)  # Ensure int32 after minimum
    
    # Create query_start_loc (cumulative sum) - ensure int32
    cumsum_q = torch.cumsum(q_lens, dim=0).to(torch.int32)
    query_start_loc = torch.cat([torch.tensor([0], dtype=torch.int32), cumsum_q])
    
    # Create block table - ensure int32
    max_blocks = (max_seq_len + page_size - 1) // page_size + 10
    block_table = torch.arange(batch_size * max_blocks, dtype=torch.int32).view(batch_size, max_blocks)
    
    # Move to device and ensure int32
    query_start_loc = query_start_loc.to(device=device, dtype=torch.int32)
    seq_lens = seq_lens.to(device=device, dtype=torch.int32)
    block_table = block_table.to(device=device, dtype=torch.int32)
    
    # Calculate buffer sizes
    max_virtual_batches = batch_size * ((max_seq_len + attn_chunk_size - 1) // attn_chunk_size)
    q_capacity = max_virtual_batches + 1
    k_capacity = max_virtual_batches
    block_rows = max_virtual_batches
    block_cols = attn_chunk_size // page_size
    
    local_q_buf = torch.zeros(q_capacity, dtype=torch.int32, device=device)
    local_k_buf = torch.zeros(k_capacity, dtype=torch.int32, device=device)
    local_block_buf = torch.zeros((block_rows, block_cols), dtype=torch.int32, device=device)
    
    return (
        attn_chunk_size,
        query_start_loc,
        seq_lens,
        block_table,
        page_size,
        local_q_buf,
        local_k_buf,
        local_block_buf
    )


def run_python_implementation(
    attn_chunk_size: int,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    page_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, torch.Tensor]:
    """Run Python implementation."""
    query_start_loc_np = query_start_loc.cpu().numpy()
    seq_lens_np = seq_lens.cpu().numpy()
    
    # Adjust chunk size like Python does
    max_seq_len = seq_lens_np.max()
    effective_chunk_size = min(attn_chunk_size, max_seq_len)
    effective_chunk_size = (effective_chunk_size // page_size) * page_size
    if effective_chunk_size < page_size:
        effective_chunk_size = page_size
    
    # Slice block_table to match max_seq_len
    max_seq_len_adjusted = int(max_seq_len)
    sliced_block_table = block_table[:seq_lens.shape[0], :max_seq_len_adjusted]
    
    return make_local_attention_virtual_batches(
        effective_chunk_size,
        query_start_loc_np,
        seq_lens_np,
        sliced_block_table,
        page_size,
    )


def compare_outputs(
    cuda_outputs: list,
    python_outputs: Tuple,
    test_name: str,
    verbose: bool = True
) -> bool:
    """Compare CUDA and Python outputs."""
    seqlens_q_local_cuda, cu_seqlens_q_local_cuda, seqlens_k_local_cuda, block_table_local_cuda, metadata_cuda = cuda_outputs
    seqlens_q_local_py, cu_seqlens_q_local_py, seqlens_k_local_py, block_table_local_py = python_outputs
    
    # Convert to numpy for comparison
    seqlens_q_local_cuda_np = seqlens_q_local_cuda.cpu().numpy()
    cu_seqlens_q_local_cuda_np = cu_seqlens_q_local_cuda.cpu().numpy()
    seqlens_k_local_cuda_np = seqlens_k_local_cuda.cpu().numpy()
    block_table_local_cuda_np = block_table_local_cuda.cpu().numpy()
    block_table_local_py_np = block_table_local_py.cpu().numpy()
    
    all_match = True
    
    # Compare seqlens_q_local
    if not np.array_equal(seqlens_q_local_cuda_np, seqlens_q_local_py):
        all_match = False
        if verbose:
            print(f"❌ {test_name}: seqlens_q_local mismatch")
            print(f"  CUDA: {seqlens_q_local_cuda_np}")
            print(f"  Python: {seqlens_q_local_py}")
    
    # Compare cu_seqlens_q_local
    if not np.array_equal(cu_seqlens_q_local_cuda_np, cu_seqlens_q_local_py):
        all_match = False
        if verbose:
            print(f"❌ {test_name}: cu_seqlens_q_local mismatch")
            print(f"  CUDA: {cu_seqlens_q_local_cuda_np}")
            print(f"  Python: {cu_seqlens_q_local_py}")
    
    # Compare seqlens_k_local
    if not np.array_equal(seqlens_k_local_cuda_np, seqlens_k_local_py):
        all_match = False
        if verbose:
            print(f"❌ {test_name}: seqlens_k_local mismatch")
            print(f"  CUDA: {seqlens_k_local_cuda_np}")
            print(f"  Python: {seqlens_k_local_py}")
    
    # Compare block_table_local
    if not np.array_equal(block_table_local_cuda_np, block_table_local_py_np):
        all_match = False
        if verbose:
            print(f"❌ {test_name}: block_table_local mismatch")
            print(f"  CUDA shape: {block_table_local_cuda_np.shape}")
            print(f"  Python shape: {block_table_local_py_np.shape}")
            print(f"  CUDA:\n{block_table_local_cuda_np}")
            print(f"  Python:\n{block_table_local_py_np}")
    
    # Check metadata
    if verbose and all_match:
        metadata_np = metadata_cuda.cpu().numpy()
        print(f"✅ {test_name}: All outputs match!")
        print(f"  Metadata: q_len={metadata_np[0]}, k_len={metadata_np[1]}, "
              f"b0={metadata_np[2]}, b1={metadata_np[3]}, "
              f"max_q={metadata_np[4]}, max_k={metadata_np[5]}")
        print(f"  Virtual batches: {len(seqlens_q_local_py)}")
    
    return all_match


def test_specific_case(cuda_module, test_name: str, batch_size: int, attn_chunk_size: int, 
                       page_size: int, seed: int = None, verbose: bool = True) -> bool:
    """Test a specific configuration."""
    try:
        # Generate test case
        args = generate_test_case(batch_size, attn_chunk_size, page_size, "cuda", seed)
        
        # Debug: Check all tensor dtypes
        if verbose:
            print(f"\nDebug - Input tensor dtypes for {test_name}:")
            print(f"  query_start_loc: {args[1].dtype}")
            print(f"  seq_lens: {args[2].dtype}")
            print(f"  block_table: {args[3].dtype}")
            print(f"  local_q_buf: {args[5].dtype}")
            print(f"  local_k_buf: {args[6].dtype}")
            print(f"  local_block_buf: {args[7].dtype}")
        
        # Run CUDA implementation
        cuda_outputs = cuda_module.make_local_attention_virtual_batches_fully_fused(*args)
        
        # Run Python implementation
        python_outputs = run_python_implementation(args[0], args[1], args[2], args[3], args[4])
        
        # Compare outputs
        return compare_outputs(cuda_outputs, python_outputs, test_name, verbose)
    except Exception as e:
        print(f"❌ {test_name}: Exception occurred: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_edge_cases(cuda_module, verbose: bool = True) -> int:
    """Test edge cases."""
    passed = 0
    total = 0
    
    edge_cases = [
        ("Single batch, small seq", 1, 64, 16, 42),
        ("Multiple batches, aligned", 4, 128, 16, 43),
        ("Large chunk size", 8, 512, 32, 44),
        ("Small chunk size", 5, 32, 8, 45),
        ("Many small batches", 16, 64, 16, 46),
        ("Varying sizes", 10, 256, 32, 47),
    ]
    
    for test_name, batch_size, attn_chunk_size, page_size, seed in edge_cases:
        total += 1
        # Only show debug info for first test
        show_debug = (total == 1) and verbose
        if test_specific_case(cuda_module, test_name, batch_size, attn_chunk_size, page_size, seed, show_debug):
            passed += 1
    
    return passed, total


def main():
    """Main test function."""
    print("=" * 80)
    print("Local Attention Virtual Batches - CUDA vs Python Comparison Test")
    print("=" * 80)
    
    try:
        import local_attention_cuda
        cuda_module = local_attention_cuda
        print("✅ CUDA module loaded successfully")
    except ImportError as e:
        print(f"❌ Failed to import CUDA module: {e}")
        print("Please compile the CUDA extension first.")
        print("\nYou can compile it using:")
        print("  python setup.py install")
        print("or use torch.utils.cpp_extension.load() in your code")
        sys.exit(1)
    
    print()
    
    # Run tests
    print("Running edge case tests...")
    print("-" * 80)
    passed, total = test_edge_cases(cuda_module, verbose=True)
    print()
    print("=" * 80)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ All tests passed! CUDA and Python implementations match.")
        sys.exit(0)
    else:
        print(f"❌ {total - passed} tests failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()