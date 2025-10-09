import torch
from sglang.srt.batch_invariant_ops.batch_invariant_ops import set_batch_invariant_mode
device_type = getattr(torch.accelerator.current_accelerator(), "type", "cpu")
torch.set_default_device(device_type)

# Just to get the logging out of the way haha
with set_batch_invariant_mode(True):
    pass

def test_batch_invariance(M=32, K=128, N=1024, dtype=torch.float32, verbose=False):
    a = torch.linspace(-100, 100, M*K, dtype=dtype).reshape(M, K)

    # change b to be non-contiguous
    b = torch.linspace(-100, 100, K*N, dtype=dtype).reshape(N, K)

    if verbose:
        print(f"a is contiguous: {a.is_contiguous()}")
        print(f"b is contiguous: {b.is_contiguous()}")

    b = b.transpose(0, 1)
    if verbose:
        print(f"transposedb is contiguous: {b.is_contiguous()}")
        print(f"a shape: {a.shape}, strides: {a.stride()}")
        print(f"b shape: {b.shape}, strides: {b.stride()}")
    
    # Method 1: Matrix-vector multiplication (batch size 1)
    out1 = torch.mm(a[:1], b)

    # Method 2: Matrix-matrix multiplication, then slice (full batch)
    out2_pre = torch.mm(a, b)
    if verbose:
        print(out2_pre[0, :5])
    out2 = out2_pre[:1]

    # Check if results are identical
    diff = (out1 - out2).abs().max()
    return diff.item() == 0, diff

def run_iters(iters=1, M=32, K=128, N=1024, verbose=False):
    for dtype in [ torch.float32 , torch.bfloat16 ]:
        is_deterministic = True
        difflist = []
        for i in range (iters):
            isd, df = test_batch_invariance(M, K, N, dtype, verbose)
            is_deterministic = is_deterministic and isd
            difflist.append(df)
        print( f"Batch Deterministic: {is_deterministic} run-to-run max/min/diff {max(difflist)}/{min(difflist)}/{max(difflist)-min(difflist)} for {dtype} in {iters} iterations")


def run_test_suite():
    """Run test cases covering small, medium, and large matrix sizes"""
    test_cases = [
        # Small sizes
        ("Small-1", 8, 64, 128),
        ("Small-2", 16, 128, 256),
        ("Small-3", 4, 32, 64),
        # Medium sizes
        ("Medium-1", 32, 128, 1024),
        ("Medium-2", 64, 512, 2048),
        ("Medium-3", 24, 192, 768),
        # Large sizes
        ("Large-1", 128, 1024, 4096),
        ("Large-2", 256, 2048, 8192),
        ("Large-3", 96, 768, 3072),
    ]
    
    print("=" * 80)
    print("Running Test Suite - Small, Medium, and Large Matrix Sizes")
    print("=" * 80)
    
    for name, M, K, N in test_cases:
        print(f"\n[{name}] Testing M={M}, K={K}, N={N}")
        print("-" * 60)
        
        # Test with standard PyTorch
        print("  Standard PyTorch:")
        with set_batch_invariant_mode(False):
            run_iters(iters=1, M=M, K=K, N=N, verbose=False)
        
        # Test with batch-invariant operations
        print("  Batch-Invariant Mode:")
        with set_batch_invariant_mode(True):
            run_iters(iters=1, M=M, K=K, N=N, verbose=False)

if __name__ == "__main__":
    # Run the comprehensive test suite
    run_test_suite()