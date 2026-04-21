"""
Standalone test for flashinfer BatchDecodeWithPagedKVCacheWrapper with NVFP4 KV cache.

Can be run in two modes:
  1. Synthetic mode (default): generates random valid NVFP4 data and runs the kernel.
  2. Replay mode: loads a dump file produced by the NaN detection path in forward_decode
     and replays the exact inputs that triggered the NaN.

Usage:
  # Synthetic test (page_size=1, various batch/seq combinations):
  python test_nvfp4_decode.py

  # Replay a NaN dump:
  python test_nvfp4_decode.py --dump /tmp/nvfp4_decode_nan_dump.pt
"""

import argparse
import sys

import torch

try:
    from flashinfer import BatchDecodeWithPagedKVCacheWrapper

    _flashinfer_available = True
except ImportError:
    _flashinfer_available = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_valid_fp8_scales(shape, device):
    """Create valid float8_e4m3fn scale tensor (avoid NaN 0x7F values)."""
    valid_fp8_ints = [32, 40, 48, 56, 64, 80, 96, 112]  # 0.125 .. 1.0 in fp8e4m3
    raw = torch.randint(0, len(valid_fp8_ints), shape, device=device)
    buf = torch.zeros(shape, dtype=torch.uint8, device=device)
    for i, v in enumerate(valid_fp8_ints):
        buf[raw == i] = v
    return buf


def _quantize_to_nvfp4(kv_bf16):
    """
    Quantize [total_tokens, nkh, head_dim] BF16 → NVFP4 format.
    Returns:
      fp4_data:  [total_tokens, nkh, head_dim//2] uint8
      fp8_scales:[total_tokens, nkh, head_dim//16] uint8
      global_scale: float
    """
    total_tokens, nkh, head_dim = kv_bf16.shape
    block_size = 16
    # Per-block absmax, reshaped for block scaling
    kv_blocks = kv_bf16.view(total_tokens, nkh, head_dim // block_size, block_size)
    block_max = kv_blocks.abs().amax(dim=-1).float()  # [T, nkh, head_dim//block_size]

    # Global scale from overall max
    global_max = block_max.max().item()
    global_scale = max(global_max / 6.0, 1e-30)

    # Per-block FP8 scale: block_max / global_scale, clamped to FP8 range
    fp8_max = torch.finfo(torch.float8_e4m3fn).max  # 448.0
    block_scale = (block_max / global_scale).clamp(1e-30, fp8_max)
    fp8_scales = block_scale.to(torch.float8_e4m3fn).view(torch.uint8)
    # shape: [total_tokens, nkh, head_dim//16]

    # Quantize values to FP4 E2M1 (range [-6, 6])
    # Scale each block by its block scale * global_scale
    scale_expanded = block_scale.unsqueeze(-1).expand_as(kv_blocks)  # [T, nkh, hd//16, 16]
    kv_scaled = kv_bf16.view(total_tokens, nkh, head_dim // block_size, block_size).float() / (
        scale_expanded * global_scale + 1e-30
    )
    kv_clamped = kv_scaled.clamp(-6.0, 6.0)

    # Simple FP4 E2M1 quantization (round to nearest representable value)
    # FP4 E2M1 values: 0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6
    fp4_levels = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=kv_bf16.device)

    kv_flat = kv_clamped.view(-1)
    sign = kv_flat.sign()
    mag = kv_flat.abs()
    # Find nearest FP4 magnitude
    diffs = (mag.unsqueeze(-1) - fp4_levels.unsqueeze(0)).abs()
    nearest_idx = diffs.argmin(dim=-1)  # 0..7
    quantized = sign * fp4_levels[nearest_idx]
    kv_quantized = quantized.view(total_tokens, nkh, head_dim)

    # Pack two FP4 values per byte (low nibble = first, high nibble = second)
    # FP4 E2M1 encoding: sign[3] | exp[2:1] | mant[0]
    def to_fp4_bits(val):
        """Convert float scalar to 4-bit FP4 E2M1 integer."""
        sign_bit = 1 if val < 0 else 0
        mag = abs(float(val))
        # FP4 E2M1: exp bias = 1, mantissa 1 bit
        # Values: 0=0b0000, 0.5=0b0001(denorm), 1=0b0010, 1.5=0b0011,
        #         2=0b0100, 3=0b0101, 4=0b0110, 6=0b0111
        fp4_map = {0.0: 0, 0.5: 1, 1.0: 2, 1.5: 3, 2.0: 4, 3.0: 5, 4.0: 6, 6.0: 7}
        bits = fp4_map.get(round(mag * 2) / 2, 0) | (sign_bit << 3)
        return bits

    # For simplicity, pack on CPU
    kv_np = kv_quantized.cpu().numpy()
    import numpy as np
    packed = np.zeros((total_tokens, nkh, head_dim // 2), dtype=np.uint8)
    for t in range(total_tokens):
        for h in range(nkh):
            for d in range(0, head_dim, 2):
                lo = to_fp4_bits(kv_np[t, h, d])
                hi = to_fp4_bits(kv_np[t, h, d + 1])
                packed[t, h, d // 2] = lo | (hi << 4)

    fp4_data = torch.from_numpy(packed).to(kv_bf16.device)
    return fp4_data, fp8_scales, global_scale


def run_batch_decode_nvfp4(
    batch_size,
    seq_lens,
    num_qo_heads,
    num_kv_heads,
    head_dim,
    device="cuda",
    verbose=True,
):
    """
    Run BatchDecodeWithPagedKVCacheWrapper with NVFP4 KV cache.
    page_size=1 (SGLang default).

    Returns True if output is NaN-free.
    """
    page_size = 1
    total_tokens = sum(seq_lens)

    # Build KV data
    kv_bf16 = torch.randn(total_tokens, num_kv_heads, head_dim, device=device, dtype=torch.bfloat16)
    fp4_data, fp8_scales, global_scale = _quantize_to_nvfp4(kv_bf16)

    # Unsqueeze page dim: [total_tokens, 1, nkh, d//2] and [total_tokens, 1, nkh, d//16]
    k_buf = fp4_data.unsqueeze(1)        # [T, 1, nkh, d//2]
    v_buf = fp4_data.unsqueeze(1)        # use same data for K and V
    k_sf = fp8_scales.unsqueeze(1)       # [T, 1, nkh, d//16]
    v_sf = fp8_scales.unsqueeze(1)

    # Build paged KV indices (page_size=1: each token is its own page)
    # Indices: token 0..seq_lens[0]-1 for req 0, seq_lens[0]..seq_lens[0]+seq_lens[1]-1 for req 1, etc.
    kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    kv_indptr[1:] = torch.cumsum(torch.tensor(seq_lens, dtype=torch.int32), dim=0).to(device)
    kv_indices = torch.arange(total_tokens, dtype=torch.int32, device=device)
    kv_last_page_len = torch.ones(batch_size, dtype=torch.int32, device=device)

    # Query
    q = torch.randn(batch_size, num_qo_heads, head_dim, device=device, dtype=torch.bfloat16)

    # Create wrapper
    workspace = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device)
    wrapper = BatchDecodeWithPagedKVCacheWrapper(workspace, "NHD", use_tensor_cores=True)

    wrapper.begin_forward(
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        data_type=torch.uint8,
        q_data_type=torch.bfloat16,
    )

    # k_scale/v_scale must be plain Python floats (not tensors)
    o = wrapper.run(
        q,
        (k_buf, v_buf),
        k_scale=float(global_scale),
        v_scale=float(global_scale),
        kv_cache_sf=(k_sf, v_sf),
    )

    has_nan = torch.isnan(o).any().item()
    if verbose:
        status = "NaN!" if has_nan else "OK"
        print(
            f"  bs={batch_size:3d}  seq_lens={seq_lens[:4]}{'...' if len(seq_lens)>4 else ''}  "
            f"nqh={num_qo_heads} nkh={num_kv_heads} d={head_dim}  → {status}"
        )
    return not has_nan


def replay_dump(dump_path, device="cuda"):
    """Replay a NaN dump file produced by forward_decode's NaN detection."""
    print(f"\n=== Replaying dump: {dump_path} ===")
    data = torch.load(dump_path, map_location="cpu")

    q = data["q"].to(device)
    k_buf = data["k_buf"].to(device)
    v_buf = data["v_buf"].to(device)
    k_sf_buf = data["k_sf_buf"].to(device)
    v_sf_buf = data["v_sf_buf"].to(device)
    # k_scale/v_scale must be plain float
    k_scale = float(data["k_scale"]) if data["k_scale"] is not None else None
    v_scale = float(data["v_scale"]) if data["v_scale"] is not None else None
    kv_indptr = data["kv_indptr"].to(device)
    kv_indices = data["kv_indices"].to(device)
    num_qo_heads = data["num_qo_heads"]
    num_kv_heads = data["num_kv_heads"]
    head_dim = data["head_dim"]

    batch_size = kv_indptr.shape[0] - 1
    kv_last_page_len = torch.ones(batch_size, dtype=torch.int32, device=device)

    print(f"  batch_size={batch_size}, num_qo_heads={num_qo_heads}, num_kv_heads={num_kv_heads}, head_dim={head_dim}")
    print(f"  q shape: {q.shape}, k_buf shape: {k_buf.shape}")
    print(f"  kv_indptr: {kv_indptr.cpu().tolist()}")
    seq_lens = (kv_indptr[1:] - kv_indptr[:-1]).cpu().tolist()
    print(f"  seq_lens: {seq_lens}")

    workspace = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device)
    wrapper = BatchDecodeWithPagedKVCacheWrapper(workspace, "NHD", use_tensor_cores=True)

    wrapper.begin_forward(
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        1,  # page_size=1
        data_type=torch.uint8,
        q_data_type=torch.bfloat16,
    )

    o = wrapper.run(
        q,
        (k_buf, v_buf),
        k_scale=k_scale,   # plain float or None
        v_scale=v_scale,   # plain float or None
        kv_cache_sf=(k_sf_buf, v_sf_buf),
    )

    nan_count = torch.isnan(o).sum().item()
    print(f"  Output shape: {o.shape}, NaN count: {nan_count}/{o.numel()}")
    if nan_count > 0:
        nan_mask = torch.isnan(o)
        print(f"  NaN positions (first 10): {nan_mask.nonzero()[:10].cpu().tolist()}")
    return nan_count == 0


def test_synthetic():
    """Run synthetic tests covering various (batch_size, seq_len) combinations."""
    if not _flashinfer_available:
        print("flashinfer not available, skipping")
        return

    print("\n=== Synthetic NVFP4 decode tests (page_size=1) ===")
    print(f"Device: {torch.cuda.get_device_name(0)}")

    configs = [
        # (batch_size, seq_lens, nqh, nkh, head_dim)
        # --- bs=1 boundary scan ---
        (1,  [1],       8, 2, 128),
        (1,  [7],       8, 2, 128),  # matches early server dump
        (1,  [64],      8, 2, 128),
        (1,  [128],     8, 2, 128),
        (1,  [256],     8, 2, 128),
        (1,  [512],     8, 2, 128),
        (1,  [513],     8, 2, 128),
        (1,  [768],     8, 2, 128),
        (1,  [1024],    8, 2, 128),
        # --- bs=2 boundary ---
        (2,  [1, 1],    8, 2, 128),
        (2,  [7, 7],    8, 2, 128),
        (2,  [64, 64],  8, 2, 128),
        (2,  [128, 128],8, 2, 128),
        # --- larger batches ---
        (4,  [128]*4,   8, 2, 128),
        (4,  [256]*4,   8, 2, 128),
        (8,  [64]*8,    8, 2, 128),
        (8,  [128]*8,   8, 2, 128),
        (16, [64]*16,   8, 2, 128),
        (16, [128]*16,  8, 2, 128),
        (32, [64]*32,   8, 2, 128),
        (32, [128]*32,  8, 2, 128),
        # --- Qwen3-8B-like config: 32 qheads, 8 kheads, head_dim=128 ---
        (1,  [7],       32, 8, 128),
        (1,  [128],     32, 8, 128),
        (4,  [128]*4,   32, 8, 128),
        (8,  [64]*8,    32, 8, 128),
        # --- GQA group=1 ---
        (4,  [128]*4,   8, 8, 128),
        (8,  [128]*8,   8, 8, 128),
        # --- Mixed seq lens ---
        (4,  [1, 10, 100, 500],   8, 2, 128),
        (4,  [1, 50, 200, 1000],  8, 2, 128),
        # --- head_dim=64 ---
        (4,  [128]*4,   8, 2, 64),
        (8,  [128]*8,   8, 2, 64),
    ]

    passed = 0
    failed = 0
    for (bs, seq_lens, nqh, nkh, hd) in configs:
        ok = run_batch_decode_nvfp4(bs, seq_lens, nqh, nkh, hd)
        if ok:
            passed += 1
        else:
            failed += 1

    print(f"\nResults: {passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump", type=str, default=None, help="Path to NaN dump file")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available")
        sys.exit(1)

    if not _flashinfer_available:
        print("flashinfer not available")
        sys.exit(1)

    if args.dump:
        ok = replay_dump(args.dump)
        sys.exit(0 if ok else 1)
    else:
        ok = test_synthetic()
        sys.exit(0 if ok else 1)
