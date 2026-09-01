"""Validate the PoC kernel's packed byte-addressing + ue8m0 dequant math
against the production dequant_k_cache_paged_ref formula, using numpy only
(no torch/triton). This isolates the highest-risk part of the Stage-3 kernel:
the compressed-slot address arithmetic.
"""
import numpy as np

DIM_NOPE = 448
DIM_ROPE = 64
TILE_SIZE = 64
NUM_SCALE_TILES = DIM_NOPE // TILE_SIZE      # 7
NOPE_ROPE_BYTES = DIM_NOPE + DIM_ROPE * 2    # 576
PADDED_SCALE_PER_TOKEN = NUM_SCALE_TILES + 1 # 8
D = DIM_NOPE + DIM_ROPE                       # 512


def e4m3fnuz_decode(byte):
    """Decode a uint8 as OCP fp8 e4m3fnuz -> float (enough for parity test).
    fnuz: 1-4-3, bias 8, no inf, sign+all-exp+all-mant==0x80 is NaN."""
    b = int(byte)
    s = (b >> 7) & 1
    e = (b >> 3) & 0xF
    m = b & 0x7
    if b == 0x80:
        return np.nan
    if e == 0:
        val = (m / 8.0) * 2.0 ** (1 - 8)      # subnormal, bias 8
    else:
        val = (1.0 + m / 8.0) * 2.0 ** (e - 8)
    return -val if s else val


def build_random_packed(C, page_size, rng):
    raw = page_size * (NOPE_ROPE_BYTES + PADDED_SCALE_PER_TOKEN)
    bytes_per_page = ((raw + NOPE_ROPE_BYTES - 1) // NOPE_ROPE_BYTES) * NOPE_ROPE_BYTES
    num_pages = (C + page_size - 1) // page_size + 1
    # random bytes; keep scale exponents in a sane range so dequant is finite
    buf = rng.integers(0, 256, size=(num_pages, bytes_per_page), dtype=np.uint8)
    return buf, bytes_per_page


def prod_ref(buf, locs, page_size, bytes_per_page):
    """Vectorized numpy port of dequantize_k_cache_paged_ref."""
    flat_u8 = buf.reshape(-1)
    flat_bf16_view_ok = True  # we emulate bf16 via uint16 pairs below
    s_offset_bytes = page_size * NOPE_ROPE_BYTES
    out = np.zeros((len(locs), D), dtype=np.float32)
    for i, loc in enumerate(locs):
        page_idx = loc // page_size
        in_page = loc % page_size
        page_byte_base = page_idx * bytes_per_page
        tdb = page_byte_base + in_page * NOPE_ROPE_BYTES
        tsb = page_byte_base + s_offset_bytes + in_page * PADDED_SCALE_PER_TOKEN
        # nope
        for t in range(NUM_SCALE_TILES):
            su8 = int(flat_u8[tsb + t])
            scale = 2.0 ** (su8 - 127)
            if scale < 2.0 ** -126:
                scale = 0.0
            for k in range(TILE_SIZE):
                d = t * TILE_SIZE + k
                out[i, d] = e4m3fnuz_decode(flat_u8[tdb + d]) * scale
        # rope: bf16 at byte (tdb+448)+2*(d-448)
        for r in range(DIM_ROPE):
            byte = tdb + DIM_NOPE + 2 * r
            lo = int(flat_u8[byte]); hi = int(flat_u8[byte + 1])
            u16 = (hi << 8) | lo
            out[i, DIM_NOPE + r] = bf16_to_f32(u16)
    return out


def kernel_addr(buf, locs, page_size, bytes_per_page):
    """Reproduce EXACTLY the address expressions in
    paged_decode_split_src._paged_decode_fused_split_src_kernel for the
    compressed branch, then decode bytes the same way, to prove the kernel's
    offsets equal the production reference's offsets."""
    flat_u8 = buf.reshape(-1)
    S_OFFSET_BYTES = page_size * NOPE_ROPE_BYTES
    out = np.zeros((len(locs), D), dtype=np.float32)
    d_offs = np.arange(D)
    nope_mask = d_offs < DIM_NOPE
    rope_mask = (d_offs >= DIM_NOPE) & (d_offs < DIM_NOPE + DIM_ROPE)
    g_idx_per_d = d_offs // TILE_SIZE
    for i, loc in enumerate(locs):
        page_idx = loc // page_size
        in_page = loc % page_size
        page_byte_base = page_idx * bytes_per_page
        token_data_base = page_byte_base + in_page * NOPE_ROPE_BYTES
        token_scale_base = page_byte_base + S_OFFSET_BYTES + in_page * PADDED_SCALE_PER_TOKEN
        # nope: fp8_off = token_data_base + d ; scale_off = token_scale_base + d//64
        fp8_off = token_data_base + d_offs
        scale_off = token_scale_base + g_idx_per_d
        for d in range(D):
            if nope_mask[d]:
                su8 = int(flat_u8[scale_off[d]])
                scale = 2.0 ** (su8 - 127)
                if scale < 2.0 ** -126:
                    scale = 0.0
                out[i, d] = e4m3fnuz_decode(flat_u8[fp8_off[d]]) * scale
        # rope: bf16 elem off = (token_data_base + DIM_NOPE)//2 + (d-DIM_NOPE)
        rope_base = (token_data_base + DIM_NOPE) // 2
        for d in range(D):
            if rope_mask[d]:
                elem = rope_base + (d - DIM_NOPE)
                byte = elem * 2
                lo = int(flat_u8[byte]); hi = int(flat_u8[byte + 1])
                u16 = (hi << 8) | lo
                out[i, d] = bf16_to_f32(u16)
    return out


def bf16_to_f32(u16):
    return np.frombuffer(np.uint32(u16 << 16).tobytes(), dtype=np.float32)[0]


def main():
    rng = np.random.default_rng(0)
    all_ok = True
    for page_size in (1, 64, 128):
        C = 300
        buf, bpp = build_random_packed(C, page_size, rng)
        locs = rng.integers(0, C, size=37).tolist()
        ref = prod_ref(buf, locs, page_size, bpp)
        ker = kernel_addr(buf, locs, page_size, bpp)
        # compare, treating NaN==NaN
        eq = np.array_equal(np.nan_to_num(ref, nan=1e30), np.nan_to_num(ker, nan=1e30))
        print(f"page_size={page_size:3d} bpp={bpp:5d} C={C} -> "
              f"{'OK' if eq else 'FAIL'} (max_abs_diff="
              f"{np.nanmax(np.abs(ref-ker)):.3e})")
        all_ok &= eq
    print("ADDRESS MATH:", "ALL OK" if all_ok else "SOME FAILED")


if __name__ == "__main__":
    main()
