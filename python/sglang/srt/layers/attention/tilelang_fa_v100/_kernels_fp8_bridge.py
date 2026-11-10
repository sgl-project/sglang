"""TileLang FP8 paged-KV gather/dequantization for SM70.

Volta cannot consume FP8 in tensor cores. Long prefill therefore resolves each
logical page once into a reusable FP16 workspace, amortizing the linear copy
over thousands of query rows. E4M3 uses a 256-entry FP16 lookup. E5M2 shares
FP16's sign/exponent placement, so widening each byte by eight zero bits avoids
the dependent lookup and is exact for every bit pattern.
"""

from functools import lru_cache

import tilelang
import tilelang.language as T

from ._kernels_paged import pass_configs


@tilelang.jit(pass_configs=pass_configs)
def _fp8_paged_gather_kernel(
    batch: int,
    heads_kv: int,
    dim: int,
    page_size: int,
    num_pages: int,
    max_blocks: int,
    e5m2: bool,
    threads: int = 128,
):
    @T.prim_func
    def main(
        KCache: T.Tensor([num_pages, page_size, heads_kv, dim], T.uint8),
        VCache: T.Tensor([num_pages, page_size, heads_kv, dim], T.uint8),
        Lut: T.Tensor([256], T.float16),
        PageTable: T.Tensor([batch, max_blocks], T.int32),
        SeqLens: T.Tensor([batch], T.int32),
        KOutput: T.Tensor([batch * max_blocks, page_size, heads_kv, dim], T.float16),
        VOutput: T.Tensor([batch * max_blocks, page_size, heads_kv, dim], T.float16),
    ):
        with T.Kernel(max_blocks, heads_kv, batch, threads=threads) as (
            logical_page,
            head,
            sequence,
        ):
            active_pages = T.ceildiv(SeqLens[sequence], page_size)
            if logical_page < active_pages:
                physical_page = PageTable[sequence, logical_page]
                output_page = sequence * max_blocks + logical_page
                for offset, d in T.Parallel(page_size, dim):
                    k_raw = KCache[physical_page, offset, head, d]
                    v_raw = VCache[physical_page, offset, head, d]
                    if e5m2:
                        k_bits = T.Cast("uint16", k_raw) << T.uint16(8)
                        v_bits = T.Cast("uint16", v_raw) << T.uint16(8)
                        KOutput[output_page, offset, head, d] = T.reinterpret(
                            k_bits, T.float16
                        )
                        VOutput[output_page, offset, head, d] = T.reinterpret(
                            v_bits, T.float16
                        )
                    else:
                        KOutput[output_page, offset, head, d] = Lut[
                            T.cast(k_raw, T.int32)
                        ]
                        VOutput[output_page, offset, head, d] = Lut[
                            T.cast(v_raw, T.int32)
                        ]

    return main


@lru_cache(maxsize=None)
def get_fp8_paged_gather_kernel(
    batch: int,
    heads_kv: int,
    dim: int,
    page_size: int,
    num_pages: int,
    max_blocks: int,
    e5m2: bool,
    threads: int = 128,
):
    return _fp8_paged_gather_kernel(
        batch,
        heads_kv,
        dim,
        page_size,
        num_pages,
        max_blocks,
        e5m2,
        threads,
    )
