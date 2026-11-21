"""Shape-only Humming kernel choice for expert-parallel NVFP4 W4A16 experts.

Humming picks a GEMM config, and with it the block M the routes are aligned to,
from ``valid_shape_m``, which it reads as the number of routes the GEMM serves.
CUDA graphs fix the choice at capture, so it can depend on shapes only. Plain
Python with no imports, so the rule can be checked without a GPU.
"""

from __future__ import annotations

# A Humming tuning table: [min_m, max_m, config] rows; a row serves
# min_m < valid_shape_m <= max_m.
TuningTable = list[list]

# Measured with humming-kernels 0.1.12 and its stream-K tables, not re-measured
# on 0.1.18 (H200, Muse Spark EP8: top-8, 32 of 256 experts): sizing tiles past
# 64 routes per local expert ran 0.81x of Marlin at 768 uniform tokens, sizing
# for only the expected share 0.48-0.85x at 32-256 skewed tokens.
MAX_ASSUMED_ROUTES_PER_EXPERT = 64


def _tile(block_shape: list, warp_shape: list, num_stages: int) -> dict:
    return {
        "mma_type": "wgmma",
        "block_shape": block_shape,
        "warp_shape": warp_shape,
        "use_stream_k": False,
        "use_f16_accum": False,
        "num_stages": num_stages,
    }


def _warp_specialized_tile(block_shape: list, warp_shape: list) -> dict:
    # Humming 0.1.18's warp-specialized TMA tile, keyed as its SM90 heuristic
    # emits it for these GEMMs.
    return {
        "mma_type": "wgmma",
        "block_shape": block_shape,
        "warp_shape": warp_shape,
        "num_stages": 4,
        "num_ctas_per_sm": 1,
        "use_warp_spec": True,
        "use_tma": True,
        "use_stream_k": False,
        "smem_reuse_mode": "none",
        "producer_stage_unroll": 1,
        "wgmma_split_issue_wait": True,
        "wgmma_use_late_as": False,
        "raster_group_m": 1,
        "use_tma_a": False,
        "use_tma_c": False,
        "use_tma_as": False,
        "use_tma_as2": False,
    }


# Humming 0.1.18's own SM90 pick for these GEMMs up to 200 routes.
_MMA_SMALL_M_TILE = {
    "mma_type": "mma",
    "block_shape": [8, 128, 64],
    "warp_shape": [8, 64, 32],
    "use_stream_k": False,
    "num_stages": 4,
    "num_ctas_per_sm": 4,
    "use_warp_spec": False,
    "use_tma": False,
    "use_tma_a": False,
    "use_tma_c": False,
    "use_tma_as": False,
    "use_tma_as2": False,
}

# Timed on H200 with humming-kernels 0.1.18 for Muse Spark's expert GEMMs (w13
# 8192 x 4096, w2 4096 x 4096, 32 local experts), each GEMM alone without
# stream-K, at decode graph sizes of 8 to 240 tokens under uniform, normal and
# skewed routing. A row takes the pair sharing one block M with the lowest w13 +
# w2 time (geometric mean, normal routing counted twice) when it beats the
# previous pair by 2% and slows no routing by more than 2%; neighbouring rows
# share one pair when it stays within 1% of each row's best. Each row serves
# from the previous measured valid_shape_m (exclusive) to its own (inclusive).
# Rows: [min_m, max_m, w13 config, w2 config].
_H200_DECODE_TILES = [
    [0, 64, _MMA_SMALL_M_TILE, _tile([8, 128, 128], [8, 32, 64], 6)],
    [
        64,
        128,
        _tile([16, 128, 256], [16, 32, 64], 4),
        _tile([16, 128, 256], [16, 32, 64], 4),
    ],
    [
        128,
        192,
        _tile([24, 128, 256], [24, 32, 64], 4),
        _tile([24, 128, 256], [24, 32, 64], 4),
    ],
    [
        192,
        256,
        _tile([32, 256, 64], [32, 32, 64], 6),
        _tile([32, 128, 128], [32, 32, 64], 6),
    ],
    [
        256,
        384,
        _tile([24, 256, 128], [24, 64, 64], 3),
        _tile([24, 256, 128], [24, 64, 64], 3),
    ],
    [
        384,
        512,
        _tile([32, 256, 128], [32, 64, 64], 3),
        _tile([32, 256, 64], [32, 32, 64], 6),
    ],
    [
        512,
        640,
        _warp_specialized_tile([32, 512, 64], [32, 32, 64]),
        _tile([32, 256, 128], [32, 64, 64], 3),
    ],
    [
        640,
        768,
        _tile([48, 256, 128], [48, 64, 64], 4),
        _tile([48, 128, 256], [48, 32, 64], 3),
    ],
    [
        768,
        1024,
        _warp_specialized_tile([40, 512, 64], [40, 32, 64]),
        _tile([40, 256, 128], [40, 64, 64], 3),
    ],
    [
        1024,
        1280,
        _tile([40, 256, 128], [40, 64, 64], 3),
        _tile([40, 256, 128], [40, 64, 64], 3),
    ],
    [
        1280,
        1408,
        _warp_specialized_tile([56, 512, 64], [56, 32, 64]),
        _warp_specialized_tile([56, 512, 64], [56, 32, 64]),
    ],
    [
        1408,
        1536,
        _tile([32, 256, 128], [32, 64, 64], 3),
        _tile([32, 256, 128], [32, 64, 64], 3),
    ],
    [
        1536,
        1664,
        _warp_specialized_tile([64, 512, 64], [64, 32, 64]),
        _tile([64, 256, 128], [64, 64, 64], 3),
    ],
    [
        1664,
        1792,
        _tile([48, 256, 128], [48, 32, 64], 3),
        _tile([48, 256, 128], [48, 64, 64], 3),
    ],
    [
        1792,
        1920,
        _tile([64, 256, 128], [64, 64, 64], 3),
        _tile([64, 256, 128], [64, 64, 64], 3),
    ],
]


def _splice_decode_tiles(table: TuningTable, sublayer: int) -> TuningTable:
    # Humming's own rows serve above the tiles: at EP8, valid_shape_m 2048 is
    # every step of 241 to 2048 tokens, and the search timed it at 256 only.
    hi = _H200_DECODE_TILES[-1][1]
    rows = [[row[0], row[1], dict(row[2 + sublayer])] for row in _H200_DECODE_TILES]
    return rows + [
        [max(lo, hi), max_m, config] for lo, max_m, config in table if max_m > hi
    ]


def ep_valid_shape_m(
    *, num_tokens: int, top_k: int, num_local_experts: int, num_global_experts: int
) -> int:
    """The route count one EP rank's Humming expert GEMMs are picked for.

    The slowest rank sets the step, so assume the busiest case, every route of
    the batch landing on this rank, but at most MAX_ASSUMED_ROUTES_PER_EXPERT
    per local expert unless the expected share is larger: past that, padding
    block M on lightly hit experts costs more than the weight re-reads it
    saves on hot ones. Without EP this is num_tokens * top_k, SGLang's rule.
    """
    expected = num_tokens * top_k * num_local_experts // num_global_experts
    busiest = num_tokens * min(top_k, num_local_experts)
    cap = MAX_ASSUMED_ROUTES_PER_EXPERT * num_local_experts
    return max(1, expected, min(busiest, cap))


def config_for(table: TuningTable, valid_shape_m: int) -> dict:
    for min_m, max_m, config in table:
        if min_m < valid_shape_m <= max_m:
            return config
    raise ValueError(f"No Humming config covers valid_shape_m={valid_shape_m}")


def block_m_for(table: TuningTable, valid_shape_m: int) -> int:
    return config_for(table, valid_shape_m)["block_shape"][0]


def ep_tuning_tables(
    w13: TuningTable, w2: TuningTable, *, h200_class: bool
) -> tuple[TuningTable, TuningTable]:
    """Humming's heuristic tables without stream-K, with the H200 decode tiles.

    The decode tiles need N % 512 == 0 and K % 256 == 0, which Muse Spark's
    expert sizes (multiples of 4096) meet.
    """
    # Stream-K sums a tile's partial K slices in BF16 in the output buffer
    # (atomically past three slices), where Marlin reduces split-K in FP32.
    # Without it each output is one FP32 accumulation over full K, rounded
    # once to BF16, and the same on every replay.
    w13, w2 = (
        [
            [min_m, max_m, {**config, "use_stream_k": False}]
            for min_m, max_m, config in table
        ]
        for table in (w13, w2)
    )
    if h200_class:
        w13, w2 = _splice_decode_tiles(w13, 0), _splice_decode_tiles(w2, 1)
    return w13, w2
