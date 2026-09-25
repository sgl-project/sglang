"""gfx950 fused DSA indexer decode path: four raw-HIP kernels replacing the
12-launch ROCm chain.  Enabled wherever supported; the gate is
``utils.gfx950_fused_indexer_runtime_ok``."""

from sglang.kernels.ops.attention.dsa.hip_gfx950.fused_decode import (  # noqa: F401
    CACHE_TOK_STRIDE,
    MAX_ROWS,
    PAGE_SIZE,
    Gfx950FusedIndexer,
    model_shape_supported,
    prealloc_workspace,
    supported_hardware,
)
