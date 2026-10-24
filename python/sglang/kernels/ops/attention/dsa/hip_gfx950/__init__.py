"""gfx950 fused DSA indexer decode path: four raw-HIP kernels replacing the
12-launch ROCm chain.  Enabled wherever supported; the gate is
``utils.gfx950_fused_indexer_runtime_ok``, the runtime wrapper
``srt.layers.attention.dsa.gfx950_fused_indexer``."""
