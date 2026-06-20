"""Native MLX model implementations for sglang MLX backend.

Each module under this package provides a `Model` class compatible with
sglang's MLX hardware backend.  These are independent of `mlx_lm.models.*`
so the sglang codebase can own the full model definition, weight
sanitization, and quantization predicate.
"""
