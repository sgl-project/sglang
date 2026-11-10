# Port of GooseLLM's tilelang chunk_scaled_dot_kkt_fwd (V100/SM70 GDN KKT kernel).
# See _kkt.py for the kernel; __init__ re-exports the public entry point.
from sglang.kernels.ops.attention.fla.tilelang._kkt import chunk_scaled_dot_kkt_fwd

__all__ = ["chunk_scaled_dot_kkt_fwd"]
