"""MiniMax-M3.1-Flash low-batch FFN (1-32 decode rows) in CuTe DSL, adapted
from vLLM's specialized decode. Entry points are in ``ffn``; this package
import stays metadata-only (no cutlass import, no compilation)."""
