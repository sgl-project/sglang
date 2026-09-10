"""TuneArgs — house-style dataclass with add_cli_args / from_cli_args.

Mirrors the SGLang benchmark modules: reuse ServerArgs for engine flags (model-path,
tp-size, dtype, kv-cache-dtype) when running inside a live tree; put only tool-specific
flags here. In --mock mode the profile is supplied directly so no engine is needed.
"""

from __future__ import annotations

import dataclasses
from typing import Optional


@dataclasses.dataclass
class TuneArgs:
    # committed corpus output dir (defaults to the packaged configs/ next to the attention layer)
    config_dir: str = "python/sglang/srt/layers/attention/configs"
    local_cache_dir: Optional[str] = None
    phases: str = "decode,prefill"
    isolate: bool = True  # subprocess-per-candidate
    timeout_s: float = 120.0

    # --mock: run GPU-free with a synthetic latency model
    mock: bool = False
    mock_device: Optional[str] = None
    mock_sm: int = 90

    # profile (used directly in --mock; derived from ModelConfig in the real path)
    qo_heads: int = 32
    kv_heads: int = 8
    head_dim: int = 128
    dtype: str = "bfloat16"
    kv_cache_dtype: str = "auto"
    mla: bool = False
    tp_size: int = 1
    ep_size: int = 1
    dp_size: int = 1

    @staticmethod
    def add_cli_args(parser) -> None:
        p = parser
        p.add_argument("--config-dir", type=str, default=TuneArgs.config_dir)
        p.add_argument("--local-cache-dir", type=str, default=None)
        p.add_argument("--phases", type=str, default="decode,prefill")
        p.add_argument("--no-isolate", dest="isolate", action="store_false")
        p.add_argument("--timeout-s", type=float, default=120.0)
        p.add_argument(
            "--mock", action="store_true", help="run GPU-free with a synthetic model"
        )
        p.add_argument("--mock-device", type=str, default=None)
        p.add_argument("--mock-sm", type=int, default=90)
        p.add_argument("--qo-heads", type=int, default=32)
        p.add_argument("--kv-heads", type=int, default=8)
        p.add_argument("--head-dim", type=int, default=128)
        p.add_argument("--dtype", type=str, default="bfloat16")
        p.add_argument("--kv-cache-dtype", type=str, default="auto")
        p.add_argument("--mla", action="store_true")
        p.add_argument("--tp-size", type=int, default=1)
        p.add_argument("--ep-size", type=int, default=1)
        p.add_argument("--dp-size", type=int, default=1)
        p.set_defaults(isolate=True)

    @classmethod
    def from_cli_args(cls, args) -> TuneArgs:
        fields = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in vars(args).items() if k in fields})
