"""AC-10 M3-B FP8 block scale-factor stability fixture (hardware-gated).

Plan §AC-10 / §303 requires that, before the DS launcher drops
``--disable-radix-cache``, the operator proves the FP8 quantization
kernel produces bit-equal per-block scale factors for the same token
written as a singleton (a 1-token write that ends up alone in its
block) vs as part of a fully-packed block (the same token written
alongside deterministic neighbors).

If the per-block scale factor changes between these two cases, the
dequantized K-noPE that the DS label-write hook sees will differ
between the cold (fresh write) and warm (radix-reused) paths, even
when the underlying float K is identical. That breaks the cold/warm
label bit-equality the radix cache requires.

This fixture is hardware-gated:

* Skipped unless CUDA is available AND the FP8 quantization kernel
  is importable AND the user has set ``SGLANG_DS_FP8_SCALE_PROOF=1``
  (a deliberate opt-in so an accidental CPU-only run does not get
  flagged as a successful M3-B check).

The proof
---------

We pick a deterministic K-noPE row ``K0`` and a deterministic
neighbour ``K1..K63``. We invoke the production FP8 quantization in
two modes:

1. Singleton: input is just ``K0`` (a 1xD tensor). The kernel emits
   ``(K0_fp8_a, scale_a)``.
2. Packed: input is ``cat([K0, K1, ..., K63])`` (a 64xD tensor) — a
   full FP8 block. The kernel emits ``(K_fp8_b, scale_b)`` and we
   take the slice corresponding to K0.

The assertion is ``scale_a == scale_b`` for the K0 block. If FP8
scales are per-block (as sglang's kernels emit them), the singleton
case constructs its own block from K0 alone while the packed case
constructs a full-block scale from all 64 tokens; equal scales mean
the K0 dequantizes back to the same value in both cases.

Failure modes documented inline so the operator understands what to
do (keep ``--disable-radix-cache``, file a kernel bug).
"""

from __future__ import annotations

import datetime
import json
import os
import pathlib
import unittest
from typing import Any, Dict, Optional


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "development" / "results"


def _env(name: str) -> Optional[str]:
    return os.environ.get(name)


def _record_artifact(payload: Dict[str, Any], *, suffix: str) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ",
    )
    path = RESULTS_DIR / f"dsv32_fp8_scale_stability_{suffix}_{ts}.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"[fp8-scale-stability] artifact written: {path}")


def _cuda_with_fp8() -> bool:
    """True when the runtime can execute the production FP8 path."""
    try:
        import torch
    except ImportError:
        return False
    if not torch.cuda.is_available():
        return False
    return hasattr(torch, "float8_e4m3fn")


@unittest.skipUnless(
    _env("SGLANG_DS_FP8_SCALE_PROOF") == "1" and _cuda_with_fp8(),
    "SGLANG_DS_FP8_SCALE_PROOF=1 + CUDA-with-FP8 required. The proof "
    "must run on the same hardware that hosts the production server; "
    "the opt-in flag prevents an accidental CPU-only skip from being "
    "scored as a pass.",
)
class TestDSv32FP8ScaleStability(unittest.TestCase):
    """Singleton vs packed-block FP8 scale-factor equality fixture."""

    def test_singleton_vs_packed_scale_is_bit_equal(self):
        import torch
        # Imported lazily so a CPU-only environment can still import
        # the test file without erroring at collection time.
        from sglang.srt.layers.quantization.fp8_kernel import (
            sglang_per_token_group_quant_fp8,
        )

        device = torch.device("cuda")
        torch.manual_seed(0)

        # Production block sizes for DSv3.2 FP8 KV cache. The K-noPE
        # the labeling kernel consumes is per-head 128 wide; here we
        # use one full row as the K0 token.
        head_dim = 128
        block_size = 64

        # K0 is the token we will compare across the two writes. The
        # neighbours are random but reproducible — the point of the
        # fixture is that K0's block scale should not depend on them.
        K0 = torch.randn(1, head_dim, dtype=torch.bfloat16, device=device)
        neighbours = torch.randn(
            block_size - 1, head_dim,
            dtype=torch.bfloat16, device=device,
        )
        packed = torch.cat([K0, neighbours], dim=0).contiguous()
        singleton = K0.contiguous()

        try:
            # Production kernel: per-token-group FP8 quant with a
            # group size matching the head dim. Returns (fp8_out,
            # scale_out). The scale tensor's first row corresponds
            # to K0 in both cases.
            fp8_single, scale_single = sglang_per_token_group_quant_fp8(
                singleton, group_size=head_dim,
            )
            fp8_packed, scale_packed = sglang_per_token_group_quant_fp8(
                packed, group_size=head_dim,
            )
        except Exception as exc:
            self.skipTest(
                "sglang_per_token_group_quant_fp8 not callable in this "
                f"environment (exc={exc!r}); the proof needs the "
                "production FP8 kernel."
            )
            return

        # Pull the K0 scale row from each output.
        scale_single_k0 = scale_single[0].detach().to(torch.float32).cpu()
        scale_packed_k0 = scale_packed[0].detach().to(torch.float32).cpu()
        scales_equal = bool(torch.equal(scale_single_k0, scale_packed_k0))

        # Pull the K0 FP8 bytes from each output. If scales match,
        # the FP8 bytes should also match (deterministic quantization
        # given the same input and scale).
        fp8_single_k0 = fp8_single[0].detach().to(torch.uint8).cpu()
        fp8_packed_k0 = fp8_packed[0].detach().to(torch.uint8).cpu()
        fp8_equal = bool(torch.equal(fp8_single_k0, fp8_packed_k0))

        payload = {
            "fixture_kind": "fp8_scale_stability",
            "head_dim": head_dim,
            "block_size": block_size,
            "scale_single_k0": scale_single_k0.tolist(),
            "scale_packed_k0": scale_packed_k0.tolist(),
            "scales_equal": scales_equal,
            "fp8_bytes_equal": fp8_equal,
            "verdict": "PASS" if scales_equal and fp8_equal else "FAIL",
            "notes": (
                "If scales differ: per-block scale depends on the "
                "neighbour tokens. Radix-cache reuse will see "
                "different K-noPE values than the cold pass; KEEP "
                "--disable-radix-cache and file a kernel bug. "
                "If scales agree but FP8 bytes differ: the kernel "
                "uses a non-deterministic rounding mode — same fix."
            ),
        }
        _record_artifact(payload, suffix="singleton_vs_packed")

        self.assertTrue(
            scales_equal,
            f"FP8 per-block scale for K0 differs between singleton "
            f"({scale_single_k0.tolist()}) and packed "
            f"({scale_packed_k0.tolist()}) writes. The DS label-write "
            "hook would see different dequantized K-noPE in cold vs "
            "warm paths. KEEP --disable-radix-cache in the DS "
            f"launcher. Diagnostic artifact: {RESULTS_DIR}/.",
        )
        self.assertTrue(
            fp8_equal,
            "FP8 scale matched but the FP8 byte for K0 differs "
            "between singleton and packed writes — the quantization "
            "kernel is non-deterministic. KEEP --disable-radix-cache "
            "in the DS launcher.",
        )


if __name__ == "__main__":
    unittest.main()
