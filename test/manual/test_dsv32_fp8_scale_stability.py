"""AC-10 M3-B FP8 block scale-factor stability fixture (production path).

Plan §AC-10 / §303 requires that, before the DS launcher drops
``--disable-radix-cache``, the operator proves the FP8 quantization +
KV-cache store kernel that DSA uses for the index-K cache produces
bit-equal per-token scale bytes for the same K row whether it is
written as a singleton (alone in its page) or as part of a
fully-populated page with deterministic neighbors.

The relevant production path is ``DSAIndexer._store_index_k_cache``,
which prefers ``fused_store_index_k_cache(key, buf, out_cache_loc,
page_size)`` (the JIT kernel from
``sglang.jit_kernel.fused_store_index_cache``). The buffer layout
per page is:

  ``buf[page_idx, 0 : page_size*128]`` — FP8 K bytes (uint8),
  ``buf[page_idx, page_size*128 : page_size*132]`` — fp32 per-token
    scales (4 bytes per token).

A passing run produces, for the SAME K0 row written into two
distinct pages (singleton vs packed), bit-equal FP8 bytes AND
bit-equal scale bytes for K0's slot. If either diverges, the DS
label-write hook will see different dequantized K-noPE under
radix-cache reuse and the AC-10 guard MUST stay in place.

This fixture is hardware-gated:

* Skipped unless ``SGLANG_DS_FP8_SCALE_PROOF=1`` AND CUDA is
  available AND ``fused_store_index_k_cache`` JIT-compiles for the
  target (CPU-only runs cannot be misread as passes).
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


def _opt_in_and_cuda() -> bool:
    if _env("SGLANG_DS_FP8_SCALE_PROOF") != "1":
        return False
    try:
        import torch
    except ImportError:
        return False
    return torch.cuda.is_available()


@unittest.skipUnless(
    _opt_in_and_cuda(),
    "SGLANG_DS_FP8_SCALE_PROOF=1 + CUDA required. Deliberate opt-in "
    "so a CPU-only run cannot be misread as a passing M3-B check.",
)
class TestDSv32FP8IndexCacheScaleStability(unittest.TestCase):
    """Singleton vs packed-page FP8 scale equality via the production
    ``fused_store_index_k_cache`` kernel."""

    def test_singleton_vs_packed_page_k0_bytes_match(self):
        import torch

        # Lazy import so the JIT module is not loaded on CPU.
        try:
            from sglang.jit_kernel.fused_store_index_cache import (
                can_use_dsa_fused_store,
                fused_store_index_k_cache,
            )
        except Exception as exc:
            self.skipTest(
                "sglang.jit_kernel.fused_store_index_cache import "
                f"failed (exc={exc!r}); the proof needs the fused "
                "store JIT kernel."
            )
            return

        device = torch.device("cuda")
        page_size = 64
        index_head_dim = 128  # DSv3.2 DSA index head dim
        page_bytes = page_size * (index_head_dim + 4)
        key_dtype = torch.bfloat16
        loc_dtype = torch.int64

        if not can_use_dsa_fused_store(key_dtype, loc_dtype, page_size):
            self.skipTest(
                "fused_store_index_k_cache cannot JIT-compile for "
                f"(key={key_dtype}, loc={loc_dtype}, page_size={page_size}) "
                "on this hardware. The proof would have to use the "
                "fallback act_quant path — not implemented here, file "
                "a follow-up fixture."
            )
            return

        torch.manual_seed(0)
        # K0 is the row we will compare across the two writes.
        K0 = torch.randn(1, index_head_dim, dtype=key_dtype, device=device)
        # 63 deterministic neighbors for the packed write.
        neighbours = torch.randn(
            page_size - 1, index_head_dim,
            dtype=key_dtype, device=device,
        )

        # Two distinct index_k_with_scale_buffer-shaped tensors: one
        # for the singleton write, one for the packed write. Layout
        # mirrors DeepSeekV4SingleKVPool.index_k_with_scale_buffer.
        num_pages = 2  # one page for singleton, one for packed
        buf_singleton = torch.zeros(
            num_pages, page_bytes, dtype=torch.uint8, device=device,
        )
        buf_packed = torch.zeros(
            num_pages, page_bytes, dtype=torch.uint8, device=device,
        )

        # Singleton: write K0 alone at out_cache_loc=0 (= page 0,
        # slot 0). Page 0 remains otherwise empty.
        singleton_loc = torch.tensor([0], dtype=loc_dtype, device=device)
        fused_store_index_k_cache(
            K0, buf_singleton, singleton_loc, page_size,
        )

        # Packed: write the full block [K0, K1, ..., K63] starting at
        # out_cache_loc=0 (= page 0, slot 0). K0 lands at slot 0 of a
        # fully populated page.
        packed_key = torch.cat([K0, neighbours], dim=0).contiguous()
        packed_loc = torch.arange(
            page_size, dtype=loc_dtype, device=device,
        )
        fused_store_index_k_cache(
            packed_key, buf_packed, packed_loc, page_size,
        )

        # K0 lives at byte offset 0..128 of page 0 in both buffers.
        # Its scale lives at byte offset page_size*128 .. +4.
        k0_fp8_singleton = buf_singleton[0, :index_head_dim].cpu()
        k0_fp8_packed = buf_packed[0, :index_head_dim].cpu()
        scale_off = page_size * index_head_dim
        k0_scale_singleton = buf_singleton[
            0, scale_off : scale_off + 4
        ].cpu()
        k0_scale_packed = buf_packed[0, scale_off : scale_off + 4].cpu()

        fp8_equal = bool(torch.equal(k0_fp8_singleton, k0_fp8_packed))
        scale_equal = bool(torch.equal(k0_scale_singleton, k0_scale_packed))

        payload = {
            "fixture_kind": "fp8_index_cache_scale_stability",
            "path_used": "fused_store_index_k_cache",
            "page_size": page_size,
            "index_head_dim": index_head_dim,
            "singleton_loc": singleton_loc.tolist(),
            "packed_loc": packed_loc.tolist(),
            "k0_fp8_singleton_hex": k0_fp8_singleton.numpy().tobytes().hex(),
            "k0_fp8_packed_hex": k0_fp8_packed.numpy().tobytes().hex(),
            "k0_scale_singleton_hex": (
                k0_scale_singleton.numpy().tobytes().hex()
            ),
            "k0_scale_packed_hex": k0_scale_packed.numpy().tobytes().hex(),
            "fp8_bytes_equal": fp8_equal,
            "scale_bytes_equal": scale_equal,
            "verdict": "PASS" if (fp8_equal and scale_equal) else "FAIL",
            "notes": (
                "If scale bytes differ: per-token scale depends on "
                "neighbour tokens within the page. Radix-cache reuse "
                "would see different K-noPE in cold vs warm paths; "
                "KEEP --disable-radix-cache. If scale bytes match but "
                "FP8 bytes differ: kernel is non-deterministic — same "
                "remediation."
            ),
        }
        _record_artifact(payload, suffix="singleton_vs_packed_page")

        self.assertTrue(
            scale_equal,
            "FP8 per-token scale for K0 differs between singleton "
            f"({k0_scale_singleton.numpy().tobytes().hex()}) and "
            f"packed ({k0_scale_packed.numpy().tobytes().hex()}) "
            "page writes. The DS label-write hook would see "
            "different dequantized K-noPE in cold vs warm paths. "
            f"KEEP --disable-radix-cache. Artifact: {RESULTS_DIR}/.",
        )
        self.assertTrue(
            fp8_equal,
            "FP8 scale matched but FP8 byte for K0 differs between "
            "singleton and packed page writes — quantization kernel "
            "is non-deterministic. KEEP --disable-radix-cache.",
        )


if __name__ == "__main__":
    unittest.main()
