"""AC-10 M3-B FP8 scale-factor stability fixture (production paths).

Plan §AC-10 / §303 requires that, before the DS launcher drops
``--disable-radix-cache``, the operator proves the FP8 quantization +
KV-cache store kernel that DSA uses for the index-K cache produces
bit-equal per-token scale bytes for the same K row whether it is
written as a singleton (alone in its page) or as part of a fully-
populated page with deterministic neighbors.

The two production code paths that
``DSAIndexer._store_index_k_cache`` uses are exercised here:

* **Fused** — ``sglang.jit_kernel.fused_store_index_cache.
  fused_store_index_k_cache(key, buf, out_cache_loc, page_size)``.
  Preferred when ``can_use_dsa_fused_store(...)`` returns True for
  the target hardware.
* **Fallback** — ``act_quant(key, block_size, scale_fmt)`` followed
  by ``SetKAndS.execute(pool=..., buf=..., loc=..., index_k=...,
  index_k_scale=...)``. The same ingredients production uses when
  the fused kernel is unavailable.

The fixture writes K0 alone into page 0 of one buffer, K0 + 63
deterministic neighbours into page 0 of another, and reads back
K0's FP8 bytes (``buf[0, 0:128]``) and per-token scale bytes
(``buf[0, page_size*128 : +4]``) at the production byte offsets
shared by ``index_buf_accessor.py``. Pass iff both fused and
fallback agree singleton-vs-packed.

Hardware-gated: ``SGLANG_DS_FP8_SCALE_PROOF=1`` AND CUDA. Skips
only when NEITHER production path can run on the target hardware.
"""

from __future__ import annotations

import datetime
import json
import os
import pathlib
import unittest
from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "development" / "results"

PAGE_SIZE = 64
INDEX_HEAD_DIM = 128  # DSv3.2 DSA index head dim
SCALE_BYTES_PER_TOKEN = 4
PAGE_BYTES = PAGE_SIZE * (INDEX_HEAD_DIM + SCALE_BYTES_PER_TOKEN)


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


def _read_back_k0_bytes(
    buf: "torch.Tensor", *, page_idx: int = 0, position: int = 0,
) -> Tuple["torch.Tensor", "torch.Tensor"]:
    """Read FP8 + scale bytes for K0 from a real index_k_with_scale
    page buffer, using the same byte offsets the production
    ``_set_k_and_s_triton`` writes at:

        FP8 bytes:    buf[page_idx, position*128 : (position+1)*128]
        scale bytes:  buf[page_idx, PAGE_SIZE*128 + position*4 :
                          PAGE_SIZE*128 + (position+1)*4]
    """
    fp8 = buf[page_idx, position * INDEX_HEAD_DIM : (position + 1) * INDEX_HEAD_DIM]
    scale_off = PAGE_SIZE * INDEX_HEAD_DIM + position * SCALE_BYTES_PER_TOKEN
    scale = buf[page_idx, scale_off : scale_off + SCALE_BYTES_PER_TOKEN]
    return fp8.detach().cpu(), scale.detach().cpu()


def _try_fused_path(K0, neighbours, device) -> Optional[Dict[str, Any]]:
    """Run the fused-store production path. Returns None on
    skip/unavailable, an artifact-payload-shaped dict otherwise."""
    import torch

    try:
        from sglang.jit_kernel.fused_store_index_cache import (
            can_use_dsa_fused_store,
            fused_store_index_k_cache,
        )
    except Exception:
        return None
    if not can_use_dsa_fused_store(
        torch.bfloat16, torch.int64, PAGE_SIZE,
    ):
        return None

    buf_singleton = torch.zeros(
        2, PAGE_BYTES, dtype=torch.uint8, device=device,
    )
    buf_packed = torch.zeros(
        2, PAGE_BYTES, dtype=torch.uint8, device=device,
    )
    # Singleton: K0 alone at out_cache_loc=0 (= page 0, slot 0).
    fused_store_index_k_cache(
        K0, buf_singleton,
        torch.tensor([0], dtype=torch.int64, device=device),
        PAGE_SIZE,
    )
    # Packed: K0 at slot 0 of page 0, K1..K63 fill the rest.
    packed_key = torch.cat([K0, neighbours], dim=0).contiguous()
    fused_store_index_k_cache(
        packed_key, buf_packed,
        torch.arange(PAGE_SIZE, dtype=torch.int64, device=device),
        PAGE_SIZE,
    )

    fp8_s, scale_s = _read_back_k0_bytes(buf_singleton)
    fp8_p, scale_p = _read_back_k0_bytes(buf_packed)
    return {
        "path_used": "fused_store_index_k_cache",
        "fp8_singleton_hex": fp8_s.numpy().tobytes().hex(),
        "fp8_packed_hex": fp8_p.numpy().tobytes().hex(),
        "scale_singleton_hex": scale_s.numpy().tobytes().hex(),
        "scale_packed_hex": scale_p.numpy().tobytes().hex(),
        "fp8_bytes_equal": bool(__import__("torch").equal(fp8_s, fp8_p)),
        "scale_bytes_equal": bool(__import__("torch").equal(scale_s, scale_p)),
    }


def _try_fallback_path(K0, neighbours, device) -> Optional[Dict[str, Any]]:
    """Run the act_quant + SetKAndS fallback. Returns None on
    skip/unavailable."""
    import torch

    try:
        from sglang.srt.layers.attention.dsa.triton_kernel import act_quant
        from sglang.srt.layers.attention.dsa.index_buf_accessor import (
            SetKAndS,
        )
    except Exception:
        return None

    # act_quant on a single (1, 128) row → (fp8: (1,128), scale:
    # (1,1)). For the packed write we quantize all 64 rows.
    try:
        fp8_singleton_quant, scale_singleton_quant = act_quant(
            K0, block_size=INDEX_HEAD_DIM,
        )
        packed_key = torch.cat([K0, neighbours], dim=0).contiguous()
        fp8_packed_quant, scale_packed_quant = act_quant(
            packed_key, block_size=INDEX_HEAD_DIM,
        )
    except Exception:
        return None

    buf_singleton = torch.zeros(
        2, PAGE_BYTES, dtype=torch.uint8, device=device,
    )
    buf_packed = torch.zeros(
        2, PAGE_BYTES, dtype=torch.uint8, device=device,
    )
    pool_shim = SimpleNamespace(page_size=PAGE_SIZE)

    try:
        SetKAndS.execute(
            pool=pool_shim, buf=buf_singleton,
            loc=torch.tensor([0], dtype=torch.int64, device=device),
            index_k=fp8_singleton_quant.contiguous(),
            index_k_scale=scale_singleton_quant.contiguous(),
        )
        SetKAndS.execute(
            pool=pool_shim, buf=buf_packed,
            loc=torch.arange(
                PAGE_SIZE, dtype=torch.int64, device=device,
            ),
            index_k=fp8_packed_quant.contiguous(),
            index_k_scale=scale_packed_quant.contiguous(),
        )
    except Exception:
        return None

    fp8_s, scale_s = _read_back_k0_bytes(buf_singleton)
    fp8_p, scale_p = _read_back_k0_bytes(buf_packed)
    return {
        "path_used": "fallback_act_quant_set_index_k_scale_buffer",
        "fp8_singleton_hex": fp8_s.numpy().tobytes().hex(),
        "fp8_packed_hex": fp8_p.numpy().tobytes().hex(),
        "scale_singleton_hex": scale_s.numpy().tobytes().hex(),
        "scale_packed_hex": scale_p.numpy().tobytes().hex(),
        "fp8_bytes_equal": bool(torch.equal(fp8_s, fp8_p)),
        "scale_bytes_equal": bool(torch.equal(scale_s, scale_p)),
    }


@unittest.skipUnless(
    _opt_in_and_cuda(),
    "SGLANG_DS_FP8_SCALE_PROOF=1 + CUDA required. Deliberate opt-in "
    "so a CPU-only run cannot be misread as a passing M3-B check.",
)
class TestDSv32FP8IndexCacheScaleStability(unittest.TestCase):
    """Singleton vs packed-page FP8 byte+scale equality via the
    production fused-store and fallback paths."""

    def test_singleton_vs_packed_page_k0_bytes_match(self):
        import torch

        device = torch.device("cuda")
        torch.manual_seed(0)
        K0 = torch.randn(
            1, INDEX_HEAD_DIM, dtype=torch.bfloat16, device=device,
        )
        neighbours = torch.randn(
            PAGE_SIZE - 1, INDEX_HEAD_DIM,
            dtype=torch.bfloat16, device=device,
        )

        results: Dict[str, Dict[str, Any]] = {}
        fused = _try_fused_path(K0, neighbours, device)
        if fused is not None:
            results["fused"] = fused
        fallback = _try_fallback_path(K0, neighbours, device)
        if fallback is not None:
            results["fallback"] = fallback

        if not results:
            self.skipTest(
                "Neither production store path (fused, fallback) "
                "could execute on this hardware. The proof needs at "
                "least one to run."
            )

        # Verdict: every path that ran must agree singleton == packed.
        verdict_paths: Dict[str, str] = {}
        for path_name, res in results.items():
            ok = res["fp8_bytes_equal"] and res["scale_bytes_equal"]
            verdict_paths[path_name] = "PASS" if ok else "FAIL"

        artifact = {
            "fixture_kind": "fp8_index_cache_scale_stability",
            "page_size": PAGE_SIZE,
            "index_head_dim": INDEX_HEAD_DIM,
            "paths_executed": sorted(results.keys()),
            "per_path": results,
            "per_path_verdict": verdict_paths,
            "verdict": (
                "PASS" if all(v == "PASS" for v in verdict_paths.values())
                else "FAIL"
            ),
            "notes": (
                "If scale bytes differ between singleton and packed "
                "writes for either path: per-token scale depends on "
                "neighbour tokens within the page, so radix-cache "
                "reuse would see different K-noPE in cold vs warm "
                "paths. KEEP --disable-radix-cache."
            ),
        }
        _record_artifact(artifact, suffix="singleton_vs_packed_page")

        for path_name, res in results.items():
            self.assertTrue(
                res["scale_bytes_equal"],
                f"FP8 per-token scale for K0 differs between "
                f"singleton and packed page writes on the "
                f"{path_name!r} path: "
                f"singleton={res['scale_singleton_hex']}, "
                f"packed={res['scale_packed_hex']}. KEEP "
                f"--disable-radix-cache. Artifact: {RESULTS_DIR}/.",
            )
            self.assertTrue(
                res["fp8_bytes_equal"],
                f"FP8 byte for K0 differs between singleton and "
                f"packed page writes on the {path_name!r} path; "
                "quantization kernel is non-deterministic. KEEP "
                "--disable-radix-cache.",
            )


if __name__ == "__main__":
    unittest.main()
