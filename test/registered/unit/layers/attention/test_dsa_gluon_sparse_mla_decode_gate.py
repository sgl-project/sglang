"""The gate that sends DSA "triton" decode / verify to the gfx950 Gluon kernels.

Anything the gate lets through runs a different kernel family, so every static
term must be able to turn it off on its own, and a call the kernels do not
implement must fall back to the Triton path.
"""

import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest import mock

import torch

import sglang.kernels.ops.attention.dsa as dsa_kernels
from sglang.srt.layers.attention import dsa_backend
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

_MODULE = "sglang.kernels.ops.attention.dsa.gluon_sparse_mla_decode_gfx950"
_GOOD = dict(
    decode_impl="triton",
    kv_cache_dtype=torch.float8_e4m3fn,
    kv_lora_rank=512,
    qk_nope_head_dim=192,
    qk_rope_head_dim=64,
    v_head_dim=256,
    num_q_heads=8,
    dcp_size=1,
)


class TestGluonSparseMlaDecodeGate(CustomTestCase):
    def setUp(self):
        # Stand-in kernel module, so the gate is testable without Gluon.
        self.kernels = ModuleType(_MODULE)
        self.kernels.gluon_sparse_mla_decode_supported = lambda *args: True
        patches = (
            mock.patch.object(dsa_backend, "_IS_GFX95", True),
            mock.patch.dict(sys.modules, {_MODULE: self.kernels}),
            mock.patch.object(
                dsa_kernels, _MODULE.rsplit(".", 1)[1], self.kernels, create=True
            ),
        )
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def test_loads_when_every_term_holds(self):
        for heads in (8, 16):
            with self.subTest(heads=heads):
                loaded = dsa_backend._load_gluon_sparse_mla_decode(
                    **{**_GOOD, "num_q_heads": heads}
                )
                self.assertIs(loaded, self.kernels)

    def test_each_term_alone_turns_it_off(self):
        for name, broken in (
            ("decode_impl", "tilelang"),
            ("decode_impl", "aiter"),
            ("kv_cache_dtype", torch.bfloat16),
            ("kv_lora_rank", 448),
            ("qk_rope_head_dim", 0),
            # DeepSeek-V3.2 (128/128) and GLM-5.3-Flash (256/0) stay on Triton.
            ("qk_nope_head_dim", 128),
            ("v_head_dim", 128),
            ("qk_nope_head_dim", 256),
            ("num_q_heads", 12),
            ("num_q_heads", 32),
            ("dcp_size", 2),
        ):
            with self.subTest(term=name, value=broken):
                self.assertIsNone(
                    dsa_backend._load_gluon_sparse_mla_decode(**{**_GOOD, name: broken})
                )

    def test_off_gfx95(self):
        with mock.patch.object(dsa_backend, "_IS_GFX95", False):
            self.assertIsNone(dsa_backend._load_gluon_sparse_mla_decode(**_GOOD))

    def test_split_q_falls_back_to_triton(self):
        """q_all is None when the caller passed q_nope and q_rope separately."""
        use = dsa_backend.DeepseekSparseAttnBackend._use_gluon_sparse_mla_decode
        backend = SimpleNamespace(gluon_sparse_mla_decode=self.kernels)
        kv = page_table = torch.empty(0)
        self.assertFalse(use(backend, None, kv, page_table))
        self.assertTrue(use(backend, torch.empty(0), kv, page_table))
        self.kernels.gluon_sparse_mla_decode_supported = lambda *args: False
        self.assertFalse(use(backend, torch.empty(0), kv, page_table))
        backend.gluon_sparse_mla_decode = None
        self.assertFalse(use(backend, torch.empty(0), kv, page_table))


class TestGluonWithoutBarrier(CustomTestCase):
    """Without a Gluon CTA barrier the specialised producers race; the path must stay off."""

    def test_loader_returns_none(self):
        import triton.experimental.gluon.language as gl

        saved = {
            n: getattr(gl, n) for n in ("barrier", "thread_barrier") if n in vars(gl)
        }
        for n in saved:
            delattr(gl, n)
        self.addCleanup(lambda: [setattr(gl, n, f) for n, f in saved.items()])
        patches = (
            mock.patch.object(dsa_backend, "_IS_GFX95", True),
            mock.patch.dict(sys.modules),
        )
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        sys.modules.pop(_MODULE, None)
        self.assertIsNone(dsa_backend._load_gluon_sparse_mla_decode(**_GOOD))


if __name__ == "__main__":
    unittest.main()
