import sys
import types
import unittest
from unittest.mock import MagicMock, patch

import torch

from sglang.multimodal_gen.runtime.layers.attention.backends.sol_attn import (
    SolAttnImpl,
    _get_sol_attn_runtime_config,
    _parse_layer_ranges,
)

_SOL = "sglang.multimodal_gen.runtime.layers.attention.backends.sol_attn"


def _server_args(**cfg):
    server_args = MagicMock()
    server_args.attention_backend_config = cfg
    return server_args


class TestSolAttnBackend(unittest.TestCase):
    def test_parse_layer_ranges(self):
        self.assertEqual(_parse_layer_ranges("0,1,3-5"), frozenset({0, 1, 3, 4, 5}))

    def test_dense_backend_aliases(self):
        for raw, expected in (
            ("fa", "fa"),
            ("sage", "sage_attn"),
            ("sage_attn", "sage_attn"),
        ):
            server_args = MagicMock()
            server_args.attention_backend_config = {"dense_backend": raw}
            with patch(
                "sglang.multimodal_gen.runtime.layers.attention.backends.sol_attn.get_global_server_args",
                return_value=server_args,
            ):
                self.assertEqual(
                    _get_sol_attn_runtime_config()["dense_backend"], expected
                )
        server_args = MagicMock()
        server_args.attention_backend_config = {"dense_backend": "torch_sdpa"}
        with (
            patch(
                "sglang.multimodal_gen.runtime.layers.attention.backends.sol_attn.get_global_server_args",
                return_value=server_args,
            ),
            self.assertRaises(ValueError),
        ):
            _get_sol_attn_runtime_config()

    def test_dense_guard_uses_early_steps(self):
        impl = SolAttnImpl(
            num_heads=8,
            head_size=128,
            causal=False,
            softmax_scale=128**-0.5,
            prefix="blocks.5.attn",
        )
        ctx = MagicMock()
        ctx.current_timestep = 3
        server_args = MagicMock()
        server_args.attention_backend_config = {
            "dense_steps": 10,
            "dense_layers": "0,1",
        }
        with (
            patch(
                "sglang.multimodal_gen.runtime.layers.attention.backends.sol_attn.get_global_server_args",
                return_value=server_args,
            ),
            patch(
                "sglang.multimodal_gen.runtime.managers.forward_context.get_forward_context",
                return_value=ctx,
            ),
        ):
            self.assertTrue(impl._should_use_dense())

    def test_sparse_layer_after_dense_guard(self):
        impl = SolAttnImpl(
            num_heads=8,
            head_size=128,
            causal=False,
            softmax_scale=128**-0.5,
            prefix="blocks.5.attn",
        )
        ctx = MagicMock()
        ctx.current_timestep = 20
        server_args = MagicMock()
        server_args.attention_backend_config = {
            "dense_steps": 10,
            "dense_layers": "0,1",
        }
        with (
            patch(
                "sglang.multimodal_gen.runtime.layers.attention.backends.sol_attn.get_global_server_args",
                return_value=server_args,
            ),
            patch(
                "sglang.multimodal_gen.runtime.managers.forward_context.get_forward_context",
                return_value=ctx,
            ),
        ):
            self.assertFalse(impl._should_use_dense())


class TestSolAttnRocm(unittest.TestCase):
    def _impl(self, prefix="blocks.5.attn"):
        return SolAttnImpl(
            num_heads=2,
            head_size=128,
            causal=False,
            softmax_scale=0.1,
            prefix=prefix,
        )

    def test_dense_backend_defaults_to_aiter_sage_on_rocm(self):
        for is_rocm, expected in ((True, "aiter_sage"), (False, "fa")):
            with (
                patch(f"{_SOL}.get_global_server_args", return_value=_server_args()),
                patch(f"{_SOL}.current_platform.is_rocm", return_value=is_rocm),
            ):
                self.assertEqual(
                    _get_sol_attn_runtime_config()["dense_backend"], expected
                )

    def test_rocm_calls_aiter_gfx1201_sol_attention(self):
        aiter = types.ModuleType("aiter")
        aiter.gfx1201_sol_attention = MagicMock(side_effect=lambda q, k, v, **_: q)
        q = torch.randn(6, 2, 128, dtype=torch.bfloat16)
        server_args = _server_args(tau=1.5, sink_tokens=64, sink_start=0)
        with (
            patch.dict(sys.modules, {"aiter": aiter}),
            patch(f"{_SOL}.get_global_server_args", return_value=server_args),
            patch(f"{_SOL}.current_platform.is_rocm", return_value=True),
        ):
            out = self._impl()._run_sol_attn_thd(q, q, q)
        self.assertTrue(torch.equal(out, q))
        self.assertEqual(
            aiter.gfx1201_sol_attention.call_args.kwargs,
            {"tau": 1.5, "sink_tokens": 64, "sink_start": 0, "softmax_scale": 0.1},
        )

    def test_rocm_rejects_other_thresholds(self):
        q = torch.randn(6, 2, 128, dtype=torch.bfloat16)
        with (
            patch(
                f"{_SOL}.get_global_server_args",
                return_value=_server_args(thresh_type="exact"),
            ),
            patch(f"{_SOL}.current_platform.is_rocm", return_value=True),
            self.assertRaises(ValueError),
        ):
            self._impl()._run_sol_attn_thd(q, q, q)

    def test_rocm_dense_guard_uses_aiter_sage(self):
        impl = self._impl(prefix="blocks.0.attn")
        dense = MagicMock()
        q = torch.randn(6, 2, 128, dtype=torch.bfloat16)
        cu_seqlens = torch.tensor([0, 6], dtype=torch.int32)
        with (
            patch(f"{_SOL}.get_global_server_args", return_value=_server_args()),
            patch(f"{_SOL}.current_platform.is_rocm", return_value=True),
            patch.object(impl, "_dense_aiter_sage", return_value=dense),
        ):
            impl.forward_varlen(q, q, q, cu_seqlens=cu_seqlens, max_seqlen=6)
        dense.forward_varlen.assert_called_once()

    def test_rocm_dense_guard_keeps_packed_trailing_padding(self):
        fav3 = types.ModuleType("aiter.ops.triton.attention.fav3_sage")
        fav3.fav3_sage_wrapper_func = MagicMock()
        modules = {
            name: types.ModuleType(name)
            for name in (
                "aiter",
                "aiter.ops",
                "aiter.ops.triton",
                "aiter.ops.triton.attention",
            )
        }
        modules[fav3.__name__] = fav3
        impl = SolAttnImpl(
            num_heads=2,
            head_size=128,
            causal=False,
            softmax_scale=0.1,
            prefix="blocks.0.attn",
            packed_trailing_padding=True,
        )
        with patch.dict(sys.modules, modules):
            dense = impl._dense_aiter_sage()
        self.assertTrue(dense.packed_trailing_padding)
        self.assertEqual(dense.softmax_scale, 0.1)

    def test_rocm_platform_accepts_sol_attn_on_gfx1201_only(self):
        from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
        from sglang.multimodal_gen.runtime.platforms.rocm import RocmPlatform

        aiter = types.ModuleType("aiter")
        aiter.gfx1201_sol_attention = MagicMock()
        for arch, ok in (("gfx1201", True), ("gfx942:sramecc+:xnack-", False)):
            with (
                patch.dict(sys.modules, {"aiter": aiter}),
                patch.object(RocmPlatform, "get_gcn_arch_name", return_value=arch),
            ):
                if ok:
                    self.assertTrue(
                        RocmPlatform.get_attn_backend_cls_str(
                            AttentionBackendEnum.SOL_ATTN, 128, torch.bfloat16
                        ).endswith("sol_attn.SolAttnBackend")
                    )
                else:
                    with self.assertRaises(ValueError):
                        RocmPlatform.get_attn_backend_cls_str(
                            AttentionBackendEnum.SOL_ATTN, 128, torch.bfloat16
                        )


if __name__ == "__main__":
    unittest.main()
