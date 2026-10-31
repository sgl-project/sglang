"""CPU tests execute the actual upstream-worktree eligibility method."""

import ast
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

SOURCE = (
    Path(__file__).resolve().parents[3]
    / "python/sglang/srt/layers/attention/dsv4/compressor.py"
)


class Tensor:
    def __init__(self, rows, width, dtype):
        self.shape = (rows, width)
        self.ndim = 2
        self.dtype = dtype
        self.device = 0
        self.is_cuda = True
        self.requires_grad = False
        self.contiguous = True

    def is_contiguous(self):
        return self.contiguous


def build():
    ns = types.SimpleNamespace
    options = ns(
        enabled=True, minimum=128, capture=False, capability=(9, 0), interleave=True
    )
    parallel = ns(attn_cp_size=8, attn_tp_size=1, attn_dp_size=1)
    method = lambda: None
    method.__module__ = "sglang.srt.layers.attention.dsv4.compressor_v2"
    backend = ns(
        online_c128_mtp=None,
        enable_deepseek_v4_fp4_indexer=False,
        token_to_kv_pool=ns(uniform_fp8=False),
        _forward_compress_all_in_one=method,
    )
    global_values = dict(
        _is_hip=False,
        _is_npu=False,
        torch=ns(
            __version__="2.14.1+cu130",
            bfloat16="bf16",
            float32="fp32",
            cuda=ns(
                is_current_stream_capturing=lambda: options.capture,
                get_device_capability=lambda device: options.capability,
            ),
        ),
        envs=ns(
            SGLANG_OPT_COMPRESSOR_RANK_MAJOR=ns(get=lambda: options.enabled),
            SGLANG_OPT_COMPRESSOR_RANK_MAJOR_MIN_TOKENS=ns(get=lambda: options.minimum),
        ),
        get_parallel=lambda: parallel,
        get_attn_backend=lambda: backend,
    )
    tree = ast.parse(SOURCE.read_text())
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Compressor"
    )
    fn = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "supports_rank_major_scores"
    )
    exec(
        compile(ast.Module(body=[fn], type_ignores=[]), str(SOURCE), "exec"),
        global_values,
    )
    meta = ns(
        total_seq_lens=50000,
        per_rank_actual_token=[6250] * 8,
        per_rank_logical_token=None,
        gather_index=None,
    )
    batch = ns(
        attn_cp_metadata=meta,
        batch_size=1,
        forward_mode="EXTEND",
        extend_prefix_lens_cpu=[0],
        spec_info=None,
        return_logprob=False,
        return_hidden_states_before_norm=False,
    )
    self = ns(ratio=4, head_dim=512, is_in_indexer=False)
    x, score = Tensor(6250, 4096, "bf16"), Tensor(6250, 2048, "fp32")
    modules = {
        name: types.ModuleType(name)
        for name in (
            "sglang",
            "sglang.srt",
            "sglang.srt.layers",
            "sglang.srt.layers.cp",
            "sglang.srt.model_executor",
            "sglang.srt.model_executor.forward_batch_info",
        )
    }
    modules["sglang.srt.layers.cp"].is_interleave = lambda: options.interleave
    modules["sglang.srt.model_executor.forward_batch_info"].ForwardMode = ns(
        EXTEND="EXTEND"
    )
    return ns(
        fn=lambda: global_values["supports_rank_major_scores"](self, x, score, batch),
        options=options,
        parallel=parallel,
        backend=backend,
        batch=batch,
        meta=meta,
        x=x,
        score=score,
        self=self,
        modules=modules,
    )


class Guards(unittest.TestCase):
    def test_supported_lengths(self):
        for n in (128, 512, 4096, 8192, 50000, 64000, 65536):
            s = build()
            s.meta.total_seq_lens = n
            s.meta.per_rank_actual_token = [n // 8] * 8
            s.x.shape = (n // 8, 4096)
            s.score.shape = (n // 8, 2048)
            with self.subTest(n=n), patch.dict(sys.modules, s.modules):
                self.assertTrue(s.fn())

    def test_fallbacks(self):
        changes = [
            lambda s: setattr(s.options, "enabled", False),
            lambda s: setattr(s.options, "interleave", False),
            lambda s: setattr(s.options, "capture", True),
            lambda s: setattr(s.options, "capability", (10, 0)),
            lambda s: setattr(s.parallel, "attn_cp_size", 4),
            lambda s: setattr(s.parallel, "attn_tp_size", 2),
            lambda s: setattr(s.batch, "batch_size", 2),
            lambda s: setattr(s.batch, "forward_mode", "DECODE"),
            lambda s: setattr(s.batch, "extend_prefix_lens_cpu", [128]),
            lambda s: setattr(s.batch, "spec_info", object()),
            lambda s: setattr(s.batch, "return_logprob", True),
            lambda s: setattr(s.meta, "gather_index", object()),
            lambda s: setattr(s.meta, "total_seq_lens", 65544),
            lambda s: setattr(s.score, "dtype", "bf16"),
            lambda s: setattr(s.score, "contiguous", False),
            lambda s: setattr(s.meta, "per_rank_actual_token", [6251] * 8),
            lambda s: setattr(s.backend.token_to_kv_pool, "uniform_fp8", True),
            lambda s: setattr(s.options, "minimum", 50001),
        ]
        for i, change in enumerate(changes):
            s = build()
            change(s)
            with self.subTest(case=i), patch.dict(sys.modules, s.modules):
                self.assertFalse(s.fn())

    def test_gemm_padding_is_not_a_logical_token(self):
        s = build()
        s.x.shape = (6256, 4096)
        s.score.shape = (6256, 2048)
        s.meta.per_rank_actual_token = [6256] * 8
        s.meta.per_rank_logical_token = [6250] * 8
        with patch.dict(sys.modules, s.modules):
            self.assertTrue(s.fn())


if __name__ == "__main__":
    unittest.main(verbosity=2)
