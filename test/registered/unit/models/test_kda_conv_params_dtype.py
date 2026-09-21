"""Regression for KDA short-conv dtype (#40504).

SGLang used to hardcode ``params_dtype=torch.float32`` on KDA ``qkv_conv1d``.
Native-bf16 checkpoints (Kimi-Linear, GLM-5.3-Flash, Ling) were silently
up-cast on ``param.data.copy_``, doubling decode conv-weight bandwidth.
"""

from __future__ import annotations

import ast
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from sglang.srt.model_loader.utils import set_default_torch_dtype
from sglang.srt.model_loader.weight_utils import (
    _KDA_CONV_DTYPE_CACHE,
    resolve_kda_conv_params_dtype,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")

_REPO_ROOT = Path(__file__).resolve().parents[4]
_KDA_MODEL_FILES = (
    _REPO_ROOT / "python/sglang/srt/models/kimi_linear.py",
    _REPO_ROOT / "python/sglang/srt/models/glm5_next.py",
    _REPO_ROOT / "python/sglang/srt/models/kimi_k3.py",
)
_KIMI_LINEAR_CONV_KEY = "model.layers.0.self_attn.q_conv1d.weight"
_KIMI_K3_CONV_KEY = "language_model.model.layers.0.self_attn.q_conv1d.weight"


def _write_conv_checkpoint(
    folder: str,
    *,
    dtype: torch.dtype,
    key: str = _KIMI_LINEAR_CONV_KEY,
    sharded: bool = False,
) -> str:
    weight = torch.zeros(8, 1, 4, dtype=dtype)
    if sharded:
        shard = "model-00001-of-00001.safetensors"
        save_file({key: weight}, os.path.join(folder, shard))
        with open(os.path.join(folder, "model.safetensors.index.json"), "w") as f:
            json.dump({"weight_map": {key: shard}}, f)
    else:
        save_file({key: weight}, os.path.join(folder, "model.safetensors"))
    return folder


def _qkv_conv1d_params_dtype_exprs(path: Path) -> list[str]:
    tree = ast.parse(path.read_text())
    exprs: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = getattr(func, "id", None) or getattr(func, "attr", None)
        if name != "MergedColumnParallelLinear":
            continue
        kwargs = {kw.arg: kw.value for kw in node.keywords if kw.arg}
        prefix = kwargs.get("prefix")
        prefix_src = ast.unparse(prefix) if prefix is not None else ""
        if "qkv_conv1d" not in prefix_src:
            continue
        params = kwargs.get("params_dtype")
        exprs.append(ast.unparse(params) if params is not None else "<missing>")
    return exprs


class TestKdaConvParamsDtype(CustomTestCase):
    def tearDown(self):
        _KDA_CONV_DTYPE_CACHE.clear()

    def test_bf16_kimi_linear_checkpoint_keeps_bf16(self):
        """Native-bf16 convs must not be allocated as fp32 (the #40504 up-cast)."""
        with tempfile.TemporaryDirectory() as folder:
            _write_conv_checkpoint(folder, dtype=torch.bfloat16)
            with set_default_torch_dtype(torch.float32):
                self.assertEqual(resolve_kda_conv_params_dtype(folder), torch.bfloat16)

    def test_fp32_kimi_k3_checkpoint_keeps_fp32(self):
        """Kimi-K3 ships fp32 convs on purpose; do not downcast them."""
        with tempfile.TemporaryDirectory() as folder:
            _write_conv_checkpoint(
                folder, dtype=torch.float32, key=_KIMI_K3_CONV_KEY, sharded=True
            )
            with set_default_torch_dtype(torch.bfloat16):
                self.assertEqual(resolve_kda_conv_params_dtype(folder), torch.float32)

    def test_missing_conv_key_falls_back_to_default_dtype(self):
        with tempfile.TemporaryDirectory() as folder:
            save_file(
                {"model.embed_tokens.weight": torch.zeros(4, 4, dtype=torch.bfloat16)},
                os.path.join(folder, "model.safetensors"),
            )
            with set_default_torch_dtype(torch.bfloat16):
                self.assertEqual(resolve_kda_conv_params_dtype(folder), torch.bfloat16)

    def test_kimi_delta_attention_matches_bf16_checkpoint(self):
        """Construction-path guard: KimiDeltaAttention (and BailingKDA) keep bf16.

        Fails on the pre-fix ``params_dtype=torch.float32`` hardcode. Also
        checks that the ``conv_weights`` view captured in ``__init__`` still
        aliases the parameter after an in-place load.
        """
        from sglang.test.test_utils import maybe_stub_sgl_kernel

        maybe_stub_sgl_kernel()

        from sglang.srt.configs.kimi_linear import KimiLinearConfig
        from sglang.srt.models.kimi_linear import KimiDeltaAttention

        parallel = SimpleNamespace(tp_size=1, tp_rank=0, attn_tp_size=1, attn_tp_rank=0)
        with tempfile.TemporaryDirectory() as folder:
            _write_conv_checkpoint(folder, dtype=torch.bfloat16)
            with (
                patch(
                    "sglang.srt.models.kimi_linear.get_parallel",
                    return_value=parallel,
                ),
                patch(
                    "sglang.srt.runtime_context.get_model",
                    return_value=SimpleNamespace(model_path=folder),
                ),
                set_default_torch_dtype(torch.bfloat16),
            ):
                config = KimiLinearConfig(
                    hidden_size=32,
                    num_attention_heads=4,
                    torch_dtype=torch.bfloat16,
                    linear_attn_config={
                        "head_dim": 8,
                        "num_heads": 4,
                        "short_conv_kernel_size": 4,
                        "kda_layers": [1],
                        "full_attn_layers": [],
                    },
                )
                attn = KimiDeltaAttention(
                    layer_idx=0,
                    hidden_size=32,
                    config=config,
                    prefix="model.layers.0.self_attn",
                )

            self.assertEqual(attn.qkv_conv1d.weight.dtype, torch.bfloat16)
            self.assertEqual(attn.attn.conv_weights.dtype, torch.bfloat16)
            storage_ptr = attn.qkv_conv1d.weight.data_ptr()
            loaded = torch.ones_like(attn.qkv_conv1d.weight, dtype=torch.bfloat16)
            attn.qkv_conv1d.weight.data.copy_(loaded)
            self.assertEqual(attn.qkv_conv1d.weight.dtype, torch.bfloat16)
            self.assertEqual(attn.qkv_conv1d.weight.data_ptr(), storage_ptr)
            self.assertEqual(attn.attn.conv_weights.data_ptr(), storage_ptr)

    def test_kda_models_pass_resolved_conv_dtype(self):
        """Bookkeeping: every KDA ``qkv_conv1d`` must take the probed dtype.

        Re-hardcoding ``params_dtype=torch.float32`` is the #40504 regression;
        BailingKDA inherits the kimi_linear site and has no third copy.
        """
        for path in _KDA_MODEL_FILES:
            with self.subTest(path=path.name):
                exprs = _qkv_conv1d_params_dtype_exprs(path)
                self.assertTrue(
                    exprs, f"no qkv_conv1d MergedColumnParallelLinear in {path}"
                )
                for expr in exprs:
                    self.assertEqual(expr, "resolve_kda_conv_params_dtype()")


if __name__ == "__main__":
    unittest.main()
