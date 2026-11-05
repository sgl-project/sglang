# SPDX-License-Identifier: Apache-2.0
"""CPU ABI checks for Cute DCP causal bounds, empty shards and LSE base.

The real Cute _run_decode_kernel body calls a finite FlashInfer stub here. This
checks the actual native keyword contract; it cannot establish GPU arithmetic,
collective progress, graph capture/replay, or cross-engine numerical tolerance.
"""

import ast
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from sglang.srt.kernel_plugins.nvidia.kimi_mla import native_mla_decode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, stage="base-a", runner_config="cpu")


def native_cute_kernel_method():
    """Load just the actual method without importing CUDA/FlashInfer on CPU."""
    source = (
        Path(__file__).resolve().parents[5]
        / "python/sglang/srt/layers/attention/cutedsl_mla_backend.py"
    )
    tree = ast.parse(source.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    method = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "_run_decode_kernel"
    )
    method.returns = None
    for arg in (*method.args.args, *method.args.kwonlyargs):
        arg.annotation = None
    scope = {
        "torch": torch,
        "envs": SimpleNamespace(
            SGLANG_SKIP_SOFTMAX_DECODE_THRESHOLD_SCALE_FACTOR=SimpleNamespace(
                get=lambda: 0
            )
        ),
    }
    exec(
        compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"),
        scope,
    )
    return scope["_run_decode_kernel"], scope


def fixture():
    query = torch.arange(2 * 12 * 576).to(torch.uint8).view(2, 12, 576)
    output = torch.empty(2, 12, 512, dtype=torch.bfloat16)
    native_layer = SimpleNamespace(tp_q_head_num=96, v_head_dim=512)
    metadata = SimpleNamespace(
        block_kv_indices=torch.zeros(2, 2048, dtype=torch.int32),
        seq_lens_k=torch.tensor([0, 131072], dtype=torch.int32),
        global_seq_lens_k=torch.tensor([1, 1048576], dtype=torch.int32),
        max_seq_len_k=131072,
    )
    parallel = SimpleNamespace(
        dcp_enabled=True,
        dcp_size=8,
        dcp_rank=3,
        attn_dcp_size=8,
        dcp_replicate_q_proj=True,
        dcp_comm_backend="fi_a2a",
        dcp_group=object(),
    )
    raw = torch.full((2, 1, 96, 512), 7, dtype=torch.bfloat16)
    lse = torch.ones(2, 96)
    kernel = Mock(return_value=(raw, lse))
    method, scope = native_cute_kernel_method()
    scope["flashinfer"] = SimpleNamespace(
        decode=SimpleNamespace(trtllm_batch_decode_with_kv_cache_mla=kernel)
    )
    backend = SimpleNamespace(
        backend="cute-dsl",
        q_indptr_decode=torch.tensor([0, 1, 2], dtype=torch.int32),
        workspace_buffer=object(),
        qk_nope_head_dim=128,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        _compute_decode_bmm1_scale=lambda layer: 0.125,
        _get_dcp_local_seq_lens=Mock(
            return_value=torch.tensor([0, 1], dtype=torch.int32)
        ),
        _forward_decode_dcp=Mock(side_effect=AssertionError("TRT DCP path")),
    )
    backend._run_decode_kernel = lambda **kw: method(backend, **kw)
    batch = SimpleNamespace(
        batch_size=2,
        seq_lens=torch.tensor([1, 8], dtype=torch.int32),
        forward_mode=SimpleNamespace(is_decode=lambda: True),
    )
    args = {
        "backend": backend,
        "cache": torch.zeros(1, 64, 576, dtype=torch.uint8),
        "attention": SimpleNamespace(
            attn_mqa=object(), attn_mqa_for_dcp_decode=native_layer
        ),
        "metadata": metadata,
        "forward_batch": batch,
    }
    return parallel, query, output, args, raw, lse, kernel


def modules(query, expected):
    logits = ModuleType("sglang.srt.layers.logits_processor")
    logits.get_in_autotune_dummy_run = lambda: False
    fixup_module = ModuleType("sglang.kernels.ops.attention.fixup_zero_kv")

    def fixup(out, lse, lens, indptr, max_seq_len):
        assert max_seq_len == 1
        assert lens[0] == 0
        assert indptr.tolist() == [0, 1, 2]
        out[0].zero_()
        lse[0].fill_(-torch.inf)

    fixup_module.fixup_zero_kv_rows = Mock(side_effect=fixup)
    dcp = ModuleType("sglang.srt.layers.dcp")
    gathered = torch.cat([query + rank for rank in range(8)], dim=1)
    dcp.all_gather_q_for_mla_decode = Mock(
        return_value=(gathered[..., :512], gathered[..., 512:])
    )

    def merge(out, lse, group, **kw):
        assert not torch.count_nonzero(out[0])
        assert torch.isneginf(lse[0]).all()
        assert kw["is_lse_base_on_e"] is True
        return expected

    dcp.dcp_a2a_lse_reduce = Mock(side_effect=merge)
    dcp.cp_lse_ag_out_rs_mla = Mock(
        side_effect=lambda *a, **kw: merge(*a, **kw).transpose(0, 1)
    )
    return {m.__name__: m for m in (logits, fixup_module, dcp)}, gathered


@pytest.mark.parametrize("comm", ("a2a", "fi_a2a", "allgather"))
def test_cute_uses_real_kernel_abi_global_causal_and_natural_lse(comm):
    parallel, query, output, args, _, _, kernel = fixture()
    parallel.dcp_comm_backend = comm
    expected = torch.ones_like(output)
    patched, gathered = modules(query, expected)
    with (
        patch("sglang.srt.runtime_context.get_parallel", return_value=parallel),
        patch.dict(sys.modules, patched),
    ):
        native_mla_decode(query, output, **args)
    torch.testing.assert_close(output, expected)
    kw = kernel.call_args.kwargs
    assert kw["backend"] == "cute-dsl" and kw["enable_dcp"] is True
    assert kw["cp_world"] == 8 and kw["cp_rank"] == 3
    assert kw["return_lse"] is True
    assert kw["causal_seqlens_kv_global"].tolist() == [1, 1048576]
    assert kw["seq_lens"].tolist() == [0, 131072]
    assert kw["block_tables"] is args["metadata"].block_kv_indices
    assert kw["max_seq_len"] == 131072
    assert torch.equal(kw["query"].view(torch.uint8).squeeze(1), gathered)
    args["backend"]._forward_decode_dcp.assert_not_called()
    patched["sglang.srt.layers.dcp"].all_gather_q_for_mla_decode.assert_called_once()


def test_cute_incomplete_hoisted_metadata_uses_native_global_lengths():
    parallel, query, output, args, _, _, kernel = fixture()
    args["metadata"].global_seq_lens_k = None
    patched, _ = modules(query, torch.zeros_like(output))
    with (
        patch("sglang.srt.runtime_context.get_parallel", return_value=parallel),
        patch.dict(sys.modules, patched),
    ):
        native_mla_decode(query, output, **args)
    assert kernel.call_args.kwargs["causal_seqlens_kv_global"].tolist() == [1, 8]
    assert kernel.call_args.kwargs["seq_lens"].tolist() == [0, 1]
    args["backend"]._get_dcp_local_seq_lens.assert_called_once()


def test_cute_never_invents_a_missing_global_causal_bound():
    parallel, query, output, args, _, _, kernel = fixture()
    args["metadata"].global_seq_lens_k = None
    args["forward_batch"].seq_lens = None
    patched, _ = modules(query, torch.zeros_like(output))
    with (
        patch("sglang.srt.runtime_context.get_parallel", return_value=parallel),
        patch.dict(sys.modules, patched),
        pytest.raises(ValueError, match="global causal lengths"),
    ):
        native_mla_decode(query, output, **args)
    kernel.assert_not_called()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
