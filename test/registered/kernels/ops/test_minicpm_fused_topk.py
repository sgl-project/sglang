"""Fused top-k selection kernels for the MiniCPM sparse attention path."""

from __future__ import annotations

import pytest
import tilelang.math
from tilelang import tvm

from sglang.srt.layers.attention.minicpm.fuse_kernel import (
    _fused_attn_pooling_online_topk,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=90, stage="base-b-kernel-unit", runner_config="1-gpu-large")

_HEADS = 32
_HEAD_KV = 2
_GROUPS = _HEADS // _HEAD_KV
_DIM = 128
_KERNEL_SIZE = 32
_KERNEL_STRIDE = 16
_BLOCK_SIZE = 64
_INIT_BLOCKS = 1
_LOCAL_BLOCKS = 2048 // _BLOCK_SIZE
_SPARSE_TOPK = 64 + _LOCAL_BLOCKS
_MAX_CACHE_LEN = 40960

_POOLED_K_LEN = (_MAX_CACHE_LEN + _BLOCK_SIZE - 1) // _BLOCK_SIZE
_OUTPUT_TOPK = min(_SPARSE_TOPK, _POOLED_K_LEN)
_KERNEL_TOPK = tilelang.math.next_power_of_2(_OUTPUT_TOPK)

_KERNEL_KWARGS = dict(
    groups=_GROUPS,
    heads=_HEADS,
    dim=_DIM,
    topk=_KERNEL_TOPK,
    pooled_k_len=tilelang.math.next_power_of_2(_POOLED_K_LEN),
    m_block_dim=_GROUPS,
    block_M=_GROUPS,
    block_stride=_BLOCK_SIZE // _KERNEL_STRIDE,
    pad_len=_KERNEL_SIZE // _KERNEL_STRIDE - 1,
    num_offs=_KERNEL_SIZE // _KERNEL_STRIDE + _BLOCK_SIZE // _KERNEL_STRIDE - 1,
    kernel_stride=_KERNEL_STRIDE,
    block_size=_BLOCK_SIZE,
    init_blocks=_INIT_BLOCKS,
    local_blocks=_LOCAL_BLOCKS,
    dtype_str="bfloat16",
)


def _tir_nodes(statement, predicate):
    nodes = []

    def collect(node):
        if predicate(node):
            nodes.append(node)

    tvm.tirx.stmt_functor.post_order_visit(statement, collect)
    return nodes


def _tir_statements(statement):
    if isinstance(statement, tvm.tirx.SeqStmt):
        for child in statement.seq:
            yield from _tir_statements(child)
    elif (
        isinstance(statement, tvm.tirx.AttrStmt)
        and statement.attr_key == "lexical_alloc_scope"
    ):
        yield from _tir_statements(statement.body)
    else:
        yield statement


def test_fused_topk_synchronizes_shared_mma_inputs():
    """Shared input writes must complete before other warps load MMA operands."""
    factory = _fused_attn_pooling_online_topk
    function = factory.get_tir(
        batch_size=1,
        max_seqlen_q_grid=1,
        is_causal=False,
        **_KERNEL_KWARGS,
    )
    target = tvm.target.Target({"kind": "cuda", "arch": "sm_120"})
    with tvm.transform.PassContext(opt_level=3, config=factory.pass_configs), target:
        artifact = tilelang.lower(function, target=target, enable_device_compile=False)

    tir = tvm.tirx
    load_op = tvm.ir.Op.get("tl.ptx_ldmatrix")
    sync_op = tvm.ir.Op.get("tirx.tvm_storage_sync")
    producer_consumer_edges = []
    for function in artifact.device_mod.functions.values():
        sequences = _tir_nodes(
            function.body, lambda node: isinstance(node, tir.SeqStmt)
        )
        for sequence in sequences:
            pending_shared_write = False
            synchronized = False
            for statement in _tir_statements(sequence):
                loads = _tir_nodes(
                    statement,
                    lambda node: (
                        isinstance(node, tir.Call) and node.op.same_as(load_op)
                    ),
                )
                stores = _tir_nodes(
                    statement,
                    lambda node: (
                        isinstance(node, tir.BufferStore)
                        and node.buffer.scope().startswith("shared")
                        and node.buffer.dtype == "bfloat16"
                    ),
                )
                if stores and not loads:
                    pending_shared_write = True
                    synchronized = False
                elif isinstance(statement, tir.Evaluate) and isinstance(
                    statement.value, tir.Call
                ):
                    call = statement.value
                    if call.op.same_as(sync_op) and call.args[0].value.startswith(
                        "shared"
                    ):
                        synchronized = True
                if loads and not stores and pending_shared_write:
                    producer_consumer_edges.append((len(loads), synchronized))
                    pending_shared_write = False

    # Statistics and pooled scoring each read the shared Q/K tiles once per loop.
    assert len(producer_consumer_edges) == 2, producer_consumer_edges
    assert all(count == 2 and synced for count, synced in producer_consumer_edges), (
        producer_consumer_edges
    )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
