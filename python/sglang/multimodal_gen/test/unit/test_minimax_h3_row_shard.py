# SPDX-License-Identifier: Apache-2.0
"""MiniMax-H3's row-sharded block stack against the all-reduce path.

The row-sharded path is only worth having because it is bit-exact, so every
assertion on block outputs here is on bits, not tolerances. The two-GPU test
runs a block-FP8 DiT block both ways at TP=2 on the runner the path takes --
DeepGEMM with UE8M0 scales -- and needs two SM120 GPUs; the rest checks the row bookkeeping on CPU.
"""

from __future__ import annotations

import os
import socket
from types import SimpleNamespace

import pytest
import torch
import torch.multiprocessing as mp

# minimax_h3 first, the way the runtime reaches the row-shard module: importing
# the linear layers first trips an import cycle in the quantization package.
from sglang.multimodal_gen.runtime.models.dits import (  # isort: skip
    minimax_h3,
    minimax_h3_row_shard,
)


@pytest.mark.parametrize("chunks", [1, 2, 4])
def test_a_rank_holds_what_each_chunks_reduce_scatter_hands_it(chunks):
    # Reduce-scattering chunk c of the global rows gives rank r its r-th half
    # of that chunk; the shard has to be those halves in chunk order, or the
    # residual a rank carries would not be the rows its collectives produce.
    world, rows = 2, 64
    full = torch.arange(rows * 3).view(rows, 3)
    whole = rows // chunks
    half = whole // world
    for rank in range(world):
        shard = minimax_h3_row_shard.shard_rows(
            full, chunks=chunks, rank=rank, world=world
        )
        expected = torch.cat(
            [
                full[c * whole + rank * half : c * whole + (rank + 1) * half]
                for c in range(chunks)
            ]
        )
        assert torch.equal(shard, expected)


@pytest.mark.parametrize("chunks", [1, 2, 4])
def test_unsharding_restores_the_original_row_order(chunks):
    # unshard_rows reorders an all-gather of every rank's shard; stand in for
    # the collective with a concatenation in rank order.
    world, rows = 2, 64
    full = torch.arange(rows * 3).view(rows, 3)
    shards = [
        minimax_h3_row_shard.shard_rows(full, chunks=chunks, rank=rank, world=world)
        for rank in range(world)
    ]
    group = SimpleNamespace(
        world_size=world, all_gather=lambda tensor, dim: torch.cat(shards, dim=dim)
    )
    restored = minimax_h3_row_shard.unshard_rows(shards[0], chunks=chunks, group=group)
    assert torch.equal(restored, full)


def test_chunk_count_backs_off_to_what_the_rows_divide_into():
    def chunks(rows, tp_size=2):
        return minimax_h3_row_shard.row_chunks(
            column_linears=(), row_linears=(), tp_size=tp_size, rows=rows
        )

    # c chunks of two halves, each a multiple of four rows, needs 8c | rows;
    # the largest c up to four that divides is taken.
    assert chunks(64 * 7) == 4
    assert chunks(24 * 5) == 3
    assert chunks(8 * 5) == 1
    assert chunks(12) == 0
    # Only TP=2 is claimed bit-exact.
    assert chunks(64 * 7, tp_size=4) == 0


def test_layers_the_path_cannot_reproduce_keep_the_all_reduce():
    # A bf16 layer quantizes nothing for the fused kernel to reproduce, and a
    # LoRA wrapper is not a plain parallel linear.
    not_fp8 = SimpleNamespace(bias=None, quant_method=object())
    wrapper = torch.nn.Module()
    for column_linears, row_linears in (((not_fp8,), ()), ((), (wrapper,))):
        assert (
            minimax_h3_row_shard.row_chunks(
                column_linears=column_linears,
                row_linears=row_linears,
                tp_size=2,
                rows=64,
            )
            == 0
        )


# --------------------------------------------------------------------------
# Two GPUs: a block-FP8 DiT block at TP=2, row-sharded against all-reduce.
# --------------------------------------------------------------------------

_ROWS = 64 * 12
_MODALITIES = 3


def _two_sm120_gpus() -> bool:
    return (
        torch.cuda.is_available()
        and torch.cuda.device_count() >= 2
        and all(torch.cuda.get_device_capability(i)[0] == 12 for i in range(2))
    )


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _fill_block_fp8(block: torch.nn.Module, *, rank: int) -> None:
    """Random block-FP8 weights, each rank its own shard; the bf16 parameters
    (the norms) are replicated across TP ranks, as they are in the model."""
    from sglang.multimodal_gen.runtime.layers.quantization.fp8 import Fp8LinearMethod

    generator = torch.Generator("cuda").manual_seed(100 + rank)
    for module in block.modules():
        method = getattr(module, "quant_method", None)
        if not isinstance(method, Fp8LinearMethod):
            continue
        weight = torch.randn(
            module.weight.shape, device="cuda", generator=generator
        ).clamp_(-4, 4)
        module.weight.data.copy_(weight.to(torch.float8_e4m3fn))
        scale = (
            torch.rand(
                module.weight_scale_inv.shape, device="cuda", generator=generator
            )
            * 0.02
            + 0.005
        )
        # Powers of two, as sglang's own block quantizer writes them: DeepGEMM
        # on SM120 takes them as UE8M0 as they are, where the loader does not
        # requantize a checkpoint.
        module.weight_scale_inv.data.copy_(torch.exp2(torch.ceil(torch.log2(scale))))
        method.process_weights_after_loading(module)
    generator = torch.Generator("cuda").manual_seed(99)
    for param in block.parameters():
        if param.dtype == torch.bfloat16:
            with torch.no_grad():
                param.copy_(
                    1
                    + 0.1 * torch.randn(param.shape, device="cuda", generator=generator)
                )


def _row_mixing_attention(local_inner: int):
    # Attention proper is the same call on the same rows in both paths; stand
    # in with something deterministic that still reads every row.
    def attend(qkv, **_kwargs):
        q, k, v = qkv.split(local_inner, dim=-1)
        weights = torch.softmax(q.float() @ k.float().t() / 64, dim=-1)
        return (weights @ v.float()).to(qkv.dtype)

    return attend


def _two_gpu_worker(rank: int, port: int, scale_dtype: torch.dtype, failures) -> None:
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        LOCAL_RANK=str(rank),
        WORLD_SIZE="2",
    )
    try:
        torch.cuda.set_device(rank)
        from sglang.multimodal_gen.configs.models.dits.minimax_h3 import (
            MiniMaxH3DiTArchConfig,
        )
        from sglang.multimodal_gen.runtime.distributed.parallel_state import (
            maybe_init_distributed_environment_and_model_parallel,
        )
        from sglang.multimodal_gen.runtime.layers.quantization.fp8 import Fp8Config

        maybe_init_distributed_environment_and_model_parallel(tp_size=2, sp_size=1)
        arch = MiniMaxH3DiTArchConfig()
        arch.hidden_size = 1024
        arch.ffn_hidden_size = 2048
        arch.num_attention_heads = 8
        arch.attention_head_dim = 128
        quant_config = Fp8Config(
            is_checkpoint_fp8_serialized=True,
            activation_scheme="dynamic",
            weight_block_size=[128, 128],
        )
        with torch.device("cuda"):
            block = minimax_h3.MiniMaxH3DiTBlock(
                arch, quant_config, prefix="blocks.0", use_adaln_cache=True
            )
        _fill_block_fp8(block, rank=rank)
        block.attn.attend = _row_mixing_attention(block.attn.local_inner_dim)

        g = torch.Generator("cuda").manual_seed(0)
        hidden = (torch.randn(_ROWS, 1024, device="cuda", generator=g) * 2).to(
            torch.bfloat16
        )
        indices = torch.randint(0, _MODALITIES, (_ROWS,), device="cuda", generator=g)
        adaln_params = tuple(
            (0.2 * torch.randn(_MODALITIES, 1024, device="cuda", generator=g)).to(
                torch.bfloat16
            )
            for _ in range(6)
        )
        common = dict(
            adaln_input=None,
            rope_cache=None,
            cu_seqlens=torch.tensor([0, _ROWS], dtype=torch.int32, device="cuda"),
            max_seqlen=_ROWS,
            adaln_params=adaln_params,
        )
        expected = block(hidden.clone(), combined_indices=indices, **common)

        out_proj = block.attn.out_proj
        taken = minimax_h3_row_shard.prequantized_scale_dtype(block.attn.qkv_proj)
        assert taken == scale_dtype, f"runner takes {taken} scales"
        chosen = minimax_h3_row_shard.row_chunks(
            column_linears=(block.attn.qkv_proj, block.mlp.fc1),
            row_linears=(out_proj, block.mlp.fc2),
            tp_size=2,
            rows=_ROWS,
        )
        assert chosen == 4, f"the path did not engage: row_chunks gave {chosen}"
        # Layerwise offload leaves a non-resident layer a placeholder weight;
        # the decision must not read it.
        resident = block.attn.qkv_proj.weight.data
        block.attn.qkv_proj.weight.data = resident.new_empty(0)
        assert (
            minimax_h3_row_shard.prequantized_scale_dtype(block.attn.qkv_proj)
            == scale_dtype
        )
        block.attn.qkv_proj.weight.data = resident
        for chunks in (1, 2, 4):
            # The block writes its residual in place, and with one chunk a
            # shard is a view of the full rows.
            shard = lambda t: minimax_h3_row_shard.shard_rows(  # noqa: E731
                t, chunks=chunks, rank=rank, world=2
            ).clone()
            with minimax_h3_row_shard.gemms_beside_collectives(
                block.attn.qkv_proj, hidden.device
            ):
                out = block(
                    shard(hidden),
                    combined_indices=shard(indices),
                    row_chunks=chunks,
                    **common,
                )
            full = minimax_h3_row_shard.unshard_rows(
                out, chunks=chunks, group=out_proj.tp_group
            )
            torch.cuda.synchronize()
            if not torch.equal(full, expected):
                diff = (full.float() - expected.float()).abs()
                raise AssertionError(
                    f"chunks={chunks}: {int((diff > 0).sum())} elements differ, "
                    f"max {diff.max().item()}"
                )
    except BaseException as exc:  # reported to the parent
        failures.put(f"rank {rank}: {exc!r}")
        raise


def _deep_gemm_has_sm120_kernels() -> bool:
    from sglang.srt.layers.deep_gemm_wrapper import configurer

    return configurer._sm120_deep_gemm_apis_available()


@pytest.mark.skipif(not _two_sm120_gpus(), reason="needs two SM120 GPUs")
def test_block_fp8_row_sharded_block_is_bit_exact_at_tp2(monkeypatch):
    if not _deep_gemm_has_sm120_kernels():
        pytest.skip("the installed DeepGEMM has no SM120 kernels")
    # DeepGEMM wins the auto dispatch where it is installed. Read at import, so
    # it has to be in the environment the workers start with.
    monkeypatch.setenv("SGLANG_ENABLE_JIT_DEEPGEMM", "1")
    ctx = mp.get_context("spawn")
    failures = ctx.Queue()
    port = _free_port()
    procs = [
        ctx.Process(
            target=_two_gpu_worker,
            args=(rank, port, torch.int32, failures),
        )
        for rank in range(2)
    ]
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join(timeout=600)
    messages = []
    while not failures.empty():
        messages.append(failures.get())
    assert all(proc.exitcode == 0 for proc in procs) and not messages, messages
