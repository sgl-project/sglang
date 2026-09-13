"""Exercise native H3 QKV loading and attention with real TP2 x U2 groups.

Run with pytest, or torchrun --standalone --nproc-per-node=4 <this file>.
No checkpoint download is needed; dimensions and weight loaders are native H3.
"""

import json
import os
import subprocess
import sys

import pytest
import torch


@torch.inference_mode()
def _worker():
    from types import SimpleNamespace
    from unittest.mock import patch

    import torch.distributed as dist

    from sglang.multimodal_gen.configs.models.dits.minimax_h3 import (
        MiniMaxH3DiTArchConfig,
        MiniMaxH3DiTConfig,
    )
    from sglang.multimodal_gen.runtime.distributed.parallel_state import (
        get_tp_rank,
        get_ulysses_ctx,
        maybe_init_distributed_environment_and_model_parallel,
    )
    from sglang.multimodal_gen.runtime.layers import usp
    from sglang.multimodal_gen.runtime.layers.attention.backends.sdpa import SDPABackend
    from sglang.multimodal_gen.runtime.models.dits.minimax_h3 import (
        MiniMaxH3Attention,
        MiniMaxH3DiTModel,
        MiniMaxH3Rope,
        _rope_cos_sin_cache,
    )

    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    maybe_init_distributed_environment_and_model_parallel(
        tp_size=2, sp_size=2, ulysses_degree=2, dist_timeout=120
    )
    device = torch.device("cuda", rank)
    _, u_rank = get_ulysses_ctx()
    arch = MiniMaxH3DiTArchConfig()
    # Check model construction and RoPE ownership without allocating weights.
    for enabled, compile_enabled, expected_sharding in (
        (False, False, False),
        (True, False, True),
        (True, True, False),
    ):
        args = SimpleNamespace(
            performance_mode="memory",
            enable_torch_compile=compile_enabled,
            enable_breakable_cuda_graph=False,
            use_fsdp_inference=False,
            lora_path=None,
        )
        with (
            patch.dict(
                os.environ, {"SGLANG_MINIMAX_H3_ULYSSES_GATHER_QKV": str(int(enabled))}
            ),
            patch(
                "sglang.multimodal_gen.runtime.server_args.get_global_server_args",
                return_value=args,
            ),
            torch.device("meta"),
        ):
            model = MiniMaxH3DiTModel(MiniMaxH3DiTConfig(), hf_config={})
        assert model._use_ulysses_gather_qkv == expected_sharding
        expected_heads = 14 if expected_sharding else 28
        assert all(
            b.attn.qkv_proj.weight.shape[0] == 3 * expected_heads * 128
            for b in model.blocks
        )
        assert all(
            not b.attn._use_ulysses_gather_qkv for b in model.token_refiner.blocks
        )
        model.rope = MiniMaxH3Rope(arch.rope_inv_freq_len).to(device)
        model.rope.inv_freq.fill_(0.01)
        position_ids = torch.arange(64 * 3, device=device).view(1, 64, 3)
        cache, positions = model.build_rope_cache(position_ids, device=device)
        global_cache = _rope_cos_sin_cache(
            model.rope(position_ids), dtype=torch.bfloat16
        )
        expected_cache = (
            global_cache
            if expected_sharding
            else global_cache[u_rank * 32 : (u_rank + 1) * 32]
        )
        torch.testing.assert_close(cache, expected_cache, rtol=0, atol=0)
        assert positions.numel() == (64 if expected_sharding else 32)
        del model

    torch.manual_seed(36244)
    dense_qkv = torch.randn(3 * 56 * 128, 5376, dtype=torch.bfloat16) * 0.01
    dense_out = torch.randn(5376, 56 * 128, dtype=torch.bfloat16) * 0.01
    with torch.device(device):
        baseline = MiniMaxH3Attention(arch, None, prefix="blocks.0.attn")
        candidate = MiniMaxH3Attention(
            arch, None, prefix="blocks.0.attn", use_ulysses_gather_qkv=True
        )
    for attention in (baseline, candidate):
        attention.qkv_proj.weight.weight_loader(attention.qkv_proj.weight, dense_qkv)
        attention.out_proj.weight.weight_loader(attention.out_proj.weight, dense_out)
        attention.q_norm.weight.fill_(1)
        attention.k_norm.weight.fill_(1)
        attention._set_attention_backend(SDPABackend)
    assert baseline.qkv_proj.weight.shape == (3 * 28 * 128, 5376)
    assert candidate.qkv_proj.weight.shape == (3 * 14 * 128, 5376)
    # Independent ownership oracle: slice native [head, Q/K/V, dim, hidden].
    head_start = get_tp_rank() * 28 + u_rank * 14
    expected_weight = (
        dense_qkv.view(56, 3, 128, 5376)[head_start : head_start + 14]
        .permute(1, 0, 2, 3)
        .reshape(3 * 14 * 128, 5376)
    )
    torch.testing.assert_close(
        candidate.qkv_proj.weight.cpu(), expected_weight, rtol=0, atol=0
    )
    del dense_qkv, dense_out, expected_weight

    reports = []
    for iteration, local_rows in enumerate((16, 63, 128, 16)):
        # All TP ranks see the same rows; U ranks see distinct contiguous rows.
        torch.manual_seed(36244 + iteration)
        full_x = torch.randn(2 * local_rows, 5376, dtype=torch.bfloat16, device=device)
        x = full_x.narrow(0, u_rank * local_rows, local_rows).contiguous()
        raw = baseline.qkv_proj(x)[0]
        q, k, v = [t.view(local_rows, 28, 128) for t in raw.chunk(3, dim=-1)]
        expected = torch.cat(usp._usp_input_all_to_all_packed_qkv(q, k, v), dim=1)
        actual = usp._usp_minimax_h3_gather_project_qkv(x, candidate.qkv_proj.weight)
        # Isolate transport from GEMM shape-dependent BF16 reduction rounding.
        transport_reference = torch.empty_like(actual)
        for owner in range(2):
            torch.mm(
                full_x.narrow(0, owner * local_rows, local_rows),
                candidate.qkv_proj.weight.t(),
                out=transport_reference.narrow(0, owner * local_rows, local_rows),
            )
        torch.testing.assert_close(actual, transport_reference, rtol=0, atol=0)
        # Check both GEMM shapes against a common FP32 accumulation oracle.
        torch.backends.cuda.matmul.allow_tf32 = False
        oracle = (full_x.float() @ candidate.qkv_proj.weight.float().t()).bfloat16()
        projected_reference = expected.reshape_as(actual)
        print(
            json.dumps(
                {
                    "rank": rank,
                    "rows": local_rows,
                    "baseline_oracle_max_abs": (
                        projected_reference.float() - oracle.float()
                    )
                    .abs()
                    .max()
                    .item(),
                    "candidate_oracle_max_abs": (actual.float() - oracle.float())
                    .abs()
                    .max()
                    .item(),
                    "transport_bitwise": True,
                }
            ),
            flush=True,
        )
        # Cancellation makes elementwise relative error ill-conditioned near
        # zero. Bound normalized L2 error for each BF16 reduction independently;
        # exact ownership and transport were checked above without a tolerance.
        for projection in (actual, projected_reference):
            relative_l2 = (
                projection.float() - oracle.float()
            ).norm() / oracle.float().norm()
            assert relative_l2.item() < 0.01, relative_l2.item()
        torch.testing.assert_close(
            actual.view(2 * local_rows, 42, 128), expected, rtol=0.02, atol=0.02
        )

        frequencies = torch.randn(
            2 * local_rows, 3 * arch.rope_inv_freq_len, device=device
        )
        full_cache = _rope_cos_sin_cache(frequencies, dtype=torch.bfloat16)
        local_cache = full_cache.narrow(0, u_rank * local_rows, local_rows)
        # Includes a second packed segment, so a sequence-order error changes
        # attention boundaries as well as positional rotation.
        lengths = (0, 2 * local_rows - 5, 2 * local_rows)
        kwargs = dict(
            cu_seqlens=torch.tensor(lengths, dtype=torch.int32, device=device),
            cu_seqlens_host=lengths,
            max_seqlen=lengths[1],
            ulysses_active=True,
        )
        reference = baseline(
            x,
            rope_cache=(local_cache, torch.arange(local_rows, device=device)),
            **kwargs,
        )
        # Alternate streams and change shape to catch stale communication or
        # scratch reuse. Both calls use the actual projection/norm/RoPE path.
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            result = candidate(
                x,
                rope_cache=(full_cache, torch.arange(2 * local_rows, device=device)),
                **kwargs,
            )
        torch.cuda.current_stream().wait_stream(stream)
        torch.testing.assert_close(result, reference, rtol=0.02, atol=0.02)
        reports.append(
            {
                "rows": local_rows,
                "bitwise": torch.equal(result, reference),
                "max_abs": (result.float() - reference.float()).abs().max().item(),
            }
        )

    released = usp._release_minimax_h3_gather_qkv_staging()
    assert released > 0
    assert usp._release_minimax_h3_gather_qkv_staging() == 0
    assert not any(k[0].startswith("h3_gather_qkv") for k in usp._A2A_STAGING_BUFFERS)
    assert any(not k[0].startswith("h3_gather_qkv") for k in usp._A2A_STAGING_BUFFERS)
    print(
        json.dumps(
            {
                "rank": rank,
                "tp_rank": get_tp_rank(),
                "u_rank": u_rank,
                "cases": reports,
                "released_bytes": released,
            }
        ),
        flush=True,
    )
    dist.barrier()
    dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="requires four CUDA GPUs")
def test_minimax_h3_tp2_u2_attention():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=4",
            __file__,
        ],
        capture_output=True,
        text=True,
        timeout=240,
    )
    assert result.returncode == 0, result.stdout + result.stderr


if __name__ == "__main__":
    _worker()
