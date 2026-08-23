import pytest
import torch

from sglang.kernels.ops.quantization.nvfp4_kv_cache import (
    dequantize_nvfp4_kv_for_speculative_extend,
)
from sglang.srt.layers.quantization.kvfp4_tensor import E2M1_VALUES
from sglang.srt.utils import is_sm100_supported, is_sm120_supported
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")


def _dequantize_reference(fp4_data, block_scales, global_scale):
    unpacked = torch.empty(
        *fp4_data.shape[:-1],
        fp4_data.shape[-1] * 2,
        dtype=torch.uint8,
        device=fp4_data.device,
    )
    unpacked[..., 0::2] = fp4_data & 0xF
    unpacked[..., 1::2] = (fp4_data >> 4) & 0xF
    values = torch.tensor(E2M1_VALUES, dtype=torch.float32, device=fp4_data.device)
    return (
        (
            values[unpacked.long()]
            * block_scales.float().repeat_interleave(16, dim=-1)
            * global_scale
        )
        .to(torch.bfloat16)
        .to(torch.float8_e4m3fn)
    )


@pytest.mark.skipif(
    not (is_sm100_supported() or is_sm120_supported()),
    reason="NVFP4 KV cache requires SM100 or SM120",
)
@pytest.mark.parametrize(
    "prefix_len_delta",
    [pytest.param(0, id="target_verify"), pytest.param(4, id="draft_extend")],
)
def test_speculative_workspace_graph_replay_uses_runtime_lengths(prefix_len_delta):
    torch.manual_seed(7)
    device = "cuda"
    size, heads, head_dim = 64, 4, 128
    real_batch_size, graph_batch_size = 2, 3
    current_tokens_per_req, max_context = 4, 16

    k_fp4 = torch.randint(
        0, 256, (size, heads, head_dim // 2), dtype=torch.uint8, device=device
    )
    v_fp4 = torch.randint(
        0, 256, (size, heads, head_dim // 2), dtype=torch.uint8, device=device
    )
    k_block_scales = (torch.rand(size, heads, head_dim // 16, device=device) * 0.2).to(
        torch.float8_e4m3fn
    )
    v_block_scales = (torch.rand(size, heads, head_dim // 16, device=device) * 0.2).to(
        torch.float8_e4m3fn
    )
    k_global_scale = torch.tensor([0.02], dtype=torch.float32, device=device)
    v_global_scale = torch.tensor([0.03], dtype=torch.float32, device=device)

    req_to_token = torch.zeros(
        real_batch_size, max_context, dtype=torch.int32, device=device
    )
    req_to_token[0] = torch.arange(2, 18, dtype=torch.int32, device=device)
    req_to_token[1] = torch.arange(30, 46, dtype=torch.int32, device=device)
    # The third row models CUDA Graph padding: graph buffers use request row 0
    # and reserved KV slot 0 for padded requests.
    req_pool_indices = torch.tensor([0, 1, 0], dtype=torch.int32, device=device)
    sequence_lens = torch.tensor(
        [1 + prefix_len_delta, 2 + prefix_len_delta, 1],
        dtype=torch.int32,
        device=device,
    )

    # QKV splits can have a larger per-token stride. Exercise that layout so
    # target verify does not add a contiguous() allocation inside the graph.
    num_current_rows = graph_batch_size * current_tokens_per_req
    k_current_storage = torch.randn(
        num_current_rows,
        heads * head_dim + 13,
        dtype=torch.bfloat16,
        device=device,
    )
    v_current_storage = torch.randn(
        num_current_rows,
        heads * head_dim + 17,
        dtype=torch.bfloat16,
        device=device,
    )
    k_current = k_current_storage[:, : heads * head_dim].view(
        num_current_rows, heads, head_dim
    )
    v_current = v_current_storage[:, : heads * head_dim].view(
        num_current_rows, heads, head_dim
    )
    assert not k_current.is_contiguous()
    assert not v_current.is_contiguous()

    dq_k = torch.full(
        (size, heads, head_dim),
        float("nan"),
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    dq_v = torch.full_like(dq_k, float("nan"))

    def make_current_locs(real_prefix_lens):
        locs = torch.zeros(
            graph_batch_size,
            current_tokens_per_req,
            dtype=torch.int32,
            device=device,
        )
        for req_offset, prefix_len in enumerate(real_prefix_lens):
            locs[req_offset].copy_(
                req_to_token[
                    req_offset,
                    prefix_len : prefix_len + current_tokens_per_req,
                ]
            )
        return locs.flatten()

    current_locs = make_current_locs([1, 2])

    def run_kernel():
        dequantize_nvfp4_kv_for_speculative_extend(
            k_fp4,
            v_fp4,
            k_block_scales,
            v_block_scales,
            k_global_scale,
            v_global_scale,
            k_current,
            v_current,
            dq_k,
            dq_v,
            req_to_token,
            req_pool_indices,
            sequence_lens,
            current_locs,
            current_tokens_per_req,
            prefix_len_delta,
        )

    # Compile before capture, then capture with short prefixes.
    run_kernel()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run_kernel()

    # Replay the same graph with different device-resident prefix lengths. If
    # capture-time host values were baked in, slots 3..7 would remain NaN.
    dq_k.fill_(float("nan"))
    dq_v.fill_(float("nan"))
    replay_prefix_lens = [6, 7]
    sequence_lens.copy_(
        torch.tensor(
            [
                *(length + prefix_len_delta for length in replay_prefix_lens),
                1,
            ],
            device=device,
        )
    )
    current_locs.copy_(make_current_locs(replay_prefix_lens))
    graph.replay()
    torch.cuda.synchronize()

    k_prefix_reference = _dequantize_reference(k_fp4, k_block_scales, k_global_scale)
    v_prefix_reference = _dequantize_reference(v_fp4, v_block_scales, v_global_scale)
    for req_offset, prefix_len in enumerate(replay_prefix_lens):
        prefix_slots = req_to_token[req_offset, :prefix_len].long()
        torch.testing.assert_close(
            dq_k[prefix_slots].float(),
            k_prefix_reference[prefix_slots].float(),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            dq_v[prefix_slots].float(),
            v_prefix_reference[prefix_slots].float(),
            rtol=0,
            atol=0,
        )

        current_slots = req_to_token[
            req_offset,
            prefix_len : prefix_len + current_tokens_per_req,
        ].long()
        current_rows = slice(
            req_offset * current_tokens_per_req,
            (req_offset + 1) * current_tokens_per_req,
        )
        torch.testing.assert_close(
            dq_k[current_slots].float(),
            k_current[current_rows].to(torch.float8_e4m3fn).float(),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            dq_v[current_slots].float(),
            v_current[current_rows].to(torch.float8_e4m3fn).float(),
            rtol=0,
            atol=0,
        )
