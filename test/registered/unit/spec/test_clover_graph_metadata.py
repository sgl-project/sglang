"""Clover graph metadata must refresh live slots without replacing captured views."""

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.attention.linear.clover_backend import CloverGDNAttnBackend
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def make_backend(device="cpu", block_size=4):
    backend = object.__new__(CloverGDNAttnBackend)
    backend.device = device
    backend.pad_slot_id = -1
    backend.topk = 1
    backend._fused_state_indices_ok = False
    backend._graph_seq_len_fill_value = 1
    mapping = torch.tensor([0, 2, 1, 3], device=device, dtype=torch.int32)
    physical = torch.tensor([0, 3, 2, 1], device=device, dtype=torch.int32)
    backend.req_to_token_pool = SimpleNamespace(
        get_mamba_indices=lambda reqs: mapping[reqs.long()],
        translate_mamba_indices=lambda slots: physical[slots.long()],
        mamba_pool=SimpleNamespace(size=4, enable_linear_replayssm=False),
    )
    for name in (
        "state_indices_list",
        "query_start_loc_list",
        "retrieve_next_token_list",
        "retrieve_next_sibling_list",
        "retrieve_parent_token_list",
    ):
        setattr(backend, name, [])
    backend.init_cuda_graph_state(3, 3 * block_size)
    return backend, mapping, physical


@pytest.mark.parametrize("block_size", [1, 4, 32])
def test_replay_refreshes_reordered_restored_and_padded_state_slots(block_size):
    backend, mapping, physical = make_backend(block_size=block_size)
    captured = backend._capture_metadata(
        3, torch.tensor([1, 2, 3]), ForwardMode.DLLM_EXTEND, None
    )
    indices, offsets = captured.mamba_cache_indices, captured.query_start_loc
    pointers = indices.data_ptr(), offsets.data_ptr()
    assert indices.tolist() == [2, 3, 1]
    assert offsets.tolist() == [i * block_size for i in range(4)]

    # Reorder, early finish, then restore a radix prefix into different physical
    # state storage. The same captured views must observe all three updates.
    cases = [
        ([2, 1, 3], [132, 68, 36], 0, [3, 2, 1]),
        ([3, 1, 2], [68, 36, 1], 1, [1, 2, -1]),
        ([1, 3, 2], [132, 1, 1], 2, [2, -1, -1]),
    ]
    for reqs, lengths, padding, expected in cases:
        replay = backend._replay_metadata(
            3,
            torch.tensor(reqs),
            ForwardMode.DLLM_EXTEND,
            None,
            torch.tensor(lengths),
            num_padding=padding,
        )
        assert indices.tolist() == expected
        assert pointers == (
            replay.mamba_cache_indices.data_ptr(),
            replay.query_start_loc.data_ptr(),
        )
        # Padded rows retain block-wide offsets; the -1 state sentinel masks them.
        assert offsets.tolist() == [i * block_size for i in range(4)]
    mapping[1] = 1
    physical[1] = 2
    replay = backend._replay_metadata(
        3,
        torch.tensor([1, 2, 3]),
        ForwardMode.DLLM_EXTEND,
        None,
        torch.tensor([260, 132, 68]),
        num_padding=0,
        mamba_track_indices=torch.tensor([1, 2, 3]),
    )
    assert indices.tolist() == [2, 2, 1]
    assert replay.mamba_track_indices.tolist() == [2, 2, 1]
    assert indices.data_ptr() == pointers[0]
    # Each graph bucket owns its views; replaying bs=1 cannot corrupt bs=3.
    backend._capture_metadata(1, torch.tensor([3]), ForwardMode.DLLM_EXTEND, None)
    assert indices.tolist() == [2, 2, 1]


@pytest.mark.parametrize("ragged", [False, True])
@pytest.mark.parametrize(
    "mode,causal", [(ForwardMode.EXTEND, True), (ForwardMode.DLLM_EXTEND, False)]
)
def test_flashinfer_context_and_preview_have_distinct_attention_masks(
    monkeypatch, ragged, mode, causal
):
    from unittest.mock import Mock

    from sglang.srt.layers.attention.flashinfer_backend import FlashInferAttnBackend
    from sglang.srt.layers.radix_attention import AttentionType

    backend = object.__new__(FlashInferAttnBackend)
    backend.is_dllm_model = True
    backend.prefill_uses_dequant_workspace = False
    backend.kv_cache_quant_method = SimpleNamespace(needs_global_scale=lambda: False)
    backend._get_wrapper_idx = lambda layer: 0
    wrapper = SimpleNamespace(forward=Mock(return_value=torch.zeros(4, 1, 8)))
    backend.prefill_wrapper_ragged = wrapper
    backend.forward_metadata = SimpleNamespace(
        prefill_wrappers=[wrapper],
        use_ragged=ragged,
        extend_no_prefix=True,
        multi_item_params=None,
        swa_out_cache_loc=None,
    )
    backend.token_to_kv_pool = Mock()
    from sglang.srt.layers import radix_attention
    from sglang.srt.models.clover_moe import CloverAttentionDecoderLayer

    qkv = torch.zeros(4, 8)
    model_layer = object.__new__(CloverAttentionDecoderLayer)
    torch.nn.Module.__init__(model_layer)
    model_layer.attn = radix_attention.RadixAttention(
        1, 8, 8**-0.5, num_kv_heads=1, layer_id=0
    )
    model_layer.attn_preview = radix_attention.RadixAttention(
        1, 8, 8**-0.5, num_kv_heads=1, layer_id=0, attn_type=AttentionType.ENCODER_ONLY
    )
    model_layer.attn_output_gate = False
    model_layer._prepare_qkv_gate = lambda **kw: (qkv, qkv, qkv, None)
    model_layer.o_proj = lambda output: (output, None)
    monkeypatch.setattr(radix_attention, "get_attn_backend", lambda: backend)
    forward_batch = SimpleNamespace(
        forward_mode=mode,
        global_num_token_non_padded_cpu=4,
        out_cache_loc=torch.arange(4),
        out_cache_loc_is_physical=False,
        swa_out_cache_loc=None,
    )
    model_layer.self_attention(None, qkv, forward_batch)
    assert model_layer.attn.attn_type == AttentionType.DECODER
    assert model_layer.attn_preview.attn_type == AttentionType.ENCODER_ONLY
    assert wrapper.forward.call_args.kwargs["causal"] is causal


@pytest.mark.parametrize("mirrored", [False, True])
def test_native_prefill_checkpoint_reads_intermediate_state(mirrored):
    """A 235-token prefill donates depth 192 without replacing its live final state."""
    backend = object.__new__(CloverGDNAttnBackend)
    backend.device = "cpu"
    backend._mamba_chunk_size = 64
    backend.conv_states_shape = (4, 2, 3)
    batch = SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        batch_size=2,
        mamba_track_mask=torch.tensor([True, False]),
        extend_seq_lens=torch.tensor([235, 32]),
        extend_prefix_lens=torch.tensor([0, 235]),
        mamba_track_indices=torch.tensor([2, 3]),
        mamba_track_seqlens=torch.tensor([235, -1]),
        mamba_prefill_track_mask_cpu=[True, False] if mirrored else None,
        mamba_track_seqlens_cpu=[235, -1] if mirrored else None,
        extend_seq_lens_cpu=[235, 32],
        extend_prefix_lens_cpu=[0, 235],
        mamba_track_aligned_lens=lambda: torch.tensor([192, 0]),
    )
    plan = backend._init_track_ssm_indices(torch.tensor([0, 1]), batch)
    assert plan[1].tolist() == [3]  # start of the fourth 64-token kernel chunk
    assert plan[2].tolist() == [2]
    assert plan[4].numel() == 0
    indices = backend._init_track_conv_indices(torch.tensor([0, 235, 267]), batch)
    assert indices.tolist() == [[189, 190, 191]]
    state = torch.arange(16).view(4, 2, 2).float()
    original = state.clone()
    h = torch.arange(20).view(1, 5, 2, 2).float() + 100
    metadata = SimpleNamespace(
        has_mamba_track_mask=True,
        track_ssm_h_src=plan[1],
        track_ssm_h_dst=plan[2],
        track_ssm_recompute_dst=None,
        track_ssm_final_src=plan[4],
        track_ssm_final_dst=plan[5],
    )
    backend._track_mamba_state_extend(batch, h, state, metadata)
    torch.testing.assert_close(state[2], h[0, 3])
    torch.testing.assert_close(state[:2], original[:2])
    torch.testing.assert_close(state[3], original[3])


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
