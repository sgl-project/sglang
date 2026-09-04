import math

import pytest
import torch

from sglang.multimodal_gen.runtime.layers.attention.backends import (
    video_sparse_attn as vsa_module,
)
from sglang.multimodal_gen.runtime.layers.attention.backends.video_sparse_attn import (
    VSA_TILE_SIZE,
    VideoSparseAttentionImpl,
    VideoSparseAttentionMetadataBuilder,
    _CakePlanSignature,
    _compute_cur_topk,
    _validate_cake_q2k_indices,
)


def test_video_sparse_attention_tile_buffer_reuse_and_untile():
    metadata = VideoSparseAttentionMetadataBuilder().build(
        current_timestep=0,
        raw_latent_shape=(5, 7, 9),
        patch_size=(1, 1, 1),
        VSA_sparsity=0.5,
        device=torch.device("cpu"),
    )

    impl = object.__new__(VideoSparseAttentionImpl)
    total_seq_length = metadata.total_seq_length
    x = torch.arange(2 * total_seq_length * 3 * 4, dtype=torch.float32).reshape(
        2, total_seq_length, 3, 4
    )

    tiled = impl.preprocess_qkv(x, metadata)
    assert metadata.tile_buf is tiled
    assert torch.equal(
        metadata.untile_combined_index,
        metadata.non_pad_index[metadata.reverse_tile_partition_indices],
    )
    assert torch.equal(impl.postprocess_output(tiled, metadata), x)

    next_x = x + 1
    next_tiled = impl.preprocess_qkv(next_x, metadata)
    assert next_tiled.data_ptr() == tiled.data_ptr()
    assert torch.equal(impl.postprocess_output(next_tiled, metadata), next_x)

    pad_mask = torch.ones(next_tiled.shape[1], dtype=torch.bool)
    pad_mask[metadata.non_pad_index.cpu()] = False
    assert torch.all(next_tiled[:, pad_mask] == 0)


def test_vsa_forward_cur_topk_uses_padded_kv_block_count(monkeypatch):
    metadata = VideoSparseAttentionMetadataBuilder().build(
        current_timestep=0,
        raw_latent_shape=(5, 32, 32),
        patch_size=(1, 1, 1),
        VSA_sparsity=0.75,
        device=torch.device("cpu"),
    )
    num_kv_blocks = metadata.variable_block_sizes.numel()
    block_elements = math.prod(VSA_TILE_SIZE)
    padded_seq_len = num_kv_blocks * block_elements
    expected_topk = math.ceil((1 - metadata.VSA_sparsity) * num_kv_blocks)
    unpadded_topk = math.ceil(
        (1 - metadata.VSA_sparsity) * (metadata.total_seq_length / block_elements)
    )
    captured = {}

    def fake_video_sparse_attn(
        query,
        key,
        value,
        variable_block_sizes,
        topk,
        block_size,
        compress_attn_weight,
    ):
        captured["topk"] = topk
        captured["block_size"] = block_size
        assert torch.equal(variable_block_sizes, metadata.variable_block_sizes)
        return query

    monkeypatch.setattr(vsa_module, "video_sparse_attn", fake_video_sparse_attn)

    query = torch.ones(1, padded_seq_len, 1, 1)
    impl = object.__new__(VideoSparseAttentionImpl)
    impl.stage2_backend = "vsa"
    output = impl.forward(query, query, query, query, metadata)

    assert unpadded_topk < expected_topk
    assert captured["topk"] == expected_topk
    assert captured["block_size"] == VSA_TILE_SIZE
    assert output.shape == query.shape


def test_vsa_cur_topk_clamps_to_valid_block_range():
    metadata = VideoSparseAttentionMetadataBuilder().build(
        current_timestep=0,
        raw_latent_shape=(5, 32, 32),
        patch_size=(1, 1, 1),
        VSA_sparsity=1.0,
        device=torch.device("cpu"),
    )
    num_kv_blocks = metadata.variable_block_sizes.numel()

    assert _compute_cur_topk(metadata) == 1

    metadata.VSA_sparsity = -0.01
    assert _compute_cur_topk(metadata) == num_kv_blocks


def test_cake_stage2_uses_direct_topk_metadata(monkeypatch):
    metadata = VideoSparseAttentionMetadataBuilder().build(
        current_timestep=0,
        raw_latent_shape=(2, 8, 8),
        patch_size=(1, 1, 1),
        VSA_sparsity=0.5,
        device=torch.device("cpu"),
    )
    num_blocks = metadata.variable_block_sizes.numel()
    expected_topk = _compute_cur_topk(metadata)
    sequence = num_blocks * math.prod(VSA_TILE_SIZE)
    query = torch.randn(1, sequence, 1, 128, dtype=torch.bfloat16)
    gate = torch.zeros_like(query)
    captured = {}

    def fake_plan_cake_vsa(*args, **kwargs):
        captured["plan_args"] = args
        captured["plan_kwargs"] = kwargs
        return {
            "q2k_indices": kwargs["q2k_indices"],
            "q2k_num": kwargs["q2k_num"],
            "workspace": {},
        }

    def fake_run_cake_vsa(plan, q, k, v, **kwargs):
        captured["run_plan"] = plan
        captured["run_shapes"] = (q.shape, k.shape, v.shape)
        captured["run_kwargs"] = kwargs
        return torch.zeros_like(q)

    monkeypatch.setattr(vsa_module, "_validate_cake_inputs", lambda *_args: None)
    monkeypatch.setattr(vsa_module, "_plan_cake_vsa", fake_plan_cake_vsa)
    monkeypatch.setattr(vsa_module, "_run_cake_vsa", fake_run_cake_vsa)
    monkeypatch.setattr(vsa_module, "_cake_stream_key", lambda _device: (0, 1))

    impl = object.__new__(VideoSparseAttentionImpl)
    impl.stage2_backend = "cake"
    impl.cake_step_indices = None
    impl.head_size = 128
    impl.softmax_scale = 128**-0.5
    impl._cake_q2k_num = {}
    impl._cake_plan_templates = {}
    impl._cake_plan_workspaces = {}
    output = impl.forward(query, query, query, gate, metadata)

    q2k_indices = captured["plan_kwargs"]["q2k_indices"]
    q2k_num = captured["plan_kwargs"]["q2k_num"]
    assert q2k_indices.dtype == torch.int32
    assert q2k_indices.is_contiguous()
    assert q2k_indices.shape == (1, num_blocks, expected_topk)
    assert torch.all(q2k_indices[..., 1:] >= q2k_indices[..., :-1])
    assert torch.equal(
        q2k_num,
        torch.full((1, num_blocks), expected_topk, dtype=torch.int32),
    )
    assert captured["plan_kwargs"]["kv_block_lens"] is metadata.variable_block_sizes
    assert captured["run_shapes"] == (
        torch.Size((sequence, 1, 128)),
        torch.Size((sequence, 1, 128)),
        torch.Size((sequence, 1, 128)),
    )
    assert captured["run_kwargs"] == {
        "out": None,
        "lse": None,
        "return_lse": False,
        "backend": "cake",
    }
    assert captured["run_plan"]["q2k_indices"] is q2k_indices
    assert captured["run_plan"]["workspace"] == {}
    assert torch.count_nonzero(output) == 0


def test_cake_plan_cache_uses_dynamic_q2k_and_stream_local_workspace(monkeypatch):
    metadata = VideoSparseAttentionMetadataBuilder().build(
        current_timestep=0,
        raw_latent_shape=(2, 8, 8),
        patch_size=(1, 1, 1),
        VSA_sparsity=0.5,
        device=torch.device("cpu"),
    )
    num_blocks = metadata.variable_block_sizes.numel()
    sequence = num_blocks * math.prod(VSA_TILE_SIZE)
    query = torch.randn(1, sequence, 1, 128, dtype=torch.bfloat16)
    gate = torch.zeros_like(query)
    topk_two_a = torch.tensor(
        [[[[3, 1], [2, 0], [1, 0], [3, 2]]]], dtype=torch.int64
    ).reshape(1, 1, num_blocks, 2)
    topk_two_b = torch.tensor(
        [[[[2, 1], [3, 1], [3, 0], [2, 0]]]], dtype=torch.int64
    ).reshape(1, 1, num_blocks, 2)
    topk_one = torch.tensor([[[[0], [1], [2], [3]]]], dtype=torch.int64).reshape(
        1, 1, num_blocks, 1
    )
    pending_topk = iter((topk_two_a, topk_two_b, topk_one, topk_two_b))
    plan_calls = []
    run_plans = []
    stream_keys = iter(((0, 11), (0, 22), (0, 11), (0, 11)))

    def fake_compressed_attention(
        query_hsd, _key_hsd, _value_hsd, _variable_block_sizes, topk
    ):
        indices = next(pending_topk)
        assert indices.shape[-1] == topk
        return torch.zeros_like(query_hsd), indices

    def fake_plan_cake_vsa(*args, **kwargs):
        plan_calls.append((args, kwargs))
        return {
            "q2k_indices": kwargs["q2k_indices"],
            "q2k_num": kwargs["q2k_num"],
            "static_marker": len(plan_calls),
            "workspace": {},
        }

    def fake_run_cake_vsa(plan, q, _k, _v, **_kwargs):
        run_plans.append(plan)
        return torch.zeros_like(q)

    monkeypatch.setattr(vsa_module, "_validate_cake_inputs", lambda *_args: None)
    monkeypatch.setattr(vsa_module, "_compressed_attention", fake_compressed_attention)
    monkeypatch.setattr(vsa_module, "_plan_cake_vsa", fake_plan_cake_vsa)
    monkeypatch.setattr(vsa_module, "_run_cake_vsa", fake_run_cake_vsa)
    monkeypatch.setattr(
        vsa_module, "_cake_stream_key", lambda _device: next(stream_keys)
    )

    impl = object.__new__(VideoSparseAttentionImpl)
    impl.head_size = 128
    impl.softmax_scale = 128**-0.5
    impl._cake_q2k_num = {}
    impl._cake_plan_templates = {}
    impl._cake_plan_workspaces = {}

    impl._forward_cake(query, query, query, gate, metadata, cur_topk=2)
    impl._forward_cake(query, query, query, gate, metadata, cur_topk=2)

    assert len(plan_calls) == 1
    assert len(run_plans) == 2
    assert run_plans[0] is not run_plans[1]
    assert torch.equal(
        run_plans[0]["q2k_indices"], topk_two_a[0].sort(-1).values.to(torch.int32)
    )
    assert torch.equal(
        run_plans[1]["q2k_indices"], topk_two_b[0].sort(-1).values.to(torch.int32)
    )
    assert run_plans[0]["q2k_indices"] is not run_plans[1]["q2k_indices"]
    assert run_plans[0]["workspace"] is not run_plans[1]["workspace"]
    template = next(iter(impl._cake_plan_templates.values()))
    assert "q2k_indices" not in template.static_fields
    assert "workspace" not in template.static_fields
    with pytest.raises(TypeError):
        template.static_fields["mutate"] = True  # type: ignore[index]

    impl._forward_cake(query, query, query, gate, metadata, cur_topk=1)

    assert len(plan_calls) == 2
    assert len(impl._cake_plan_templates) == 2
    assert run_plans[2]["workspace"] is not run_plans[0]["workspace"]

    impl._forward_cake(query, query, query, gate, metadata, cur_topk=2)

    assert len(plan_calls) == 2
    assert run_plans[3]["workspace"] is run_plans[0]["workspace"]
    assert run_plans[3]["q2k_indices"] is not run_plans[1]["q2k_indices"]


def test_validate_cake_q2k_indices_checks_dynamic_tensor_contract():
    signature = _CakePlanSignature(
        sequence=256,
        num_heads=2,
        head_dim=128,
        topk=2,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
        dit_seq_shape=(2, 8, 8),
        sm_scale=128**-0.5,
    )
    valid = torch.zeros((2, 4, 2), dtype=torch.int32)
    _validate_cake_q2k_indices(valid, signature)

    with pytest.raises(ValueError, match="contiguous int32"):
        _validate_cake_q2k_indices(valid.to(torch.int64), signature)
    with pytest.raises(ValueError, match="contiguous int32"):
        _validate_cake_q2k_indices(
            torch.zeros((2, 2, 4), dtype=torch.int32).transpose(1, 2), signature
        )


def test_cake_stage2_step_selection_uses_native_for_excluded_step(monkeypatch):
    metadata = VideoSparseAttentionMetadataBuilder().build(
        current_timestep=2,
        raw_latent_shape=(2, 8, 8),
        patch_size=(1, 1, 1),
        VSA_sparsity=0.5,
        device=torch.device("cpu"),
    )
    sequence = metadata.variable_block_sizes.numel() * math.prod(VSA_TILE_SIZE)
    query = torch.ones(1, sequence, 1, 128, dtype=torch.bfloat16)
    captured = {}

    def fake_video_sparse_attn(query, key, value, **kwargs):
        captured["topk"] = kwargs["topk"]
        return query

    monkeypatch.setattr(vsa_module, "video_sparse_attn", fake_video_sparse_attn)
    impl = object.__new__(VideoSparseAttentionImpl)
    impl.stage2_backend = "cake"
    impl.cake_step_indices = frozenset({0, 1})

    output = impl.forward(query, query, query, query, metadata)

    assert captured["topk"] == _compute_cur_topk(metadata)
    assert torch.equal(output, query)
