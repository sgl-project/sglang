"""Small RoPE configuration for real Indexer/backend tests."""


def rotary_test_config():
    """Small-table tests use the same frequencies as the checkpoint fixture."""
    from types import SimpleNamespace

    return SimpleNamespace(
        head_dim=192,
        index_head_dim=256,
        layer_types_with_mtp=["sliding_attention", "full_attention"],
        rope_theta=[10000, 10000000],
        partial_rotary_factors=[1.0, 1 / 3],
        sparse_config={"sparse_indexer_rope_dim": 32},
        rope_scaling={
            "rope_type": "llama3",
            "factor": 1.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 32.0,
            "original_max_position_embeddings": 1048576,
        },
    )
