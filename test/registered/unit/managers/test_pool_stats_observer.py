from sglang.srt.managers.scheduler_components.pool_stats_observer import PoolStats


def test_hybrid_ssm_kv_token_stats_use_mamba_capacity_pressure():
    stats = PoolStats(
        full_num_used=25,
        full_token_usage=0.25,
        full_available_size=75,
        full_evictable_size=0,
        is_hybrid_ssm=True,
        mamba_num_used=8,
        mamba_usage=0.8,
        mamba_available_size=2,
        mamba_evictable_size=0,
    )

    assert stats.get_kv_token_stats() == (25, 0.8)


def test_hybrid_ssm_kv_token_stats_keep_higher_full_kv_pressure():
    stats = PoolStats(
        full_num_used=90,
        full_token_usage=0.9,
        full_available_size=10,
        full_evictable_size=0,
        is_hybrid_ssm=True,
        mamba_num_used=4,
        mamba_usage=0.4,
        mamba_available_size=6,
        mamba_evictable_size=0,
    )

    assert stats.get_kv_token_stats() == (90, 0.9)
