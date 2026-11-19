"""HiSparse PP target allocation and sizing must agree on index-K producers.

CPU-only layout tests; these do not establish GPU/PD generation parity.
"""

import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.srt.arg_groups.hisparse_hook import HISPARSE_VALIDATED_SPEC_ALGORITHMS
from sglang.srt.mem_cache import kv_cache_configurator as kcc
from sglang.srt.mem_cache.hisparse_memory_pool import HiSparseDSATokenToKVPool
from sglang.srt.model_executor.pool_configurator import DefaultPoolConfigurator
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

# GLM-5.2: three initial producers, then one producer every four layers.
PRODUCERS = {0, 1, *range(2, 78, 4)}
PP_STAGES = [
    # Actual no-MTP PP8 benchmark partition: 10x7 + 8.
    (0, 10),
    (10, 20),
    (20, 30),
    (30, 40),
    (40, 50),
    (50, 60),
    (60, 70),
    (70, 78),
    # Also retain the earlier uneven partition as a boundary regression.
    (0, 12),
    (12, 22),
    (22, 32),
    (32, 42),
    (42, 52),
    (52, 62),
    (62, 72),
    (72, 78),
]


@pytest.mark.parametrize("start,end", [*PP_STAGES, (3, 6)])
@pytest.mark.parametrize("ratio", [2, 16, 32])
@pytest.mark.parametrize("shared", [False, True])
def test_pp_pool_mask_allocation_and_budget(start, end, ratio, shared):
    """Build each stage's real HiSparse pool on CPU with a small logical pool;
    its index-K allocation and the sizing must count the same producers."""
    n = end - start
    types = ["full" if i in PRODUCERS or not shared else "shared" for i in range(78)]
    hf = SimpleNamespace(
        architectures=["GlmMoeDsaForCausalLM"],
        index_topk=2048,
        index_head_dim=128,
        indexer_types=types,
    )
    with get_context().override_server_args(
        enable_hisparse=True,
        hisparse_config=f'{{"host_to_device_ratio": {ratio}}}',
        speculative_algorithm=None,
        enable_dsa_cache_layer_split=False,
        enable_unified_cache_external_linker=False,
        disaggregation_mode="decode",
        page_size=64,
    ):
        kvc = SimpleNamespace(
            is_draft_worker=False,
            loc_space_scale=1,
            device="cpu",
            kv_cache_dtype=torch.float8_e4m3fn,
            pool_page_size=64,
            mambaish_config=None,
            kv_shard_rank=None,
            model_config=SimpleNamespace(
                hf_config=hf,
                kv_lora_rank=512,
                qk_rope_head_dim=64,
            ),
            layer_info=SimpleNamespace(
                start_layer=start,
                end_layer=end,
                num_effective_layers=n,
            ),
        )
        with (
            patch.object(kcc, "_is_npu", False),
            patch.object(kcc, "calculate_mla_kv_cache_dim", return_value=656),
            patch.object(kcc, "reject_out_of_tree_path"),
        ):
            pool = kcc.KVCacheConfigurator._build_dsa_kv_pool(
                kvc,
                max_total_num_tokens=128,
                max_running_requests=8,
                dsa_pool_class=object,
            )
            cell = DefaultPoolConfigurator._compute_dsa_indexer_cell_size(
                None,
                kvc=kvc,
                num_layers=n,
            )
        expected = [types[i] == "shared" for i in range(start, end)]
        assert isinstance(pool, HiSparseDSATokenToKVPool)
        assert pool.skip_topk_layers == expected
        assert len(pool.kv_buffer) == n  # MLA KV still has every layer.
        producer_count = expected.count(False)
        assert cell == producer_count * 132 * ratio

        logical_tokens = 128 * ratio
        assert pool.index_buf_size == logical_tokens
        pages = (logical_tokens + 65) // 64
        buffers = pool.index_k_with_scale_buffer
        # Zero-row placeholders retain layer IDs.
        assert [tuple(buf.shape) for buf in buffers] == [
            (0 if skip else pages, 64 * 132) for skip in expected
        ]
        assert sum(buf.nbytes for buf in buffers) == producer_count * pages * 64 * 132
        _, lengths, strides = pool.get_state_buf_infos()
        assert lengths == [0 if skip else pages * 64 * 132 for skip in expected]
        assert strides == [0 if skip else 64 * 132 for skip in expected]


@pytest.mark.parametrize("algorithm", [None, *HISPARSE_VALIDATED_SPEC_ALGORITHMS])
@pytest.mark.parametrize("draft", [False, True])
def test_speculative_and_draft_paths_remain_dense(algorithm, draft):
    with (
        get_context().override_server_args(
            enable_hisparse=True,
            speculative_algorithm=algorithm,
            disaggregation_mode="decode",
            enable_unified_cache_external_linker=False,
        ),
        patch.object(kcc, "_is_npu", False),
    ):
        assert kcc._should_elide_dsa_index_k(is_draft_worker=draft) == (
            algorithm is None and not draft
        )


def test_external_linker_guard_is_preserved():
    with get_context().override_server_args(
        enable_hisparse=True,
        speculative_algorithm=None,
        enable_unified_cache_external_linker=True,
    ):
        assert not kcc._should_elide_dsa_index_k(is_draft_worker=False)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
