from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.srt.managers.schedule_policy import AddReqResult, PrefillAdder
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def test_zero_chunk_budget_rejects_before_storage_restore():
    tree_cache = MagicMock()
    tree_cache.disable = False
    tree_cache.evictable_size.return_value = 0
    tree_cache.is_tree_cache.return_value = False

    allocator = MagicMock()
    allocator.available_size.return_value = 4096

    adder = PrefillAdder.__new__(PrefillAdder)
    adder.page_size = 4
    adder.tree_cache = tree_cache
    adder.token_to_kv_pool_allocator = allocator
    adder.running_batch = None
    adder.prefill_delayer_single_pass = None
    adder.dsa_prefill_cp_in_seq_split = False
    adder.prefill_max_requests = None
    adder.can_run_list = []
    adder.dllm_config = None
    adder.rem_chunk_tokens = 0
    adder.rem_input_tokens = 4096
    adder.rem_total_token_offset = 0
    adder.cur_rem_token_offset = 0
    adder.is_hybrid_swa = False
    adder.is_all_swa = False
    adder.is_hybrid_ssm_cache = False
    adder._mamba_slot_cost = 0
    adder.rem_mamba_slots = None

    req = SimpleNamespace(
        sampling_params=SimpleNamespace(max_new_tokens=16, ignore_eos=False),
        output_ids=[],
        full_untruncated_fill_ids=list(range(12)),
        prefix_indices=torch.arange(4, dtype=torch.int64),
        host_hit_length=4,
        last_node=object(),
        retracted_stain=False,
        needs_host_load_back=lambda: True,
    )

    result = adder.add_one_req(
        req,
        has_chunked_req=False,
        truncation_align_size=None,
    )

    assert result is AddReqResult.OTHER
    tree_cache.init_load_back.assert_not_called()
    assert adder.can_run_list == []
