"""FlashInfer dense-MHA/GQA CP-v2 correctness tests."""

import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import requests
import torch

from sglang.srt.distributed.device_communicators.custom_all_reduce_utils import (
    update_environment_variables,
)
from sglang.srt.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    init_distributed_environment,
    initialize_model_parallel,
)
from sglang.srt.layers.cp.base import get_cp_strategy, init_cp_strategy
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    CustomTestCase,
    find_available_port,
    kill_process_tree,
    popen_launch_server,
)

register_cuda_ci(est_time=300, stage="base-b", runner_config="2-gpu-large")

NUM_GPUS = 2
HEAD_DIM = 64
DTYPE = torch.float16
MODEL_PATH = "Qwen/Qwen3-0.6B"


def _dense_causal_reference(
    q,
    key,
    value,
    prefix_len,
    num_q_heads,
    num_kv_heads,
):
    repeat = num_q_heads // num_kv_heads
    key = key.repeat_interleave(repeat, dim=1).float()
    value = value.repeat_interleave(repeat, dim=1).float()
    scores = torch.einsum("qhd,khd->hqk", q.float(), key) / math.sqrt(HEAD_DIM)
    q_positions = torch.arange(q.shape[0], device=q.device) + prefix_len
    k_positions = torch.arange(key.shape[0], device=q.device)
    causal_mask = k_positions.unsqueeze(0) > q_positions.unsqueeze(1)
    scores.masked_fill_(causal_mask.unsqueeze(0), float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    return torch.einsum("hqk,khd->qhd", probs, value).to(DTYPE)


def _build_case_fixture(
    device,
    num_q_heads,
    num_kv_heads,
    prefix_lens,
    extend_lens,
    allocator_page_size,
):
    from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

    torch.manual_seed(0)
    seq_lens = [
        prefix_len + extend_len
        for prefix_len, extend_len in zip(prefix_lens, extend_lens)
    ]
    page_counts = [
        (seq_len + allocator_page_size - 1) // allocator_page_size
        for seq_len in seq_lens
    ]
    num_pages = sum(page_counts) + 2
    pool = MHATokenToKVPool(
        size=num_pages * allocator_page_size,
        page_size=allocator_page_size,
        dtype=DTYPE,
        head_num=num_kv_heads,
        head_dim=HEAD_DIM,
        layer_num=1,
        device=device,
        enable_memory_saver=False,
        enable_alt_stream=False,
    )
    allocator = PagedTokenToKVPoolAllocator(
        size=pool.size,
        page_size=allocator_page_size,
        dtype=DTYPE,
        device=device,
        kvcache=pool,
        need_sort=False,
    )
    max_seq_len = max(seq_lens)
    req_to_token = torch.zeros(
        len(seq_lens), max_seq_len, dtype=torch.int32, device=device
    )

    q_parts = []
    extend_key_parts = []
    extend_value_parts = []
    reference_parts = []
    prefix_key_parts = []
    prefix_value_parts = []
    for req_id, (prefix_len, extend_len) in enumerate(zip(prefix_lens, extend_lens)):
        allocated_pages = []
        for _ in range(page_counts[req_id]):
            page = allocator.alloc(allocator_page_size)
            assert page is not None
            allocated_pages.append(page)
        slots = torch.cat(allocated_pages)[: seq_lens[req_id]]
        req_to_token[req_id, : seq_lens[req_id]] = slots

        q = torch.randn(extend_len, num_q_heads, HEAD_DIM, dtype=DTYPE, device=device)
        prefix_key = torch.randn(
            prefix_len, num_kv_heads, HEAD_DIM, dtype=DTYPE, device=device
        )
        prefix_value = torch.randn(
            prefix_len, num_kv_heads, HEAD_DIM, dtype=DTYPE, device=device
        )
        extend_key = torch.randn(
            extend_len, num_kv_heads, HEAD_DIM, dtype=DTYPE, device=device
        )
        extend_value = torch.randn(
            extend_len, num_kv_heads, HEAD_DIM, dtype=DTYPE, device=device
        )
        full_key = torch.cat([prefix_key, extend_key], dim=0)
        full_value = torch.cat([prefix_value, extend_value], dim=0)

        q_parts.append(q)
        extend_key_parts.append(extend_key)
        extend_value_parts.append(extend_value)
        prefix_key_parts.append(prefix_key)
        prefix_value_parts.append(prefix_value)
        reference_parts.append(
            _dense_causal_reference(
                q,
                full_key,
                full_value,
                prefix_len,
                num_q_heads,
                num_kv_heads,
            )
        )

    pool_key_cache, pool_value_cache = pool.get_kv_buffer(0)
    key_cache = torch.zeros_like(pool_key_cache)
    value_cache = torch.zeros_like(pool_value_cache)
    out_cache_locs = []
    for req_id, prefix_len in enumerate(prefix_lens):
        prefix_slots = req_to_token[req_id, :prefix_len].long()
        key_cache[prefix_slots] = prefix_key_parts[req_id]
        value_cache[prefix_slots] = prefix_value_parts[req_id]
        out_cache_locs.append(
            req_to_token[req_id, prefix_len : seq_lens[req_id]].long()
        )

    return SimpleNamespace(
        seq_lens=seq_lens,
        req_to_token=req_to_token,
        q=torch.cat(q_parts, dim=0),
        key=torch.cat(extend_key_parts, dim=0),
        value=torch.cat(extend_value_parts, dim=0),
        reference=torch.cat(reference_parts, dim=0),
        key_cache=key_cache,
        value_cache=value_cache,
        out_cache_loc=torch.cat(out_cache_locs, dim=0),
        pool=pool,
    )


def _pad_metadata_for_collectives(metadata, world_size):
    logical_tokens = list(metadata.per_rank_actual_token)
    alignment = 2 * world_size
    physical_tokens = (max(logical_tokens) + alignment - 1) // alignment * alignment
    metadata.per_rank_logical_token = logical_tokens
    metadata.per_rank_actual_token = [physical_tokens] * world_size
    metadata.max_rank_len = [physical_tokens] * world_size


def _run_flashinfer_case(
    device,
    world_size,
    num_q_heads,
    num_kv_heads,
    prefix_lens,
    extend_lens,
    allocator_page_size,
):
    from flashinfer import BatchPrefillWithPagedKVCacheWrapper

    from sglang.srt.layers.attention.flashinfer_backend import FlashInferAttnBackend

    fixture = _build_case_fixture(
        device,
        num_q_heads,
        num_kv_heads,
        prefix_lens,
        extend_lens,
        allocator_page_size,
    )
    strategy = get_cp_strategy()
    assert strategy is not None
    metadata = strategy.build_metadata(
        num_tokens=sum(extend_lens),
        seqs_len=fixture.seq_lens,
        extend_seqs_len=extend_lens,
    )
    _pad_metadata_for_collectives(metadata, world_size)
    forward_batch = SimpleNamespace(
        attn_cp_metadata=metadata,
        forward_mode=ForwardMode.EXTEND,
        req_pool_indices=torch.arange(
            len(extend_lens), dtype=torch.int32, device=device
        ),
        out_cache_loc=fixture.out_cache_loc,
    )

    local_q = strategy.shard_hidden_states(fixture.q, forward_batch)
    local_key = strategy.shard_hidden_states(fixture.key, forward_batch)
    local_value = strategy.shard_hidden_states(fixture.value, forward_batch)
    expected_local = strategy.shard_hidden_states(fixture.reference, forward_batch)

    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    wrapper = BatchPrefillWithPagedKVCacheWrapper(workspace, "NHD", backend="fa2")
    backend = object.__new__(FlashInferAttnBackend)
    backend.req_to_token_pool = SimpleNamespace(req_to_token=fixture.req_to_token)
    backend.prefill_wrappers_paged = [wrapper]
    backend.indices_updater_prefill = SimpleNamespace(
        num_qo_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=HEAD_DIM,
        q_data_type=DTYPE,
        data_type=DTYPE,
    )
    backend.prefill_split_tile_size = None
    backend._init_forward_metadata_cp(forward_batch)

    pool = fixture.pool
    key_cache, value_cache = pool.get_kv_buffer(0)
    key_cache[: fixture.key_cache.shape[0]].copy_(fixture.key_cache)
    value_cache[: fixture.value_cache.shape[0]].copy_(fixture.value_cache)
    backend.token_to_kv_pool = pool
    layer = SimpleNamespace(
        layer_id=0,
        tp_q_head_num=num_q_heads,
        tp_k_head_num=num_kv_heads,
        tp_v_head_num=num_kv_heads,
        head_dim=HEAD_DIM,
        scaling=HEAD_DIM**-0.5,
        logit_cap=0.0,
        is_cross_attention=False,
        k_scale=None,
        v_scale=None,
        k_scale_float=1.0,
        v_scale_float=1.0,
    )

    with patch(
        "sglang.srt.layers.cp.zigzag.get_token_to_kv_pool",
        return_value=pool,
    ):
        output = backend.forward_extend(
            local_q,
            local_key,
            local_value,
            layer,
            forward_batch,
        )

    torch.testing.assert_close(
        output.view_as(expected_local),
        expected_local,
        rtol=1e-2,
        atol=2e-2,
    )
    torch.testing.assert_close(
        key_cache[fixture.out_cache_loc],
        fixture.key,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        value_cache[fixture.out_cache_loc],
        fixture.value,
        rtol=0,
        atol=0,
    )


def _worker(local_rank, world_size, port):
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    update_environment_variables(
        {
            "RANK": str(local_rank),
            "LOCAL_RANK": str(local_rank),
            "WORLD_SIZE": str(world_size),
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(port),
        }
    )
    init_distributed_environment(
        world_size=world_size,
        rank=local_rank,
        local_rank=local_rank,
        backend="nccl",
    )
    initialize_model_parallel(
        tensor_model_parallel_size=world_size,
        attention_context_model_parallel_size=world_size,
    )
    init_cp_strategy(
        SimpleNamespace(
            enable_prefill_cp=True,
            cp_strategy="zigzag",
            attn_cp_size=world_size,
        )
    )

    try:
        cases = [
            (8, 8, [0], [9], 1),
            (8, 2, [3, 5], [33, 34], 16),
        ]
        for case in cases:
            _run_flashinfer_case(device, world_size, *case)
    finally:
        init_cp_strategy(SimpleNamespace(enable_prefill_cp=False))
        destroy_model_parallel()
        destroy_distributed_environment()


class TestFlashInferPrefillCP(CustomTestCase):
    @unittest.skipUnless(
        torch.cuda.is_available() and torch.cuda.device_count() >= NUM_GPUS,
        f"Need {NUM_GPUS} CUDA devices",
    )
    def test_two_gpu_mha_and_gqa(self):
        torch.multiprocessing.spawn(
            _worker,
            args=(NUM_GPUS, find_available_port(29500)),
            nprocs=NUM_GPUS,
            join=True,
        )


class TestFlashInferPrefillCPServer(CustomTestCase):
    def _launch_server(self, enable_cp):
        base_url = f"http://127.0.0.1:{find_available_port(30000)}"
        other_args = [
            "--tp-size",
            str(NUM_GPUS),
            "--attention-backend",
            "flashinfer",
            "--cuda-graph-backend-prefill",
            "disabled",
            "--skip-server-warmup",
        ]
        env = None
        if enable_cp:
            other_args.extend(
                [
                    "--attn-cp-size",
                    str(NUM_GPUS),
                    "--moe-dense-tp-size",
                    "1",
                    "--enable-prefill-cp",
                    "--cp-strategy",
                    "zigzag",
                ]
            )
            env = {"SGLANG_ENABLE_CP_V2": "1"}

        process = popen_launch_server(
            MODEL_PATH,
            base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
            env=env,
        )
        return process, base_url

    def _generate(self, base_url, input_ids):
        response = requests.post(
            base_url + "/generate",
            json={
                "input_ids": input_ids,
                "sampling_params": {"max_new_tokens": 4, "temperature": 0},
                "return_logprob": True,
            },
            timeout=120,
        )
        response.raise_for_status()
        return response.json()

    def _assert_output_parity(self, expected, actual):
        self.assertEqual(actual["output_ids"], expected["output_ids"])
        expected_logprobs = expected["meta_info"]["output_token_logprobs"]
        actual_logprobs = actual["meta_info"]["output_token_logprobs"]
        self.assertEqual(len(actual_logprobs), len(expected_logprobs))
        for expected_item, actual_item in zip(expected_logprobs, actual_logprobs):
            self.assertEqual(int(actual_item[1]), int(expected_item[1]))
            self.assertAlmostEqual(
                float(actual_item[0]),
                float(expected_item[0]),
                delta=2e-1,
            )

    @unittest.skipUnless(
        torch.cuda.is_available() and torch.cuda.device_count() >= NUM_GPUS,
        f"Need {NUM_GPUS} CUDA devices",
    )
    def test_qwen3_matches_tp_and_reuses_prefix(self):
        prefix = list(range(1000, 1064))
        prompt = prefix + list(range(2000, 2032))

        baseline_process, baseline_url = self._launch_server(enable_cp=False)
        try:
            baseline = self._generate(baseline_url, prompt)
        finally:
            kill_process_tree(baseline_process.pid, wait_timeout=60)

        cp_process, cp_url = self._launch_server(enable_cp=True)
        try:
            cold = self._generate(cp_url, prompt)
            requests.post(cp_url + "/flush_cache", timeout=30).raise_for_status()
            self._generate(cp_url, prefix)
            cached = self._generate(cp_url, prompt)
        finally:
            kill_process_tree(cp_process.pid, wait_timeout=60)

        self._assert_output_parity(baseline, cold)
        self._assert_output_parity(cold, cached)
        self.assertGreater(cached["meta_info"]["cached_tokens"], 0)


if __name__ == "__main__":
    unittest.main()
