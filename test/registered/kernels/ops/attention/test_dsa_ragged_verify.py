"""Packed DSA target verify through the backend's entry points, on real kernels."""

import unittest
import warnings
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import fields
from itertools import accumulate
from types import SimpleNamespace
from unittest.mock import patch

import msgspec
import torch
import triton

from sglang.kernels.ops.attention.dsa.metadata import (
    _fused_dsa_draft_extend_metadata_kernel,
)
from sglang.kernels.ops.attention.dsa.paged_mqa_logits import (
    cutedsl_paged_mqa_logits,
    deepgemm_paged_mqa_logits_split,
)
from sglang.kernels.ops.quantization.dsa_quant_k_cache import quantize_k_cache
from sglang.kernels.ops.speculative.ragged_verify_kernels import (
    _pad_verify_layout_kernel,
)
from sglang.srt.configs.kimi_linear import KimiLinearConfig
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsa.dsa_indexer_metadata import DSAIndexerMetadata
from sglang.srt.layers.attention.dsa.dsa_topk_backend import (
    DSATopKBackend,
    TopkTransformMethod,
)
from sglang.srt.layers.attention.dsa.utils import (
    resolve_fp8_paged_mqa_logits_fn,
    resolve_num_sms,
    resolve_paged_mqa_logits_metadata_fn,
)
from sglang.srt.layers.attention.dsa_backend import (
    DeepseekSparseAttnBackend,
    _flashmla_kv_padded_heads,
)
from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.runtime_context import get_parallel, get_platform
from sglang.srt.speculative.ragged_verify import (
    RaggedVerifyLayout,
    build_capture_verify_lens,
)
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase, enter_scope

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")
# The CuTe DSL paged-logits cases need SM100.
register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

# Rows a request verifies at most; a padding slot owns at most as many.
NUM_DRAFT_TOKENS = 6
# TOPK: FlashMLA's SM100 128-head sparse kernel walks top-k in blocks of 128.
NUM_SLOTS, NUM_TIER_ROWS, WIDTH, TOPK = 8, 16, 256, 128
# req_to_token rows; row 0 is the reserved padding row, so nine requests fit.
NUM_POOL_ROWS = 10
PAGE = 64


def _paged_index_cache_view(packed):
    # DeepGEMM reads the pool's raw page bytes, FP32 scales included;
    # reinterpreting this tensor as FP8 would break its fused-cache ABI.
    if packed.dtype != DSATokenToKVPool.index_k_with_scale_buffer_dtype:
        raise ValueError("paged index cache must retain the production byte dtype")
    return packed.view(
        packed.shape[0], 64, 1, DSATokenToKVPool.index_bytes_per_token_for(128)
    )


def _k31_model_config():
    # Dimensions from moonshotai/Kimi-K3.1-Pro's pinned text config;
    # NoPE disables rotation, and its 64-dimensional unrotated tail stays in KV.
    text = KimiLinearConfig(
        hidden_size=7168,
        num_attention_heads=96,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        mla_use_nope=True,
        linear_attn_config={"kda_layers": [1], "full_attn_layers": [0]},
        dsa_config={
            "dsa_layers": [0],
            "index_topk": 2048,
            "index_head_dim": 128,
            "index_num_heads": 32,
            "index_rope_dim": 0,
        },
    )
    config = object.__new__(ModelConfig)
    config.hf_config = SimpleNamespace(
        architectures=["KimiK31ForConditionalGeneration"],
        model_type="kimi_k31",
        text_config=text,
    )
    config.hf_text_config = text
    config._derive_model_shapes()
    return config


def _backend(
    *,
    max_bs,
    max_num_tokens,
    width=WIDTH,
    topk=TOPK,
    num_pool_rows=NUM_POOL_ROWS,
    k31=False,
    num_q_heads=16,
):
    backend = object.__new__(DeepseekSparseAttnBackend)
    backend.device = torch.device("cuda")
    backend.device_sm_major = torch.cuda.get_device_capability()[0]
    backend.attn_dcp_size, backend.attn_dcp_rank = 1, 0
    backend.dsa_index_kpool, backend.dsa_index_topk = 1, topk
    backend.hisparse_coordinator = backend._kv_shard_pool = None
    backend.runner_ids = None
    backend.dsa_decode_impl = "flashmla_kv" if k31 else "flashmla_sparse"
    backend.dsa_prefill_impl = backend.dsa_decode_impl
    backend.enable_auto_select_prefill_impl = False
    backend.dsa_kv_cache_store_fp8 = k31
    backend.kv_cache_dtype = torch.float8_e4m3fn if k31 else torch.bfloat16
    backend.dsa_shards_index_k = False
    backend.kv_cache_dim = 656 if k31 else 576
    backend.kv_scale_ue8m0 = k31
    backend.flashmla_kv_format = "V32"
    backend.flashmla_kv_num_q_heads = _flashmla_kv_padded_heads(num_q_heads)
    backend.flashmla_kv_num_q_heads_prefill = backend.flashmla_kv_num_q_heads
    backend.physical_page_size = backend.real_page_span = PAGE
    backend.dsa_topk_backend = DSATopKBackend.SGL_KERNEL
    backend.use_fused_topk = True
    backend.speculative_num_draft_tokens = NUM_DRAFT_TOKENS
    # Only the attention dispatch is synthetic; every metadata kernel is real.
    backend._is_in_breakable_cuda_graph = lambda: False
    backend._is_in_tc_piecewise_cuda_graph = lambda: False
    backend._get_device_sm = lambda: backend.device_sm_major * 10
    backend._is_blackwell = lambda: backend.device_sm_major == 10
    backend._arange_buf = torch.arange(16384, device="cuda", dtype=torch.int32)
    req_to_token = torch.arange(
        num_pool_rows * width, device="cuda", dtype=torch.int32
    ).view(num_pool_rows, width)
    # Pool row 0 is never allocated: every position of it names token 0,
    # so a padding request reads only KV page 0, the sink page.
    req_to_token[0].zero_()
    backend.req_to_token = req_to_token
    backend.req_to_token_pool = SimpleNamespace(req_to_token=req_to_token)
    backend._sink_pad_cache = {}
    backend._init_kpool_metadata_fusion()
    backend._memory_saver_adapter = TorchMemorySaverAdapter.create(enable=False)
    backend._trtllm_fused_q = backend._flashmla_kv_pad_q_in_place = False
    backend.init_cuda_graph_state(max_bs=max_bs, max_num_tokens=max_num_tokens)
    return backend


def _verify_batch(
    *, batch_size, prefix_lens, req_pool_indices, layout, seq_lens_cpu=None
):
    # The graph runner pads the slots past the live requests: seq_len 1, pool row 0.
    num_padding = batch_size - len(prefix_lens)
    return SimpleNamespace(
        batch_size=batch_size,
        forward_mode=ForwardMode.TARGET_VERIFY,
        seq_lens=torch.tensor(
            prefix_lens + [1] * num_padding, device="cuda", dtype=torch.int64
        ),
        seq_lens_cpu=seq_lens_cpu,
        seq_lens_sum=sum(prefix_lens) + num_padding,
        req_pool_indices=torch.tensor(
            req_pool_indices + [0] * num_padding, device="cuda", dtype=torch.int64
        ),
        ragged_verify_layout=layout,
        spec_info=None,
        global_num_tokens_cpu=None,
        extend_seq_lens_cpu=None,
    )


def _capture_layout(*, capture_lens, num_rows):
    # What the graph runner captures a token tier with; it carries host lengths.
    return RaggedVerifyLayout.from_verify_lens(
        verify_lens_cpu=capture_lens, device=torch.device("cuda"), grid=[num_rows]
    )


def _live_layout(*, verify_lens, num_rows):
    # As the planner builds a serving step's layout: on the device only,
    # so the host has no per-request length to read.
    return RaggedVerifyLayout.from_verify_lens_device(
        verify_lens=torch.tensor(verify_lens, device="cuda", dtype=torch.int32),
        graph_num_tokens=num_rows,
    )


def _capture_tier(test, backend, *, num_slots, num_rows):
    # As the graph runner captures a tier: every slot a padding request,
    # the rows spread evenly over the slots.
    capture_lens = build_capture_verify_lens(
        num_tokens=num_rows, num_slots=num_slots, num_draft_tokens=NUM_DRAFT_TOKENS
    )
    batch = _verify_batch(
        batch_size=num_slots,
        prefix_lens=[],
        req_pool_indices=[],
        layout=_capture_layout(capture_lens=capture_lens, num_rows=num_rows),
    )
    _run_entry(
        test,
        entry=lambda batch: backend.init_forward_metadata_out_graph(
            batch, in_capture=True
        ),
        batch=batch,
    )
    metadata = backend.forward_metadata
    _check_packed_rows(
        test,
        backend,
        metadata,
        prefix_lens=[1] * num_slots,
        req_pool_indices=[0] * num_slots,
        verify_lens=capture_lens,
        num_rows=num_rows,
    )
    return metadata


def _layout_state(layout):
    tensors = (layout.verify_lens, layout.qo_indptr_device, layout.extend_start_loc)
    return [(tensor.data_ptr(), tensor.tolist()) for tensor in tensors]


def _run_entry(test, *, entry, batch):
    layout = batch.ragged_verify_layout
    before = _layout_state(layout)
    entry(batch)
    # Left alone: the linear half of a hybrid backend reads the same layout next.
    test.assertIs(batch.ragged_verify_layout, layout)
    test.assertEqual(_layout_state(layout), before)
    test.assertIsNone(batch.extend_seq_lens_cpu)


def _live_rows(*, prefix_lens, req_pool_indices, verify_lens):
    # (pool row, causal length) of each live packed row, in request order;
    # row j of a request sees its prefix and its own rows 0..j.
    return [
        (req_pool_indices[r], prefix_lens[r] + j + 1)
        for r, num_verify in enumerate(verify_lens)
        for j in range(num_verify)
    ]


def _check_row_lengths(test, lengths, *, live_rows, num_draft_tokens):
    test.assertEqual(lengths[: len(live_rows)], [length for _, length in live_rows])
    # Later rows belong to padding requests (prefix 1), whose rows count 2, 3, ...;
    # each has at most num_draft_tokens rows, and which takes which row is free.
    previous = None
    for length in lengths[len(live_rows) :]:
        test.assertIn(length, (2,) if previous is None else (2, previous + 1))
        test.assertLessEqual(length, num_draft_tokens + 1)
        previous = length


def _check_slot_kv_lens(
    test, cu_seqlens_k, *, prefix_lens, verify_lens, num_rows, num_draft_tokens
):
    # Per request slot, not per row: the live requests, then padding requests,
    # which together own every row the live ones leave over.
    test.assertEqual(cu_seqlens_k[0], 0)
    kv_lens = [end - start for start, end in zip(cu_seqlens_k, cu_seqlens_k[1:])]
    test.assertEqual(
        kv_lens[: len(prefix_lens)], [p + v for p, v in zip(prefix_lens, verify_lens)]
    )
    padding = kv_lens[len(prefix_lens) :]
    for kv_len in padding:
        test.assertGreaterEqual(kv_len, 1)
        test.assertLessEqual(kv_len, 1 + num_draft_tokens)
    test.assertEqual(sum(padding), len(padding) + num_rows - sum(verify_lens))


def _check_packed_rows(
    test, backend, metadata, *, prefix_lens, req_pool_indices, verify_lens, num_rows
):
    num_draft_tokens = backend.speculative_num_draft_tokens
    live_rows = _live_rows(
        prefix_lens=prefix_lens,
        req_pool_indices=req_pool_indices,
        verify_lens=verify_lens,
    )
    lengths = metadata.dsa_seqlens_expanded.tolist()
    test.assertEqual(len(lengths), num_rows)
    _check_row_lengths(
        test, lengths, live_rows=live_rows, num_draft_tokens=num_draft_tokens
    )
    # The indexer takes its batch axis from cache_seqlens_int32,
    # and its rows per request from dsa_extend_seq_lens_list: one each, on every SM.
    test.assertEqual(metadata.cache_seqlens_int32.tolist(), lengths)
    test.assertEqual(metadata.dsa_extend_seq_lens_list, [1] * num_rows)
    test.assertEqual(tuple(metadata.paged_mqa_ctx_lens_2d.shape), (num_rows, 1))
    test.assertEqual(metadata.paged_mqa_ctx_lens_2d.reshape(-1).tolist(), lengths)
    clipped = [min(length, backend.dsa_index_topk) for length in lengths]
    test.assertEqual(metadata.dsa_cache_seqlens_int32.tolist(), clipped)
    test.assertEqual(metadata.dsa_cu_seqlens_k.tolist(), [0, *accumulate(clipped)])
    test.assertEqual(metadata.cu_seqlens_q.tolist(), list(range(num_rows + 1)))
    test.assertEqual(metadata.dsa_cu_seqlens_q.tolist(), list(range(num_rows + 1)))
    test.assertEqual(metadata.topk_v2_plan.shape[0], num_rows + 1)
    _check_slot_kv_lens(
        test,
        metadata.cu_seqlens_k.tolist(),
        prefix_lens=prefix_lens,
        verify_lens=verify_lens,
        num_rows=num_rows,
        num_draft_tokens=num_draft_tokens,
    )

    width = metadata.page_table_1.shape[1]
    test.assertLessEqual(max(lengths), width)
    owners = [owner for owner, _ in live_rows] + [0] * (num_rows - len(live_rows))
    expected = backend.req_to_token[torch.tensor(owners, device="cuda"), :width]
    # Only the columns a row's length covers: no consumer reads past them.
    visible = (
        torch.arange(width, device="cuda")[None, :]
        < torch.tensor(lengths, device="cuda")[:, None]
    )
    for name, actual, wanted, shown in (
        ("page_table_1", metadata.page_table_1, expected, visible),
        (
            "real_page_table",
            metadata.real_page_table,
            expected[:, ::PAGE] // PAGE,
            visible[:, ::PAGE],
        ),
    ):
        test.assertEqual(tuple(actual.shape), tuple(wanted.shape), name)
        torch.testing.assert_close(
            actual[shown],
            wanted[shown],
            rtol=0,
            atol=0,
            msg=lambda text, name=name: f"{name}: {text}",
        )


def _poison_rows(metadata, *, tables):
    # No packed row has length 0 and no cumulative length or token id is -1,
    # so a row the replay skips shows.
    metadata.dsa_seqlens_expanded.zero_()
    metadata.dsa_cache_seqlens_int32.zero_()
    metadata.dsa_cu_seqlens_k.fill_(-1)
    metadata.cu_seqlens_k.fill_(-1)
    # Only for cases that run no kernel over the tables.
    if tables:
        metadata.page_table_1.fill_(-1)
        metadata.real_page_table.fill_(-1)


def _storage(tensor):
    return (
        tensor.data_ptr(),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        str(tensor.dtype),
    )


def _struct_tensors(*, name_prefix, struct):
    for name in struct.__struct_fields__:
        value = getattr(struct, name)
        if isinstance(value, torch.Tensor):
            yield name_prefix + name, value
        elif isinstance(value, msgspec.Struct):
            yield from _struct_tensors(
                name_prefix=name_prefix + name + ".", struct=value
            )


def _addresses(metadata):
    # A captured graph reads an entry's buffers by address.
    result = {
        f.name: _storage(value)
        for f in fields(metadata)
        if isinstance(value := getattr(metadata, f.name), torch.Tensor)
    }
    if metadata.flashmla_metadata is not None:
        result.update(
            {
                "flashmla." + f.name: _storage(
                    getattr(metadata.flashmla_metadata, f.name)
                )
                for f in fields(metadata.flashmla_metadata)
            }
        )
    # The graph never reads the request slots; a replay must refill them in place.
    if metadata.ragged_verify_slots is not None:
        result.update(
            {
                name: _storage(tensor)
                for name, tensor in _struct_tensors(
                    name_prefix="slots.", struct=metadata.ragged_verify_slots
                )
            }
        )
    return result


@contextmanager
def _compiled_kernels():
    compiled = []
    previous = triton.knobs.runtime.jit_post_compile_hook

    def record(*, repr, **kwargs):
        compiled.append(repr)
        if previous is not None:
            return previous(repr=repr, **kwargs)
        return None

    triton.knobs.runtime.jit_post_compile_hook = record
    try:
        yield compiled
    finally:
        triton.knobs.runtime.jit_post_compile_hook = previous


def _sync_warnings(*, entry, batches):
    previous = torch.cuda.get_sync_debug_mode()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        torch.cuda.set_sync_debug_mode("warn")
        try:
            for batch in batches:
                entry(batch)
        finally:
            torch.cuda.set_sync_debug_mode(previous)
    return [
        str(warning.message)
        for warning in caught
        if "synchronizing CUDA operation" in str(warning.message)
    ]


# Live layouts per token tier; the runner gives a tier min(num_rows, max_bs) slots.
_TIER_SHAPES = {
    # Prefixes on both sides of a 64-token page and of the 128-token top-k;
    # three rows are left over for padding slots, two of which stay empty.
    "page_boundaries": dict(
        max_bs=8,
        num_rows=16,
        prefix_lens=[62, 63, 64, 127, 128],
        req_pool_indices=[5, 2, 7, 1, 4],
        verify_lens=[1, 3, 6, 2, 1],
    ),
    # Thirteen rows left over: padding slots own more than one row each.
    "one_request": dict(
        max_bs=8, num_rows=16, prefix_lens=[200], req_pool_indices=[3], verify_lens=[3]
    ),
    # No runner-padded slot and 24 rows left over:
    # only slots the backend adds can own them, four full ones.
    "all_slots_live_below_the_tier": dict(
        max_bs=8,
        num_rows=32,
        prefix_lens=[0, 61, 62, 63, 64, 126, 127, 249],
        req_pool_indices=[8, 3, 6, 1, 7, 2, 5, 4],
        verify_lens=[1] * 8,
    ),
    # Nothing left over: every padding slot is empty.
    "full_tier": dict(
        max_bs=8,
        num_rows=16,
        prefix_lens=[5, 190, 64],
        req_pool_indices=[9, 4, 6],
        verify_lens=[6, 6, 4],
    ),
    # As many slots as rows, all live: no padding slot at all.
    "tier_of_one_row_per_slot": dict(
        max_bs=8,
        num_rows=8,
        prefix_lens=[0, 61, 62, 63, 64, 126, 127, 249],
        req_pool_indices=[8, 3, 6, 1, 7, 2, 5, 4],
        verify_lens=[1] * 8,
    ),
    # 270 request slots, 258 of them live: past the layout kernel's 256-slot block.
    "more_than_256_slots": dict(
        max_bs=260,
        num_rows=320,
        prefix_lens=[(37 * i) % 200 for i in range(258)],
        req_pool_indices=[1 + (5 * i) % 263 for i in range(258)],
        verify_lens=[2 if i < 55 else 1 for i in range(258)],
        num_pool_rows=264,
    ),
}


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class _PackedVerifyTestCase(CustomTestCase):
    def setUp(self):
        super().setUp()
        if torch.cuda.get_device_capability()[0] < 9:
            self.skipTest("native FP8 DSA requires Hopper or newer")
        # The kernel routes a server takes by default,
        # and the context-parallel width eager metadata reads from the runtime context.
        for scope in (
            envs.SGLANG_FP8_PAGED_MQA_LOGITS_TORCH.override(False),
            envs.SGLANG_DSA_FUSE_TOPK.override(True),
            envs.SGLANG_OPT_USE_TOPK_V2.override(True),
            get_parallel().override(attn_cp_size=1),
        ):
            enter_scope(self, scope)


class TestDSARaggedVerifyMetadata(_PackedVerifyTestCase):
    def test_rows_follow_the_live_layout_through_every_entry(self):
        """Each entry gives a packed row its owner's table and causal length;
        rows left over go to padding requests, never a live one; none is unwritten."""
        for name, shape in _TIER_SHAPES.items():
            with self.subTest(shape=name):
                self._check_every_entry(**shape)

    def _check_every_entry(
        self,
        *,
        max_bs,
        num_rows,
        prefix_lens,
        req_pool_indices,
        verify_lens,
        num_pool_rows=NUM_POOL_ROWS,
    ):
        num_slots = min(num_rows, max_bs)
        backend = _backend(
            max_bs=max_bs, max_num_tokens=num_rows, num_pool_rows=num_pool_rows
        )
        live = dict(
            prefix_lens=prefix_lens,
            req_pool_indices=req_pool_indices,
            verify_lens=verify_lens,
            num_rows=num_rows,
        )

        # Eager sees the live requests only. Its table, with a host mirror of seq_lens,
        # is the longest prefix plus one block wide, which must still cover every row.
        for seq_lens_cpu, width in (
            (None, WIDTH),
            (
                torch.tensor(prefix_lens),
                min(max(prefix_lens) + NUM_DRAFT_TOKENS, WIDTH),
            ),
        ):
            batch = _verify_batch(
                batch_size=len(prefix_lens),
                prefix_lens=prefix_lens,
                req_pool_indices=req_pool_indices,
                layout=_live_layout(verify_lens=verify_lens, num_rows=num_rows),
                seq_lens_cpu=seq_lens_cpu,
            )
            _run_entry(self, entry=backend.init_forward_metadata, batch=batch)
            self.assertEqual(backend.forward_metadata.page_table_1.shape[1], width)
            _check_packed_rows(self, backend, backend.forward_metadata, **live)

        # A table row the capture or the replay does not fill keeps the poison.
        backend.decode_cuda_graph_metadata["page_table"].fill_(-1)
        metadata = _capture_tier(self, backend, num_slots=num_slots, num_rows=num_rows)
        addresses = _addresses(metadata)

        _poison_rows(metadata, tables=True)
        batch = _verify_batch(
            batch_size=num_slots,
            prefix_lens=prefix_lens,
            req_pool_indices=req_pool_indices,
            layout=_live_layout(verify_lens=verify_lens, num_rows=num_rows),
        )
        _run_entry(self, entry=backend.init_forward_metadata_out_graph, batch=batch)
        self.assertIs(backend.forward_metadata, metadata)
        self.assertEqual(_addresses(metadata), addresses)
        _check_packed_rows(self, backend, metadata, **live)

    def test_uniform_full_block_matches_static_verify(self):
        """When every request verifies its whole block,
        packed rows equal static verify's: same lengths and page tables, row for row."""
        prefix_lens, req_pool_indices = [62, 127, 0, 200], [3, 7, 1, 5]
        verify_lens = [NUM_DRAFT_TOKENS] * len(prefix_lens)
        num_rows = sum(verify_lens)
        num_slots = min(num_rows, NUM_SLOTS)
        live = dict(
            prefix_lens=prefix_lens,
            req_pool_indices=req_pool_indices,
            verify_lens=verify_lens,
            num_rows=num_rows,
        )
        packed = _backend(max_bs=NUM_SLOTS, max_num_tokens=num_rows)
        static = _backend(max_bs=NUM_SLOTS, max_num_tokens=num_rows)

        def static_batch():
            return _verify_batch(
                batch_size=len(prefix_lens),
                prefix_lens=prefix_lens,
                req_pool_indices=req_pool_indices,
                layout=None,
            )

        _run_entry(
            self,
            entry=packed.init_forward_metadata,
            batch=_verify_batch(
                batch_size=len(prefix_lens),
                prefix_lens=prefix_lens,
                req_pool_indices=req_pool_indices,
                layout=_live_layout(verify_lens=verify_lens, num_rows=num_rows),
            ),
        )
        static.init_forward_metadata(static_batch())
        _check_packed_rows(self, packed, packed.forward_metadata, **live)
        self._check_against_static(
            packed=packed.forward_metadata,
            static=static.forward_metadata,
            prefix_lens=prefix_lens,
        )

        _capture_tier(self, packed, num_slots=num_slots, num_rows=num_rows)
        _run_entry(
            self,
            entry=packed.init_forward_metadata_out_graph,
            batch=_verify_batch(
                batch_size=num_slots,
                prefix_lens=prefix_lens,
                req_pool_indices=req_pool_indices,
                layout=_live_layout(verify_lens=verify_lens, num_rows=num_rows),
            ),
        )
        # A static graph entry is built by the capture call and filled by a replay.
        batch = static_batch()
        static.init_forward_metadata_out_graph(batch, in_capture=True)
        static.init_forward_metadata_out_graph(batch)
        _check_packed_rows(self, packed, packed.forward_metadata, **live)
        self._check_against_static(
            packed=packed.forward_metadata,
            static=static.forward_metadata,
            prefix_lens=prefix_lens,
        )

    def _check_against_static(self, *, packed, static, prefix_lens):
        num_rows = NUM_DRAFT_TOKENS * len(prefix_lens)
        for name in (
            "dsa_seqlens_expanded",
            "dsa_cache_seqlens_int32",
            "dsa_cu_seqlens_k",
            "cu_seqlens_q",
            "dsa_cu_seqlens_q",
        ):
            self.assertEqual(
                getattr(packed, name).tolist(), getattr(static, name).tolist(), name
            )
        row_lens = static.dsa_seqlens_expanded
        for name, tokens_per_column in (("page_table_1", 1), ("real_page_table", PAGE)):
            packed_table, static_table = getattr(packed, name), getattr(static, name)
            self.assertEqual(packed_table.shape, static_table.shape, name)
            columns = torch.arange(packed_table.shape[1], device="cuda")
            visible = columns[None, :] * tokens_per_column < row_lens[:, None]
            torch.testing.assert_close(
                packed_table[visible],
                static_table[visible],
                rtol=0,
                atol=0,
                msg=lambda text, name=name: f"{name}: {text}",
            )
        # By design, static verify keeps cache_seqlens per request,
        # and on SM100 a [requests, block] context grid; a packed tier is per row.
        self.assertEqual(
            static.cache_seqlens_int32.tolist(),
            [p + NUM_DRAFT_TOKENS for p in prefix_lens],
        )
        self.assertEqual(
            tuple(static.paged_mqa_ctx_lens_2d.shape),
            (len(prefix_lens), NUM_DRAFT_TOKENS)
            if get_platform().is_sm100
            else (num_rows, 1),
        )
        self.assertEqual(
            packed.cache_seqlens_int32.tolist(), packed.dsa_seqlens_expanded.tolist()
        )
        self.assertEqual(tuple(packed.paged_mqa_ctx_lens_2d.shape), (num_rows, 1))

    def test_tiers_sharing_a_slot_count_keep_separate_entries(self):
        """Token tiers at or above the largest batch size share one slot count;
        each still replays into its own buffers."""
        backend = _backend(max_bs=NUM_SLOTS, max_num_tokens=32)
        tiers = {
            num_rows: _capture_tier(
                self, backend, num_slots=NUM_SLOTS, num_rows=num_rows
            )
            for num_rows in (16, 32)
        }
        self.assertIsNot(tiers[16], tiers[32])
        rounds = {
            16: dict(
                prefix_lens=[62, 127], req_pool_indices=[3, 7], verify_lens=[6, 4]
            ),
            32: dict(
                prefix_lens=[0, 63, 128, 5],
                req_pool_indices=[5, 1, 2, 8],
                verify_lens=[6, 1, 3, 6],
            ),
        }
        for num_rows, other in ((16, 32), (32, 16), (16, 32)):
            with self.subTest(num_rows=num_rows):
                kept = self._tier_state(tiers[other])
                _poison_rows(tiers[num_rows], tables=True)
                batch = _verify_batch(
                    batch_size=NUM_SLOTS,
                    prefix_lens=rounds[num_rows]["prefix_lens"],
                    req_pool_indices=rounds[num_rows]["req_pool_indices"],
                    layout=_live_layout(
                        verify_lens=rounds[num_rows]["verify_lens"], num_rows=num_rows
                    ),
                )
                _run_entry(
                    self, entry=backend.init_forward_metadata_out_graph, batch=batch
                )
                self.assertIs(backend.forward_metadata, tiers[num_rows])
                _check_packed_rows(
                    self,
                    backend,
                    tiers[num_rows],
                    num_rows=num_rows,
                    **rounds[num_rows],
                )
                self.assertEqual(self._tier_state(tiers[other]), kept)

    @staticmethod
    def _tier_state(metadata):
        # page_table_1 is left out: it views the one wide table all tiers share,
        # which every replay rewrites for its own rows.
        return (
            _addresses(metadata),
            [
                getattr(metadata, name).tolist()
                for name in (
                    "dsa_seqlens_expanded",
                    "dsa_cache_seqlens_int32",
                    "dsa_cu_seqlens_k",
                    "cu_seqlens_k",
                    "real_page_table",
                )
            ],
        )

    def test_replay_of_an_uncaptured_tier_is_refused(self):
        """A graph reads the buffers of the entry it was captured with,
        so a replay must not build an entry of its own for a tier that has none."""
        backend = _backend(max_bs=NUM_SLOTS, max_num_tokens=32)

        def replay(num_rows):
            return _verify_batch(
                batch_size=NUM_SLOTS,
                prefix_lens=[62, 127],
                req_pool_indices=[3, 7],
                layout=_live_layout(verify_lens=[6, 4], num_rows=num_rows),
            )

        entries = set(backend.decode_cuda_graph_metadata)
        with self.assertRaisesRegex(RuntimeError, "no packed verify graph metadata"):
            backend.init_forward_metadata_out_graph(replay(16))
        self.assertEqual(set(backend.decode_cuda_graph_metadata), entries)

        # Capturing one tier does not admit another with the same slot count.
        metadata = _capture_tier(self, backend, num_slots=NUM_SLOTS, num_rows=16)
        entries = set(backend.decode_cuda_graph_metadata)
        with self.assertRaisesRegex(RuntimeError, "no packed verify graph metadata"):
            backend.init_forward_metadata_out_graph(replay(32))
        self.assertEqual(set(backend.decode_cuda_graph_metadata), entries)
        backend.init_forward_metadata_out_graph(replay(16))
        self.assertIs(backend.forward_metadata, metadata)


class TestDSARaggedVerifyNativeGraph(_PackedVerifyTestCase):
    def test_native_indexer_sparse_attention_and_graph_replay(self):
        self._check_native_graph("deepgemm")

    def test_cutedsl_indexer_sparse_attention_and_graph_replay(self):
        if torch.cuda.get_device_capability()[0] != 10:
            self.skipTest("CuTe DSL paged DSA requires Blackwell")
        self._check_native_graph("cutedsl")

    def test_k31_fp8_paged_attention_and_graph_replay(self):
        self._check_native_graph("deepgemm", k31=True)

    def test_k31_fp8_cutedsl_paged_attention_and_graph_replay(self):
        if torch.cuda.get_device_capability()[0] != 10:
            self.skipTest("CuTe DSL paged DSA requires Blackwell")
        self._check_native_graph("cutedsl", k31=True)

    def _check_native_graph(self, logits_backend, *, k31=False):
        torch.manual_seed(19)
        config = _k31_model_config() if k31 else None
        width, topk = (4096, config.hf_text_config.index_topk) if k31 else (WIDTH, TOPK)
        num_heads = config.get_num_attention_heads(8) if k31 else 16
        num_index_heads = config.hf_text_config.index_n_heads if k31 else 64
        q_dim = config.kv_lora_rank + config.qk_rope_head_dim if k31 else 576
        sm_scale = config.scaling if k31 else 576**-0.5
        long_prefix = 4090 if k31 else 127
        backend = _backend(
            max_bs=NUM_SLOTS,
            max_num_tokens=NUM_TIER_ROWS,
            width=width,
            topk=topk,
            k31=k31,
            num_q_heads=num_heads,
        )
        metadata = _capture_tier(
            self, backend, num_slots=NUM_SLOTS, num_rows=NUM_TIER_ROWS
        )
        pointers = _addresses(metadata)
        inputs = self._native_inputs(
            k31=k31,
            config=config,
            width=width,
            num_heads=num_heads,
            num_index_heads=num_index_heads,
            q_dim=q_dim,
        )
        layer = SimpleNamespace(tp_q_head_num=num_heads, head_dim=q_dim)

        def forward(meta=metadata):
            indexer_metadata = DSAIndexerMetadata(
                meta, TopkTransformMethod.PAGED, topk_backend=DSATopKBackend.SGL_KERNEL
            )
            # B and next_n as Indexer._get_topk_paged derives them from the metadata;
            # a packed tier is one row per request.
            q_offset = sum(indexer_metadata.get_dsa_extend_len_cpu())
            num_requests = indexer_metadata.get_seqlens_int32().shape[0]
            next_n = q_offset // num_requests if num_requests > 0 else 0
            self.assertEqual(
                (q_offset, num_requests, next_n), (NUM_TIER_ROWS, NUM_TIER_ROWS, 1)
            )
            block_tables = indexer_metadata.get_page_table_64()
            max_seq_len = block_tables.shape[1] * 64
            if logits_backend == "cutedsl":
                logits = cutedsl_paged_mqa_logits(
                    inputs.index_q,
                    inputs.index_cache,
                    inputs.weights,
                    indexer_metadata.get_seqlens_int32(),
                    block_tables,
                    meta.paged_mqa_schedule_metadata,
                    max_seq_len,
                    q_offset=q_offset,
                    B=num_requests,
                    next_n=next_n,
                    is_target_verify=True,
                    dsl_expand_factor=1,
                    dsl_atom=1,
                    blocksize=64,
                    sm_count=resolve_num_sms(),
                    get_paged_mqa_logits_metadata_fn=resolve_paged_mqa_logits_metadata_fn(),
                )
            else:
                logits = deepgemm_paged_mqa_logits_split(
                    resolve_fp8_paged_mqa_logits_fn(),
                    inputs.index_q,
                    inputs.index_cache,
                    inputs.weights,
                    meta.paged_mqa_ctx_lens_2d,
                    block_tables,
                    meta.paged_mqa_schedule_metadata,
                    max_seq_len,
                    q_offset=q_offset,
                )
            selected = indexer_metadata.topk_transform(logits, topk)
            if k31:
                output = backend._forward_flashmla_kv(
                    inputs.q, inputs.kv, 512, sm_scale, layer, meta, selected, None
                ).view(NUM_TIER_ROWS, num_heads, 512)
            else:
                output = backend._forward_flashmla_sparse(
                    inputs.q,
                    inputs.kv,
                    512,
                    selected,
                    sm_scale,
                    topk_length=meta.dsa_cache_seqlens_int32,
                    attn_sink=inputs.sink,
                )
            return selected, output, logits

        # Warm the real native kernels before capture;
        # this graph uses no stand-in kernels and no synthetic logits.
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                forward()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            graph_topk, graph_output, graph_logits = forward()
        torch.cuda.current_stream().wait_stream(stream)

        # More rows than captured per slot, one request, a full tier, a permutation,
        # and every slot live with rows left over.
        for prefix_lens, req_pool_indices, verify_lens in (
            ([0, 62, long_prefix], [5, 2, 7], [1, 3, 6]),
            ([long_prefix], [3], [1]),
            ([63, long_prefix, 5], [7, 2, 4], [6, 6, 4]),
            ([0, 62, long_prefix], [2, 7, 5], [6, 1, 3]),
            (
                [3, 62, 63, 64, 0, 17, long_prefix, 40],
                [8, 1, 6, 3, 9, 4, 2, 5],
                [1] * 8,
            ),
        ):
            with self.subTest(verify_lens=verify_lens):
                live_rows = _live_rows(
                    prefix_lens=prefix_lens,
                    req_pool_indices=req_pool_indices,
                    verify_lens=verify_lens,
                )
                live = dict(
                    prefix_lens=prefix_lens,
                    req_pool_indices=req_pool_indices,
                    verify_lens=verify_lens,
                    num_rows=NUM_TIER_ROWS,
                )
                # The tables keep the previous round,
                # so an unrewritten spare row still points at a request of that round.
                _poison_rows(metadata, tables=False)
                batch = _verify_batch(
                    batch_size=NUM_SLOTS,
                    prefix_lens=prefix_lens,
                    req_pool_indices=req_pool_indices,
                    layout=_live_layout(
                        verify_lens=verify_lens, num_rows=NUM_TIER_ROWS
                    ),
                )
                _run_entry(
                    self, entry=backend.init_forward_metadata_out_graph, batch=batch
                )
                self.assertIs(backend.forward_metadata, metadata)
                self.assertEqual(_addresses(metadata), pointers)
                _check_packed_rows(self, backend, metadata, **live)
                graph.replay()

                batch = _verify_batch(
                    batch_size=len(prefix_lens),
                    prefix_lens=prefix_lens,
                    req_pool_indices=req_pool_indices,
                    layout=_live_layout(
                        verify_lens=verify_lens, num_rows=NUM_TIER_ROWS
                    ),
                )
                _run_entry(self, entry=backend.init_forward_metadata, batch=batch)
                _check_packed_rows(self, backend, backend.forward_metadata, **live)
                eager_topk, eager_output, _ = forward(backend.forward_metadata)
                # Eager has no runner-padded slots,
                # so spare rows fall on other padding requests; compare live rows only.
                num_live = len(live_rows)
                # Folded top-k emits an unordered set through atomic counters.
                # Equivalent sets can change the attention reduction's last bits.
                torch.testing.assert_close(
                    graph_topk[:num_live].sort(dim=-1).values,
                    eager_topk[:num_live].sort(dim=-1).values,
                    rtol=0,
                    atol=0,
                )
                torch.testing.assert_close(
                    graph_output[:num_live],
                    eager_output[:num_live],
                    rtol=0.015,
                    atol=0.015,
                )
                self.assertTrue(torch.isfinite(graph_output).all())
                self._check_spare_rows(
                    selected=graph_topk,
                    lengths=metadata.dsa_seqlens_expanded.tolist(),
                    num_live=num_live,
                )
                self._check_live_rows(
                    live_rows=live_rows,
                    selected=graph_topk,
                    logits=graph_logits,
                    output=graph_output,
                    inputs=inputs,
                    width=width,
                    topk=topk,
                    sm_scale=sm_scale,
                )

    def _native_inputs(self, *, k31, config, width, num_heads, num_index_heads, q_dim):
        num_kv_tokens = NUM_POOL_ROWS * width
        num_pages = num_kv_tokens // 64
        index_k = torch.randn(num_pages, 64, 128, device="cuda").to(torch.float8_e4m3fn)
        scales = torch.ones(num_pages, 64, device="cuda", dtype=torch.float32)
        packed = torch.empty(num_pages, 64 * 132, device="cuda", dtype=torch.uint8)
        packed[:, : 64 * 128] = index_k.view(torch.uint8).reshape(num_pages, -1)
        packed[:, 64 * 128 :] = scales.view(torch.uint8).reshape(num_pages, -1)
        index_cache = _paged_index_cache_view(packed)
        index_q = torch.randn(NUM_TIER_ROWS, num_index_heads, 128, device="cuda").to(
            torch.float8_e4m3fn
        )
        weights = torch.rand(NUM_TIER_ROWS, num_index_heads, device="cuda")
        q = torch.randn(
            NUM_TIER_ROWS, num_heads, q_dim, device="cuda", dtype=torch.bfloat16
        )
        kv = torch.randn(num_kv_tokens, 1, q_dim, device="cuda", dtype=torch.bfloat16)
        sink = (
            None if k31 else torch.randn(num_heads, device="cuda", dtype=torch.float32)
        )
        if k31:
            kv = quantize_k_cache(
                kv.view(num_pages, 64, 1, q_dim),
                dv=config.kv_lora_rank,
                scale_ue8m0=True,
            ).view(num_kv_tokens, 1, -1)
            self.assertEqual(kv.shape[-1], 656)
            raw = kv.view(torch.uint8).view(num_kv_tokens, 656)
            values = raw[:, :512].contiguous().view(torch.float8_e4m3fn).float()
            scales = raw[:, 512:528].contiguous().view(torch.float32)
            tail = raw[:, 528:656].contiguous().view(torch.bfloat16).float()
            reference_kv = torch.cat(
                [values * scales.repeat_interleave(128, dim=1), tail], dim=1
            )
        else:
            reference_kv = kv[:, 0].float()
        return SimpleNamespace(
            index_keys=index_k.view(num_kv_tokens, 128),
            index_cache=index_cache,
            index_q=index_q,
            weights=weights,
            q=q,
            kv=kv,
            sink=sink,
            reference_kv=reference_kv,
        )

    def _check_spare_rows(self, *, selected, lengths, num_live):
        # A padding request's table names token 0 at every position,
        # so position p of its row is slot p of KV page 0, the sink page.
        for token in range(num_live, len(lengths)):
            chosen = selected[token]
            chosen = chosen[chosen >= 0]
            self.assertEqual(sorted(chosen.tolist()), list(range(lengths[token])))

    def _check_live_rows(
        self, *, live_rows, selected, logits, output, inputs, width, topk, sm_scale
    ):
        # An independent causal reference checks page ownership,
        # and the selected sparse attention including the learned sink term.
        for token, (pool_row, length) in enumerate(live_rows):
            chosen = selected[token]
            chosen = chosen[chosen >= 0].long()
            logical = chosen - pool_row * width
            self.assertTrue(((logical >= 0) & (logical < length)).all())
            self.assertEqual(chosen.numel(), min(length, topk))
            index_keys = inputs.index_keys[
                pool_row * width : pool_row * width + length
            ].float()
            index_scores = (inputs.index_q[token].float() @ index_keys.T).relu()
            index_reference = (index_scores * inputs.weights[token, :, None]).sum(0)
            torch.testing.assert_close(
                logits[token, :length], index_reference, atol=0.003, rtol=0.003
            )
            keys = inputs.reference_kv[chosen]
            scores = (inputs.q[token].float() @ keys.T) * sm_scale
            if inputs.sink is None:
                probs = torch.softmax(scores, dim=1)
            else:
                probs = torch.softmax(
                    torch.cat([scores, inputs.sink[:, None]], dim=1), dim=1
                )[:, :-1]
            reference = probs @ keys[:, :512]
            torch.testing.assert_close(
                output[token].float(), reference, atol=0.025, rtol=0.025
            )


class TestDSARaggedVerifySteadyState(_PackedVerifyTestCase):
    """What a serving step may not do once its tier is captured."""

    @staticmethod
    def _replay_batches(*, num_slots, num_rows, layouts):
        return [
            _verify_batch(
                batch_size=num_slots,
                prefix_lens=[(41 * i) % 200 for i in range(len(verify_lens))],
                req_pool_indices=list(range(1, len(verify_lens) + 1)),
                layout=_live_layout(verify_lens=verify_lens, num_rows=num_rows),
            )
            for verify_lens in layouts
        ]

    def test_live_layouts_compile_no_kernel_variant(self):
        """Live request counts and lengths are run-time values to the metadata kernels:
        a step with a new mix compiles nothing."""
        # Fresh variant caches for the kernels a packed replay launches,
        # so variants compiled earlier in this process cannot hide a new one.
        for kernel in (
            _pad_verify_layout_kernel,
            _fused_dsa_draft_extend_metadata_kernel,
        ):
            # In the method body: CustomTestCase's CI retry reruns it, not setUp.
            enter_scope(
                self,
                patch.object(
                    kernel, "device_caches", defaultdict(kernel.create_binder)
                ),
            )
        # 17 slots: the capture's own live count is neither 1 nor a multiple of 16.
        num_slots, num_rows = 17, 32
        backend = _backend(
            max_bs=num_slots, max_num_tokens=num_rows, num_pool_rows=num_slots + 1
        )
        _capture_tier(self, backend, num_slots=num_slots, num_rows=num_rows)
        warm = self._replay_batches(
            num_slots=num_slots, num_rows=num_rows, layouts=([6, 1, 3], [2] * 7)
        )
        # Triton keys a specialized integer on being 1 and on being a multiple of 16,
        # so one and sixteen live requests are classes of their own.
        again = self._replay_batches(
            num_slots=num_slots,
            num_rows=num_rows,
            layouts=([4], [2] * 16, [1, 6, 2, 5, 3]),
        )
        for batch in warm:
            backend.init_forward_metadata_out_graph(batch)
        with _compiled_kernels() as compiled:
            for batch in again:
                backend.init_forward_metadata_out_graph(batch)
        self.assertEqual(compiled, [])

    def test_replay_synchronizes_no_more_than_static_verify(self):
        """A packed replay never reads the live lengths back:
        torch reports no more host-device syncs for it than for static verify."""
        packed = _backend(max_bs=NUM_SLOTS, max_num_tokens=NUM_TIER_ROWS)
        _capture_tier(self, packed, num_slots=NUM_SLOTS, num_rows=NUM_TIER_ROWS)
        packed_batches = self._replay_batches(
            num_slots=NUM_SLOTS,
            num_rows=NUM_TIER_ROWS,
            layouts=([6, 1, 3], [2] * 8, [1], [6, 6, 4]),
        )
        static = _backend(max_bs=NUM_SLOTS, max_num_tokens=4 * NUM_DRAFT_TOKENS)
        static_batch = _verify_batch(
            batch_size=4,
            prefix_lens=[62, 127, 0, 200],
            req_pool_indices=[3, 7, 1, 5],
            layout=None,
        )
        static.init_forward_metadata_out_graph(static_batch, in_capture=True)
        # One replay each first: lazily resolved or compiled kernels are not measured.
        packed.init_forward_metadata_out_graph(packed_batches[0])
        static.init_forward_metadata_out_graph(static_batch)
        packed_syncs = _sync_warnings(
            entry=packed.init_forward_metadata_out_graph, batches=packed_batches
        )
        static_syncs = _sync_warnings(
            entry=static.init_forward_metadata_out_graph,
            batches=[static_batch] * len(packed_batches),
        )
        self.assertLessEqual(len(packed_syncs), len(static_syncs), packed_syncs)


if __name__ == "__main__":
    unittest.main()
