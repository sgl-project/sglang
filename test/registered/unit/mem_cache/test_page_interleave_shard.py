# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for logical-page KV cache sharding.

Pins the pure arithmetic that rotated owner-classed allocation hangs on:

1. The placement bijection ``loc = Q*(N*ps) + r*ps + o`` — owner / local-row
   round-trip, disjoint equal partition across ranks.
2. ``PageInterleavePoolAllocator`` — N mirrored class free lists, rotated
   class draws (owners exactly cyclic along a chain), least-full root
   seeding, min-class admission accounting, zero stranding (a freed page is
   immediately reusable).
3. The host rotation base on ``UnifiedTreeNode`` — stamped at insert, copied
   on split, read through ``last_node``, and the pre-flight that declines an
   insert whose pages carry a different base than the chain it would join.
4. ``translate_loc_to_scratch`` — the per-batch page->scratch-page lookup mapping
   any consumer index vector onto the owner-major ``[prefix | chunk | trash]``
   scratch, checked against a brute-force reference.
5. ``begin_shard_extend`` plan capture (page positions, padded send rows,
   owner-congruence guard) with the gather stubbed out, following the
   SimpleNamespace binding pattern of ``test_dsa_layer_shard_utils.py``.
6. MTP draft pool construction, shared allocator/layout validation, and scratch
   sizing for separate target and draft pools.
7. The P/D send filter ``filter_kv_indices_for_shard_rank`` and the sender's
   skip of a non-final chunk this rank owns no page of.
8. Real sharded MLA pools: N ranks of stock-size pools hold N times one pool's
   tokens, each token stored on exactly one rank.

The real NCCL MLA/MHA/MTP gathers, zero-SM gather correctness and release,
and symmetric buffer lifetime tests live in
``test_page_interleave_gather_multi_gpu.py``. DSA indexer and attention
coverage lives in ``test_page_sharded_dsa_attention.py``.
"""

import ctypes
import gc
import math
import shutil
import unittest
import unittest.mock
from array import array
from contextlib import ExitStack, contextmanager, nullcontext
from types import SimpleNamespace

import numpy as np
import torch
from parameterized import parameterized_class

from sglang.srt.disaggregation.common.conn import CommonKVSender
from sglang.srt.disaggregation.utils import filter_kv_indices_for_shard_rank
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    _release_nccl_window_buffer,
    allocate_symmetric_buffer,
)
from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import ReqKvInfo
from sglang.srt.mem_cache import page_interleave
from sglang.srt.mem_cache.allocator.page_interleave import (
    PageInterleavePoolAllocator,
    page_interleave_shard_size,
)
from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import (
    EvictResult,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.common import _evict_until_allocatable
from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator
from sglang.srt.mem_cache.memory_pool import (
    MHATokenToKVPool,
    MLATokenToKVPool,
    ReqToTokenPool,
)
from sglang.srt.mem_cache.page_interleave import (
    PageInterleavePlacement,
    PageShardSpec,
    compute_page_shard_scratch_bytes,
    get_kv_shard_group_info,
    get_shared_kv_shard_pool,
    make_page_shard_spec,
)
from sglang.srt.mem_cache.page_interleave_pool import (
    _NCCL_SYMMETRIC_SCRATCH_MIN_VERSION,
    PageInterleaveKVPoolMixin,
    PageInterleaveMHATokenToKVPool,
    PageInterleaveMLATokenToKVPool,
    _symmetric_scratch_layout,
    _symmetric_scratch_supported,
)
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.components import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.runtime_context import get_context, get_memory, get_parallel, get_spec
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils import ceil_div
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.mem_cache_utils import finish_req
from sglang.test.test_utils import CustomTestCase, enter_scope

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

N = 4  # shard size
PS = 16  # physical page size
GS = N * PS  # full-group span (N physical pages)


class TestPageShardScratchSizing(CustomTestCase):
    @contextmanager
    def _fixture(
        self,
        *,
        enabled=True,
        draft=False,
        shard_size=4,
        use_mla=True,
        context_len=4096,
        chunk_tokens=256,
        prefill_max_requests=None,
        max_running_requests=None,
        dp_size=1,
        tp_size=4,
        num_kv_heads=8,
        head_dim=96,
        v_head_dim=64,
    ):
        group = SimpleNamespace(world_size=shard_size, rank_in_group=0)
        parallel = SimpleNamespace(
            enable_kv_cache_sharding=enabled,
            attn_cp_group=(SimpleNamespace(world_size=1) if use_mla else group),
            attn_tp_group=group,
            attn_tp_size=tp_size,
            attn_dp_size=dp_size,
        )
        schedule = SimpleNamespace(
            chunked_prefill_size=chunk_tokens,
            prefill_max_requests=prefill_max_requests,
            max_running_requests=max_running_requests,
        )
        kvc = SimpleNamespace(
            is_draft_worker=draft,
            token_to_kv_pool_allocator=None,
            use_mla_backend=use_mla,
            page_size=16,
            kv_cache_dtype=torch.bfloat16,
            model_config=SimpleNamespace(
                context_len=context_len,
                kv_lora_rank=16,
                qk_rope_head_dim=8,
                head_dim=head_dim,
                v_head_dim=v_head_dim,
                get_num_kv_heads=lambda tp: num_kv_heads // tp,
            ),
        )
        with (
            unittest.mock.patch.object(
                page_interleave, "get_parallel", return_value=parallel
            ),
            unittest.mock.patch.object(
                page_interleave, "get_schedule", return_value=schedule
            ),
        ):
            yield kvc

    def test_each_context_is_aligned_before_multiplying_by_eight(self):
        with self._fixture(context_len=4097, chunk_tokens=33) as kvc:
            spec = make_page_shard_spec(kvc)
        # Each request's prefix gather pads independently to 64 tokens.
        # Aligning only 8 * context_len would reserve too few rows.
        self.assertEqual(spec.max_prefix_tokens, 8 * 4160)
        self.assertGreater(spec.max_prefix_tokens, 32832)
        self.assertEqual(spec.chunk_tokens, 48)
        self.assertEqual(spec.max_prefix_tokens % spec.logical_page_size, 0)

    def test_target_with_mtp_budgets_one_extra_scratch_pair(self):
        with (
            self._fixture(context_len=100, chunk_tokens=64, shard_size=2) as kvc,
            get_context().override_server_args(enable_multi_layer_eagle=False),
        ):
            kvc.spec_algorithm = SpeculativeAlgorithm.EAGLE
            # Target configs can omit next-N layers; use the resolved draft count.
            kvc.model_config.num_nextn_predict_layers = None
            kvc.spec_aux_config = SimpleNamespace(eagle_draft_num_layers=1)
            expected = 2 * (8 * 128 + 64 + 16) * (16 + 8) * 2
            for layers in (1, 3):
                with self.subTest(layers=layers):
                    kvc.spec_aux_config.eagle_draft_num_layers = layers
                    self.assertEqual(compute_page_shard_scratch_bytes(kvc), expected)
                    self.assertEqual(
                        compute_page_shard_scratch_bytes(kvc, include_mtp=True),
                        2 * expected,
                    )
            kvc.spec_aux_config.eagle_draft_num_layers = None
            with self.assertRaisesRegex(ValueError, "EAGLE MTP"):
                compute_page_shard_scratch_bytes(kvc, include_mtp=True)

    def test_estimate_matches_actual_mla_and_gqa_scratch_tensors(self):
        for use_mla in (True, False):
            with (
                self.subTest(use_mla=use_mla),
                self._fixture(use_mla=use_mla, context_len=128, tp_size=2) as kvc,
            ):
                spec = make_page_shard_spec(kvc)
                estimated = compute_page_shard_scratch_bytes(kvc)
                pool = SimpleNamespace(
                    device="cpu",
                    store_dtype=kvc.kv_cache_dtype,
                    kv_cache_dim=24,
                    head_num=4,
                    head_dim=96,
                    v_head_dim=64,
                )
                pool_cls = (
                    PageInterleaveMLATokenToKVPool
                    if use_mla
                    else PageInterleaveMHATokenToKVPool
                )
                specs = pool_cls._scratch_tensor_specs(pool, spec.scratch_rows)
                actual = 2 * sum(
                    math.prod(shape) * dtype.itemsize for shape, dtype in specs.values()
                )
                self.assertEqual(estimated, actual)
                # The zero-SM buffer adds at most one window alignment per tensor.
                _, symmetric_bytes = _symmetric_scratch_layout(specs, num_slots=2)
                self.assertGreaterEqual(symmetric_bytes, estimated)
                self.assertLess(symmetric_bytes, estimated + 2 * len(specs) * 4096)


class TestZeroSmScratchLayout(CustomTestCase):
    def test_requires_nccl_with_copy_engine_race_fix(self):
        self.assertEqual(_NCCL_SYMMETRIC_SCRATCH_MIN_VERSION, 23102)
        for version in (22902, 23007, 23101, 23102):
            with (
                self.subTest(version=version),
                envs.SGLANG_ENABLE_KV_SHARD_ZERO_SM_GATHER.override(True),
                unittest.mock.patch(
                    "sglang.srt.mem_cache.page_interleave_pool.is_cuda",
                    return_value=True,
                ),
                unittest.mock.patch(
                    "sglang.srt.mem_cache.page_interleave_pool.NCCLLibrary"
                ) as library,
                unittest.mock.patch(
                    "sglang.srt.mem_cache.page_interleave_pool.in_the_same_node_as",
                    return_value=[True, True],
                ) as same_node,
            ):
                library.return_value.ncclGetRawVersion.return_value = version
                self.assertEqual(
                    _symmetric_scratch_supported(
                        memory_saver_enabled=False,
                        custom_mem_pool=None,
                        shard_group=SimpleNamespace(cpu_group=object()),
                    ),
                    version >= _NCCL_SYMMETRIC_SCRATCH_MIN_VERSION,
                )
                if version < _NCCL_SYMMETRIC_SCRATCH_MIN_VERSION:
                    same_node.assert_not_called()

    def test_slot_tensors_are_window_aligned_and_disjoint(self):
        """Both slots share one symmetric window: every tensor must start on
        NCCL's 4 KiB window alignment and no two tensors may overlap."""
        rows = 13 * 16  # neither row width below fills a whole 4 KiB block
        specs = {
            "kv": ((rows, 1, 576), torch.bfloat16),
            "index_k": ((rows // 16, 16 * 132), torch.uint8),
        }
        offsets, nbytes = _symmetric_scratch_layout(specs, num_slots=2)
        spans = sorted(
            (slot[name], slot[name] + math.prod(shape) * dtype.itemsize)
            for slot in offsets
            for name, (shape, dtype) in specs.items()
        )
        self.assertEqual(len(spans), 4)
        self.assertTrue(all(start % 4096 == 0 for start, _ in spans))
        for (_, end), (next_start, _) in zip(spans, spans[1:]):
            self.assertLessEqual(end, next_start)
        self.assertLessEqual(spans[-1][1], nbytes)


class TestSymmetricBufferFailureCleanup(CustomTestCase):
    @contextmanager
    def _fixture(self):
        comm = SimpleNamespace(
            device=torch.device("cuda:0"),
            nccl=SimpleNamespace(
                ncclMemAlloc=unittest.mock.Mock(return_value=1234),
                ncclMemFree=unittest.mock.Mock(),
            ),
            register_comm_window_raw=unittest.mock.Mock(
                return_value=ctypes.c_void_p(5678)
            ),
            deregister_comm_window=unittest.mock.Mock(),
        )
        with (
            unittest.mock.patch("torch.cuda.device", return_value=nullcontext()),
            unittest.mock.patch("torch.cuda.synchronize"),
            unittest.mock.patch("torch.as_tensor") as as_tensor,
        ):
            yield comm, as_tensor

    def test_registration_failure_frees_allocation(self):
        with self._fixture() as (comm, as_tensor):
            comm.register_comm_window_raw.side_effect = RuntimeError("register failed")
            with self.assertRaisesRegex(RuntimeError, "register failed"):
                allocate_symmetric_buffer(comm, 8192)
            gc.collect()
            as_tensor.assert_not_called()
            comm.deregister_comm_window.assert_not_called()
            comm.nccl.ncclMemFree.assert_called_once_with(1234)

    def test_tensor_setup_failures_release_once(self):
        for operation in ("wrap", "zero"):
            with (
                self.subTest(operation=operation),
                self._fixture() as (comm, as_tensor),
            ):
                if operation == "wrap":
                    as_tensor.side_effect = RuntimeError("wrap failed")
                else:
                    as_tensor.return_value.zero_.side_effect = RuntimeError(
                        "zero failed"
                    )
                with self.assertRaisesRegex(RuntimeError, f"{operation} failed"):
                    allocate_symmetric_buffer(comm, 8192)
                # Drop Mock's saved owner reference as well: a registered
                # finalizer must not run a second time after eager cleanup.
                as_tensor.reset_mock()
                gc.collect()
                comm.deregister_comm_window.assert_called_once()
                self.assertEqual(
                    comm.deregister_comm_window.call_args.args[0].value, 5678
                )
                comm.nccl.ncclMemFree.assert_called_once_with(1234)

    def test_cleanup_failure_preserves_setup_error_without_double_free(self):
        with self._fixture() as (comm, as_tensor):
            as_tensor.side_effect = RuntimeError("wrap failed")
            comm.deregister_comm_window.side_effect = RuntimeError("deregister failed")
            with (
                self.assertLogs(
                    "sglang.srt.distributed.device_communicators.pynccl_allocator",
                    level="ERROR",
                ),
                self.assertRaisesRegex(RuntimeError, "wrap failed"),
            ):
                allocate_symmetric_buffer(comm, 8192)
            as_tensor.reset_mock()
            gc.collect()
            comm.deregister_comm_window.assert_called_once()
            comm.nccl.ncclMemFree.assert_called_once_with(1234)

    def test_deregister_failure_still_frees_allocation(self):
        with self._fixture() as (comm, _):
            comm.deregister_comm_window.side_effect = RuntimeError("deregister failed")
            with self.assertRaisesRegex(RuntimeError, "deregister failed"):
                _release_nccl_window_buffer(comm, 1234, ctypes.c_void_p(5678))
            comm.nccl.ncclMemFree.assert_called_once_with(1234)


def _make_mtp_configurator(use_mla=True):
    # Keep real pool/allocator types for validation; allocate only CPU metadata.
    pool_class = (
        PageInterleaveMLATokenToKVPool if use_mla else PageInterleaveMHATokenToKVPool
    )
    target = pool_class.__new__(pool_class)
    target.shard_spec = PageShardSpec(1, 2, 16, 128, 64)
    target.shard_group = SimpleNamespace(rank_in_group=1, world_size=2)
    target.size, target.page_size, target.dtype = 128, 16, torch.bfloat16
    target.kv_lora_rank, target.qk_rope_head_dim = 8, 4
    target.head_num, target.head_dim, target.v_head_dim = 2, 4, 6
    kvc = KVCacheConfigurator.__new__(KVCacheConfigurator)
    kvc.token_to_kv_pool_allocator = PageInterleavePoolAllocator(
        size=target.size,
        physical_page_size=target.page_size,
        shard_size=target.shard_spec.shard_size,
        dtype=target.dtype,
        device="cpu",
        kvcache=target,
        need_sort=False,
        shard_spec=target.shard_spec,
    )
    kvc.device = "cpu"
    kvc.is_draft_worker = True
    kvc.page_size = target.page_size
    kvc.use_mla_backend = use_mla
    kvc.kv_cache_dtype = target.dtype
    kvc.kv_cache_dtype_str = "bfloat16"
    kvc.spec_algorithm = SpeculativeAlgorithm.EAGLE
    kvc.model_config = SimpleNamespace(
        num_nextn_predict_layers=1,
        context_len=9999,
        kv_lora_rank=8,
        qk_rope_head_dim=4,
        head_dim=4,
        v_head_dim=6,
        get_num_kv_heads=lambda tp: 4 // tp,
        hf_config=SimpleNamespace(architectures=["DeepseekV3ForCausalLM"]),
    )
    kvc.layer_info = SimpleNamespace(start_layer=5, end_layer=6, num_effective_layers=1)
    kvc.is_hybrid_swa = False
    kvc.sliding_window_size = None
    kvc.mambaish_config = None
    kvc.post_capture_kv_active = False
    return kvc, target


def _build_mtp_pool(kvc, *, capacity=128, is_dsa_model=False, is_dsv4_model=False):
    return kvc._build_token_to_kv_pool(
        sizes=SimpleNamespace(max_total_num_tokens=capacity),
        is_dsa_model=is_dsa_model,
        is_dsv4_model=is_dsv4_model,
        req_to_token_pool=None,
    )


class TestMTPKVShardConfig(CustomTestCase):
    def setUp(self):
        super().setUp()
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(
            get_context().override_server_args(
                page_size=16, chunked_prefill_size=1024, enable_multi_layer_eagle=False
            )
        )
        stack.enter_context(
            get_parallel().override(
                tp_size=2, attn_tp_size=2, moe_tp_size=2, attn_dcp_size=1
            )
        )

    def test_draft_shares_target_layout_but_has_its_own_pool(self):
        for use_mla in (True, False):
            with self.subTest(use_mla=use_mla):
                kvc, target = _make_mtp_configurator(use_mla)
                pool_class = type(target)
                # Mock allocation only; preserve real types and distinct instances.
                with (
                    unittest.mock.patch.object(
                        pool_class, "__init__", return_value=None
                    ) as init,
                    unittest.mock.patch.object(
                        page_interleave,
                        "get_kv_shard_group",
                        side_effect=AssertionError,
                    ),
                ):
                    self.assertIs(get_shared_kv_shard_pool(kvc), target)
                    self.assertEqual(get_kv_shard_group_info(kvc), (1, 2))
                    self.assertIs(make_page_shard_spec(kvc), target.shard_spec)
                    pool = _build_mtp_pool(kvc)
                self.assertIsInstance(pool, pool_class)
                self.assertIsNot(pool, target)
                kwargs = init.call_args.kwargs
                self.assertIs(kwargs["shard_spec"], target.shard_spec)
                self.assertIs(kwargs["shard_group"], target.shard_group)
                for name, expected in dict(
                    size=128,
                    page_size=16,
                    dtype=torch.bfloat16,
                    layer_num=1,
                    start_layer=5,
                    end_layer=6,
                ).items():
                    self.assertEqual(kwargs[name], expected, name)
                allocator = kvc.token_to_kv_pool_allocator
                self.assertIs(
                    kvc._build_token_to_kv_pool_allocator(
                        sizes=SimpleNamespace(max_total_num_tokens=128),
                        token_to_kv_pool=pool,
                        is_dsv4_model=False,
                        req_to_token_pool=None,
                        token_to_kv_pool_allocator=allocator,
                    ),
                    allocator,
                )
                # Scratch uses inherited bounds despite different draft context/chunk sizes.
                row_bytes = (8 + 4) * 2 if use_mla else 2 * (4 + 6) * 2
                self.assertEqual(
                    compute_page_shard_scratch_bytes(kvc),
                    2 * (128 + 64 + 16) * row_bytes,
                )
                with self.assertRaisesRegex(ValueError, "target"):
                    compute_page_shard_scratch_bytes(kvc, include_mtp=True)

    def test_without_shared_allocator_keeps_existing_pool(self):
        for draft, allocator in ((True, None), (True, object()), (False, None)):
            with self.subTest(draft=draft, allocator=allocator):
                kvc, _ = _make_mtp_configurator()
                kvc.is_draft_worker = draft
                kvc.token_to_kv_pool_allocator = allocator
                self.assertIsNone(get_shared_kv_shard_pool(kvc))
                if draft:
                    self.assertEqual(get_kv_shard_group_info(kvc), (None, 1))
                    self.assertIsNone(make_page_shard_spec(kvc))
                    self.assertEqual(compute_page_shard_scratch_bytes(kvc), 0)
                with unittest.mock.patch.object(
                    KVCacheConfigurator, "_build_mla_kv_pool"
                ) as build:
                    self.assertIs(_build_mtp_pool(kvc), build.return_value)
                build.assert_called_once_with(
                    max_total_num_tokens=128, mla_pool_class=MLATokenToKVPool
                )

    def test_rejects_invalid_shared_allocator(self):
        kvc, _ = _make_mtp_configurator()
        for name, value, message in (
            ("_kvcache", object(), "target's sharded KV pool"),
            ("shard_spec", PageShardSpec(0, 2, 16, 128, 64), "different shard layouts"),
            ("shard_size", 4, "different shard layouts"),
            ("page_size", 32, "different shard layouts"),
            ("size", 128, "different shard layouts"),
        ):
            with (
                self.subTest(attribute=name),
                unittest.mock.patch.object(kvc.token_to_kv_pool_allocator, name, value),
                self.assertRaisesRegex(ValueError, message),
            ):
                get_shared_kv_shard_pool(kvc)

    def test_rejects_incompatible_pool_geometry(self):
        for use_mla in (True, False):
            kvc, _ = _make_mtp_configurator(use_mla)
            geometry = (
                ("kv_lora_rank", "qk_rope_head_dim")
                if use_mla
                else ("head_dim", "v_head_dim")
            )
            cases = [
                (kvc, "page_size", 32, "physical page size"),
                (kvc, "kv_cache_dtype", torch.float16, "geometry and dtype"),
                (kvc, "use_mla_backend", not use_mla, "attention geometry"),
                *(
                    (kvc.model_config, field, 16, "geometry and dtype")
                    for field in geometry
                ),
            ]
            if not use_mla:
                cases.append(
                    (
                        kvc.model_config,
                        "get_num_kv_heads",
                        lambda tp: 4,
                        "geometry and dtype",
                    )
                )
            for obj, name, value, message in cases:
                with (
                    self.subTest(use_mla=use_mla, attribute=name),
                    unittest.mock.patch.object(obj, name, value),
                    self.assertRaisesRegex(ValueError, message),
                ):
                    _build_mtp_pool(kvc)
            for capacity in (64, 256):
                with (
                    self.subTest(use_mla=use_mla, capacity=capacity),
                    self.assertRaisesRegex(ValueError, "same per-rank capacity"),
                ):
                    _build_mtp_pool(kvc, capacity=capacity)

    def test_rejects_unsupported_draft_config(self):
        kvc, _ = _make_mtp_configurator()
        cases = [
            (kvc, "spec_algorithm", algorithm, "EAGLE MTP")
            for algorithm in (
                SpeculativeAlgorithm.EAGLE3,
                SpeculativeAlgorithm.DFLASH,
                SpeculativeAlgorithm.FROZEN_KV_MTP,
            )
        ] + [
            (kvc.model_config, "num_nextn_predict_layers", None, "EAGLE MTP"),
            (kvc.model_config, "num_nextn_predict_layers", 0, "EAGLE MTP"),
            (kvc, "is_hybrid_swa", True, "dense MLA and MHA"),
            (kvc, "mambaish_config", object(), "dense MLA and MHA"),
            (kvc, "sliding_window_size", 128, "dense MLA and MHA"),
            (kvc, "post_capture_kv_active", True, "plain per-layer KV layout"),
            (kvc, "kv_cache_dtype_str", "mxfp8", "plain per-layer KV layout"),
        ]
        for obj, name, value, message in cases:
            with (
                self.subTest(attribute=name, value=value),
                unittest.mock.patch.object(obj, name, value),
                self.assertRaisesRegex(ValueError, message),
            ):
                _build_mtp_pool(kvc)
        for options in ({"is_dsa_model": True}, {"is_dsv4_model": True}):
            with (
                self.subTest(options=options),
                self.assertRaisesRegex(ValueError, "dense MLA and MHA"),
            ):
                _build_mtp_pool(kvc, **options)
        for context, options, message in (
            (get_spec(), {"enable_multi_layer_eagle": True}, "EAGLE MTP"),
            (get_parallel(), {"attn_dcp_size": 2}, "incompatible with DCP"),
            (
                get_memory(),
                {"enable_page_major_kv_layout": True},
                "plain per-layer KV layout",
            ),
        ):
            with (
                self.subTest(options=options),
                context.override(**options),
                self.assertRaisesRegex(ValueError, message),
            ):
                _build_mtp_pool(kvc)


def _make_spec(shard_rank=0, max_prefix_groups=64, chunk_pages=32):
    return PageShardSpec(
        shard_rank=shard_rank,
        shard_size=N,
        page_size=PS,
        max_prefix_tokens=max_prefix_groups * GS,
        chunk_tokens=chunk_pages * PS,
    )


def _make_allocator(pages_per_rank=32, need_sort=False):
    return PageInterleavePoolAllocator(
        size=pages_per_rank * PS,  # physical token slots of one rank
        physical_page_size=PS,
        shard_size=N,
        dtype=torch.bfloat16,
        device="cpu",
        kvcache=None,
        need_sort=need_sort,
    )


def _alloc_extend_batch(alloc, prefix_lens, seq_lens, rotation_bases, last_locs=None):
    """Drive alloc_extend for a batch; rotation_bases is resolved in place."""
    if last_locs is None:
        last_locs = [-1] * len(prefix_lens)
    return alloc.alloc_extend(
        prefix_lens=torch.tensor(prefix_lens, dtype=torch.int64),
        prefix_lens_cpu=torch.tensor(prefix_lens, dtype=torch.int64),
        seq_lens=torch.tensor(seq_lens, dtype=torch.int64),
        seq_lens_cpu=torch.tensor(seq_lens, dtype=torch.int64),
        last_loc=torch.tensor(last_locs, dtype=torch.int64),
        extend_num_tokens=sum(s - p for p, s in zip(prefix_lens, seq_lens)),
        rotation_bases=rotation_bases,
    )


def _alloc_extend(alloc, prefix_len, seq_len, rotation_base, last_loc=-1):
    return _alloc_extend_batch(
        alloc, [prefix_len], [seq_len], [rotation_base], [last_loc]
    )


class TestClassedAllocator(CustomTestCase):
    def test_index_space_widened_classes_mirror_ranks(self):
        alloc = _make_allocator(pages_per_rank=32)
        self.assertEqual(alloc.size, 32 * PS * N)  # logical slots
        self.assertEqual(alloc.page_size, PS)  # the PHYSICAL page quantum
        self.assertEqual(page_interleave_shard_size(alloc), N)
        # Class r holds exactly rank r's allocatable pages: l % N == r,
        # local pages 1..32 (page 0 reserved on every rank).
        self.assertEqual(alloc.class_free_page_counts(), [32] * N)
        for r in range(N):
            pages = alloc.class_free_pages[r]
            self.assertTrue(bool((pages % N == r).all()))
            self.assertTrue(torch.equal(pages // N, torch.arange(1, 33)))

    def test_rotation_worked_example_zero_stranding(self):
        """Two-turn worked example at ps=16: turn 1 allocates cyclic owners
        from the root base; the turn-boundary free returns its page whole and
        immediately reusable; turn 2 continues the rotation and reuses the
        freed page before any fresh one."""
        alloc = _make_allocator()
        total = alloc.available_size()

        # Turn 1: 122 tokens = 8 position-pages, root base 0.
        base = alloc.least_full_class()
        self.assertEqual(base, 0)  # all classes equal -> lowest id
        out = _alloc_extend(alloc, 0, 122, base)
        pages = out[::PS] // PS
        # Owners exactly cyclic from the base; in-page offsets positional.
        self.assertTrue(torch.equal(pages % N, torch.arange(8) % N))
        self.assertTrue(torch.equal(out % PS, torch.arange(122) % PS))

        # Boundary: cache 112 (7 pages), free the sub-ps tail's page whole.
        alloc.free(out[112:122])
        # Page 7's owner is (0 + 7) % 4 = 3: back on class 3, reusable now.
        self.assertEqual(alloc.class_free_page_counts(), [30, 30, 30, 31])

        # Turn 2: prefix 112, extend to 244 (9 new pages P7..P15).
        out2 = _alloc_extend(alloc, 112, 244, base, last_loc=int(out[111]))
        pages2 = out2[::PS] // PS
        self.assertTrue(torch.equal(pages2 % N, (7 + torch.arange(9)) % N))
        # The freed page is the class-3 head: reused before any fresh page.
        self.assertEqual(int(pages2[0]), int(pages[7]))
        # Nothing stranded: freeing the chain restores full capacity.
        alloc.free(out[:112])
        alloc.free(out2)
        self.assertEqual(alloc.available_size(), total)
        self.assertEqual(alloc.class_free_page_counts(), [32] * N)

    def test_min_class_admission_accounting(self):
        """available_size is the MIN-CLASS floor: draining one class must
        zero the admission budget even while the aggregate stays large —
        an aggregate gate would over-admit into the alloc path's fail-loud
        RuntimeError when the tight class is protected."""
        alloc = _make_allocator(pages_per_rank=4)
        outs = [_alloc_extend(alloc, 0, PS, rotation_base=3) for _ in range(4)]
        self.assertEqual(alloc.class_free_page_counts(), [4, 4, 4, 0])
        self.assertEqual(alloc.available_size(), 0)
        self.assertEqual(alloc.aggregate_free_size(), 12 * PS)
        # A draw needing the empty class defers (None), never raises.
        self.assertIsNone(_alloc_extend(alloc, 0, N * PS, rotation_base=0))
        # A free of one class-3 page lifts the floor by one page per class.
        alloc.free(outs[0])
        self.assertEqual(alloc.available_size(), N * PS)

    def test_least_full_root_seeding(self):
        """Roots draw from the class with the most free pages (ties: lowest
        id). Uniform 1-page roots therefore spread with skew <= 1."""
        alloc = _make_allocator(pages_per_rank=32)
        for i in range(2 * N + 1):
            base = alloc.least_full_class()
            _alloc_extend(alloc, 0, PS, rotation_base=base)
            counts = alloc.class_free_page_counts()
            self.assertLessEqual(max(counts) - min(counts), 1, counts)
        # 9 single-page roots at N=4: classes filled 3,2,2,2.
        self.assertEqual(alloc.class_free_page_counts(), [29, 30, 30, 30])

    def test_chain_rotation_run_property(self):
        """Within one chain (root + arbitrary ps-aligned extensions) the
        owners are exactly cyclic, so per-rank owned page counts differ by
        <= 1 — the padded-allgather block contract ceil(K/N). Guards the
        class-interleave scatter in alloc_extend."""
        for shard_size in (2, 4, 8):
            alloc = PageInterleavePoolAllocator(
                size=256 * PS,
                physical_page_size=PS,
                shard_size=shard_size,
                dtype=torch.bfloat16,
                device="cpu",
                kvcache=None,
                need_sort=False,
            )
            lens = [3 * PS, 5 * PS, PS, 7 * PS]  # chunked extensions
            base = alloc.least_full_class()
            chain = []
            prefix = 0
            for ext in lens:
                out = _alloc_extend(alloc, prefix, prefix + ext, base)
                chain.append(out)
                prefix += ext
            locs = torch.cat(chain)
            pages = locs[::PS] // PS
            owners = pages % shard_size
            expect = torch.arange(pages.numel()) % shard_size
            self.assertTrue(torch.equal(owners, (int(owners[0]) + expect) % shard_size))
            per_rank = torch.bincount(owners, minlength=shard_size)
            self.assertLessEqual(int(per_rank.max() - per_rank.min()), 1)

    def test_free_splits_by_owner_class(self):
        alloc = _make_allocator()
        out = _alloc_extend(alloc, 0, 6 * PS, rotation_base=1)
        before = alloc.class_free_page_counts()
        # Free pages 2 and 3 of the chain (owners 3 and 0) in one call, via
        # the free-group batching path the scheduler uses.
        alloc.free_group_begin()
        alloc.free(out[2 * PS : 3 * PS])
        alloc.free(out[3 * PS : 4 * PS])
        alloc.free_group_end()
        after = alloc.class_free_page_counts()
        deltas = [a - b for a, b in zip(after, before)]
        self.assertEqual(deltas, [1, 0, 0, 1])  # classes (1+2)%4=3 and (1+3)%4=0

    def test_grouped_free_owns_indices_before_caller_mutation(self):
        """Deferred frees must snapshot req_to_token views: the scheduler may
        overwrite the backing row before free_group_end consumes them."""
        alloc = _make_allocator()
        out = _alloc_extend(alloc, 0, 2 * PS, rotation_base=2)
        first_page = out[:PS]
        owner = int(first_page[0] // PS % N)
        before = alloc.class_free_page_counts()

        alloc.free_group_begin()
        alloc.free(first_page)
        first_page.zero_()
        alloc.free_group_end()

        after = alloc.class_free_page_counts()
        self.assertEqual(after[owner], before[owner] + 1)
        self.assertEqual(
            [after[r] - before[r] for r in range(N)],
            [1 if r == owner else 0 for r in range(N)],
        )

    def test_free_segment_returns_pages_to_their_classes(self):
        # The radix cache frees through free_segment/free_segments. The paged
        # base routes those to the stock free_pages list, which this allocator
        # never reads, so the override must land them in the class lists.
        alloc = _make_allocator()
        total = alloc.available_size()
        out = _alloc_extend(alloc, 0, 3 * PS, rotation_base=2)
        self.assertLess(alloc.available_size(), total)
        alloc.free_segment(out, start_pos=0)
        self.assertEqual(alloc.available_size(), total)
        self.assertEqual(alloc.class_free_page_counts(), [32] * N)

    def test_free_segments_splits_at_a_page_boundary(self):
        alloc = _make_allocator()
        total = alloc.available_size()
        out = _alloc_extend(alloc, 0, 4 * PS, rotation_base=0)
        # Two disjoint ascending segments of one request's kv row.
        alloc.free_segments([(out[: 2 * PS], 0), (out[2 * PS :], 2 * PS)])
        self.assertEqual(alloc.available_size(), total)
        self.assertEqual(alloc.class_free_page_counts(), [32] * N)

    def test_need_sort_merges_per_class(self):
        alloc = _make_allocator(pages_per_rank=4, need_sort=True)
        out = _alloc_extend(alloc, 0, 4 * N * PS, rotation_base=0)  # everything
        self.assertEqual(alloc.available_size(), 0)
        alloc.free(out)  # lands in the per-class release lists
        self.assertEqual(alloc.available_size(), 4 * N * PS)
        # A fresh draw forces the per-class merge+sort and succeeds.
        out2 = _alloc_extend(alloc, 0, N * PS, rotation_base=0)
        self.assertIsNotNone(out2)
        pages = out2[::PS] // PS
        self.assertTrue(torch.equal(pages % N, torch.arange(N) % N))

    def test_unsupported_paths_fail_loud(self):
        alloc = _make_allocator()
        with self.assertRaises(NotImplementedError):
            alloc.alloc(GS)
        with self.assertRaises(NotImplementedError):
            alloc.alloc_decode(
                torch.tensor([PS + 1]), torch.tensor([PS + 1]), torch.tensor([PS - 1])
            )

    def test_batch_alloc_per_request_rotation(self):
        """bs > 1: each request draws its own cyclic run; out_cache_loc is
        the batch-order concatenation (write_cache_indices' contract), and a
        None base is resolved from the least-full class AT THAT REQUEST'S
        TURN — the draw must see earlier requests' pops in the same batch,
        or uniform short batches would all pile onto one class."""
        alloc = _make_allocator()
        # req0: extension of a base-1 chain with a 2-page prefix;
        # req1 and req2: new chains (drawn in place).
        bases = [1, None, None]
        out = _alloc_extend_batch(
            alloc,
            prefix_lens=[2 * PS, 0, 0],
            seq_lens=[5 * PS, 3 * PS, PS],
            rotation_bases=bases,
            # req0's last prefix page must carry owner (1 + 1) % 4 = 2.
            last_locs=[(5 * N + 2) * PS + PS - 1, -1, -1],
        )
        self.assertEqual(out.numel(), 3 * PS + 3 * PS + PS)
        # Batch-order concatenation, per-request cyclic owners.
        pages = out[::PS] // PS
        self.assertTrue(
            torch.equal(pages[:3] % N, (1 + 2 + torch.arange(3)) % N)  # req0
        )
        b1, b2 = bases[1], bases[2]
        self.assertIsNotNone(b1)
        self.assertIsNotNone(b2)
        self.assertTrue(torch.equal(pages[3:6] % N, (b1 + torch.arange(3)) % N))
        self.assertEqual(int(pages[6]) % N, b2)
        # req1's draw saw req0's pops (classes 3,0,1 used once each -> class
        # 2 is fullest... all equal except used {3,0,1} -> least-full = 2);
        # req2's draw saw req1's pops on top.
        self.assertEqual(b1, 2)
        self.assertEqual(b2, 1)  # after req1 used {2,3,0}: class 1 fullest
        # The whole batch is one no-duplicate allocation.
        self.assertEqual(len(torch.unique(out)), out.numel())

    def test_batch_alloc_defers_whole_when_a_class_is_short(self):
        """A batch either commits whole or returns None (mirrored decision):
        partial commits would desync the free lists from the retry."""
        alloc = _make_allocator(pages_per_rank=2)
        counts_before = alloc.class_free_page_counts()
        out = _alloc_extend_batch(
            alloc,
            prefix_lens=[0, 0],
            seq_lens=[4 * PS, 5 * PS],  # 9 pages: class need exceeds 2 somewhere
            rotation_bases=[0, 0],
        )
        self.assertIsNone(out)
        self.assertEqual(alloc.class_free_page_counts(), counts_before)


class TestEvictUntilAllocatable(CustomTestCase):
    """The evict-then-allocate contract under min-class accounting: one
    evict() sized in tokens can raise the tight class by less than the
    tokens it freed (evicted pages spread across classes), so the alloc
    path iterates. Guards the two termination conditions of
    _evict_until_allocatable."""

    def _allocator_with_tight_class(self):
        alloc = _make_allocator(pages_per_rank=4)
        # Four 1-page chains, all in class 3: the tight class.
        outs = [_alloc_extend(alloc, 0, PS, rotation_base=3) for _ in range(4)]
        assert alloc.available_size() == 0
        return alloc, outs

    def _tree_stub(self, alloc, frees):
        stub = SimpleNamespace(calls=0)

        def evict(params):
            stub.calls += 1
            if not frees:
                return EvictResult(num_tokens_evicted=0)
            head = frees.pop(0)
            alloc.free(head)
            return EvictResult(num_tokens_evicted=head.numel())

        stub.evict = evict
        return stub

    def test_iterates_until_min_class_covers(self):
        alloc, outs = self._allocator_with_tight_class()
        # Each round frees ONE class-3 page (a whole 1-page chain): reaching
        # a min-class floor of 2 pages takes 2 rounds.
        tree = self._tree_stub(alloc, list(outs))
        _evict_until_allocatable(tree, alloc, 2 * N * PS)
        self.assertGreaterEqual(alloc.available_size(), 2 * N * PS)
        self.assertEqual(tree.calls, 2)

    def test_terminates_when_tree_dry(self):
        alloc, _ = self._allocator_with_tight_class()
        tree = self._tree_stub(alloc, [])  # nothing evictable
        _evict_until_allocatable(tree, alloc, PS)
        self.assertEqual(alloc.available_size(), 0)  # need unmet, but no hang
        self.assertEqual(tree.calls, 1)


def _unified_tree(page_size=4, pool_size=256, disable=False):
    """A CPU-only UnifiedRadixCache with only the Full component."""
    dtype = torch.float16
    kv_pool = MHATokenToKVPool(
        size=pool_size,
        page_size=page_size,
        dtype=dtype,
        head_num=2,
        head_dim=8,
        layer_num=1,
        device="cpu",
        enable_memory_saver=False,
    )
    allocator = PagedTokenToKVPoolAllocator(
        size=pool_size,
        page_size=page_size,
        dtype=dtype,
        device="cpu",
        kvcache=kv_pool,
        need_sort=False,
    )
    req_pool = ReqToTokenPool(
        size=8,
        max_context_len=128,
        device="cpu",
        enable_memory_saver=False,
    )
    return UnifiedRadixCache(
        CacheInitParams(
            disable=disable,
            req_to_token_pool=req_pool,
            token_to_kv_pool_allocator=allocator,
            page_size=page_size,
            eviction_policy="lru",
            tree_components=(ComponentType.FULL,),
        )
    )


def _insert(tree, tokens, rotation_base=None, value=None):
    if value is None:
        value = tree.token_to_kv_pool_allocator.alloc(len(tokens))
    return tree.insert(
        InsertParams(
            key=RadixKey(array("q", tokens)),
            value=value.to(dtype=torch.int64),
            rotation_base=rotation_base,
        )
    )


def _match_len(tree, tokens):
    res = tree.match_prefix(MatchPrefixParams(key=RadixKey(array("q", tokens))))
    return res.device_prefix_len


class _TreeCoreBackendCase(CustomTestCase):
    def setUp(self):
        super().setUp()
        if self.tree_core_backend == "rust" and shutil.which("cargo") is None:
            self.skipTest("the Rust backend builds with cargo")
        override = envs.SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND.override(
            self.tree_core_backend
        )
        override.__enter__()
        self.addCleanup(override.__exit__, None, None, None)


@parameterized_class(("tree_core_backend",), [("python",), ("rust",)])
class TestUnifiedRotationBase(_TreeCoreBackendCase):
    """The host rotation base on UnifiedTreeNode: the one new piece of
    metadata. The Full component's value is a device tensor, so the base must
    survive inserts and splits purely host-side or the alloc path gains a D2H
    sync."""

    def test_insert_stamps_split_copies(self):
        tree = _unified_tree()
        _insert(tree, list(range(12)), rotation_base=2)
        # A shorter lookup splits the node at the match boundary: BOTH halves
        # keep the chain's base (position-page P keeps owner (b+P)%N on both
        # sides of any split).
        probe = list(range(8)) + [99, 98, 97, 96]
        _insert(tree, probe, rotation_base=2)
        res = tree.match_prefix(MatchPrefixParams(key=RadixKey(array("q", probe))))
        self.assertEqual(tree.rotation_base_of(res.last_device_node), 2)
        for tokens in (list(range(8)), list(range(12))):
            matched = tree.match_prefix(
                MatchPrefixParams(key=RadixKey(array("q", tokens)))
            )
            self.assertEqual(tree.rotation_base_of(matched.last_device_node), 2)

    def test_new_chain_gets_its_own_base(self):
        tree = _unified_tree()
        _insert(tree, list(range(8)), rotation_base=1)
        _insert(tree, list(range(100, 108)), rotation_base=3)
        r1 = tree.match_prefix(MatchPrefixParams(key=RadixKey(array("q", range(8)))))
        r2 = tree.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", range(100, 108))))
        )
        self.assertEqual(tree.rotation_base_of(r1.last_device_node), 1)
        self.assertEqual(tree.rotation_base_of(r2.last_device_node), 3)

    def test_extension_tail_node_stamped_from_request(self):
        tree = _unified_tree()
        _insert(tree, list(range(8)), rotation_base=1)
        # A longer insert of the same chain dedups the prefix and stamps the
        # tail node with the (same, chain-constant) base.
        _insert(tree, list(range(16)), rotation_base=1)
        res = tree.match_prefix(MatchPrefixParams(key=RadixKey(array("q", range(16)))))
        self.assertEqual(tree.rotation_base_of(res.last_device_node), 1)

    def test_unsharded_inserts_keep_none(self):
        tree = _unified_tree()
        _insert(tree, list(range(8)))
        res = tree.match_prefix(MatchPrefixParams(key=RadixKey(array("q", range(8)))))
        self.assertIsNone(tree.rotation_base_of(res.last_device_node))

    def test_rotation_base_of_reads_through_the_cache_boundary(self):
        """The alloc path holds a NodeId, not a node: the base must be
        readable through the tree-cache API (BasePrefixCache.rotation_base_of
        defaults to None, so an unsharded cache sends the alloc path to the
        request's recorded base)."""
        tree = _unified_tree()
        _insert(tree, list(range(8)), rotation_base=2)
        res = tree.match_prefix(MatchPrefixParams(key=RadixKey(array("q", range(8)))))
        self.assertEqual(tree.rotation_base_of(res.last_device_node), 2)
        self.assertIsNone(tree.rotation_base_of(None))
        self.assertIsNone(tree.rotation_base_of(tree.tree_core.root_node_handle()))


@parameterized_class(("tree_core_backend",), [("python",), ("rust",)])
class TestShardedCoreGate(_TreeCoreBackendCase):
    """A tree core that does not model rotation_base would never decline a
    cross-base graft. Pairing one with a sharded allocator must fail at
    construction, not produce wrong-owner gathers at serve time."""

    def test_core_supports_rotation_base(self):
        tree = _unified_tree()
        self.assertTrue(tree.tree_core.supports_rotation_base)

    def test_sharded_allocator_rejects_a_core_without_rotation_base(self):
        tree = _unified_tree()
        params = CacheInitParams(
            disable=False,
            req_to_token_pool=tree.req_to_token_pool,
            token_to_kv_pool_allocator=_make_allocator(pages_per_rank=8),
            page_size=PS,
            eviction_policy="lru",
            tree_components=(ComponentType.FULL,),
        )
        with unittest.mock.patch.object(
            type(tree.tree_core), "supports_rotation_base", False
        ):
            with self.assertRaisesRegex(ValueError, "rotation bases"):
                UnifiedRadixCache(params)

    def test_unsharded_allocator_accepts_any_core(self):
        tree = _unified_tree()
        params = CacheInitParams(
            disable=False,
            req_to_token_pool=ReqToTokenPool(
                size=8, max_context_len=128, device="cpu", enable_memory_saver=False
            ),
            token_to_kv_pool_allocator=tree.token_to_kv_pool_allocator,
            page_size=4,
            eviction_policy="lru",
            tree_components=(ComponentType.FULL,),
        )
        with unittest.mock.patch.object(
            type(tree.tree_core), "supports_rotation_base", False
        ):
            UnifiedRadixCache(params)  # no raise: sharding is off


class _GraftReq:
    """Minimal Req stand-in for checkpoint."""

    def __init__(self, fill_ids, req_pool_idx=0):
        self.fill_ids = list(fill_ids)
        self.origin_input_ids = array("q", fill_ids)
        self.full_untruncated_fill_ids = self.origin_input_ids
        self.output_ids = array("q", [])
        self.kv = ReqKvInfo(req_pool_idx=req_pool_idx)
        self.extra_key = None
        self.cache_salt = None
        self.prefix_len = 0
        self.last_node = None
        self.priority = 0
        self.kv_rotation_base = None
        self.lock = None
        self.finished_reason = None
        self.session = None
        self.session_id = None

    def get_fill_ids(self):
        return array("q", self.fill_ids)

    def refresh_fill_ids(self):
        pass  # fill ids are fixed for the stand-in

    def finished(self):
        return self.finished_reason is not None


@parameterized_class(("tree_core_backend",), [("python",), ("rust",)])
class TestRotationGraftDecline(_TreeCoreBackendCase):
    """The overlap disagg-prefill loop plans batch t+1 before batch t's radix
    insert lands, so two requests sharing a prefix can allocate under
    different rotation bases. Grafting the second one's tail under the first
    chain leaves the cached path's page owners not one cyclic run, so a later
    reader either crashes on a negative allgather pad or silently reads the
    wrong rank's scratch rows. Inserts must refuse the graft.

    Unlike the flat radix tree, the unified insert also transfers page
    ownership inside the MATCHED region (an evicted node restored from the
    request's fresh pages, a component re-pointing a matched node's Full value
    at them). The decline therefore runs before the walk and refuses the whole
    insert, which additionally keeps the walk's duplicate frees from running
    under a request that is about to keep its own pages.
    """

    PS = 4  # tree quantum for these tests

    def _tree_with_spy(self):
        """Record every KV row release. The unified tree frees through two
        seams: the caller's free_kv_row -> free_segments, and the insert
        walk's FreeDeviceKV -> free_segment."""
        tree = _unified_tree(page_size=self.PS)
        allocator = tree.token_to_kv_pool_allocator
        freed = []
        real_free_segments = allocator.free_segments
        real_free_segment = allocator.free_segment

        def spy_segments(segments):
            freed.extend(torch.as_tensor(seg).clone() for seg, _start in segments)
            return real_free_segments(segments)

        def spy_segment(free_index, start_pos):
            freed.append(torch.as_tensor(free_index).clone())
            return real_free_segment(free_index, start_pos=start_pos)

        allocator.free_segments = spy_segments
        allocator.free_segment = spy_segment
        return tree, freed

    def _seed_chain(self, tree, tokens, base):
        _insert(tree, tokens, rotation_base=base)

    def _own_row(self, tree, req, n_tokens):
        """Give the request its own allocated KV row and return the locs."""
        locs = tree.token_to_kv_pool_allocator.alloc(n_tokens).to(dtype=torch.int64)
        tree.req_to_token_pool.req_to_token[req.kv.req_pool_idx, :n_tokens] = locs
        return locs

    def test_foreign_base_tail_declined(self):
        tree = _unified_tree(page_size=self.PS)
        self._seed_chain(tree, list(range(12)), base=1)
        # Same 8-token prefix, different suffix, allocated under base 3.
        key = list(range(8)) + [90, 91, 92, 93]
        res = _insert(tree, key, rotation_base=3)
        self.assertTrue(res.rotation_tail_declined)
        self.assertEqual(res.prefix_len, 8)
        # The suffix is NOT cached: a full-key match stops at the seam.
        self.assertEqual(_match_len(tree, key), 8)

    def test_empty_page_aligned_key_insert(self):
        """A finished request with fewer cached tokens than one tree page
        inserts an EMPTY page-aligned key; the empty-key early return must
        precede the rotation pre-flight."""
        tree = _unified_tree(page_size=self.PS)
        res = _insert(tree, [1, 2], rotation_base=1)
        self.assertEqual(res.prefix_len, 0)
        self.assertFalse(res.rotation_tail_declined)

    def test_same_base_tail_attaches(self):
        tree = _unified_tree(page_size=self.PS)
        self._seed_chain(tree, list(range(12)), base=1)
        key = list(range(8)) + [90, 91, 92, 93]
        res = _insert(tree, key, rotation_base=1)
        self.assertFalse(res.rotation_tail_declined)
        self.assertEqual(_match_len(tree, key), 12)

    def test_no_matched_chain_never_declines(self):
        """A request that matches nothing starts its own chain: the guard
        only fires against an EXISTING chain's base."""
        tree = _unified_tree(page_size=self.PS)
        self._seed_chain(tree, list(range(12)), base=1)
        res = _insert(tree, list(range(50, 62)), rotation_base=3)
        self.assertFalse(res.rotation_tail_declined)
        self.assertEqual(_match_len(tree, list(range(50, 62))), 12)

    def test_unsharded_insert_onto_a_based_chain_never_declines(self):
        """rotation_base=None means sharding is off for this insert: the
        pre-flight must not fire, or every unsharded path would stop caching."""
        tree = _unified_tree(page_size=self.PS)
        self._seed_chain(tree, list(range(8)), base=1)
        key = list(range(8)) + [90, 91, 92, 93]
        res = _insert(tree, key, rotation_base=None)
        self.assertFalse(res.rotation_tail_declined)
        self.assertEqual(_match_len(tree, key), 12)

    def test_full_match_under_a_foreign_base_declines(self):
        """No tail to graft, but the unified insert would still hand the
        request's pages to the matched chain (unevict-on-insert, component
        Full re-point). The pre-flight declines that too."""
        tree = _unified_tree(page_size=self.PS)
        self._seed_chain(tree, list(range(12)), base=1)
        res = _insert(tree, list(range(12)), rotation_base=3)
        self.assertTrue(res.rotation_tail_declined)

    def test_insert_decline_keeps_request_on_own_pages(self):
        tree, freed = self._tree_with_spy()
        self._seed_chain(tree, list(range(8)), base=1)
        req = _GraftReq(list(range(8)) + [90, 91, 92, 93])
        req.kv_rotation_base = 3
        own_locs = self._own_row(tree, req, 12)
        tree.checkpoint(req, up_to=len(req.fill_ids))
        # No dedup free, no rebind: the request keeps its own locs whole.
        self.assertEqual([t.tolist() for t in freed], [])
        row = tree.req_to_token_pool.req_to_token[req.kv.req_pool_idx, :12]
        self.assertTrue(torch.equal(row.to(dtype=torch.int64), own_locs))
        self.assertEqual(req.kv.cache_protected_len, 0)
        self.assertTrue(
            torch.equal(tree.req_to_token_pool.req_to_token[0, :12], own_locs)
        )

    def test_checkpoint_decline_frees_duplicates_and_suffix(self):
        tree, freed = self._tree_with_spy()
        self._seed_chain(tree, list(range(8)), base=1)
        req = _GraftReq(list(range(8)) + [90, 91, 92, 93])
        req.kv_rotation_base = 3
        own_locs = self._own_row(tree, req, 12)
        finish_req(tree, req, 12)
        released = torch.cat(freed)
        # Everything past the protected prefix is released: the duplicates of
        # the matched region AND the declined tail (nothing leaks, nothing is
        # grafted).
        self.assertEqual(set(released.tolist()), set(own_locs.tolist()))
        self.assertEqual(_match_len(tree, req.fill_ids), 8)

    def test_checkpoint_same_base_keeps_the_tail_cached(self):
        """Control for the decline test: with an agreeing base the tail is
        grafted and only the matched duplicates are freed."""
        tree, freed = self._tree_with_spy()
        self._seed_chain(tree, list(range(8)), base=1)
        req = _GraftReq(list(range(8)) + [90, 91, 92, 93])
        req.kv_rotation_base = 1
        own_locs = self._own_row(tree, req, 12)
        req.last_node = tree.root_node_handle()
        finish_req(tree, req, 12)
        self.assertEqual(_match_len(tree, req.fill_ids), 12)
        released = torch.cat(freed) if freed else torch.empty(0, dtype=torch.int64)
        # Only the 8 duplicate rows go back; the tail stays live in the tree.
        self.assertEqual(set(released.tolist()), set(own_locs[:8].tolist()))


def _chain_pages(base, n_pages, local_start=5):
    """Logical page ids of one chain: page P has owner (base + P) % N and an
    arbitrary (here: increasing) local page on its owner."""
    counter = {r: local_start for r in range(N)}
    pages = []
    for p in range(n_pages):
        r = (base + p) % N
        pages.append(counter[r] * N + r)
        counter[r] += 1
    return pages


def _chain_row(pages, seq_len):
    row = torch.empty(seq_len, dtype=torch.int32)
    for i in range(seq_len):
        row[i] = pages[i // PS] * PS + i % PS
    return row


def _make_pool_stub(spec, shard_rank=0, debug=True, table_pages=4096):
    """A SimpleNamespace carrying exactly the state begin_shard_extend /
    translate_loc_to_scratch read."""
    stub = SimpleNamespace()
    stub.shard_spec = spec
    stub.shard_rank = shard_rank
    stub.device = "cpu"
    stub.start_layer = 0
    stub._chunk_base = spec.max_prefix_tokens
    stub._trash_base = spec.max_prefix_tokens + spec.chunk_tokens
    stub._page_pos = torch.full(
        (table_pages,), stub._trash_base // PS, dtype=torch.int32
    )
    stub._local_page_stride = table_pages
    stub._epoch = 0
    stub._write_plan_key = stub._write_plan = None
    stub._translate_cache = {}
    stub._debug_plan_checks = debug
    stub.translate_loc_to_scratch = lambda loc: (
        PageInterleaveKVPoolMixin.translate_loc_to_scratch(stub, loc)
    )
    stub.prefetched = []
    stub._prefetch_layer = lambda layer_id: stub.prefetched.append(layer_id)
    return stub


def _run_begin(stub, prefix_lens, seq_lens, rows):
    width = max(r.numel() for r in rows)
    padded = [
        torch.cat([r, torch.zeros(width - r.numel(), dtype=torch.int32)]) for r in rows
    ]
    PageInterleaveKVPoolMixin.begin_shard_extend(
        stub,
        torch.stack(padded),
        torch.arange(len(rows)),
        prefix_lens,
        seq_lens,
    )
    return stub


def _reference_prefix_slots(per_request_prefix_pages):
    """Brute-force reference of the owner-major slot assignment: the batch's
    unique prefix pages sorted by (owner, local page), rank r's pages
    contiguous at r * block; block = sum of per-request ceil(K_i / N)."""
    block = sum(ceil_div(len(pages), N) for pages in per_request_prefix_pages)
    uniq = sorted({p for pages in per_request_prefix_pages for p in pages})
    slots = {}
    counts = {r: 0 for r in range(N)}
    for page in sorted(uniq, key=lambda p: (p % N, p // N)):
        owner = page % N
        slots[page] = owner * block + counts[owner]
        counts[owner] += 1
    return slots, block


class TestBeginShardExtendPlan(CustomTestCase):
    def test_plan_with_rotated_prefix(self):
        """7 prefix pages of a base-2 chain + 9 chunk pages (last partial):
        owner-major slots, send rows owner-filtered in the same order and
        padded to the block bound ceil(7/4) = 2 pages."""
        pages = _chain_pages(base=2, n_pages=16)
        prefix_len, seq_len = 7 * PS, 16 * PS - 5
        row = _chain_row(pages, seq_len)
        slots, block = _reference_prefix_slots([pages[:7]])
        for rank in range(N):
            stub = _run_begin(
                _make_pool_stub(_make_spec(), rank), [prefix_len], [seq_len], [row]
            )
            self.assertEqual(stub._block_pages, block)
            self.assertTrue(stub._shard_extend_active)
            self.assertEqual(stub._epoch, 1)
            self.assertEqual(stub.prefetched, [0])  # first layer kicked
            for page, slot in slots.items():
                self.assertEqual(int(stub._page_pos[page]), slot)
            for j, page in enumerate(pages[7:]):
                self.assertEqual(int(stub._page_pos[page]), stub._chunk_base // PS + j)
            own = sorted((p for p in pages[:7] if p % N == rank), key=lambda p: p // N)
            expect = torch.cat(
                [torch.arange((p // N) * PS, (p // N + 1) * PS) for p in own]
            )
            if len(own) < block:  # padded with the trash page (local page 0)
                expect = torch.cat([expect, torch.arange((block - len(own)) * PS)])
            self.assertTrue(torch.equal(stub._send_rows, expect))

    def test_multi_request_plan_shared_prefix_dedup(self):
        """bs > 1: request 0 and request 1 share a 3-page cached prefix
        (request 1 extends it by 2 pages); request 2 is an unrelated base-2
        chain. Shared pages must gather into ONE slot (no duplicate plan
        entries), the block is the per-request ceil sum, and every request's
        locs translate through the same table."""
        chain_a = _chain_pages(base=0, n_pages=5)
        chain_c = _chain_pages(base=2, n_pages=4, local_start=20)
        # rows: request 0 = A[:3] prefix + 1 chunk page; request 1 = A[:5]
        # prefix + 2 chunk pages; request 2 = C[:2] prefix + 2 chunk pages.
        chunk0 = _chain_pages(base=3, n_pages=1, local_start=40)
        chunk1 = _chain_pages(base=1, n_pages=2, local_start=50)
        chunk2 = _chain_pages(base=0, n_pages=2, local_start=60)
        rows = [
            _chain_row(chain_a[:3] + chunk0, 4 * PS),
            _chain_row(chain_a[:5] + chunk1, 7 * PS),
            _chain_row(chain_c[:2] + chunk2, 4 * PS - 3),
        ]
        stub = _run_begin(
            _make_pool_stub(_make_spec()),
            [3 * PS, 5 * PS, 2 * PS],
            [4 * PS, 7 * PS, 4 * PS - 3],
            rows,
        )
        slots, block = _reference_prefix_slots([chain_a[:3], chain_a[:5], chain_c[:2]])
        self.assertEqual(block, 1 + 2 + 1)
        self.assertEqual(stub._block_pages, block)
        for page, slot in slots.items():
            self.assertEqual(int(stub._page_pos[page]), slot)
        # Chunk slots are absolute scratch pages in batch order.
        for j, page in enumerate(chunk0 + chunk1 + chunk2):
            self.assertEqual(int(stub._page_pos[page]), stub._chunk_base // PS + j)
        # Shared pages: both requests' locs hit the SAME scratch rows.
        shared_loc_r0 = rows[0][:PS].long()
        shared_loc_r1 = rows[1][:PS].long()
        t0 = PageInterleaveKVPoolMixin.translate_loc_to_scratch(stub, shared_loc_r0)
        t1 = PageInterleaveKVPoolMixin.translate_loc_to_scratch(stub, shared_loc_r1)
        self.assertTrue(torch.equal(t0, t1))
        # Per-rank send lists fit the block and pad with the trash page.
        all_prefix = sorted(set(chain_a[:5] + chain_c[:2]))
        for rank in range(N):
            stub_r = _run_begin(
                _make_pool_stub(_make_spec(), rank),
                [3 * PS, 5 * PS, 2 * PS],
                [4 * PS, 7 * PS, 4 * PS - 3],
                rows,
            )
            own = sorted((p for p in all_prefix if p % N == rank), key=lambda p: p // N)
            self.assertLessEqual(len(own), block)
            self.assertEqual(stub_r._send_rows.numel(), block * PS)
            expect_head = torch.cat(
                [torch.arange((p // N) * PS, (p // N + 1) * PS) for p in own]
            )
            self.assertTrue(
                torch.equal(stub_r._send_rows[: len(own) * PS], expect_head)
            )

    def test_send_order_follows_local_page_not_position(self):
        """A freed-and-reused page can give a chain a LOWER local page id at
        a later position. Slot assignment and send packing must both order
        by local page id (they only need to agree — a mismatch reads the
        wrong rank rows)."""
        # Owner-0 pages appear at positions 0 and 4 with locals 9 then 3.
        pages = [9 * N + 0, 5 * N + 1, 5 * N + 2, 5 * N + 3, 3 * N + 0]
        row = _chain_row(pages, 5 * PS)
        stub = _run_begin(
            _make_pool_stub(_make_spec(), shard_rank=0),
            [5 * PS],
            [5 * PS + PS],
            [torch.cat([row, _chain_row([7 * N + 1], PS)])],
        )
        slots, block = _reference_prefix_slots([pages])
        self.assertEqual(block, 2)
        # local 3 gets owner-0's first slot although it sits at position 4.
        self.assertEqual(int(stub._page_pos[3 * N + 0]), 0)
        self.assertEqual(int(stub._page_pos[9 * N + 0]), 1)
        expect = torch.cat(
            [torch.arange(3 * PS, 4 * PS), torch.arange(9 * PS, 10 * PS)]
        )
        self.assertTrue(torch.equal(stub._send_rows, expect))

    def test_plan_without_prefix(self):
        pages = _chain_pages(base=0, n_pages=2)
        stub = _run_begin(
            _make_pool_stub(_make_spec()), [0], [PS + 5], [_chain_row(pages, PS + 5)]
        )
        self.assertEqual(stub._block_pages, 0)
        self.assertTrue(stub._shard_extend_active)
        self.assertEqual(stub.prefetched, [])  # nothing to gather
        self.assertIsNone(stub._send_rows)
        self.assertEqual(int(stub._page_pos[pages[0]]), stub._chunk_base // PS)
        self.assertEqual(int(stub._page_pos[pages[1]]), stub._chunk_base // PS + 1)

    def test_unaligned_prefix_rejected(self):
        # The tree quantum is the PHYSICAL page: a prefix that is not a
        # ps-multiple can never come out of match_prefix.
        pages = _chain_pages(base=0, n_pages=4)
        with self.assertRaises(AssertionError):
            _run_begin(
                _make_pool_stub(_make_spec()),
                [PS + 3],
                [4 * PS],
                [_chain_row(pages, 4 * PS)],
            )

    def test_owner_congruence_guard(self):
        """A rotation-base bug that breaks a request's prefix-owner
        cyclicity invalidates the sync-free block bound (a rank can own more
        than ceil(K/N) pages); the debug guard must catch it at plan time."""
        pages = _chain_pages(base=1, n_pages=8)
        pages[2], pages[5] = pages[5], pages[2]  # same multiset, not cyclic
        with self.assertRaises(AssertionError) as ctx:
            _run_begin(
                _make_pool_stub(_make_spec()),
                [6 * PS],
                [8 * PS],
                [_chain_row(pages, 8 * PS)],
            )
        self.assertIn("cyclic", str(ctx.exception))


class TestLayerTransferCounterRefused(CustomTestCase):
    """Layer-wise KV load-back must fail loud at registration.

    The gather reads pool rows directly (`_gather_pairs`), so it never passes
    through the base getters' `layer_transfer_counter.wait_until` hook, and
    `begin_shard_extend` kicks the first gather before any getter runs.
    """

    class _Base:
        def register_layer_transfer_counter(self, counter):
            self.counter = counter

    class _Pool(PageInterleaveKVPoolMixin, _Base):
        def __init__(self):
            pass

    def test_a_real_counter_is_refused(self):
        pool = self._Pool()
        with self.assertRaises(NotImplementedError) as cm:
            pool.register_layer_transfer_counter(object())
        self.assertIn("logical-page KV sharding", str(cm.exception))

    def test_none_stays_a_no_op(self):
        """The SWA/hybrid wrappers disable the counter by passing None."""
        pool = self._Pool()
        pool.register_layer_transfer_counter(None)
        self.assertIsNone(pool.counter)


class TestScratchTranslation(CustomTestCase):
    def _plan(self, base=2, n_prefix=7, n_chunk=9, rank=1):
        pages = _chain_pages(base=base, n_pages=n_prefix + n_chunk)
        seq_len = (n_prefix + n_chunk) * PS
        stub = _run_begin(
            _make_pool_stub(_make_spec(), rank),
            [n_prefix * PS],
            [seq_len],
            [_chain_row(pages, seq_len)],
        )
        return stub, pages[:n_prefix], pages[n_prefix:]

    def _reference_row(self, stub, prefix_pages, chunk_pages, loc):
        """Brute-force reference: owner-major (owner, local-page)-sorted
        prefix slots, sequence-order chunk."""
        spec = stub.shard_spec
        page, off = loc // PS, loc % PS
        if page in prefix_pages:
            slots, _ = _reference_prefix_slots([prefix_pages])
            return slots[page] * PS + off
        if page in chunk_pages:
            k = chunk_pages.index(page)
            return spec.max_prefix_tokens + k * PS + off
        return stub._trash_base + off

    def test_translation_matches_reference(self):
        stub, prefix_pages, chunk_pages = self._plan()
        locs = (
            [p * PS + o for p in prefix_pages + chunk_pages for o in (0, 3, PS - 1)]
            + list(range(0, N))  # reserved pages -> trash
            + [3000, 3001]  # off-plan -> trash
        )
        got = PageInterleaveKVPoolMixin.translate_loc_to_scratch(
            stub, torch.tensor(locs, dtype=torch.int64)
        )
        expect = torch.tensor(
            [self._reference_row(stub, prefix_pages, chunk_pages, l) for l in locs],
            dtype=torch.int64,
        )
        self.assertTrue(torch.equal(got, expect))

    def test_translation_is_injective_over_the_plan(self):
        stub, prefix_pages, chunk_pages = self._plan(base=3, n_prefix=5, n_chunk=4)
        locs = [p * PS + o for p in prefix_pages + chunk_pages for o in range(PS)]
        rows = PageInterleaveKVPoolMixin.translate_loc_to_scratch(
            stub, torch.tensor(locs, dtype=torch.int64)
        )
        self.assertEqual(len(torch.unique(rows)), len(locs))
        # Prefix rows stay inside the (padded) gather span, chunk rows inside
        # the chunk region.
        n_prefix_tokens = len(prefix_pages) * PS
        self.assertTrue(
            bool((rows[:n_prefix_tokens] < N * stub._block_pages * PS).all())
        )
        self.assertTrue(
            bool(
                (rows[n_prefix_tokens:] >= stub.shard_spec.max_prefix_tokens).all()
                and (rows[n_prefix_tokens:] < stub._trash_base).all()
            )
        )

    def test_int32_page_table_input(self):
        stub, prefix_pages, chunk_pages = self._plan(base=0, n_prefix=4, n_chunk=1)
        table = torch.tensor(
            [prefix_pages[0] * PS, prefix_pages[1] * PS, chunk_pages[0] * PS, 0],
            dtype=torch.int32,
        )
        rows = PageInterleaveKVPoolMixin.translate_loc_to_scratch(stub, table)
        self.assertEqual(rows.dtype, torch.int64)
        # Page-aligned inputs land on page-aligned scratch rows (the FA3
        # stride-divide contract).
        self.assertTrue(bool((rows[:3] % PS == 0).all()))
        self.assertEqual(int(rows[3]), stub._trash_base)

    def test_translation_cache_cleared_with_new_plan(self):
        pages = _chain_pages(base=0, n_pages=2)
        stub = _make_pool_stub(_make_spec())

        # The first batch treats page 0 as part of the current chunk.
        _run_begin(stub, [0], [PS], [_chain_row(pages[:1], PS)])
        loc = _chain_row(pages[:1], PS).long()
        first = PageInterleaveKVPoolMixin._translate_loc_cached(stub, loc)
        again = PageInterleaveKVPoolMixin._translate_loc_cached(stub, loc)
        self.assertIs(again, first)

        # The next batch reuses the same loc tensor after page 0 becomes a
        # cached prefix. Installing the new plan must discard the old mapping.
        _run_begin(stub, [PS], [2 * PS], [_chain_row(pages, 2 * PS)])
        fresh = PageInterleaveKVPoolMixin._translate_loc_cached(stub, loc)
        self.assertIsNot(fresh, first)
        self.assertFalse(torch.equal(fresh, first))


class TestWritePlan(CustomTestCase):
    def test_owner_filter_cached_per_loc_tensor(self):
        spec = _make_spec(shard_rank=2)
        stub = SimpleNamespace()
        stub.placement = PageInterleavePlacement(spec)
        stub.shard_rank = 2
        stub._epoch = 1
        stub._write_plan_key = stub._write_plan = None

        loc = torch.arange(5 * GS, 7 * GS)  # two whole groups
        owned_idx, local_rows = PageInterleaveKVPoolMixin._get_write_plan(stub, loc)
        self.assertEqual(owned_idx.numel(), 2 * PS)
        # Owned rows are ps-contiguous runs at [Q*ps, (Q+1)*ps).
        self.assertTrue(
            torch.equal(
                local_rows,
                torch.cat([torch.arange(5 * PS, 6 * PS), torch.arange(6 * PS, 7 * PS)]),
            )
        )
        # Same tensor + same epoch -> cached (identity).
        again = PageInterleaveKVPoolMixin._get_write_plan(stub, loc)
        self.assertIs(again[0], owned_idx)
        # Epoch bump invalidates.
        stub._epoch = 2
        fresh = PageInterleaveKVPoolMixin._get_write_plan(stub, loc)
        self.assertIsNot(fresh[0], owned_idx)

    def test_partial_tail_page_may_own_nothing(self):
        spec = _make_spec(shard_rank=3)
        stub = SimpleNamespace()
        stub.placement = PageInterleavePlacement(spec)
        stub.shard_rank = 3
        stub._epoch = 1
        stub._write_plan_key = stub._write_plan = None
        # 10 tokens: all inside owner-0's page of the group.
        loc = torch.arange(8 * GS, 8 * GS + 10)
        owned_idx, local_rows = PageInterleaveKVPoolMixin._get_write_plan(stub, loc)
        self.assertEqual(owned_idx.numel(), 0)
        self.assertEqual(local_rows.numel(), 0)


class TestShardedPoolFootprint(CustomTestCase):
    """Real MLA pools on CPU: the shard group stores one copy of the KV."""

    PAGES_PER_RANK = 8
    POOL_KWARGS = dict(
        page_size=PS,
        dtype=torch.bfloat16,
        kv_lora_rank=16,
        qk_rope_head_dim=8,
        layer_num=1,
        device="cpu",
        enable_memory_saver=False,
    )

    def setUp(self):
        super().setUp()
        # The MLA write path asserts against decode context parallelism.
        enter_scope(self, get_parallel().override(dcp_enabled=False))

    def _sharded_pool(self, shard_size, shard_rank):
        spec = PageShardSpec(
            shard_rank=shard_rank,
            shard_size=shard_size,
            page_size=PS,
            max_prefix_tokens=shard_size * PS,
            chunk_tokens=PS,
        )
        group = SimpleNamespace(
            world_size=shard_size,
            rank_in_group=shard_rank,
            cpu_group=None,
            device="cpu",
        )
        with unittest.mock.patch(
            "sglang.srt.mem_cache.page_interleave_pool.PyNcclCommunicator"
        ):
            return PageInterleaveMLATokenToKVPool(
                self.PAGES_PER_RANK * PS,
                shard_spec=spec,
                shard_group=group,
                **self.POOL_KWARGS,
            )

    def test_full_logical_capacity_stores_each_token_once(self):
        stock = MLATokenToKVPool(self.PAGES_PER_RANK * PS, **self.POOL_KWARGS)
        for shard_size in (2, 4, 8):
            with self.subTest(shard_size=shard_size):
                pools = [self._sharded_pool(shard_size, r) for r in range(shard_size)]
                for pool in pools:
                    self.assertEqual(
                        pool.get_kv_size_bytes(), stock.get_kv_size_bytes()
                    )
                alloc = PageInterleavePoolAllocator(
                    size=stock.size,
                    physical_page_size=PS,
                    shard_size=shard_size,
                    dtype=torch.bfloat16,
                    device="cpu",
                    kvcache=pools[0],
                    need_sort=False,
                )
                num_tokens = shard_size * stock.size
                loc = _alloc_extend(alloc, 0, num_tokens, None)
                self.assertEqual(alloc.available_size(), 0)
                # Token i carries (i % 128 + 1, i // 128 + 1), exact in bf16.
                token = torch.arange(num_tokens)
                cache_k = torch.zeros(num_tokens, 1, 24, dtype=torch.bfloat16)
                cache_k[:, 0, 0] = (token % 128 + 1).to(torch.bfloat16)
                cache_k[:, 0, 1] = (token // 128 + 1).to(torch.bfloat16)
                stored = []
                for pool in pools:
                    pool.set_kv_buffer(SimpleNamespace(layer_id=0), loc, cache_k, None)
                    tags = pool.kv_buffer[0][:, 0, :2].float()
                    rows = torch.nonzero(tags[:, 0]).squeeze(1)
                    # Every row past the padded page is filled, and no more.
                    self.assertTrue(
                        torch.equal(rows, torch.arange(PS, PS + stock.size))
                    )
                    stored.append(
                        ((tags[rows, 1] - 1) * 128 + tags[rows, 0] - 1).long()
                    )
                # One copy across the group: each token on exactly one rank.
                self.assertTrue(torch.equal(torch.cat(stored).sort().values, token))


class TestShardSendFilter(CustomTestCase):
    """P/D send side: rank r sends the pages with l % N == r, as its local page
    l // N, at their request-global send positions."""

    def test_ranks_partition_a_rotated_chunk(self):
        # A base-2 rotated chain: owners are not position-congruent.
        pages = np.array(_chain_pages(base=2, n_pages=9), dtype=np.int64)
        positions = []
        for rank in range(N):
            local, pos = filter_kv_indices_for_shard_rank(
                kv_indices=pages,
                index_slice=slice(20, 29),
                shard_rank=rank,
                shard_size=N,
            )
            np.testing.assert_array_equal(pages[pos - 20] % N, rank)
            np.testing.assert_array_equal(local, pages[pos - 20] // N)
            positions.append(pos)
        np.testing.assert_array_equal(
            np.sort(np.concatenate(positions)), np.arange(20, 29)
        )

    def test_empty_chunk_is_skipped_unless_final(self):
        # One page per owner 0..3, sent as two 2-page chunks.
        pages = np.array(_chain_pages(base=0, n_pages=4), dtype=np.int64)
        for rank, expect_skip in ((3, [True, False]), (1, [False, False])):
            sender = SimpleNamespace(
                kv_mgr=SimpleNamespace(kv_shard_rank=rank, kv_shard_size=N),
                curr_idx=0,
                num_kv_indices=len(pages),
            )
            skips = [
                CommonKVSender._prepare_send_indices(sender, chunk)[3]
                for chunk in (pages[:2], pages[2:])
            ]
            self.assertEqual(skips, expect_skip)


if __name__ == "__main__":
    unittest.main()
