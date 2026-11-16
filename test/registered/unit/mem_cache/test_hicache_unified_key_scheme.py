import threading
import unittest
from contextlib import ExitStack
from dataclasses import replace
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.arg_groups.hicache_hook import (
    handle_hicache,
    handle_hicache_ratio_default,
)
from sglang.srt.arg_groups.kv_cache_hook import handle_cache_compatibility
from sglang.srt.arg_groups.overrides import resolution_result
from sglang.srt.arg_groups.speculative_hook import (
    _resolve_dflash_draft_kv_ring,
    handle_speculative_decoding,
)
from sglang.srt.environ import envs
from sglang.srt.managers.cache_controller import CacheOperation, HiCacheController
from sglang.srt.mem_cache.hicache_key_scheme import (
    DraftKVNamespace,
    derive_namespace,
    draft_namespace_digest,
    namespace_digest,
    validate_unified_chunk_tiling,
)
from sglang.srt.mem_cache.hicache_storage import HiCacheFile, HiCacheStorageConfig
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import HostPoolGroup
from sglang.srt.mem_cache.kv_cache_builder import (
    build_kv_cache,
    check_unified_hicache_draft,
    maybe_register_hicache_draft,
)
from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool
from sglang.srt.mem_cache.pool_host.page_unified import PageUnifiedLayout
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.base_spec_worker import HiCacheDraftMode, HiCacheDraftPlan
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _resolve(**overrides):
    """Run the real hicache resolution; the rewrites land in the resolution record."""
    fields = dict(
        model_path="dummy",
        enable_hierarchical_cache=True,
        hicache_storage_backend="file",
        hicache_storage_key_scheme="unified",
    )
    fields.update(overrides)
    args = ServerArgs(**fields)
    # The ratio default is a separate hook; handle_hicache refuses an unsized
    # host pool without it.
    handle_hicache_ratio_default(args)
    handle_hicache(args)
    return args


class TestUnifiedKeySchemeArgs(CustomTestCase):
    def test_unified_pins_the_layout_and_io_backend(self):
        """The layout is object identity, so it is pinned rather than checked.

        A unified deployment that silently kept another layout would publish
        byte-permuted KV under keys a reader trusts.
        """
        args = _resolve(hicache_mem_layout="page_first", hicache_io_backend="direct")
        self.assertEqual(resolution_result(args, "hicache_mem_layout"), "page_unified")
        self.assertEqual(resolution_result(args, "hicache_io_backend"), "kernel")

    def test_rank_suffix_leaves_the_layout_alone(self):
        args = _resolve(
            hicache_storage_key_scheme="rank-suffix", hicache_mem_layout="page_first"
        )
        self.assertEqual(resolution_result(args, "hicache_mem_layout"), "page_first")

    def test_page_unified_storage_requires_distinct_keys(self):
        with self.assertRaisesRegex(ValueError, "requires.*unified"):
            _resolve(
                hicache_storage_key_scheme="rank-suffix",
                hicache_mem_layout="page_unified",
            )
        args = _resolve(
            hicache_storage_key_scheme="rank-suffix",
            hicache_mem_layout="page_unified",
            hicache_storage_backend=None,
        )
        self.assertEqual(resolution_result(args, "hicache_mem_layout"), "page_unified")

    def test_partition_configs_require_the_unified_scheme(self):
        with self.assertRaisesRegex(ValueError, "require --hicache-storage-key-scheme"):
            _resolve(
                hicache_storage_key_scheme="rank-suffix",
                hicache_storage_head_partition=2,
            )

    def test_non_positive_partition_config_is_refused(self):
        for field in (
            "hicache_storage_head_partition",
            "hicache_storage_layer_partition",
        ):
            with (
                self.subTest(field=field),
                self.assertRaisesRegex(ValueError, "must be positive"),
            ):
                _resolve(**{field: 0})

    def test_unified_needs_a_supported_storage_backend(self):
        with self.assertRaisesRegex(ValueError, "needs"):
            _resolve(hicache_storage_backend=None)
        for backend in ("hf3fs", "nixl"):
            with self.subTest(backend=backend), self.assertRaises(NotImplementedError):
                _resolve(hicache_storage_backend=backend)

    def test_unified_accepts_the_mooncake_backend(self):
        args = _resolve(
            hicache_storage_backend="mooncake",
            hicache_storage_head_partition=2,
            hicache_storage_layer_partition=8,
        )
        self.assertEqual(resolution_result(args, "hicache_mem_layout"), "page_unified")
        self.assertEqual(resolution_result(args, "hicache_io_backend"), "kernel")

    def test_unified_refuses_speculative_decoding(self):
        # Draft layers sit on the host pool's layer axis but outside the grid.
        # The refusal needs the resolved algorithm, so it comes from the hook
        # that runs after the speculative one, not from handle_hicache().
        with self.assertRaisesRegex(NotImplementedError, "EAGLE3"):
            _resolve_through_speculative(speculative_algorithm="EAGLE3")

    def test_unified_supports_full_page_attention_cp(self):
        args = _resolve(attn_cp_size=2, enable_prefill_cp=True)
        self.assertEqual(resolution_result(args, "hicache_mem_layout"), "page_unified")

    def test_unified_refuses_decode_context_parallelism(self):
        with self.assertRaisesRegex(NotImplementedError, "decode context parallelism"):
            _resolve(dcp_size=2)

    def test_unified_refuses_shared_decode_host_receive_pool(self):
        with self.assertRaisesRegex(ValueError, "layer_first"):
            _resolve(
                disaggregation_mode="decode",
                disaggregation_decode_host_receive_threshold=0.5,
            )


def _resolve_through_speculative(patches=(), **overrides):
    """The hicache, speculative and cache-compatibility hooks, in pipeline order."""
    fields = dict(
        model_path="dummy",
        device="cuda",
        enable_hierarchical_cache=True,
        hicache_storage_backend="file",
        hicache_storage_key_scheme="unified",
        speculative_eagle_topk=1,
        speculative_num_steps=3,
        speculative_num_draft_tokens=4,
    )
    fields.update(overrides)
    args = ServerArgs(**fields)
    args._model_config = SimpleNamespace(
        hf_config=SimpleNamespace(
            architectures=["LlamaForCausalLM"],
            get_text_config=lambda: SimpleNamespace(),
        )
    )
    handle_hicache_ratio_default(args)
    handle_hicache(args)
    with ExitStack() as stack:
        for context in patches:
            stack.enter_context(context)
        handle_speculative_decoding(args)
    handle_cache_compatibility(args)
    return args


_SPEC_HOOK = "sglang.srt.arg_groups.speculative_hook"


def _dflash_with_ring(ring_unsupported_reason):
    """DFLASH without a draft checkpoint: only the ring resolution runs for real."""
    return (
        patch(
            f"{_SPEC_HOOK}._handle_dflash", side_effect=_resolve_dflash_draft_kv_ring
        ),
        patch(
            "sglang.srt.speculative.dflash_draft_ring.draft_kv_ring_unsupported_reason",
            return_value=ring_unsupported_reason,
        ),
    )


class TestUnifiedKeySchemeSpeculative(CustomTestCase):
    def test_algorithms_without_draft_kv_in_hicache_are_accepted(self):
        args = _resolve_through_speculative(speculative_algorithm="FROZEN_KV_MTP")
        self.assertEqual(
            resolution_result(args, "speculative_algorithm"), "FROZEN_KV_MTP"
        )

    def test_promoted_alias_is_judged_by_its_resolved_algorithm(self):
        # A Gemma4 assistant draft promotes NEXTN to FROZEN_KV_MTP in the
        # speculative hook; the raw name alone would be refused as EAGLE.
        promoted = patch(
            f"{_SPEC_HOOK}._resolve_speculative_algorithm_alias",
            return_value="FROZEN_KV_MTP",
        )
        args = _resolve_through_speculative(
            patches=(promoted,), speculative_algorithm="NEXTN"
        )
        self.assertEqual(
            resolution_result(args, "speculative_algorithm"), "FROZEN_KV_MTP"
        )
        # Unpromoted, NEXTN is EAGLE, whose draft KV is stored.
        args = _resolve_through_speculative(
            speculative_algorithm="NEXTN", hicache_storage_draft_id="org/draft"
        )
        self.assertEqual(resolution_result(args, "speculative_algorithm"), "EAGLE")

    def test_dflash_is_accepted_only_with_the_draft_kv_ring(self):
        args = _resolve_through_speculative(
            patches=_dflash_with_ring(None), speculative_algorithm="DFLASH"
        )
        self.assertTrue(resolution_result(args, "speculative_dflash_draft_kv_ring"))
        with self.assertRaisesRegex(NotImplementedError, "only with the draft KV ring"):
            _resolve_through_speculative(
                patches=_dflash_with_ring("the draft is not MLA"),
                speculative_algorithm="DFLASH",
            )

    def test_every_other_algorithm_is_refused_by_name(self):
        # These handlers need a draft checkpoint or more arguments.
        handlers = {
            "DSPARK": (patch(f"{_SPEC_HOOK}._handle_dspark"),),
            "NGRAM": (patch(f"{_SPEC_HOOK}._handle_ngram"),),
            "UNO": (patch(f"{_SPEC_HOOK}._handle_uno"),),
        }
        for algorithm in ("EAGLE3", "STANDALONE", "DSPARK", "NGRAM", "UNO"):
            with (
                self.subTest(algorithm=algorithm),
                self.assertRaisesRegex(
                    NotImplementedError, f"speculative-algorithm {algorithm} "
                ),
            ):
                _resolve_through_speculative(
                    patches=handlers.get(algorithm, ()),
                    speculative_algorithm=algorithm,
                )

    def test_rank_suffix_scheme_is_not_restricted(self):
        _resolve_through_speculative(
            speculative_algorithm="EAGLE3", hicache_storage_key_scheme="rank-suffix"
        )

    def test_decode_context_parallelism_is_still_refused_first(self):
        with self.assertRaisesRegex(NotImplementedError, "decode context parallelism"):
            _resolve_through_speculative(speculative_algorithm="DFLASH", dcp_size=2)

    def test_host_pools_outside_hierarchical_cache_refuse_every_algorithm(self):
        # PD decode offload builds its own controller with unified keys.
        for hierarchical_cache in (False, True):
            with (
                self.subTest(hierarchical_cache=hierarchical_cache),
                self.assertRaisesRegex(
                    NotImplementedError, "only with --enable-hierarchical-cache"
                ),
            ):
                _resolve_through_speculative(
                    speculative_algorithm="FROZEN_KV_MTP",
                    enable_hierarchical_cache=hierarchical_cache,
                    disaggregation_mode="decode",
                    disaggregation_decode_enable_offload_kvcache=True,
                )

    def test_an_inert_unified_flag_stays_inert(self):
        # Nothing builds a HiCache host pool: the flag has no effect today,
        # with or without speculative decoding.
        _resolve_through_speculative(
            speculative_algorithm="EAGLE3", enable_hierarchical_cache=False
        )
        # The external linker has no L3 key scheme at all.
        _resolve_through_speculative(
            speculative_algorithm="EAGLE3",
            enable_hierarchical_cache=False,
            hicache_storage_backend=None,
            enable_unified_cache_external_linker=True,
        )


class TestUnifiedKeySchemeDraftKV(CustomTestCase):
    """NEXTN-style MTP: the static requirements and the draft id."""

    def _resolve(self, **overrides):
        fields = dict(
            speculative_algorithm="NEXTN", speculative_draft_model_path="dummy"
        )
        fields.update(overrides)
        # The alias resolution reads the draft checkpoint's config.
        algorithm = fields["speculative_algorithm"]
        resolved = patch(
            f"{_SPEC_HOOK}._resolve_speculative_algorithm_alias",
            return_value="EAGLE" if algorithm == "NEXTN" else algorithm,
        )
        return _resolve_through_speculative(patches=(resolved,), **fields)

    def _draft_id(self, **overrides):
        return resolution_result(self._resolve(**overrides), "hicache_storage_draft_id")

    def test_draft_id_is_resolved_once_at_startup(self):
        # Part of the target checkpoint: the target namespace names it.
        self.assertEqual(self._draft_id(), "bundled")
        # No draft path at all: the draft loads from the target checkpoint.
        self.assertEqual(self._draft_id(speculative_draft_model_path=None), "bundled")
        self.assertEqual(
            self._draft_id(speculative_draft_model_path=None, revision="abc"), "bundled"
        )
        # The same path under another target revision is another checkpoint.
        self.assertEqual(self._draft_id(revision="abc"), "dummy@main")
        self.assertEqual(
            self._draft_id(speculative_draft_model_path="org/draft"), "org/draft@main"
        )
        self.assertEqual(
            self._draft_id(
                speculative_draft_model_path="org/draft",
                speculative_draft_model_revision="abc",
            ),
            "org/draft@abc",
        )
        # The same checkpoint quantized for the draft alone is another draft.
        self.assertEqual(
            self._draft_id(speculative_draft_model_quantization="fp8"), "bundled+fp8"
        )
        # A declared id wins: local mount paths differ between hosts.
        self.assertEqual(
            self._draft_id(
                speculative_draft_model_path="/mnt/a/draft",
                hicache_storage_draft_id="org/draft@abc",
            ),
            "org/draft@abc",
        )

    def test_draft_id_without_a_stored_draft_is_refused(self):
        cases = {
            "rank_suffix": dict(hicache_storage_key_scheme="rank-suffix"),
            "no_algorithm": dict(speculative_algorithm=None),
            "no_draft_kv": dict(speculative_algorithm="FROZEN_KV_MTP"),
            "empty": dict(hicache_storage_draft_id=""),
        }
        for name, overrides in cases.items():
            with self.subTest(name), self.assertRaises(ValueError):
                self._resolve(**{"hicache_storage_draft_id": "x", **overrides})

    def test_paths_where_draft_kv_is_not_safe_to_store_are_refused(self):
        cases = {
            "multi-layer EAGLE": dict(enable_multi_layer_eagle=True),
            "buffer_only": dict(hicache_host_memory_mode="buffer_only"),
            "random weights": dict(speculative_draft_load_format="dummy"),
            "random weights ": dict(load_format="dummy"),
            "prefill context parallelism": dict(attn_cp_size=2, enable_prefill_cp=True),
            "PD disaggregation": dict(disaggregation_mode="prefill"),
            "--enable-streaming-session": dict(enable_streaming_session=True),
            "--enable-session-radix-cache": dict(enable_session_radix_cache=True),
        }
        for message, overrides in cases.items():
            with (
                self.subTest(message),
                self.assertRaisesRegex(NotImplementedError, message.strip()),
            ):
                self._resolve(**overrides)
        with (
            envs.SGLANG_SPEC_SKIP_ZERO_STEP_DRAFT_EXTEND.override(True),
            self.assertRaisesRegex(NotImplementedError, "draft rows unwritten"),
        ):
            self._resolve()
        # None of this concerns an algorithm that stores no draft KV.
        self._resolve(speculative_algorithm="FROZEN_KV_MTP", load_format="dummy")
        self._resolve(
            speculative_algorithm="FROZEN_KV_MTP", enable_streaming_session=True
        )
        # Nor the rank-suffix scheme, which keeps streaming sessions.
        self._resolve(
            hicache_storage_key_scheme="rank-suffix", enable_streaming_session=True
        )


class TestUnifiedDraftModeGate(CustomTestCase):
    def _check(
        self,
        mode,
        *,
        scheme="unified",
        algorithm=SpeculativeAlgorithm.EAGLE,
        external_linker=False,
    ):
        plan = None if mode is None else HiCacheDraftPlan(mode=mode)
        memory = SimpleNamespace(
            hicache_storage_key_scheme=scheme,
            enable_unified_cache_external_linker=external_linker,
        )
        with patch(
            "sglang.srt.mem_cache.kv_cache_builder.get_memory", return_value=memory
        ):
            check_unified_hicache_draft(algorithm, plan)

    def test_modes_without_draft_kv_pass(self):
        self._check(None, algorithm=SpeculativeAlgorithm.NONE)
        self._check(HiCacheDraftMode.NONE, algorithm=SpeculativeAlgorithm.FROZEN_KV_MTP)

    def _check_packed(self, algorithm, draft_layers):
        plan = HiCacheDraftPlan(
            mode=HiCacheDraftMode.PACKED, device_pools=(object(),) * draft_layers
        )
        memory = SimpleNamespace(
            hicache_storage_key_scheme="unified",
            enable_unified_cache_external_linker=False,
        )
        with patch(
            "sglang.srt.mem_cache.kv_cache_builder.get_memory", return_value=memory
        ):
            check_unified_hicache_draft(algorithm, plan)

    def test_one_packed_nextn_layer_passes(self):
        self._check_packed(SpeculativeAlgorithm.EAGLE, 1)

    def test_deeper_packed_drafts_are_refused(self):
        # A deeper layer's rows depend on tokens the page key does not cover.
        for layers in (2, 3):
            with (
                self.subTest(layers=layers),
                self.assertRaisesRegex(
                    NotImplementedError, f"one packed draft layer; {layers}"
                ),
            ):
                self._check_packed(SpeculativeAlgorithm.EAGLE, layers)

    def test_packed_drafts_of_other_algorithms_are_refused(self):
        # DSPARK compressed-KV packs too; EAGLE3 and FROZEN_KV_MTP are in
        # the EAGLE family but never pack.
        for algorithm in ("DSPARK", "EAGLE3", "FROZEN_KV_MTP"):
            with (
                self.subTest(algorithm=algorithm),
                self.assertRaisesRegex(
                    NotImplementedError, f"draft mode 'packed' for {algorithm}"
                ),
            ):
                self._check_packed(SpeculativeAlgorithm[algorithm], 1)

    def test_sidecar_and_fused_drafts_are_refused_by_name(self):
        for mode in (HiCacheDraftMode.SIDECAR, HiCacheDraftMode.FUSED):
            with (
                self.subTest(mode=mode),
                self.assertRaisesRegex(
                    NotImplementedError, f"draft mode '{mode.value}' for EAGLE"
                ),
            ):
                self._check(mode)

    def _stored(self, *, is_eagle=True, host_layers=5, target_layers=4, pools=1):
        tree_cache = SimpleNamespace(
            tree_core=SimpleNamespace(is_eagle=is_eagle),
            cache_controller=SimpleNamespace(
                storage_host_pool=SimpleNamespace(
                    layer_num=host_layers, target_layer_num=target_layers
                )
            ),
        )
        plan = HiCacheDraftPlan(
            mode=HiCacheDraftMode.PACKED, device_pools=(object(),) * pools
        )
        memory = SimpleNamespace(
            hicache_storage_key_scheme="unified",
            enable_unified_cache_external_linker=False,
        )
        # Through the function build_kv_cache() calls once the tree exists.
        with patch(
            "sglang.srt.mem_cache.kv_cache_builder.get_memory", return_value=memory
        ):
            maybe_register_hicache_draft(tree_cache=tree_cache, draft_plan=plan)

    def test_accepted_packed_draft_must_be_what_the_cache_stores(self):
        self._stored()
        # The draft row of slot i needs token i + 1 in the page key.
        with self.assertRaisesRegex(NotImplementedError, "bigram page hashes"):
            self._stored(is_eagle=False)
        # A host pool that did not pack the draft would store target-only
        # pages, and readers would hit them without draft KV.
        with self.assertRaisesRegex(NotImplementedError, "expected 1 packed.*found 0"):
            self._stored(host_layers=4)

    def test_rank_suffix_scheme_is_not_restricted(self):
        for mode in HiCacheDraftMode:
            self._check(mode, scheme="rank-suffix")

    def test_external_linker_ignores_the_key_scheme(self):
        # It builds draft pools of its own and never derives L3 keys.
        for mode in HiCacheDraftMode:
            self._check(mode, external_linker=True)

    def test_gate_runs_before_the_cache_is_built(self):
        # Every other argument is a placeholder: the gate must raise before
        # the cache tree is built and storage attaches, so none is read.
        with (
            patch(
                "sglang.srt.mem_cache.kv_cache_builder.get_memory",
                return_value=SimpleNamespace(
                    hicache_storage_key_scheme="unified",
                    enable_unified_cache_external_linker=False,
                ),
            ),
            self.assertRaisesRegex(NotImplementedError, "draft mode 'fused'"),
        ):
            build_kv_cache(
                server_args=None,
                model_config=None,
                tp_worker=None,
                page_size=64,
                spec_algorithm=SpeculativeAlgorithm.DFLASH,
                enable_metrics=False,
                enable_kv_cache_events=False,
                enable_hierarchical_cache=True,
                hicache_draft_plan=HiCacheDraftPlan(mode=HiCacheDraftMode.FUSED),
            )


class TestControllerUnifiedKeys(CustomTestCase):
    def _controller(self, *, mla=False, grouped=True):
        device_pool = object.__new__(MLATokenToKVPool) if mla else SimpleNamespace()
        # FP8 variants share a uint8 storage view but require distinct keys.
        device_pool.dtype = torch.float8_e4m3fn
        device_pool.store_dtype = torch.uint8
        # Buffer-info host pools expose the physical dtype and no device_pool.
        host_pool = SimpleNamespace(
            dtype=torch.uint8,
            head_num=0 if mla else 4,
            head_group_num=1,
            start_layer=4,
            end_layer=8,
            layer_num=4,
            page_unified_layout=PageUnifiedLayout(
                page_size=16,
                layer_num=4,
                head_num=1 if mla else 4,
                head_group_num=1,
                head_dim=128,
                itemsize=1,
                is_mla=mla,
            ),
        )
        controller = object.__new__(HiCacheController)
        controller.page_size = 16
        controller.storage_host_pool = host_pool
        if grouped:
            anchor = SimpleNamespace(host_pool=host_pool, device_pool=device_pool)
            controller.mem_pool_host = SimpleNamespace(
                entries=[anchor], anchor_entry=anchor
            )
            # A composite device pool need not describe its anchor's physical KV.
            controller.mem_pool_device = SimpleNamespace(dtype=torch.float16)
        else:
            controller.mem_pool_host = host_pool
            controller.mem_pool_device = device_pool
        return controller

    def _suffixes(self, controller, *, attn_cp_size=1, pp_rank=1, **draft):
        memory = SimpleNamespace(
            hicache_storage_key_scheme="unified",
            hicache_storage_head_partition=4,
            hicache_storage_layer_partition=4,
        )
        parallel = SimpleNamespace(
            tp_rank=5,
            tp_size=8,
            attn_tp_rank=1,
            attn_tp_size=2,
            pp_rank=pp_rank,
            pp_size=2,
        )
        patches = [
            patch(
                "sglang.srt.managers.cache_controller.get_memory", return_value=memory
            ),
            patch(
                "sglang.srt.managers.cache_controller.get_parallel",
                return_value=parallel,
            ),
        ]
        if draft:
            # Only a pool with packed draft layers reads these.
            memory.hicache_storage_draft_id = draft["draft_id"]
            patches.append(
                patch(
                    "sglang.srt.managers.cache_controller.get_spec",
                    return_value=SimpleNamespace(
                        speculative_algorithm=draft["algorithm"]
                    ),
                )
            )
        with ExitStack() as stack:
            for context in patches:
                stack.enter_context(context)
            return controller._build_unified_plan(
                "org/model", attn_cp_size=attn_cp_size
            ).suffixes

    def _packed_controller(self, *, mla=False, draft_dtypes=(torch.float8_e4m3fn,)):
        """The fake pool with one-layer draft pools packed behind its 4 layers."""
        controller = self._controller(mla=mla)
        host_pool = controller.storage_host_pool
        host_pool.target_layer_num = 4
        host_pool.layer_num = 4 + len(draft_dtypes)
        host_pool.page_unified_layout = PageUnifiedLayout(
            page_size=16,
            layer_num=host_pool.layer_num,
            head_num=1 if mla else 4,
            head_group_num=1,
            head_dim=128,
            itemsize=1,
            is_mla=mla,
        )
        # Like the target: a logical dtype over a uint8 storage view.
        controller.mem_pool_host.anchor_entry.packed_draft_device_pools = tuple(
            SimpleNamespace(dtype=dtype, store_dtype=torch.uint8)
            for dtype in draft_dtypes
        )
        return controller

    def _draft_suffix(self, target_suffix, **overrides):
        fields = dict(
            target=target_suffix.split("_")[0],
            algorithm="EAGLE",
            draft_id="bundled",
            placement="packed",
            dtype="float8_e4m3fn",
            rank_replicated=False,
            total_kv_heads=8,
            head_group=4,
            row_width=128,
            layer_num=1,
            object_layout="page-unified-v1",
        )
        fields.update(overrides)
        return draft_namespace_digest(DraftKVNamespace(**fields)) + "_D0-1"

    def test_packed_draft_layer_is_named_after_the_target_chunks(self):
        target = self._suffixes(self._controller())
        suffixes = self._suffixes(
            self._packed_controller(), algorithm="EAGLE", draft_id="bundled"
        )
        self.assertEqual(suffixes, [*target, self._draft_suffix(target[0]) + "_H1"])

        # MLA: replicated across ranks, so no head coordinate.
        target = self._suffixes(self._controller(mla=True))
        suffixes = self._suffixes(
            self._packed_controller(mla=True), algorithm="EAGLE", draft_id="org/d@main"
        )
        draft = self._draft_suffix(
            target[0],
            draft_id="org/d@main",
            rank_replicated=True,
            total_kv_heads=0,
            head_group=0,
        )
        self.assertEqual(suffixes, [*target, draft])

    def test_packed_draft_identity_is_the_drafts_own_logical_dtype(self):
        # The pools pack together because both store as uint8; the target
        # chunks are shared, the draft chunks must not be.
        target = self._suffixes(self._controller())
        e5m2 = self._suffixes(
            self._packed_controller(draft_dtypes=(torch.float8_e5m2,)),
            algorithm="EAGLE",
            draft_id="bundled",
        )
        self.assertEqual(
            e5m2, [*target, self._draft_suffix(target[0], dtype="float8_e5m2") + "_H1"]
        )
        self.assertNotEqual(e5m2[-1], self._draft_suffix(target[0]) + "_H1")

    def test_packed_draft_spans_tile_the_page_block(self):
        controller = self._packed_controller()
        controller.storage_host_pool.layout = "page_unified"
        controller.storage_host_pool.storage_format_tag = None
        controller.mem_pool_host.layout = "page_unified"
        controller.enable_storage_metrics = False
        memory = SimpleNamespace(
            hicache_storage_key_scheme="unified",
            hicache_storage_head_partition=4,
            hicache_storage_layer_partition=4,
            hicache_storage_draft_id="bundled",
        )
        parallel = SimpleNamespace(
            tp_rank=1,
            tp_size=2,
            attn_tp_rank=1,
            attn_tp_size=2,
            attn_dcp_size=1,
            attn_dcp_rank=0,
            pp_rank=1,
            pp_size=2,
        )
        module = "sglang.srt.managers.cache_controller"
        with (
            patch(f"{module}.get_memory", return_value=memory),
            patch(f"{module}.get_parallel", return_value=parallel),
            patch(
                f"{module}.get_spec",
                return_value=SimpleNamespace(speculative_algorithm="EAGLE"),
            ),
            patch(f"{module}.is_dp_attention_enabled", return_value=False),
            patch.object(controller, "get_attn_cp_rank_and_size", return_value=(0, 1)),
        ):
            config = controller._generate_storage_config("org/model")
        # 4 target layers then the draft layer, 16 tokens x (K + V) x 4 x 128.
        self.assertEqual(config.unified_chunk_spans, [(0, 65536), (65536, 16384)])
        self.assertTrue(config.unified_suffixes[-1].endswith("_D0-1_H1"))

    def test_packed_drafts_that_cannot_be_named_are_refused(self):
        two = (torch.float8_e4m3fn, torch.float8_e4m3fn)
        mixed = (torch.float8_e4m3fn, torch.float8_e5m2)
        cases = {
            # DSPARK compressed-KV packs too.
            "not_nextn": (
                dict(algorithm="DSPARK"),
                {},
                "only for NEXTN-style MTP; --speculative-algorithm DSPARK is not",
            ),
            # The refusal names what it found, not "None".
            "no_algorithm": (
                dict(algorithm=None),
                {},
                "a host pool with 1 packed draft layer.s. and no speculative "
                "algorithm is not supported",
            ),
            "deeper_draft": ({}, dict(draft_dtypes=two), "one packed draft layer"),
            "mixed_dtypes": ({}, dict(draft_dtypes=mixed), "one packed draft layer"),
            "no_draft_id": (dict(draft_id=None), {}, "needs a draft id"),
            "prefill_cp": (dict(attn_cp_size=2), {}, "prefill context parallelism"),
        }
        for name, (call, pool, message) in cases.items():
            with (
                self.subTest(name),
                self.assertRaisesRegex(NotImplementedError, message),
            ):
                self._suffixes(
                    self._packed_controller(**pool),
                    **{"algorithm": "EAGLE", "draft_id": "bundled", **call},
                )
        # The host pool says one draft layer, the pool group holds none.
        controller = self._packed_controller()
        controller.mem_pool_host.anchor_entry.packed_draft_device_pools = ()
        with self.assertRaisesRegex(NotImplementedError, "from 0 draft pool"):
            self._suffixes(controller, algorithm="EAGLE", draft_id="bundled")
        # Only the last pipeline stage hosts a draft.
        controller = self._packed_controller()
        controller.storage_host_pool.start_layer = 0
        controller.storage_host_pool.end_layer = 4
        with self.assertRaisesRegex(ValueError, "final pipeline stage"):
            self._suffixes(controller, pp_rank=0, algorithm="EAGLE", draft_id="bundled")

    def test_anchor_dtype_and_attention_topology_define_mha_keys(self):
        namespace = derive_namespace(
            model_id="org/model",
            dtype="float8_e4m3fn",
            page_size=16,
            rank_replicated=False,
            total_kv_heads=8,
            head_group=4,
            layer_partition=4,
        )
        self.assertEqual(
            self._suffixes(self._controller()),
            [f"{namespace_digest(namespace)}_L4-8_H1"],
        )

    def test_anchor_mla_pool_has_no_head_coordinate(self):
        namespace = derive_namespace(
            model_id="org/model",
            dtype="float8_e4m3fn",
            page_size=16,
            rank_replicated=True,
            total_kv_heads=0,
            head_group=0,
            layer_partition=4,
        )
        self.assertEqual(
            self._suffixes(self._controller(mla=True)),
            [f"{namespace_digest(namespace)}_L4-8"],
        )

    def test_target_only_pool_keys_never_read_the_speculative_algorithm(self):
        # A draft mode that keeps no draft KV in HiCache must derive exactly
        # the keys of a deployment without speculative decoding. For a
        # target-only pool the plan has no reason to look at the algorithm,
        # and must not: any dependence on it would split the keyspace.
        untouchable = AssertionError("target-only keys must not read the algorithm")
        for mla, expected in ((False, "_L4-8_H1"), (True, "_L4-8")):
            namespace = derive_namespace(
                model_id="org/model",
                dtype="float8_e4m3fn",
                page_size=16,
                rank_replicated=mla,
                total_kv_heads=0 if mla else 8,
                head_group=0 if mla else 4,
                layer_partition=4,
            )
            with (
                self.subTest(mla=mla),
                patch("sglang.srt.runtime_context.get_spec", side_effect=untouchable),
                # Also a by-name import, the style of that module.
                patch(
                    "sglang.srt.managers.cache_controller.get_spec",
                    side_effect=untouchable,
                    create=True,
                ),
            ):
                self.assertEqual(
                    self._suffixes(self._controller(mla=mla)),
                    [namespace_digest(namespace) + expected],
                )

    def test_standalone_and_grouped_pools_share_logical_identity(self):
        for mla in (False, True):
            with self.subTest(mla=mla):
                self.assertEqual(
                    self._suffixes(self._controller(mla=mla, grouped=False)),
                    self._suffixes(self._controller(mla=mla)),
                )

    def test_full_page_cp_keeps_attention_tp_shard_keys(self):
        for mla in (False, True):
            with self.subTest(mla=mla):
                controller = self._controller(mla=mla)
                self.assertEqual(
                    self._suffixes(controller, attn_cp_size=4),
                    self._suffixes(controller, attn_cp_size=1),
                )

    def test_runtime_attach_refuses_dcp_pool(self):
        controller = self._controller(mla=True)
        controller.storage_host_pool.dcp_size = 2
        with self.assertRaisesRegex(NotImplementedError, "decode context parallelism"):
            self._suffixes(controller)

    def test_cp_writer_election_preserves_each_attention_tp_shard(self):
        # With attention DP, the storage TP rank is attention-group-local.
        # Every MHA head shard needs a writer, while MLA needs a single writer.
        expected_writers = {
            (False, False): [(0, 0), (1, 0), (0, 1), (1, 1)],
            (False, True): [(0, 0), (0, 1)],
            (True, False): [(0, 0), (1, 0)],
            (True, True): [(0, 0)],
        }
        for (unified, mla), expected in expected_writers.items():
            with self.subTest(unified=unified, mla=mla):
                writers = []
                for cp_rank in range(2):
                    for tp_rank in range(2):
                        controller = self._controller(mla=mla)
                        controller.enable_storage = False
                        controller.storage_host_pool.layout = (
                            "page_unified" if unified else "page_first"
                        )
                        controller.mem_pool_host.size = 256
                        controller.mem_pool_host.logical_size = 256
                        controller.host_memory_mode = "cache"
                        controller.storage_stop_event = threading.Event()
                        config = HiCacheStorageConfig(
                            tp_rank=tp_rank,
                            tp_size=2,
                            pp_rank=0,
                            pp_size=1,
                            attn_cp_rank=cp_rank,
                            attn_cp_size=2,
                            is_mla_model=mla,
                            enable_storage_metrics=False,
                            is_page_first_layout=not unified,
                            model_name="org/model",
                            unified_suffixes=["shared"] if unified else None,
                        )
                        self.assertEqual(
                            config.is_storage_writer, (tp_rank, cp_rank) in expected
                        )
                        memory = SimpleNamespace(
                            hicache_storage_key_scheme=(
                                "unified" if unified else "rank-suffix"
                            )
                        )
                        with (
                            patch(
                                "sglang.srt.managers.cache_controller.get_memory",
                                return_value=memory,
                            ),
                            patch(
                                "sglang.srt.managers.cache_controller.get_parallel",
                                return_value=SimpleNamespace(attn_dcp_size=1),
                            ),
                            patch.object(controller, "_stop_storage_threads"),
                            patch.object(controller, "_start_storage_threads"),
                            patch.object(
                                controller,
                                "_generate_storage_config",
                                return_value=config,
                            ),
                            patch.object(
                                controller, "_create_sync_groups", return_value=[]
                            ),
                            patch(
                                "sglang.srt.mem_cache.storage.StorageBackendFactory.create_backend",
                                return_value=MagicMock(),
                            ),
                        ):
                            controller.attach_storage_backend("file")
                        if not controller.backup_skip:
                            writers.append((tp_rank, cp_rank))
                self.assertEqual(writers, expected)

    def test_attention_dp_config_uses_local_tp_and_cp_ranks(self):
        controller = self._controller(mla=True)
        controller.mem_pool_device = controller.mem_pool_host.anchor_entry.device_pool
        controller.mem_pool_host.layout = "page_unified"
        controller.storage_host_pool.storage_format_tag = None
        controller.enable_storage_metrics = False
        parallel = SimpleNamespace(
            tp_rank=6,
            tp_size=8,
            attn_tp_rank=0,
            attn_tp_size=2,
            attn_cp_rank=1,
            attn_cp_size=2,
            attn_dp_rank=1,
            attn_dcp_size=1,
            attn_dcp_rank=0,
            pp_rank=0,
            pp_size=1,
        )
        with (
            patch(
                "sglang.srt.managers.cache_controller.get_parallel",
                return_value=parallel,
            ),
            patch(
                "sglang.srt.managers.cache_controller.is_dp_attention_enabled",
                return_value=True,
            ),
            patch.object(controller, "get_attn_cp_rank_and_size", return_value=(1, 2)),
            patch.object(controller, "_build_unified_plan", return_value=None),
        ):
            config = controller._generate_storage_config("org/model")
        self.assertEqual((config.tp_rank, config.tp_size, config.dp_rank), (0, 2, 1))
        self.assertEqual((config.attn_cp_rank, config.attn_cp_size), (1, 2))
        self.assertTrue(config.is_mla_model)

    def test_config_pairs_grid_keys_with_physical_page_spans(self):
        # MHA key order alternates head groups, while each group's layers are
        # contiguous in memory. Include a short final PP layer chunk.
        cases = (
            (False, 4, [(0, 16384), (32768, 16384), (16384, 16384), (49152, 16384)]),
            (False, 3, [(0, 16384), (24576, 16384), (16384, 8192), (40960, 8192)]),
            (True, 4, [(0, 4096), (4096, 4096)]),
            (True, 3, [(0, 4096), (4096, 2048)]),
        )
        for mla, layer_num, expected_spans in cases:
            with self.subTest(mla=mla, layer_num=layer_num):
                controller = self._controller(mla=mla, grouped=False)
                host_pool = controller.storage_host_pool
                host_pool.layout = "page_unified"
                host_pool.storage_format_tag = None
                host_pool.layer_num = layer_num
                host_pool.end_layer = 4 + layer_num
                host_pool.head_group_num = 1 if mla else 2
                host_pool.page_unified_layout = PageUnifiedLayout(
                    page_size=16,
                    layer_num=layer_num,
                    head_num=1 if mla else 4,
                    head_group_num=1 if mla else 2,
                    head_dim=128,
                    itemsize=1,
                    is_mla=mla,
                )
                controller.enable_storage_metrics = False
                parallel = SimpleNamespace(
                    tp_rank=1,
                    tp_size=2,
                    attn_tp_rank=1,
                    attn_tp_size=2,
                    attn_dcp_size=1,
                    attn_dcp_rank=0,
                    pp_rank=1,
                    pp_size=2,
                )
                memory = SimpleNamespace(
                    hicache_storage_key_scheme="unified",
                    hicache_storage_head_partition=2,
                    hicache_storage_layer_partition=2,
                )
                with (
                    patch(
                        "sglang.srt.managers.cache_controller.get_memory",
                        return_value=memory,
                    ),
                    patch(
                        "sglang.srt.managers.cache_controller.get_parallel",
                        return_value=parallel,
                    ),
                    patch(
                        "sglang.srt.managers.cache_controller.is_dp_attention_enabled",
                        return_value=False,
                    ),
                    patch.object(
                        controller, "get_attn_cp_rank_and_size", return_value=(0, 1)
                    ),
                ):
                    config = controller._generate_storage_config("org/model")

                namespace = derive_namespace(
                    model_id="org/model",
                    dtype="float8_e4m3fn",
                    page_size=16,
                    rank_replicated=mla,
                    total_kv_heads=0 if mla else 8,
                    head_group=0 if mla else 2,
                    layer_partition=2,
                )
                digest = namespace_digest(namespace)
                head_suffixes = ("",) if mla else ("_H2", "_H3")
                self.assertEqual(
                    config.unified_suffixes,
                    [
                        f"{digest}_{layers}{heads}"
                        for layers in ("L4-6", f"L6-{4 + layer_num}")
                        for heads in head_suffixes
                    ],
                )
                self.assertEqual(config.unified_chunk_spans, expected_spans)

    def test_hybrid_write_back_preserves_cpu_host_indices(self):
        controller = object.__new__(HybridCacheController)
        controller.io_backend = "kernel"
        controller.mem_pool_host = SimpleNamespace(
            layout="page_unified",
            supports_per_pool_backup_indices=True,
            anchor_entry=SimpleNamespace(
                host_pool=SimpleNamespace(can_use_write_back_jit=True)
            ),
        )
        operation = CacheOperation(torch.arange(16), torch.arange(16, 32), node_id=1)
        with patch.object(
            controller,
            "move_indices",
            side_effect=AssertionError("Staged write-back needs CPU host indices"),
        ):
            host_indices, device_indices, transfers = controller._move_write_operation(
                operation
            )
        self.assertIs(host_indices, operation.host_indices)
        self.assertIs(device_indices, operation.device_indices)
        self.assertIsNone(transfers)

    def test_incompatible_host_grids_are_rejected(self):
        for unsupported in ("sidecar", "missing_geometry", "packed_draft"):
            with self.subTest(unsupported=unsupported):
                controller = self._controller()
                if unsupported == "sidecar":
                    controller.mem_pool_host.entries.append(SimpleNamespace())
                elif unsupported == "missing_geometry":
                    del controller.storage_host_pool.page_unified_layout
                else:
                    controller.storage_host_pool.layer_num += 1
                with self.assertRaises(NotImplementedError):
                    self._suffixes(controller)


class TestUnifiedChunkTiling(CustomTestCase):
    def test_spans_must_name_every_page_byte_exactly_once(self):
        # Key order is not byte order, so spans arrive unsorted.
        validate_unified_chunk_tiling([(8, 8), (0, 8), (16, 4)], 20)
        for name, spans in (
            ("gap", [(0, 8), (12, 8)]),
            ("overlap", [(0, 8), (4, 16)]),
            ("short", [(0, 8), (8, 8)]),
            ("long", [(0, 8), (8, 16)]),
            ("offset_start", [(4, 16)]),
            ("empty_chunk", [(0, 20), (20, 0)]),
            ("no_chunks", []),
        ):
            with self.subTest(case=name), self.assertRaises(ValueError):
                validate_unified_chunk_tiling(spans, 20)

    def test_controller_refuses_a_plan_that_leaves_page_bytes_unnamed(self):
        # The page block holds one layer more than the plan names: its bytes
        # would never be stored, and a zero-copy read would leave them stale.
        controller = TestControllerUnifiedKeys._controller(None, grouped=False)
        host_pool = controller.storage_host_pool
        host_pool.layout = "page_unified"
        host_pool.storage_format_tag = None
        host_pool.page_unified_layout = PageUnifiedLayout(
            page_size=16,
            layer_num=5,
            head_num=4,
            head_group_num=1,
            head_dim=128,
            itemsize=1,
        )
        controller.enable_storage_metrics = False
        memory = SimpleNamespace(
            hicache_storage_key_scheme="unified",
            hicache_storage_head_partition=4,
            hicache_storage_layer_partition=4,
        )
        parallel = SimpleNamespace(
            tp_rank=1,
            tp_size=2,
            attn_tp_rank=1,
            attn_tp_size=2,
            attn_dcp_size=1,
            attn_dcp_rank=0,
            pp_rank=1,
            pp_size=2,
        )
        module = "sglang.srt.managers.cache_controller"
        with (
            patch(f"{module}.get_memory", return_value=memory),
            patch(f"{module}.get_parallel", return_value=parallel),
            patch(f"{module}.is_dp_attention_enabled", return_value=False),
            patch.object(controller, "get_attn_cp_rank_and_size", return_value=(0, 1)),
            self.assertRaisesRegex(ValueError, "cover 65536 bytes.*has 81920"),
        ):
            controller._generate_storage_config("org/model")


def _draft_entry(*, mla=False, head_group_num=2, name="draft", dtype=torch.bfloat16):
    """A draft KV pool of its own geometry: 5 layers of 64-wide rows, 4 kv
    heads on this rank (or one latent row for MLA), in bfloat16."""
    return SimpleNamespace(
        name=name,
        is_primary_index_anchor=False,
        # An MLA pool view says whether its rows are replicated across ranks.
        device_pool=SimpleNamespace(dtype=dtype, storage_rank_replicated=mla),
        host_pool=SimpleNamespace(
            page_unified_layout=PageUnifiedLayout(
                page_size=16,
                layer_num=5,
                head_num=1 if mla else 4,
                head_group_num=1 if mla else head_group_num,
                head_dim=64,
                itemsize=2,
                is_mla=mla,
            )
        ),
    )


class _DraftSidecarPlanning:
    """Runs the controller's planner for a pool beside the KV anchor."""

    MODULE = "sglang.srt.managers.cache_controller"
    TARGET = dict(
        model_id="org/model",
        dtype="float8_e4m3fn",
        page_size=16,
        rank_replicated=False,
        total_kv_heads=8,
        head_group=2,
        layer_partition=4,
    )

    def _planning(
        self,
        controller,
        *,
        algorithm="DFLASH",
        draft_id="org/dflash@main",
        attn_tp_rank=1,
        pp_size=1,
        cp=(0, 1),
        head_partition=2,
        backend="file",
    ):
        controller.storage_backend_type = backend
        controller.unified_kv_namespace = derive_namespace(**self.TARGET)
        memory = SimpleNamespace(
            hicache_storage_key_scheme="unified",
            hicache_storage_head_partition=head_partition,
            hicache_storage_layer_partition=4,
            hicache_storage_draft_id=draft_id,
        )
        parallel = SimpleNamespace(
            tp_rank=attn_tp_rank,
            tp_size=2,
            attn_tp_rank=attn_tp_rank,
            attn_tp_size=2,
            attn_dcp_size=1,
            attn_dcp_rank=0,
            pp_rank=pp_size - 1,
            pp_size=pp_size,
        )
        stack = ExitStack()
        stack.enter_context(patch(f"{self.MODULE}.get_memory", return_value=memory))
        stack.enter_context(patch(f"{self.MODULE}.get_parallel", return_value=parallel))
        stack.enter_context(
            patch(
                f"{self.MODULE}.get_spec",
                return_value=SimpleNamespace(speculative_algorithm=algorithm),
            )
        )
        stack.enter_context(
            patch(f"{self.MODULE}.is_dp_attention_enabled", return_value=False)
        )
        stack.enter_context(
            patch.object(controller, "get_attn_cp_rank_and_size", return_value=cp)
        )
        return stack

    def _draft_digest(self, **overrides):
        """The draft digest from literals: nothing here is read from the plan."""
        fields = dict(
            target=namespace_digest(derive_namespace(**self.TARGET)),
            algorithm="DFLASH",
            draft_id="org/dflash@main",
            placement="sidecar",
            dtype="bfloat16",
            rank_replicated=False,
            total_kv_heads=8,
            head_group=2,
            row_width=64,
            layer_num=5,
            object_layout="page-unified-v1",
        )
        fields.update(overrides)
        return draft_namespace_digest(DraftKVNamespace(**fields))


class TestControllerUnifiedDraftSidecar(_DraftSidecarPlanning, CustomTestCase):
    def _controller(self):
        return object.__new__(HiCacheController)

    def _plan(self, entry=None, **planning):
        controller = self._controller()
        with self._planning(controller, **planning):
            return controller._plan_unified_sidecar(entry or _draft_entry())

    def test_draft_pool_pages_are_chunks_on_the_drafts_own_grid(self):
        # One (head group, 5 layers) cell: K and V of 16 tokens x 2 heads x 64
        # in bfloat16.
        cell = 5 * 2 * 16 * 2 * 64 * 2
        chunks = self._plan()
        digest = self._draft_digest()
        # Attention-TP rank 1 of 2 owns head groups 2 and 3 of 4.
        self.assertEqual(chunks.suffixes, (f"{digest}_D0-5_H2", f"{digest}_D0-5_H3"))
        self.assertEqual(chunks.spans, ((0, cell), (cell, cell)))
        self.assertTrue(chunks.is_writer)
        self.assertEqual(
            self._plan(attn_tp_rank=0).suffixes,
            (f"{digest}_D0-5_H0", f"{digest}_D0-5_H1"),
        )

    def test_draft_identity_is_the_drafts_own(self):
        # The target is fp8 with 128-wide rows; none of it names the draft.
        base = self._plan().suffixes
        self.assertTrue(base[0].startswith(self._draft_digest()))
        other_dtype = self._plan(_draft_entry(dtype=torch.float8_e5m2)).suffixes
        self.assertTrue(
            other_dtype[0].startswith(self._draft_digest(dtype="float8_e5m2"))
        )
        other_id = self._plan(draft_id="another").suffixes
        self.assertTrue(other_id[0].startswith(self._draft_digest(draft_id="another")))
        self.assertEqual(len({base[0], other_dtype[0], other_id[0]}), 3)

    def test_replicated_draft_has_one_chunk_and_one_writer(self):
        digest = self._draft_digest(
            rank_replicated=True, total_kv_heads=0, head_group=0
        )
        page = 5 * 16 * 64 * 2
        for attn_tp_rank in (0, 1):
            with self.subTest(attn_tp_rank=attn_tp_rank):
                chunks = self._plan(_draft_entry(mla=True), attn_tp_rank=attn_tp_rank)
                self.assertEqual(chunks.suffixes, (f"{digest}_D0-5",))
                self.assertEqual(chunks.spans, ((0, page),))
                # Every rank holds the same bytes: rank 0 writes them.
                self.assertEqual(chunks.is_writer, attn_tp_rank == 0)

    def test_one_writer_per_draft_object(self):
        for mla in (False, True):
            for cp_rank in (0, 1):
                with self.subTest(mla=mla, cp_rank=cp_rank):
                    if cp_rank:
                        with self.assertRaisesRegex(
                            NotImplementedError, "prefill context parallelism"
                        ):
                            self._plan(_draft_entry(mla=mla), cp=(cp_rank, 2))
                        continue
                    writers = {}
                    for attn_tp_rank in (0, 1):
                        chunks = self._plan(
                            _draft_entry(mla=mla), attn_tp_rank=attn_tp_rank
                        )
                        for suffix in chunks.suffixes:
                            writers.setdefault(suffix, []).append(chunks.is_writer)
                    self.assertEqual(len(writers), 1 if mla else 4)
                    for suffix, flags in writers.items():
                        self.assertEqual(sum(flags), 1, suffix)

    def test_pools_without_unified_keys_are_refused(self):
        cases = {
            "draft_swa": dict(entry=_draft_entry(name="draft_swa")),
            "draft_indexer": dict(entry=_draft_entry(name="draft_indexer")),
            "mamba": dict(entry=_draft_entry(name="mamba")),
            "None": dict(entry=SimpleNamespace()),
            "draft": dict(entry=SimpleNamespace(name="draft", host_pool=object())),
        }
        for name, case in cases.items():
            with (
                self.subTest(name),
                self.assertRaisesRegex(
                    NotImplementedError, f"hybrid pools: {name} has no unified keys"
                ),
            ):
                self._plan(**case)
        # The sidecar path is shared with EAGLE3 and DSPARK drafts.
        for algorithm in ("EAGLE3", "DSPARK", "EAGLE", None):
            with (
                self.subTest(algorithm=algorithm),
                self.assertRaisesRegex(
                    NotImplementedError, "hybrid pools: draft has no unified keys"
                ),
            ):
                self._plan(algorithm=algorithm)

    def test_draft_pools_that_cannot_be_named_safely_are_refused(self):
        dcp_entry = _draft_entry()
        dcp_entry.host_pool.dcp_size = 2
        sharded_mla = _draft_entry(mla=True)
        sharded_mla.device_pool.storage_rank_replicated = False
        # A pool that does not say is not taken for replicated.
        unknown_mla = _draft_entry(mla=True)
        del unknown_mla.device_pool.storage_rank_replicated
        cases = {
            "Mooncake backend": dict(backend="mooncake"),
            "single pipeline stage": dict(pp_size=2),
            "full pages": dict(entry=dcp_entry),
            "not known to be replicated": dict(entry=sharded_mla),
            "not known to be replicated ": dict(entry=unknown_mla),
            "draft id": dict(draft_id=None),
            "prefill context parallelism": dict(cp=(0, 2)),
        }
        for message, case in cases.items():
            with (
                self.subTest(message),
                self.assertRaisesRegex(NotImplementedError, message.strip()),
            ):
                self._plan(**case)
        # Without attached unified-key storage there is no target to bind to.
        controller = self._controller()
        with self._planning(controller):
            controller.unified_kv_namespace = None
            with self.assertRaisesRegex(NotImplementedError, "no unified-key storage"):
                controller._plan_unified_sidecar(_draft_entry())
        # 1 kv head per rank could be replication: the operator must say so.
        one_head = _draft_entry(head_group_num=1)
        one_head.host_pool.page_unified_layout = PageUnifiedLayout(
            page_size=16,
            layer_num=5,
            head_num=1,
            head_group_num=1,
            head_dim=64,
            itemsize=2,
        )
        with self.assertRaisesRegex(NotImplementedError, "1 kv head per rank"):
            self._plan(one_head, head_partition=None)
        # The head partition must divide the draft's local kv heads too.
        with self.assertRaisesRegex(ValueError, "must divide this rank's 4"):
            self._plan(head_partition=3)

    def test_host_pool_must_be_cut_into_the_planned_head_groups(self):
        # Built without the head partition, the page is head-major over all 4
        # heads: a chunk span would cut across layers.
        with self.assertRaisesRegex(ValueError, "2 head groups.*holds 1"):
            self._plan(_draft_entry(head_group_num=1))

    def test_attach_plans_pools_that_joined_before_storage(self):
        """A runtime re-attach: the draft pool is already in the group."""
        fixture = TestControllerUnifiedKeys()
        controller = fixture._controller()
        host_pool = controller.storage_host_pool
        host_pool.layout = "page_unified"
        host_pool.storage_format_tag = None
        controller.mem_pool_host.layout = "page_unified"
        controller.enable_storage_metrics = False
        with self._planning(controller, head_partition=4):
            target_only = controller._generate_storage_config("org/model")
            self.assertEqual(target_only.unified_pool_chunks, {})
            draft = _draft_entry(head_group_num=1)
            controller.mem_pool_host.entries.append(draft)
            config = controller._generate_storage_config("org/model")
            self.assertEqual(config.unified_suffixes, target_only.unified_suffixes)
            self.assertEqual(
                config.unified_chunk_spans, target_only.unified_chunk_spans
            )
            self.assertEqual(list(config.unified_pool_chunks), ["draft"])
            # The draft is bound to the namespace this attach derived, not to
            # one left over on the controller.
            target = derive_namespace(**{**self.TARGET, "head_group": 4})
            self.assertEqual(controller.unified_kv_namespace, target)
            digest = self._draft_digest(target=namespace_digest(target), head_group=4)
            self.assertEqual(
                config.unified_pool_chunks["draft"].suffixes, (f"{digest}_D0-5_H1",)
            )
            # Any other pool beside the anchor still fails the attach.
            controller.mem_pool_host.entries.append(_draft_entry(name="draft_swa"))
            with self.assertRaisesRegex(NotImplementedError, "single page_unified"):
                controller._generate_storage_config("org/model")
            controller.mem_pool_host.entries[-1] = _draft_entry(name="draft")
            controller.mem_pool_host.entries[-2] = SimpleNamespace()
            with self.assertRaisesRegex(NotImplementedError, "single page_unified"):
                controller._generate_storage_config("org/model")

    def test_failed_attach_leaves_no_target_namespace_behind(self):
        """The namespace is that of the attached storage: a pool that
        registers later must not take its target from an attach that failed."""
        left_over = derive_namespace(**self.TARGET)

        def controller_with(*entries):
            controller = TestControllerUnifiedKeys()._controller()
            controller.storage_host_pool.layout = "page_unified"
            controller.storage_host_pool.storage_format_tag = None
            controller.mem_pool_host.layout = "page_unified"
            controller.mem_pool_host.entries.extend(entries)
            controller.enable_storage_metrics = False
            controller.enable_storage = False
            controller.prefetch_hits_sync_groups = []
            controller.prefetch_completion_sync_groups = []
            return controller

        # The pool that is already in the group is refused while it is planned.
        controller = controller_with(_draft_entry(head_group_num=1))
        with (
            self._planning(controller, head_partition=4, draft_id=None),
            self.assertRaisesRegex(NotImplementedError, "draft id"),
        ):
            self.assertEqual(controller.unified_kv_namespace, left_over)
            controller._generate_storage_config("org/model")
        self.assertIsNone(controller.unified_kv_namespace)
        # The backend cannot be created.
        controller = controller_with()
        with (
            self._planning(controller, head_partition=4),
            patch.object(controller, "_stop_storage_threads"),
            patch(
                "sglang.srt.mem_cache.storage.StorageBackendFactory.create_backend",
                side_effect=RuntimeError("no backend"),
            ),
            self.assertRaisesRegex(RuntimeError, "no backend"),
        ):
            controller.attach_storage_backend("file", model_name="org/model")
        self.assertIsNone(controller.unified_kv_namespace)
        self.assertFalse(controller.enable_storage)


class TestUnifiedLateHostPool(_DraftSidecarPlanning, CustomTestCase):
    def _controller(self, unified_suffixes):
        controller = object.__new__(HybridCacheController)
        controller.mem_pool_host = MagicMock(spec=HostPoolGroup)
        controller.enable_storage = True
        controller.storage_backend = MagicMock()
        controller.storage_host_pool = SimpleNamespace(dcp_size=1)
        controller.storage_config = SimpleNamespace(
            unified_suffixes=unified_suffixes, unified_pool_chunks={}
        )
        controller.extra_host_mem_release_queues = {}
        return controller

    def test_named_pool_is_planned_before_it_joins_and_registers(self):
        controller = self._controller(["ukv1-abc_L0-4"])
        entry = _draft_entry()
        seen = {}
        controller.mem_pool_host.add_entry.side_effect = lambda _: seen.update(
            add=dict(controller.storage_config.unified_pool_chunks)
        )
        controller.storage_backend.register_mem_host_pool_v2.side_effect = lambda *_: (
            seen.update(
                register=dict(controller.storage_config.unified_pool_chunks),
                added=controller.mem_pool_host.add_entry.called,
            )
        )
        with self._planning(controller):
            controller.register_host_pool_entry(entry)
        chunks = controller.storage_config.unified_pool_chunks["draft"]
        self.assertEqual(len(chunks.suffixes), 2)
        # The backend reads the plan while the pool registers.
        self.assertEqual(
            seen,
            {"add": {"draft": chunks}, "register": {"draft": chunks}, "added": True},
        )
        controller.storage_backend.register_mem_host_pool_v2.assert_called_once_with(
            entry.host_pool, "draft"
        )

    def test_refused_draft_pool_leaves_no_plan_behind(self):
        controller = self._controller(["ukv1-abc_L0-4"])
        with (
            self._planning(controller, backend="mooncake"),
            self.assertRaisesRegex(NotImplementedError, "Mooncake backend"),
        ):
            controller.register_host_pool_entry(_draft_entry())
        self.assertEqual(controller.storage_config.unified_pool_chunks, {})
        controller.mem_pool_host.add_entry.assert_not_called()
        controller.storage_backend.register_mem_host_pool_v2.assert_not_called()

    def test_plan_does_not_outlive_a_pool_that_failed_to_register(self):
        earlier = SimpleNamespace(is_writer=True)
        for step in ("add_entry", "register_mem_host_pool_v2"):
            for previous in ({}, {"draft": earlier}):
                with self.subTest(step=step, previous=previous):
                    controller = self._controller(["ukv1-abc_L0-4"])
                    pool_chunks = controller.storage_config.unified_pool_chunks
                    pool_chunks.update(previous)
                    seen = []

                    def fail(*_):
                        seen.append(dict(pool_chunks))
                        raise RuntimeError("no room")

                    owner = (
                        controller.mem_pool_host
                        if step == "add_entry"
                        else controller.storage_backend
                    )
                    getattr(owner, step).side_effect = fail
                    with (
                        self._planning(controller),
                        self.assertRaisesRegex(RuntimeError, "no room"),
                    ):
                        controller.register_host_pool_entry(_draft_entry())
                    # The failing step saw the new plan; nothing keeps it.
                    self.assertEqual(len(seen[0]["draft"].suffixes), 2)
                    self.assertEqual(pool_chunks, previous)
                    self.assertIs(
                        controller.storage_config.unified_pool_chunks, pool_chunks
                    )

    def test_planned_writer_backs_up_the_pool_whatever_the_kv_writer_is(self):
        for kv_writer in (False, True):
            for is_writer in (False, True):
                with self.subTest(kv_writer=kv_writer, is_writer=is_writer):
                    controller = self._controller(["ukv1-abc_L0-4"])
                    controller.storage_backend_type = "file"
                    controller.storage_config.is_storage_writer = kv_writer
                    controller.storage_config.unified_pool_chunks["draft"] = (
                        SimpleNamespace(is_writer=is_writer)
                    )
                    transfer = SimpleNamespace(name="draft")
                    self.assertEqual(controller.should_backup(transfer), is_writer)
                    # A pool without a plan keeps the legacy rule.
                    other = SimpleNamespace(name="swa")
                    self.assertEqual(controller.should_backup(other), kv_writer)

    def test_pool_without_unified_keys_is_refused_before_it_joins(self):
        controller = self._controller(["ukv1-abc_L0-4"])
        entry = SimpleNamespace(
            name="draft", host_pool=object(), is_primary_index_anchor=False
        )
        with self.assertRaisesRegex(
            NotImplementedError, "hybrid pools: draft has no unified keys"
        ):
            controller.register_host_pool_entry(entry)
        controller.mem_pool_host.add_entry.assert_not_called()
        controller.storage_backend.register_mem_host_pool_v2.assert_not_called()
        self.assertEqual(controller.extra_host_mem_release_queues, {})

    def test_rank_suffix_registration_is_unchanged(self):
        controller = self._controller(None)
        entry = SimpleNamespace(
            name="draft", host_pool=object(), is_primary_index_anchor=False
        )
        controller.register_host_pool_entry(entry)
        controller.mem_pool_host.add_entry.assert_called_once_with(entry)
        controller.storage_backend.register_mem_host_pool_v2.assert_called_once_with(
            entry.host_pool, "draft"
        )


class TestFileBackendUnifiedKeys(CustomTestCase):
    def _config(self, unified_suffixes):
        return HiCacheStorageConfig(
            tp_rank=1,
            tp_size=4,
            pp_rank=0,
            pp_size=1,
            attn_cp_rank=0,
            attn_cp_size=1,
            is_mla_model=False,
            enable_storage_metrics=False,
            is_page_first_layout=False,
            model_name="org/model",
            unified_suffixes=unified_suffixes,
        )

    def test_unified_suffix_replaces_the_whole_rank_suffix(self):
        """No tp/pp/cp coordinate may survive, or the key stops being portable."""
        backend = HiCacheFile.__new__(HiCacheFile)
        config = self._config(["ukv1-abc_L0-80_H1"])
        config.attn_cp_rank = 3
        config.attn_cp_size = 4
        HiCacheFile._init_config_suffix(backend, config)
        self.assertEqual(backend.config_suffix, "_ukv1-abc_L0-80_H1")
        self.assertEqual(backend.tp_config_suffix, backend.config_suffix)

    def test_file_round_trip_and_eviction_ownership_follow_writer_election(self):
        # Full-page CP replicas share files; MLA also shares across TP ranks.
        cases = (
            (False, 1, 0, True),
            (False, 1, 1, False),
            (True, 0, 0, True),
            (True, 1, 0, False),
            (True, 0, 1, False),
        )
        value = torch.arange(16, dtype=torch.uint8)
        for mla, tp_rank, cp_rank, owns_files in cases:
            with (
                self.subTest(mla=mla, tp_rank=tp_rank, cp_rank=cp_rank),
                TemporaryDirectory() as directory,
                envs.SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR.override(directory),
            ):
                owner_config = replace(
                    self._config(["ukv1-abc_L0-80" + ("" if mla else "_H1")]),
                    is_mla_model=mla,
                    tp_rank=0 if mla else tp_rank,
                    attn_cp_size=2,
                    extra_config={
                        "enable_metadata_cache": True,
                        "metadata_ttl": 60,
                        "max_size": "32",
                        "min_free_space": "0",
                        "eviction_ratio": 1.0,
                    },
                )
                writer = HiCacheFile(owner_config)
                self.assertTrue(writer.set("existing", value))

                # Initializing another rank scans shared files for metadata,
                # while its evictor must track only the elected writer's files.
                backend = HiCacheFile(
                    replace(owner_config, tp_rank=tp_rank, attn_cp_rank=cp_rank)
                )
                self.assertTrue(backend.exists("existing"))
                restored = torch.empty_like(value)
                self.assertIs(backend.get("existing", restored), restored)
                torch.testing.assert_close(restored, value)
                self.assertEqual(backend.set("new", value), owns_files)
                self.assertTrue(writer.exists("existing"))

    def test_rank_suffix_path_is_unchanged(self):
        backend = HiCacheFile.__new__(HiCacheFile)
        HiCacheFile._init_config_suffix(backend, self._config(None))
        self.assertEqual(backend.config_suffix, "_org-model_1_4")

    def test_multi_chunk_grid_requires_byte_spans(self):
        """Coordinates alone cannot identify a chunk's position in a page."""
        backend = HiCacheFile.__new__(HiCacheFile)
        with self.assertRaises(ValueError):
            HiCacheFile._init_config_suffix(backend, self._config(["a", "b"]))


if __name__ == "__main__":
    unittest.main()
