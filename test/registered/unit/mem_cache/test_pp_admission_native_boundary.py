"""Main native lookup and receipt boundaries (CPU doubles, not GPU acceptance)."""

import dataclasses
import logging
import os
import sys
import unittest
from array import array
from contextlib import contextmanager, nullcontext
from itertools import product
from queue import Queue
from types import SimpleNamespace as NS
from unittest.mock import Mock, create_autospec, patch

from sglang.srt.runtime_context import (
    get_context,
    get_disagg,
    get_memory,
    get_parallel,
    get_serving,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.pp_cache_test_support import (
    ROOT,
    bind_prefill_release,
    config,
    environ,
    functions,
    lease,
    method,
)
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


@contextmanager
def admission_context(
    cfg=None,
    *,
    pp=4,
    mode="prefill",
    cache_mode="cache",
    storage="mooncake",
    hicache=True,
    role_switch=False,
):
    """Publish the process configuration read by the real environment gate."""
    if cfg is None:
        cfg = NS(
            pp_size=pp,
            disaggregation_mode=mode,
            enable_hierarchical_cache=hicache,
            hicache_storage_backend=storage,
            hicache_host_memory_mode=cache_mode,
            enable_pd_role_switch=role_switch,
        )
    fields = {
        name: getattr(cfg, name)
        for name in (
            "pp_size",
            "disaggregation_mode",
            "enable_hierarchical_cache",
            "hicache_storage_backend",
            "hicache_host_memory_mode",
            "enable_pd_role_switch",
        )
    }
    with get_context().override_server_args(**fields):
        yield


def admission_profiles(*, backends=(None, "mooncake", "file"), hicache=(False, True)):
    """Independent profile truth table, shared by startup and attach tests."""
    for flag, pp, mode, host, backend, enabled, role_switch in product(
        ("0", "1"),
        (1, 4),
        ("null", "decode", "prefill"),
        ("cache", "buffer_only"),
        backends,
        hicache,
        (False, True),
    ):
        cfg = NS(
            pp_size=pp,
            disaggregation_mode=mode,
            hicache_host_memory_mode=host,
            hicache_storage_backend=backend,
            enable_hierarchical_cache=enabled,
            enable_pd_role_switch=role_switch,
        )
        requested = (
            flag == "1"
            and pp > 1
            and mode == "prefill"
            and host == "cache"
            and backend is not None
            and enabled
        )
        yield flag, cfg, requested


validate_admission_backend = method(
    "arg_groups/hicache_hook.py",
    None,
    "validate_pp_cache_admission_backend",
    {
        "resolving_view": lambda cfg: cfg,
        "pp_cache_admission_requested": config.pp_cache_admission_requested,
    },
)


def model_validator(model_lookup):
    return method(
        "arg_groups/hicache_hook.py",
        None,
        "validate_pp_cache_admission_model",
        {
            "resolving_view": lambda cfg: cfg,
            "pp_cache_admission_requested": config.pp_cache_admission_requested,
            "model_config_of": model_lookup,
        },
    )


match_cache = functions(
    "mem_cache/common.py",
    {"match_kv_cache", "_req_radix_key"},
    {
        "RadixKey": lambda **kw: NS(**kw),
        "MatchPrefixParams": lambda **kw: NS(**kw),
        "envs": NS(SGLANG_RADIX_FORCE_MISS=NS(get=lambda: False)),
    },
)["match_kv_cache"]

init_input = method(
    "managers/schedule_batch.py",
    "Req",
    "init_next_round_input",
    {
        "array": array,
        "RadixKey": lambda **kw: NS(**kw),
        "MatchPrefixParams": lambda **kw: NS(**kw),
        "envs": NS(SGLANG_RADIX_FORCE_MISS=NS(get=lambda: False)),
        "match_kv_cache": match_cache,
    },
)


class Cache:
    page_size = 64
    lock = method(
        "mem_cache/base_prefix_cache.py",
        "BasePrefixCache",
        "lock",
        {"TreeLock": lambda node, receipt: NS(node=node, receipt=receipt)},
    )
    unlock = method("mem_cache/base_prefix_cache.py", "BasePrefixCache", "unlock", {})

    def __init__(self, device=128, total=1024):
        self.device, self.total = device, total
        self.calls, self.locks = [], []
        self.reprefill_tail = 0

    def swa_reprefill_tail_tokens(self, req=None):
        return self.reprefill_tail

    def supports_mamba(self):
        return False

    def match_prefix(self, params):
        self.calls.append(params)
        end = min(self.total, params.key.limit or 0)
        return NS(
            device_prefix_len=min(self.device, end),
            last_device_node=("d", end),
            last_host_node=("h", end),
            best_match_node=("h", end),
            host_hit_length=max(0, end - self.device),
            swa_host_hit_length=0,
            swa_branching_seqlen=None,
            mamba_host_hit_length=0,
            mamba_branching_seqlen=None,
            cache_protected_len=None,
        )

    def _lock(self, kind, node):
        receipt = (kind, node, len(self.locks))
        self.locks.append(receipt)
        return NS(to_dec_params=lambda: receipt)

    def _unlock(self, kind, node, receipt):
        assert receipt[:2] == (kind, node)
        self.locks.remove(receipt)

    def inc_lock_ref(self, node):
        return self._lock("device", node)

    def inc_host_lock_ref(self, node):
        return self._lock("host", node)

    def dec_lock_ref(self, node, receipt):
        self._unlock("device", node, receipt)

    def dec_host_lock_ref(self, node, receipt):
        self._unlock("host", node, receipt)


def request(natural=4095):
    req = NS(
        is_dllm=lambda: False,
        refresh_fill_ids=lambda: None,
        full_untruncated_fill_ids=array("q", range(4096)),
        _compute_max_prefix_len=lambda length: min(length - 1, natural),
        session=None,
        positional_embed_overrides=None,
        extra_key="adapter",
        cache_salt="tenant",
        kv=NS(),
        is_retracted=False,
    )
    req.init_next_round_input = lambda cache, **kw: init_input(req, cache, **kw)
    return req


class NativeBoundaryTests(CustomTestCase):
    def test_stale_query_is_rejected_before_accounting_and_allocation(self):
        drain = method(
            "mem_cache/unified_radix_cache.py",
            "UnifiedRadixCache",
            "_drain_storage_control_queues_impl",
            {"pp_cache_admission_enabled": lambda: True},
        )
        handle = ("reused-rid", 0)
        old = NS(handle=handle, storage_hit_count=512)
        current = NS(operation=object())
        controller = NS(
            **{
                name: Queue()
                for name in (
                    "prefetch_hit_queue",
                    "ack_prefetch_queue",
                    "ack_backup_queue",
                    "host_mem_release_queue",
                )
            }
        )
        controller.prefetch_hit_queue.put(old)
        cache = NS(
            cache_controller=controller,
            host_memory_mode="cache",
            ongoing_prefetch={handle: current},
            _invalidate_absent_from_hit_query=Mock(),
            _record_storage_prefetch_hit=Mock(
                side_effect=AssertionError("stale query accounted")
            ),
        )
        drain(cache, 1, 0, 0, 0, {}, False)
        cache._invalidate_absent_from_hit_query.assert_called_once_with(old)
        cache._record_storage_prefetch_hit.assert_not_called()
        self.assertIs(cache.ongoing_prefetch[handle], current)
        self.assertTrue(controller.prefetch_hit_queue.empty())

    def load_back_fixture(self, tokens=64, capacity=4096):
        import torch

        scope = functions(
            "mem_cache/base_prefix_cache.py",
            {"InitLoadBackParams"},
            {"__name__": __name__, "dataclasses": dataclasses},
        )
        params_type = scope["InitLoadBackParams"]
        scope.update(
            rank_consensus=lambda **kw: lambda fn: fn,
            ComponentType=NS(FULL="full"),
            CacheTransferPhase=NS(LOAD_BACK="load"),
            EvictParams=lambda **kw: NS(**kw),
            _OngoingLoadBack=lambda *args: args,
            logger=logging.getLogger(__name__),
        )
        names = {
            "init_load_back",
            "load_back",
            "_load_back_transfers",
            "load_back_is_cache_owned",
        }
        functions(
            "mem_cache/unified_radix_cache.py", names, scope, cls="UnifiedRadixCache"
        )
        indices = torch.arange(tokens)
        receipt = NS(delta=0, to_dec_params=lambda: "receipt")
        cache = NS(
            buffer_pipeline=None,
            linker=None,
            load_back_threshold=256,
            tree_core=NS(
                is_full_device_evicted=lambda node: True,
                empty_match_result=NS(device_indices=torch.empty(0)),
                collect_full_device_indices=Mock(return_value=indices),
                build_load_back_spec=Mock(return_value=(NS(host_indices=indices), {})),
                commit_load_back=Mock(return_value=[]),
            ),
            _build_sidecar_transfers=Mock(return_value=[]),
            cache_controller=NS(load=Mock(return_value=indices)),
            token_to_kv_pool_allocator=NS(),
            _component_available_size=Mock(return_value=capacity),
            evict_for_alloc=Mock(),
            inc_host_lock_ref=Mock(return_value=receipt),
            inc_lock_ref=Mock(return_value=receipt),
            dec_host_lock_ref=Mock(),
            dec_lock_ref=Mock(),
            _components_tuple=(),
            _apply_cache_actions=Mock(),
            ongoing_load_back={},
        )
        for name in names:
            setattr(
                cache, name, lambda *args, fn=scope[name], **kw: fn(cache, *args, **kw)
            )
        return cache, params_type

    def test_load_back_bypass_skips_only_size_threshold_not_quota_or_capacity(self):
        for tokens, bypass, quota, capacity, success in (
            (64, False, None, 4096, False),
            (64, True, None, 4096, True),
            (0, True, None, 4096, False),
            (64, True, 32, 4096, False),
            (64, True, None, 0, False),
            (512, False, None, 4096, True),
        ):
            with self.subTest(
                tokens=tokens, bypass=bypass, quota=quota, capacity=capacity
            ):
                cache, params_type = self.load_back_fixture(tokens, capacity)
                req = NS(rid="r", last_node="device")
                params = params_type("host", tokens, mem_quota=quota, req=req)
                self.assertFalse(params.skip_load_back_threshold)
                params.skip_load_back_threshold = bypass
                loaded_tokens, node = cache.init_load_back(params)
                self.assertEqual(loaded_tokens, tokens if success else 0)
                self.assertEqual(node, "host" if success else "device")
                self.assertEqual(cache.cache_controller.load.call_count, int(success))
                self.assertEqual(cache.dec_host_lock_ref.call_count, int(not success))
                self.assertEqual(bool(cache.ongoing_load_back), success)

    def test_adder_keeps_cap_check_and_sends_only_threshold_intent_to_cache(self):
        for cap in (None, 128, 0, 64):
            with self.subTest(cap=cap):
                cache, params_type = self.load_back_fixture()
                cache.tree_core.build_hicache_transfers = Mock(
                    return_value=[NS(host_indices=list(range(64)))]
                )
                cache.init_load_back = Mock(wraps=cache.init_load_back)
                req = NS(
                    rid="r",
                    last_node="device",
                    best_match_node="host",
                    prefix_len=64,
                    host_hit_length=64,
                    swa_host_hit_length=0,
                    full_untruncated_fill_ids=list(range(129)),
                    output_ids=[],
                    kv=NS(),
                    sampling_params=NS(ignore_eos=False, max_new_tokens=0),
                    needs_host_load_back=lambda: True,
                )

                class Rejected:
                    OTHER = object()

                add = method(
                    "managers/schedule_policy.py",
                    "PrefillAdder",
                    "add_one_req",
                    {
                        "UnifiedRadixCache": NS,
                        "InitLoadBackParams": params_type,
                        "AddReqResult": Rejected,
                        "CLIP_MAX_NEW_TOKENS": 4096,
                        "ComponentType": NS(FULL="full"),
                        "CacheTransferPhase": NS(LOAD_BACK="load"),
                        "nullcontext": nullcontext,
                    },
                )
                adder = NS(
                    prefill_max_requests=None,
                    tree_cache=cache,
                    per_req_token_overhead=64,
                    _mamba_gap_budget_for_req=lambda req, **kw: 0,
                    _request_slot_reserve_for_req=lambda req, **kw: 0,
                    rem_mamba_slots=None,
                    _lock_node=lambda *a, **kw: nullcontext(),
                    _select_prefill_admission=lambda *a, **kw: NS(
                        prefix_len=128, extend_len=1, max_new_tokens=0, is_chunked=False
                    ),
                    prefill_delayer_single_pass=None,
                    page_size=64,
                    rem_chunk_tokens=1024,
                    memory_budget=NS(
                        prepare_load_back=lambda **kw: True,
                        load_back_needs_full_tokens=True,
                    ),
                    _kv_shard_reserve_scratch=lambda **kw: True,
                    _commit_prefill_admission=Mock(),
                    budget_state=lambda: "continue",
                )
                # Keep the native arm's host hit above its configured threshold.
                # The smaller-hit native rejection is tested separately above.
                if cap is None:
                    cache.load_back_threshold = 64
                if cap in (0, 64):
                    with self.assertRaisesRegex(RuntimeError, "certified PP prefix"):
                        add(adder, req, False, None, certified_prefix_len=cap)
                    adder._commit_prefill_admission.assert_not_called()
                else:
                    self.assertEqual(
                        add(adder, req, False, None, certified_prefix_len=cap),
                        "continue",
                    )
                    adder._commit_prefill_admission.assert_called_once()
                params = cache.init_load_back.call_args.args[0]
                self.assertEqual(params.skip_load_back_threshold, cap is not None)

    def test_storage_forecast_is_read_only_and_none_differs_from_default(self):
        with environ.envs.SGLANG_ENABLE_PP_L3_CACHE_ASYNC_ADMISSION.override(True):
            for current in (None, "mooncake", "file"):
                with self.subTest(current=current), admission_context(storage=current):
                    original = config.pp_cache_admission_enabled()
                    self.assertFalse(
                        config.pp_cache_admission_enabled(storage_backend=None)
                    )
                    for proposed in ("mooncake", "file"):
                        self.assertTrue(
                            config.pp_cache_admission_enabled(storage_backend=proposed)
                        )
                    self.assertEqual(config.pp_cache_admission_enabled(), original)
                    self.assertEqual(
                        sys.modules["sglang.srt.runtime_context"]
                        .get_memory()
                        .hicache_storage_backend,
                        current,
                    )

    def test_disabled_flag_does_not_read_unpublished_runtime(self):
        with (
            environ.envs.SGLANG_ENABLE_PP_L3_CACHE_ASYNC_ADMISSION.override(False),
            patch.dict(sys.modules, {"sglang.srt.runtime_context": None}),
        ):
            self.assertFalse(config.pp_cache_admission_enabled(storage_backend="file"))

    def test_experimental_flag_is_a_deprecated_alias(self):
        flag = environ.envs.SGLANG_ENABLE_PP_L3_CACHE_ASYNC_ADMISSION
        with (
            flag.override(False),
            patch.dict(os.environ, {"SGLANG_PP_L3_CACHE_ASYNC_ADMISSION": "1"}),
        ):
            flag.clear()
            with self.assertWarnsRegex(DeprecationWarning, "deprecated"):
                self.assertTrue(flag.get())

    def test_backend_options_fail_before_model_or_runtime_initialization(self):
        cfg = NS(
            pp_size=4,
            disaggregation_mode="prefill",
            enable_hierarchical_cache=True,
            hicache_storage_backend="file",
            hicache_host_memory_mode="cache",
            radix_cache_backend=None,
            enable_flexkv=False,
            enable_pd_role_switch=False,
            enable_lora=False,
            lora_paths=None,
            enable_priority_scheduling=False,
            disable_priority_preemption=False,
            enable_dynamic_chunking=False,
            enable_mixed_chunk=False,
            dllm_algorithm=None,
            speculative_algorithm=None,
        )
        with (
            environ.envs.SGLANG_ENABLE_PP_L3_CACHE_ASYNC_ADMISSION.override(True),
            patch.dict(
                sys.modules,
                {
                    "sglang.srt.runtime_context": None,
                    "sglang.srt.configs.hybrid_arch": None,
                },
            ),
        ):
            # An L3 backend does not replace the built-in radix cache.
            validate_admission_backend(cfg)
            for field, value, message in (
                ("radix_cache_backend", "custom", "built-in UnifiedRadixCache"),
                ("enable_flexkv", True, "enable-flexkv"),
                ("enable_pd_role_switch", True, "role switching"),
                ("enable_lora", True, "static PD-prefill"),
                ("lora_paths", ["adapter"], "static PD-prefill"),
                ("enable_dynamic_chunking", True, "static PD-prefill"),
                ("enable_mixed_chunk", True, "static PD-prefill"),
                ("dllm_algorithm", "test", "static PD-prefill"),
                ("speculative_algorithm", "EAGLE", "static PD-prefill"),
                ("enable_priority_scheduling", True, "static PD-prefill"),
            ):
                bad = NS(**vars(cfg))
                setattr(bad, field, value)
                with (
                    self.subTest(field=field),
                    self.assertRaisesRegex(ValueError, message),
                ):
                    validate_admission_backend(bad)
        source = (ROOT / "arg_groups/pipeline.py").read_text()
        self.assertLess(
            source.index("run_hook(validate_pp_cache_admission_backend, server_args)"),
            source.index("if cfg.model_path.lower()"),
        )

    def test_startup_and_runtime_gates_share_the_profile_contract(self):
        model_lookup = Mock(return_value=NS(is_hybrid_swa=True))
        validate_model = model_validator(model_lookup)
        for flag, cfg, requested in admission_profiles():
            cfg.radix_cache_backend = "plugin"
            cfg.enable_flexkv = True
            with (
                self.subTest(flag=flag, **vars(cfg)),
                environ.envs.SGLANG_ENABLE_PP_L3_CACHE_ASYNC_ADMISSION.override(
                    flag == "1"
                ),
                admission_context(cfg),
                patch.dict(
                    sys.modules,
                    {"sglang.srt.configs.hybrid_arch": NS(mambaish_config=Mock())},
                ),
            ):
                self.assertEqual(
                    config.pp_cache_admission_requested(config=cfg), requested
                )
                self.assertEqual(
                    config.pp_cache_admission_enabled(),
                    requested and not cfg.enable_pd_role_switch,
                )
                model_lookup.reset_mock()
                if requested:
                    with self.assertRaisesRegex(
                        ValueError, "built-in UnifiedRadixCache"
                    ):
                        validate_admission_backend(cfg)
                    with self.assertRaisesRegex(ValueError, "SWA/Mamba"):
                        validate_model(cfg)
                    model_lookup.assert_called_once_with(cfg)
                else:
                    validate_admission_backend(cfg)
                    validate_model(cfg)
                    model_lookup.assert_not_called()

    def storage_admin(
        self,
        *,
        active=False,
        pp=4,
        mode="prefill",
        host_mode="cache",
        storage="mooncake",
        hicache=True,
        role_switch=False,
    ):
        memory = NS(
            pp_size=pp,
            disaggregation_mode=mode,
            enable_pd_role_switch=role_switch,
            enable_hierarchical_cache=hicache,
            hicache_storage_backend=storage,
            hicache_host_memory_mode=host_mode,
        )
        published = get_context().override_server_args(
            **vars(memory), served_model_name="model"
        )
        published.install()
        self.addCleanup(published.restore)
        memory = get_memory()
        # Observe calls while applying real runtime overrides to the published bags.
        context = NS(override=Mock(wraps=get_context().override))
        scheduler = NS(
            enable_hierarchical_cache=hicache,
            enable_hicache_storage=storage is not None,
            pp_prefetch_admission=object() if active else None,
            disaggregation_mode=mode,
            is_fully_idle=Mock(return_value=True),
            waiting_queue=[],
            running_batch=NS(reqs=[]),
            tree_cache=NS(
                attach_storage_backend=Mock(return_value=(True, "attached")),
                detach_storage_backend=Mock(return_value=(True, "detached")),
            ),
        )
        namespace = dict(
            os=os,
            logger=Mock(),
            AttachHiCacheStorageReqOutput=NS,
            DetachHiCacheStorageReqOutput=NS,
            get_memory=get_memory,
            get_parallel=get_parallel,
            get_disagg=get_disagg,
            get_serving=get_serving,
            get_context=lambda: context,
            DisaggregationMode=NS(PREFILL="prefill"),
            pp_cache_admission_enabled=config.pp_cache_admission_enabled,
        )
        native_attach = method(
            "managers/scheduler.py",
            "Scheduler",
            "attach_hicache_storage_wrapped",
            namespace,
        )
        native_detach = method(
            "managers/scheduler.py",
            "Scheduler",
            "detach_hicache_storage_wrapped",
            namespace,
        )

        def attach(s, req):
            return native_attach(s, req)

        def detach(s, req):
            return native_detach(s, req)

        req = NS(
            hicache_storage_backend="mooncake",
            hicache_storage_backend_extra_config_json=None,
            hicache_storage_prefetch_policy="timeout",
            hicache_write_policy=None,
        )
        return scheduler, memory, context, req, attach, detach

    def test_live_admission_detach_is_rejected_before_any_side_effect(self):
        for flag in ("0", "1"):
            with (
                self.subTest(flag=flag),
                environ.envs.SGLANG_ENABLE_PP_L3_CACHE_ASYNC_ADMISSION.override(
                    flag == "1"
                ),
            ):
                s, memory, context, req, _, detach = self.storage_admin(active=True)
                owner = s.pp_prefetch_admission
                s.is_fully_idle.side_effect = AssertionError(
                    "guard must precede idle/drain"
                )
                for _ in range(2):
                    out = detach(s, req)
                    self.assertFalse(out.success)
                    self.assertIn("Restart", out.message)
                s.tree_cache.detach_storage_backend.assert_not_called()
                context.override.assert_not_called()
                self.assertEqual(memory.hicache_storage_backend, "mooncake")
                self.assertTrue(s.enable_hicache_storage)
                self.assertIs(s.pp_prefetch_admission, owner)

    def test_l2_only_cannot_hot_attach_into_half_an_admission_path(self):
        with environ.envs.SGLANG_ENABLE_PP_L3_CACHE_ASYNC_ADMISSION.override(True):
            s, memory, context, req, attach, _ = self.storage_admin(storage=None)
            out = attach(s, req)
            self.assertFalse(out.success)
            self.assertIn("restart", out.message)
            s.tree_cache.attach_storage_backend.assert_not_called()
            s.is_fully_idle.assert_not_called()
            context.override.assert_not_called()
            self.assertFalse(s.enable_hicache_storage)
            self.assertIsNone(memory.hicache_storage_backend)
            self.assertIsNone(s.pp_prefetch_admission)

    def test_active_admission_still_allows_same_backend_policy_update(self):
        with environ.envs.SGLANG_ENABLE_PP_L3_CACHE_ASYNC_ADMISSION.override(True):
            s, memory, context, req, attach, _ = self.storage_admin(active=True)
            owner = s.pp_prefetch_admission
            for policy in ("best_effort", "timeout", "wait_complete"):
                req.hicache_storage_prefetch_policy = policy
                out = attach(s, req)
                self.assertTrue(out.success)
                self.assertEqual(memory.hicache_storage_backend, "mooncake")
                self.assertEqual(memory.hicache_storage_prefetch_policy, policy)
                self.assertIs(s.pp_prefetch_admission, owner)
            self.assertEqual(s.tree_cache.attach_storage_backend.call_count, 3)
            self.assertEqual(context.override.call_count, 3)
            s.tree_cache.detach_storage_backend.assert_not_called()

    def test_storage_attach_matches_real_activation_under_requested_backend(self):
        for flag, cfg, requested in admission_profiles(
            backends=("mooncake", "file"), hicache=(True,)
        ):
            with (
                self.subTest(flag=flag, **vars(cfg)),
                environ.envs.SGLANG_ENABLE_PP_L3_CACHE_ASYNC_ADMISSION.override(
                    flag == "1"
                ),
            ):
                with admission_context(cfg):
                    blocked = config.pp_cache_admission_enabled()
                self.assertEqual(blocked, requested and not cfg.enable_pd_role_switch)
                s, memory, context, req, attach, detach = self.storage_admin(
                    pp=cfg.pp_size,
                    mode=cfg.disaggregation_mode,
                    host_mode=cfg.hicache_host_memory_mode,
                    role_switch=cfg.enable_pd_role_switch,
                    storage=None,
                )
                req.hicache_storage_backend = cfg.hicache_storage_backend
                out = attach(s, req)
                self.assertEqual(out.success, not blocked)
                self.assertEqual(
                    s.tree_cache.attach_storage_backend.call_count, int(not blocked)
                )
                self.assertEqual(context.override.call_count, int(not blocked))
                if blocked:
                    self.assertIsNone(memory.hicache_storage_backend)
                    self.assertFalse(s.enable_hicache_storage)
                else:
                    self.assertEqual(
                        memory.hicache_storage_backend, cfg.hicache_storage_backend
                    )
                    self.assertTrue(detach(s, req).success)
                    s.tree_cache.detach_storage_backend.assert_called_once_with()
                    self.assertIsNone(memory.hicache_storage_backend)
                    self.assertFalse(s.enable_hicache_storage)

    def test_storage_attach_defers_the_decision_to_the_shared_gate(self):
        # If the common predicate acquires another condition, the handler must
        # honor its result, not independently re-evaluate an obsolete profile.
        for blocked in (False, True):
            decision = Mock(return_value=blocked)
            with patch.object(config, "pp_cache_admission_enabled", decision):
                s, _, _, req, attach, _ = self.storage_admin(storage=None)
                req.hicache_storage_backend = "file"
                self.assertEqual(attach(s, req).success, not blocked)
                decision.assert_called_once_with(storage_backend="file")

    def test_storage_admin_preserves_native_busy_and_disabled_rejections(self):
        for active in (False, True):
            s, _, context, req, attach, detach = self.storage_admin(
                active=active, hicache=False
            )
            self.assertFalse(attach(s, req).success)
            self.assertFalse(detach(s, req).success)
            s.tree_cache.attach_storage_backend.assert_not_called()
            s.tree_cache.detach_storage_backend.assert_not_called()
            context.override.assert_not_called()

        with environ.envs.SGLANG_ENABLE_PP_L3_CACHE_ASYNC_ADMISSION.override(False):
            s, _, context, req, attach, detach = self.storage_admin()
            s.is_fully_idle.return_value = False
            self.assertIn("not idle", attach(s, req).message)
            self.assertIn("not idle", detach(s, req).message)
            s.tree_cache.attach_storage_backend.assert_not_called()
            s.tree_cache.detach_storage_backend.assert_not_called()
            context.override.assert_not_called()

    def test_native_role_switching_engine_never_attaches_into_admission(self):
        with environ.envs.SGLANG_ENABLE_PP_L3_CACHE_ASYNC_ADMISSION.override(True):
            s, memory, _, req, attach, detach = self.storage_admin(
                mode="prefill", role_switch=True, storage=None
            )
            self.assertTrue(attach(s, req).success)
            self.assertIsNone(s.pp_prefetch_admission)
            self.assertTrue(detach(s, req).success)
            self.assertIsNone(memory.hicache_storage_backend)

    def test_rejection_calls_real_release_signature_without_checkpointing(self):
        from sglang.test.pp_cache_test_support import m

        overallocated = Mock()
        release = method(
            "mem_cache/common.py",
            None,
            "release_kv_cache",
            {"_release_overallocated_kv_indices": overallocated},
        )
        release_checked = create_autospec(release, side_effect=release)
        kv = NS(
            holds_kv=True,
            holds_mamba=False,
            is_kv_released=False,
            cache_protected_len=64,
            kv_allocated_len=128,
        )

        def mark_released():
            kv.holds_kv = False
            kv.is_kv_released = True

        kv.mark_kv_released = mark_released
        lock = object()
        req = NS(
            rid="r",
            kv=kv,
            lock=lock,
            owned_kv_len=lambda: 128,
            pending_bootstrap=True,
            disagg_kv_sender=None,
            refresh_fill_ids=Mock(
                side_effect=AssertionError("checkpointed rejected KV")
            ),
        )
        cache = NS(
            claim_kv_row=Mock(return_value=False),
            free_kv_row=Mock(),
            unlock=Mock(),
            on_release=Mock(),
            req_to_token_pool=NS(free=Mock()),
            checkpoint=Mock(side_effect=AssertionError("checkpointed rejected KV")),
        )
        s = NS(
            tree_cache=cache,
            clear_pending_chunk_send=Mock(),
            req_to_metadata_buffer_idx_allocator=object(),
            retire_unadmitted_request=Mock(),
        )
        prepare = Mock()
        bind_prefill_release(
            s, release_kv_cache=release_checked, release_metadata=Mock()
        )
        with patch.dict(
            sys.modules,
            {
                "sglang.srt.disaggregation.utils": NS(prepare_abort=prepare),
            },
        ):
            req.sampling_params = NS(json_schema={})
            m.PPAdmissionGate.reject_unsupported(NS(s=s), req)
        release_checked.assert_called_once_with(req, cache, checkpoint=False)
        cache.checkpoint.assert_not_called()
        req.refresh_fill_ids.assert_not_called()
        cache.free_kv_row.assert_called_once_with(kv, [(64, 128)])
        cache.unlock.assert_called_once_with(lock)
        cache.on_release.assert_called_once_with(req, checkpointed=False)
        cache.req_to_token_pool.free.assert_called_once_with(req)
        self.assertFalse(kv.holds_kv)
        self.assertTrue(kv.is_kv_released)
        self.assertIsNone(req.lock)
        self.assertFalse(req.pending_bootstrap)
        s.retire_unadmitted_request.assert_called_once_with(req)
        self.assertEqual(prepare.call_args.kwargs["status_code"], 400)

    def test_server_args_rejects_components_before_hicache_setup(self):
        cfg = NS(
            pp_size=4,
            disaggregation_mode="prefill",
            enable_hierarchical_cache=True,
            hicache_storage_backend="mooncake",
            hicache_host_memory_mode="cache",
            enable_linker_mla_dedup=False,
            enable_unified_cache_external_linker=False,
            disaggregation_decode_host_receive_threshold=1.0,
        )
        model_lookup = Mock()
        validate = model_validator(model_lookup)
        layout = Mock()
        handler = method(
            "arg_groups/hicache_hook.py",
            None,
            "handle_hicache",
            {
                "resolving_view": lambda cfg: cfg,
                "validate_hicache_host_memory_mode": Mock(),
                "validate_pp_cache_admission_model": validate,
                "resolve_layout_io_compatibility": layout,
                "resolve_storage_layout_compatibility": Mock(),
                "resolve_hicache_dcp_compatibility": Mock(),
                "resolve_unified_memory_layout_io": Mock(),
            },
        )
        with (
            environ.envs.SGLANG_ENABLE_PP_L3_CACHE_ASYNC_ADMISSION.override(True),
            patch.dict(
                sys.modules,
                {
                    "sglang.srt.runtime_context": None,  # Not published in ServerArgs.
                    "sglang.srt.configs.hybrid_arch": NS(
                        mambaish_config=lambda model: object() if model.mamba else None
                    ),
                },
            ),
        ):
            for name, swa, mamba in (
                ("FULL", False, False),
                ("DSA", False, False),
                ("SWA", True, False),
                ("Mamba", False, True),
                ("mixed", True, True),
            ):
                with self.subTest(model=name):
                    layout.reset_mock()
                    model_lookup.reset_mock()
                    model_lookup.return_value = NS(is_hybrid_swa=swa, mamba=mamba)
                    if swa or mamba:
                        with self.assertRaisesRegex(ValueError, "SWA/Mamba"):
                            handler(cfg)
                        layout.assert_not_called()
                    else:
                        handler(cfg)
                        layout.assert_called_once_with(cfg)
                    model_lookup.assert_called_once_with(cfg)

    def test_prefetch_groups_exclude_pp_only_for_selected_async_path(self):
        ranks = {"tp": [0, 1], "cp": [0, 2], "pp": [0, 4, 8, 12]}
        distributed = NS(
            get_world_size=lambda *, group: len(ranks[group]),
            get_process_group_ranks=lambda group: ranks[group],
        )
        create = Mock(side_effect=lambda **kwargs: tuple(kwargs["group_ranks"]))
        fn = method(
            "managers/cache_controller.py",
            "HiCacheController",
            "_create_sync_groups",
            {
                "torch": NS(distributed=distributed),
                "pp_cache_admission_enabled": config.pp_cache_admission_enabled,
            },
        )
        with patch.dict(
            "sys.modules",
            {
                "sglang.srt.distributed.parallel_state": NS(
                    create_custom_parallel_group=create
                )
            },
        ):
            for enabled, expected in (
                (False, [(0, 2), (0, 1), (0, 4, 8, 12)]),
                (True, [(0, 2), (0, 1)]),
            ):
                cache = NS(
                    attn_cp_group="cp",
                    attn_tp_group="tp",
                    tp_group="tp",
                    pp_group="pp",
                )
                with (
                    admission_context(),
                    environ.envs.SGLANG_ENABLE_PP_L3_CACHE_ASYNC_ADMISSION.override(
                        enabled
                    ),
                ):
                    self.assertEqual(fn(cache), expected)

    def test_async_cache_drain_has_no_pp_progress_dependency(self):
        fn = method(
            "mem_cache/unified_radix_cache.py",
            "UnifiedRadixCache",
            "_all_reduce",
            {"pp_cache_admission_enabled": config.pp_cache_admission_enabled},
        )
        cache = NS(
            pp_rank=2,
            _all_reduce_attn_groups=Mock(),
            _pp_sync=Mock(side_effect=AssertionError("cross-stage wait")),
        )
        payload, op = object(), object()
        with (
            admission_context(),
            environ.envs.SGLANG_ENABLE_PP_L3_CACHE_ASYNC_ADMISSION.override(True),
        ):
            fn(cache, payload, op)
        cache._all_reduce_attn_groups.assert_called_once_with(payload, op)
        cache._pp_sync.assert_not_called()

    def test_rejection_uses_native_beam_session_and_attempt_retirement(self):
        from sglang.test.pp_cache_test_support import m

        events = []
        req = NS(
            rid="r",
            cache_request_handle=object(),
            return_logprob=False,
            session=NS(streaming=True, abort_req=lambda: events.append("session")),
            kv=NS(holds_kv=False, holds_mamba=False),
            pending_bootstrap=True,
            disagg_kv_sender=None,
            time_stats=NS(trace_ctx=NS(abort=lambda **kw: events.append("trace"))),
            update_finish_state=lambda: events.append("finish"),
            release_mm_inputs_on_abort=lambda: events.append("mm"),
        )
        native_retire = method(
            "managers/scheduler.py", "Scheduler", "retire_unadmitted_request", {}
        )
        s = NS(
            clear_pending_chunk_send=lambda r: events.append("chunk"),
            req_to_metadata_buffer_idx_allocator=object(),
            beam_coordinator=NS(retire_group=lambda r: events.append("beam")),
            _release_aborted_request=lambda r: events.append(
                ("cache", r.cache_request_handle)
            ),
            output_streamer=NS(stream_output=lambda rs, lp: events.append("output")),
        )
        s.retire_unadmitted_request = lambda r: native_retire(s, r)

        def prepare(r, message, status_code):
            r.finished_reason = NS(message=message, status_code=status_code)

        bind_prefill_release(
            s,
            release_kv_cache=lambda *a, **kw: self.fail("no KV owned"),
            release_metadata=lambda *a: events.append("metadata"),
        )
        with patch.dict(
            "sys.modules",
            {
                "sglang.srt.disaggregation.utils": NS(prepare_abort=prepare),
            },
        ):
            req.sampling_params = NS(json_schema={})
            m.PPAdmissionGate.reject_unsupported(NS(s=s), req)
        self.assertEqual(
            events,
            [
                "chunk",
                "metadata",
                "mm",
                "session",
                "beam",
                ("cache", req.cache_request_handle),
                "trace",
                "finish",
                "output",
            ],
        )
        self.assertIsNone(req.session)
        self.assertEqual(req.finished_reason.status_code, 400)

    def test_opt_in_lookup_limits_host_and_anchor_not_only_device(self):
        r, c = request(), Cache()
        init_input(r, c, max_cache_prefix_tokens=512)
        self.assertEqual((r.prefix_len, r.host_hit_length), (128, 384))
        self.assertEqual(r.best_match_node, ("h", 512))
        self.assertEqual(c.calls[-1].key.cache_salt, "tenant")
        self.assertIs(c.calls[-1].key.token_ids, r.full_untruncated_fill_ids)

    def test_none_zero_and_later_chunk_are_distinct(self):
        r, c = request(), Cache()
        for kwargs in (
            {},
            {"max_cache_prefix_tokens": 0},
            {"max_cache_prefix_tokens": 512},
            {},
        ):
            init_input(r, c, **kwargs)
        self.assertEqual([p.key.limit for p in c.calls], [4095, 0, 512, 4095])

    def test_native_logprob_and_swa_limits_win(self):
        r, c = request(natural=64), Cache()
        init_input(r, c, max_cache_prefix_tokens=1024)
        self.assertEqual(c.calls[-1].key.limit, 64)
        r = request()
        c.reprefill_tail = 3840
        init_input(r, c, max_cache_prefix_tokens=1024)
        self.assertEqual(c.calls[-1].key.limit, 256)

    def test_embedding_override_disables_match(self):
        r, c = request(), Cache()
        r.positional_embed_overrides = object()
        init_input(r, c, max_cache_prefix_tokens=512)
        self.assertIsNone(c.calls[-1].key.limit)
        self.assertEqual(len(c.calls[-1].key.token_ids), 0)

    def test_device_only_lease_cannot_release_later_backup_reference(self):
        r, c = request(), Cache(device=1024)
        hold = lease.CachePrefixLease.acquire(c, r, 1024)
        self.assertIsNone(hold.host_node)
        backup = c.inc_host_lock_ref(r.best_match_node).to_dec_params()
        hold.release()
        hold.release()
        self.assertEqual(c.locks, [backup])

    def test_shorter_lease_preserves_exact_receipts(self):
        r, c = request(), Cache()
        hold = lease.CachePrefixLease.acquire(c, r, 1024)
        short = hold.restrict(512)
        self.assertTrue(hold.released)
        self.assertEqual(short.cap, 512)
        self.assertEqual(len(c.locks), 2)
        short.release()
        self.assertEqual(c.locks, [])

    def test_host_lock_error_unwinds_device_receipt(self):
        r, c = request(), Cache()
        with patch.object(c, "inc_host_lock_ref", side_effect=RuntimeError("host")):
            with self.assertRaisesRegex(RuntimeError, "host"):
                lease.CachePrefixLease.acquire(c, r, 1024)
        self.assertEqual(c.locks, [])

    def test_storage_attribution_uses_native_absolute_span_and_common_cap(self):
        from sglang.test.pp_cache_test_support import m
        from sglang.test.pp_cache_test_support import request as runtime_request

        split = method(
            "managers/schedule_batch.py", None, "split_cached_prefix_by_tier", {}
        )
        fulfilled = method(
            "managers/schedule_batch.py", "Req", "fulfilled_storage_hit_len", {}
        )
        req = runtime_request("r")
        cache = NS(caps={"r": 1024})
        entry = NS(req=req, cap=768, storage_span=(256, 1024))
        selection = m.PrefillAdmissionSelection({"r": entry})
        selection.prepare_request(req, cache)
        self.assertEqual((req.storage_hit_start, req.storage_hit_length), (256, 768))
        self.assertEqual(fulfilled(req, 768), 512)
        self.assertEqual(
            split(768, 256, req.storage_hit_length, req.storage_hit_start),
            (256, 0, 512),
        )
        self.assertEqual(
            split(128, 0, req.storage_hit_length, req.storage_hit_start), (128, 0, 0)
        )

    def test_runtime_gate_keeps_role_switching_engines_outside_admission(self):
        with environ.envs.SGLANG_ENABLE_PP_L3_CACHE_ASYNC_ADMISSION.override(True):
            # The early hook rejects static Prefill conflicts. A native engine
            # started in Decode must also stay native after a role transition.
            for mode in ("decode", "prefill", "decode"):
                with admission_context(mode=mode, role_switch=True):
                    self.assertFalse(config.pp_cache_admission_enabled())
            with admission_context(pp=1, role_switch=True):
                self.assertFalse(config.pp_cache_admission_enabled())
        with (
            environ.envs.SGLANG_ENABLE_PP_L3_CACHE_ASYNC_ADMISSION.override(False),
            admission_context(role_switch=True),
        ):
            self.assertFalse(config.pp_cache_admission_enabled())

    def test_dsa_boundary_reads_the_same_environment_profile(self):
        model = NS(num_hidden_layers=8)
        skips_topk = Mock(side_effect=lambda config, layer: layer == 2)
        with (
            patch.dict(
                sys.modules,
                {
                    "sglang.srt.configs.model_config": NS(
                        dsa_layer_skips_topk=skips_topk
                    ),
                    "sglang.srt.distributed.utils": NS(
                        get_pp_indices=lambda n, rank, pp: (
                            rank * (n // pp),
                            (rank + 1) * (n // pp),
                        )
                    ),
                },
            ),
            admission_context(),
        ):
            with environ.envs.SGLANG_ENABLE_PP_L3_CACHE_ASYNC_ADMISSION.override(False):
                self.assertFalse(
                    config.requires_logical_dsa_pp_topk(model_config=model)
                )
                skips_topk.assert_not_called()
            with environ.envs.SGLANG_ENABLE_PP_L3_CACHE_ASYNC_ADMISSION.override(True):
                self.assertTrue(config.requires_logical_dsa_pp_topk(model_config=model))
            with (
                admission_context(mode="decode"),
                environ.envs.SGLANG_ENABLE_PP_L3_CACHE_ASYNC_ADMISSION.override(True),
            ):
                skips_topk.reset_mock()
                self.assertFalse(
                    config.requires_logical_dsa_pp_topk(model_config=model)
                )
                skips_topk.assert_not_called()


if __name__ == "__main__":
    unittest.main()
