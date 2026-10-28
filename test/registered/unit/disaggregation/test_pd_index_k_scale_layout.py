"""PD peers and HiCache storage must agree on the index-K scale bytes: FP32 or
DeepGEMM's packed UE8M0."""

import asyncio
import importlib
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.disaggregation.common.conn import (
    CommonKVBootstrapServer,
    CommonKVManager,
)
from sglang.srt.disaggregation.utils import get_pd_kv_layout, pd_kv_layouts_match
from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsa.utils import dsa_index_k_cache_is_ue8m0
from sglang.srt.managers.cache_controller import storage_model_name
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.hybrid_cache.linker_pool_assembler import (
    DevicePoolEntry,
    DevicePoolGroup,
)
from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool
from sglang.srt.mem_cache.unified_cache.linker_factory import (
    create_unified_cache_linker,
    get_startup_linker_config,
)
from sglang.srt.runtime_context import get_context, get_memory
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_UTILS = "sglang.srt.layers.attention.dsa.utils"


def _kernel_args(backend="auto", fp4_indexer=False):
    return SimpleNamespace(
        kernel=SimpleNamespace(
            dsa_paged_mqa_logits_backend=backend,
            enable_deepseek_v4_fp4_indexer=fp4_indexer,
        )
    )


def _device(stack, *, sm100=True, backend="auto", fp4_indexer=False):
    exec_args = _kernel_args(backend, fp4_indexer)
    stack.enter_context(patch(f"{_UTILS}.is_cuda", return_value=True))
    stack.enter_context(patch(f"{_UTILS}.uses_mx_mqa_logits", return_value=sm100))
    stack.enter_context(patch(f"{_UTILS}.get_exec", return_value=exec_args))
    stack.enter_context(
        patch(
            "sglang.srt.layers.attention.dsa.paged_mqa_logits_backend.get_platform",
            return_value=SimpleNamespace(is_sm100=sm100),
        )
    )


def _dsa_pool(index_kpool=1):
    pool = DSATokenToKVPool.__new__(DSATokenToKVPool)
    pool.index_kpool = index_kpool
    return pool


class TestPdIndexKScaleLayout(CustomTestCase):
    def _dsa_layout(self, pool=None, **device):
        with ExitStack() as stack:
            _device(stack, **device)
            return get_pd_kv_layout(pool if pool is not None else _dsa_pool())

    def test_dsa_cache_format_follows_its_sm100_reader(self):
        cases = [
            (dict(backend="deepgemm"), 1, True),
            (dict(backend="cutedsl"), 1, False),
            (dict(backend="cutedsl"), 4, True),
            (dict(sm100=False), 1, False),
        ]
        for device, index_kpool, ue8m0 in cases:
            with self.subTest(device=device, index_kpool=index_kpool):
                with ExitStack() as stack:
                    _device(stack, **device)
                    self.assertEqual(
                        dsa_index_k_cache_is_ue8m0(kpool=index_kpool > 1), ue8m0
                    )
        with ExitStack() as stack:
            _device(stack)
            stack.enter_context(envs.SGLANG_FP8_PAGED_MQA_LOGITS_TORCH.override(True))
            self.assertFalse(dsa_index_k_cache_is_ue8m0(kpool=False))

    def test_dsa_peers_with_different_readers_report_different_layouts(self):
        deepgemm = self._dsa_layout(backend="deepgemm")
        self.assertEqual(deepgemm, "index_k:ue8m0")
        self.assertEqual(self._dsa_layout(backend="cutedsl"), "index_k:fp32")
        self.assertEqual(self._dsa_layout(sm100=False), "index_k:fp32")
        hybrid = SimpleNamespace(full_kv_pool=_dsa_pool())
        self.assertEqual(self._dsa_layout(hybrid, backend="deepgemm"), deepgemm)

    def test_a_peer_without_a_layout_only_matches_fp32_index_scales(self):
        self.assertTrue(pd_kv_layouts_match(None, None))
        self.assertTrue(pd_kv_layouts_match(None, "index_k:fp32"))
        self.assertTrue(pd_kv_layouts_match("v4", None))
        self.assertFalse(pd_kv_layouts_match(None, "index_k:ue8m0"))
        self.assertFalse(pd_kv_layouts_match("v4,index_k:ue8m0", None))
        self.assertFalse(pd_kv_layouts_match("index_k:fp32", "index_k:ue8m0"))
        self.assertTrue(pd_kv_layouts_match("index_k:ue8m0", "index_k:ue8m0"))

    def test_storage_keys_name_a_ue8m0_index_cache(self):
        host_pool = SimpleNamespace(storage_format_tag=None)
        for device, want in [
            (dict(backend="deepgemm"), "org/model-index_k-ue8m0"),
            (dict(backend="cutedsl"), "org/model"),
            (dict(sm100=False), "org/model"),
        ]:
            with self.subTest(device=device), ExitStack() as stack:
                _device(stack, **device)
                self.assertEqual(
                    storage_model_name("org/model", host_pool, _dsa_pool()), want
                )
        self.assertEqual(storage_model_name("org/model", host_pool), "org/model")

    def test_dsv4_layout_names_a_ue8m0_fp8_index_cache(self):
        pool = DeepSeekV4TokenToKVPool.__new__(DeepSeekV4TokenToKVPool)
        cases = [
            (True, False, "v4,index_k:ue8m0"),
            (False, False, "v4"),
            (True, True, "v4"),
        ]
        for ue8m0, fp4_indexer, want in cases:
            with (
                self.subTest(ue8m0=ue8m0, fp4_indexer=fp4_indexer),
                ExitStack() as stack,
            ):
                _device(stack, fp4_indexer=fp4_indexer)
                stack.enter_context(
                    patch.object(
                        DeepSeekV4TokenToKVPool, "pd_kv_layout", return_value="v4"
                    )
                )
                stack.enter_context(
                    patch(
                        f"{_UTILS}.dsv4_fp8_index_cache_is_ue8m0",
                        return_value=ue8m0,
                    )
                )
                self.assertEqual(get_pd_kv_layout(pool), want)


class TestDirectLinkerStorageKeys(CustomTestCase):
    def test_direct_linkers_name_a_ue8m0_index_cache(self):
        group = DevicePoolGroup(
            [
                DevicePoolEntry(
                    name=PoolName.KV,
                    indices_from_pool=PoolName.KV,
                    device_pool=None,
                    components=[[torch.zeros((2, 8), dtype=torch.uint8)]],
                    layer_mapping={0: 0},
                    page_size=1,
                    rows_are_pages=False,
                )
            ],
            num_layers=1,
            page_size=1,
        )
        params = CacheInitParams(
            disable=False,
            req_to_token_pool=None,
            token_to_kv_pool_allocator=Mock(),
            page_size=1,
        )
        for backend, module_path in (
            ("mooncake", "mooncake_store.mooncake_direct_linker"),
            ("mori", "umbp.umbp_direct_linker"),
        ):
            module = importlib.import_module(
                f"sglang.srt.mem_cache.storage.{module_path}"
            )
            for ue8m0 in (False, True):
                with (
                    self.subTest(backend=backend, ue8m0=ue8m0),
                    get_context().override_server_args(
                        unified_cache_external_linker_backend=backend,
                        unified_cache_external_linker_config={},
                    ),
                    patch(f"{_UTILS}.index_k_cache_is_ue8m0", return_value=ue8m0),
                    patch.object(
                        module, "resolve_hybrid_device_pool_group", return_value=group
                    ),
                    patch.object(
                        module.torch.distributed, "is_initialized", return_value=False
                    ),
                    patch.object(
                        module,
                        "HiCacheStorageConfig",
                        side_effect=RuntimeError("storage boundary"),
                    ) as storage_config,
                    self.assertRaisesRegex(RuntimeError, "storage boundary"),
                ):
                    create_unified_cache_linker(
                        "args",
                        params,
                        backend=get_memory().unified_cache_external_linker_backend,
                        config=get_startup_linker_config(),
                        components={"full"},
                    )
                model_name = storage_config.call_args.kwargs["model_name"]
                self.assertEqual(model_name.endswith("-index_k-ue8m0"), ue8m0)


def _bootstrap_server(kv_layout):
    server = CommonKVBootstrapServer.__new__(CommonKVBootstrapServer)
    server.__dict__.update(
        attn_tp_size=1,
        attn_cp_size=1,
        dp_size=1,
        pp_size=1,
        page_size=64,
        kv_cache_dtype="auto",
        kv_layout=kv_layout,
        dsv41_spec_layout=None,
        follow_bootstrap_room=True,
        enable_dsa_cache_layer_split=False,
        enable_kv_cache_sharding=False,
        prefill_http_port=30000,
        decode_allocation_policy="early",
        _registered_count=1,
    )
    return server


def _route_status(server, **query):
    query = dict(
        prefill_dp_rank="-1",
        prefill_cp_rank="-1",
        target_tp_rank="-1",
        target_pp_rank="-1",
        **query,
    )
    with patch.object(CommonKVBootstrapServer, "_is_ready", return_value=True):
        response = asyncio.run(server._handle_route_get(SimpleNamespace(query=query)))
    return response.status


class TestMixedVersionPd(CustomTestCase):
    def test_ue8m0_prefill_refuses_a_decoder_that_reports_no_layout(self):
        server = _bootstrap_server("index_k:ue8m0")
        self.assertEqual(_route_status(server), 400)
        self.assertEqual(_route_status(server, decode_kv_layout="index_k:ue8m0"), 200)

    def test_fp32_prefill_serves_a_decoder_that_reports_no_layout(self):
        for kv_layout in ("index_k:fp32", None):
            with self.subTest(kv_layout=kv_layout):
                self.assertEqual(_route_status(_bootstrap_server(kv_layout)), 200)

    def test_decoder_reports_its_layout_to_prefill(self):
        for kv_layout, sent in (("index_k:ue8m0", "index_k:ue8m0"), (None, "")):
            with self.subTest(kv_layout=kv_layout):
                manager = CommonKVManager.__new__(CommonKVManager)
                manager.prefill_info_table = {}
                manager.kv_layout = kv_layout
                failed = SimpleNamespace(status_code=500, text="stop")
                with patch(
                    "sglang.srt.disaggregation.common.conn.requests.get",
                    return_value=failed,
                ) as get:
                    self.assertFalse(manager.try_ensure_parallel_info("host:8998"))
                self.assertEqual(
                    get.call_args.kwargs["params"], {"decode_kv_layout": sent}
                )


if __name__ == "__main__":
    unittest.main()
