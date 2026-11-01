"""PD peers carry each window sub-pool of a unified pool as an entry of its own.

A unified pool may hold several windows. Each window's pages travel as their
own state entry, registered with the window's sub-pool; the prefill sends and
the decode receives that sub-pool's pages for it, and a transfer's rows are
recorded against it. Two peers pair entries by position, so peers whose
entries differ are refused before any transfer.
"""

import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import numpy as np
import torch

from sglang.srt.disaggregation.base.conn import KVArgs, StateType
from sglang.srt.disaggregation.common.conn import (
    CommonKVBootstrapServer,
    CommonKVManager,
    PrefillServerInfo,
)
from sglang.srt.disaggregation.utils import (
    check_state_wire,
    record_network_transfer,
    setup_state_kv_args,
    state_indices_per_entry,
    state_wire,
)
from sglang.srt.mem_cache.kv.build import UnifiedBuildContext, build_kv_storage
from sglang.srt.mem_cache.kv.decl import MHAGeometry, WindowLayers, kv_layer_decls
from sglang.srt.mem_cache.kv.plan import plan_kv_storage
from sglang.srt.mem_cache.kv.router import KV_PART
from sglang.srt.mem_cache.kv.sizing import KVSizingInputs, size_plan
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.unified_allocator_fixtures import setup_allocator_context

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_MHA = MHAGeometry(head_num=1, head_dim=8, v_head_dim=8, kv_cache_dtype=torch.bfloat16)
_WINDOWS = (16, 8, 4)


def _storage(windows):
    setup_allocator_context()
    decls = kv_layer_decls(
        full_layer_ids=(0,),
        full_geometry=_MHA,
        windows=[
            WindowLayers(layer_ids=(i + 1,), geometry=_MHA, window=w)
            for i, w in enumerate(windows)
        ],
    )
    plan = plan_kv_storage(decls, page_size=4)
    sized = size_plan(
        plan,
        KVSizingInputs(
            full_tokens=128,
            window_tokens=64,
            state_slots=0,
            max_running_requests=3,
            context_len=64,
            alignment=1,
        ),
    )
    ctx = UnifiedBuildContext(
        device="cpu",
        start_layer=0,
        end_layer=len(windows) + 1,
        enable_memory_saver=False,
        need_sort=False,
        max_num_reqs=3,
        model_context_len=64,
        lazy_compaction=True,
    )
    return build_kv_storage(plan, sized, ctx)


def _registered(windows):
    storage = _storage(windows)
    kv_args = KVArgs()
    setup_state_kv_args(kv_args, storage)
    return storage, kv_args


def _window_names(windows):
    return {f"window_{w}" for w in windows}


class TestUnifiedWindowWire(CustomTestCase):
    def test_each_window_registers_an_entry_of_its_own(self):
        storage, kv_args = _registered(_WINDOWS)
        self.assertEqual(kv_args.state_types, [StateType.SWA] * len(_WINDOWS))
        self.assertEqual(set(kv_args.state_sub_pools), _window_names(_WINDOWS))
        self.assertEqual(
            kv_args.state_sub_pools, [e.sub_pool for e in storage.wire_entries()]
        )
        for sub_pool, ptrs, item_lens in zip(
            kv_args.state_sub_pools, kv_args.state_data_ptrs, kv_args.state_item_lens
        ):
            region = storage.regions(sub_pool, part=KV_PART)
            self.assertEqual(ptrs, list(region.ptrs))
            self.assertEqual(item_lens, list(region.item_lens))

    def test_each_window_entry_carries_its_own_pages(self):
        _, kv_args = _registered(_WINDOWS)
        swa = Mock(return_value="type payload")
        indices = state_indices_per_entry(
            kv_args,
            kv_args.state_types,
            {StateType.SWA: swa},
            lambda sub_pool: f"pages of {sub_pool}",
        )
        self.assertEqual(
            indices, [f"pages of {sub_pool}" for sub_pool in kv_args.state_sub_pools]
        )
        swa.assert_not_called()

        # A static pool's entries name no sub-pool: its type's payload.
        static = SimpleNamespace(
            state_types=[StateType.SWA, StateType.MAMBA], state_sub_pools=[None, None]
        )
        self.assertEqual(
            state_indices_per_entry(
                static, static.state_types, {StateType.SWA: swa}, lambda _: "unused"
            ),
            ["type payload", None],
        )

    def test_a_transfer_records_each_window_on_its_sub_pool(self):
        storage, kv_args = _registered(_WINDOWS)
        req = object()
        ids = [np.array([i + 1]) for i in range(len(kv_args.state_types))]
        with patch.object(storage, "record_network_transfer") as record:
            record_network_transfer(
                SimpleNamespace(storage=storage),
                req,
                kv_manager=None,
                writes=True,
                page_indices=np.array([0]),
                page_size=4,
                state_types=kv_args.state_types,
                state_indices=ids,
            )
        self.assertEqual(
            record.call_args_list[1:],
            [
                call(
                    req,
                    sub_pool,
                    part,
                    unit_tokens=storage.unit_tokens(sub_pool),
                    writes=True,
                )
                for sub_pool, part in zip(kv_args.state_sub_pools, ids)
            ],
        )

    def test_peers_with_other_entries_are_refused(self):
        _, prefill = _registered(_WINDOWS)
        # Peers list the same entries whatever order the model declares its
        # windows in.
        _, decode = _registered(_WINDOWS[::-1])
        self.assertEqual(state_wire(prefill), state_wire(decode))
        check_state_wire(state_wire(prefill), state_wire(decode))
        # A peer that does not report its entries is not checked.
        check_state_wire(None, state_wire(decode))
        _, one_window = _registered(_WINDOWS[:1])
        with self.assertRaisesRegex(RuntimeError, "PD state entry mismatch"):
            check_state_wire(state_wire(prefill), state_wire(one_window))


def _wire(sub_pools):
    return [["swa", name] for name in sub_pools]


class _Request:
    def __init__(self, data=None, query=None):
        self._data = data
        self.query = query or {}

    async def json(self):
        return self._data


def _register(server, *, pp_rank, pp_size, wire, tp_rank=0, tp_size=1):
    return asyncio.run(
        server._handle_route_put(
            _Request(
                {
                    "attn_tp_size": tp_size,
                    "attn_tp_rank": tp_rank,
                    "attn_cp_size": 1,
                    "attn_cp_rank": 0,
                    "attn_dp_size": 1,
                    "attn_dp_rank": 0,
                    "pp_size": pp_size,
                    "pp_rank": pp_rank,
                    "system_dp_size": 1,
                    "system_dp_rank": 0,
                    "rank_ip": "127.0.0.1",
                    "rank_port": 1000 + pp_rank * tp_size + tp_rank,
                    "page_size": 4,
                    "kv_cache_dtype": "auto",
                    "state_wire": wire,
                }
            )
        )
    )


def _served_info(server):
    response = asyncio.run(
        server._handle_route_get(
            _Request(
                query={
                    "prefill_dp_rank": "-1",
                    "prefill_cp_rank": "-1",
                    "target_tp_rank": "-1",
                    "target_pp_rank": "-1",
                }
            )
        )
    )
    return PrefillServerInfo(**json.loads(response.text))


class TestStateEntryAgreement(CustomTestCase):
    def test_bootstrap_compares_each_pipeline_stage(self):
        stage_wires = [_wire(["window_16", "window_8"]), _wire(["window_16"])]
        server = CommonKVBootstrapServer("127.0.0.1", 0)
        for pp_rank, wire in enumerate(stage_wires):
            self.assertEqual(
                _register(server, pp_rank=pp_rank, pp_size=2, wire=wire).status, 200
            )
        info = _served_info(server)
        self.assertIsNone(info.state_wire)
        self.assertEqual(info.state_wires, stage_wires)

        def decode(pp_rank, wire, pp_size=2):
            return SimpleNamespace(
                pp_size=pp_size,
                pp_rank=pp_rank,
                kv_args=SimpleNamespace(
                    state_types=[StateType.SWA] * len(wire),
                    state_sub_pools=[entry[1] for entry in wire],
                ),
            )

        for pp_rank, wire in enumerate(stage_wires):
            CommonKVManager._check_prefill_state_entries(decode(pp_rank, wire), info)
        with self.assertRaisesRegex(RuntimeError, "PD state entry mismatch"):
            CommonKVManager._check_prefill_state_entries(
                decode(1, stage_wires[0]), info
            )

    def test_ranks_of_one_stage_must_agree(self):
        server = CommonKVBootstrapServer("127.0.0.1", 0)
        self.assertEqual(
            _register(
                server, pp_rank=0, pp_size=1, wire=_wire(["window_16"]), tp_size=2
            ).status,
            200,
        )
        response = _register(
            server,
            pp_rank=0,
            pp_size=1,
            wire=_wire(["window_8"]),
            tp_rank=1,
            tp_size=2,
        )
        self.assertEqual(response.status, 400)
        self.assertIn("one pipeline stage", response.text)


if __name__ == "__main__":
    unittest.main()
