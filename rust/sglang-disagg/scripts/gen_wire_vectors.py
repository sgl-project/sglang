#!/usr/bin/env python3
"""Generate golden wire vectors from the REAL SGLang wire implementation.

Loads `python/sglang/srt/disaggregation/{base,common,nixl}/*.py` from this
checkout with heavy dependencies (torch, zmq, msgspec, aiohttp) stubbed out,
runs the actual Python codecs, and writes
`rust/sglang-disagg/tests/vectors/wire.json` for the Rust bit-identity test.

Run from anywhere:  python3 rust/sglang-disagg/scripts/gen_wire_vectors.py
Regenerate whenever the Python wire format changes.
"""

import dataclasses
import importlib.util
import json
import struct
import sys
import types
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
DISAGG = ROOT / "python" / "sglang" / "srt" / "disaggregation"
OUT = ROOT / "rust" / "sglang-disagg" / "tests" / "vectors" / "wire.json"

# --------------------------------------------------------------------------- //
# Dependency stubs (heavy/unavailable modules the real files import)
# --------------------------------------------------------------------------- //


def stub(name, **attrs):
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[name] = m
    return m


class _MsgspecStruct:
    def __init_subclass__(cls, **kw):
        super().__init_subclass__()
        dataclasses.dataclass(cls)


stub("msgspec", Struct=_MsgspecStruct)
stub("requests")


class _Any:
    """Permissive dummy: callable, subscriptable, attribute-returning."""

    def __call__(self, *a, **k):
        return _Any()

    def __getattr__(self, k):
        return _Any()

    def __getitem__(self, k):
        return _Any()

    def __init_subclass__(cls, **kw):
        super().__init_subclass__()


def _permissive(name):
    m = stub(name)
    m.__getattr__ = lambda k: _Any  # noqa: B023 (module-level PEP 562)
    return m


_permissive("zmq")
_permissive("aiohttp").web = _Any
_permissive("torch")
_permissive("torch.distributed")

# --------------------------------------------------------------------------- //
# sglang package skeleton (stub everything not under test)
# --------------------------------------------------------------------------- //

for pkg in [
    "sglang",
    "sglang.srt",
    "sglang.srt.disaggregation",
    "sglang.srt.disaggregation.base",
    "sglang.srt.disaggregation.common",
    "sglang.srt.disaggregation.nixl",
]:
    stub(pkg)


class _EnvVal:
    value = 1024

    def get(self, default=None):
        return self.value


class _Envs:
    def __getattr__(self, k):
        return _EnvVal()


stub("sglang.srt.environ", envs=_Envs())
stub(
    "sglang.srt.runtime_context",
    get_device=lambda *a, **k: None,
    get_schedule=lambda *a, **k: None,
    get_disagg=lambda *a, **k: None,
    get_observability=lambda *a, **k: None,
    get_parallel=lambda *a, **k: None,
    get_serving=lambda *a, **k: None,
    linear_attn_tp_size=lambda *a, **k: 1,
    max_prefill_buffer_tokens=lambda *a, **k: 0,
)
stub("sglang.srt.server_args", ServerArgs=type("ServerArgs", (), {}))
stub(
    "sglang.srt.utils.network",
    NetworkAddress=type("NetworkAddress", (), {}),
    get_local_ip_auto=lambda *a, **k: "127.0.0.1",
    get_zmq_socket_on_host=lambda *a, **k: None,
)
stub(
    "sglang.srt.utils",
    create_device_event=lambda *a, **k: None,
    create_device_stream=lambda *a, **k: None,
    device_stream_context=lambda *a, **k: None,
)
stub(
    "sglang.srt.utils.common",
    run_with_deadline=lambda f, *a, **k: f(),
    ceil_align=lambda x, a: -(-x // a) * a,
)
stub("sglang.srt.observability")
stub(
    "sglang.srt.observability.trace",
    TraceNullContext=type("TraceNullContext", (), {}),
    TraceReqContext=type("TraceReqContext", (), {}),
)
stub(
    "sglang.srt.disaggregation.pp_consensus_store",
    PPConsensusStore=type("PPConsensusStore", (), {}),
)


class _DisaggregationMode:
    PREFILL = "prefill"
    DECODE = "decode"
    NULL = "null"


stub(
    "sglang.srt.disaggregation.utils",
    DisaggregationMode=_DisaggregationMode,
    build_dsa_tail_transfer_blocks=lambda *a, **k: None,
    build_transfer_entry_pairs=lambda *a, **k: None,
    compute_mamba_state_slice_byte_blocks=lambda *a, **k: None,
    is_elided_entry=lambda *a, **k: False,
    resolve_dcp_dst_entry_indices=lambda *a, **k: None,
    slice_dsa_tail_dst_ptrs_for_pp=lambda *a, **k: None,
    filter_kv_indices_for_cp_rank=lambda *a, **k: None,
    filter_kv_indices_for_shard_rank=lambda *a, **k: None,
    get_dsv41_spec_layout=lambda *a, **k: None,
)
stub("sglang.srt.mem_cache")
stub("sglang.srt.mem_cache.page_interleave", get_kv_shard_group=lambda *a, **k: None)

# --------------------------------------------------------------------------- //
# Real upstream modules under test
# --------------------------------------------------------------------------- //


def load_real(dotted, rel):
    spec = importlib.util.spec_from_file_location(dotted, DISAGG / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[dotted] = mod
    spec.loader.exec_module(mod)
    return mod


base_conn = load_real("sglang.srt.disaggregation.base.conn", "base/conn.py")
load_real("sglang.srt.disaggregation.common.bootstrap", "common/bootstrap.py")
common_utils = load_real("sglang.srt.disaggregation.common.utils", "common/utils.py")
common_conn = load_real("sglang.srt.disaggregation.common.conn", "common/conn.py")
load_real(
    "sglang.srt.disaggregation.common.staging_handler", "common/staging_handler.py"
)
nixl_conn = load_real("sglang.srt.disaggregation.nixl.conn", "nixl/conn.py")

import numpy as np  # noqa: E402

KVPoll = base_conn.KVPoll

# --------------------------------------------------------------------------- //
# Vector emission
# --------------------------------------------------------------------------- //

V = {
    "pack": [],
    "pack_buffers": [],
    "kv_status": [],
    "abort": [],
    "abort_ack": [],
    "bootstrap": [],
    "chunk_ready": [],
    "registration": [],
    "metadata": [],
    "http": {},
}


def hexs(frames):
    return [f.hex() for f in frames]


def s64s(list_):
    return [str(x) for x in list_]


# --- pack helpers (common/utils.py) ---
for lists, fmt in [
    ([], "Q"),
    ([[]], "Q"),
    ([[1, 2], [], [2**64 - 1]], "Q"),
    ([[0, 2**32 - 1], [7]], "I"),
    ([[-1, 0, 2**31 - 1, -(2**31)]], "i"),
    ([[], []], "i"),
]:
    V["pack"].append(
        {
            "lists": [s64s(l) for l in lists],
            "fmt": fmt,
            "packed": common_utils.pack_int_lists(lists, fmt).hex(),
        }
    )

for buffers in [
    [],
    [b""],
    [b"hello", b"", b"\x00\xff" * 7],
]:
    V["pack_buffers"].append(
        {
            "buffers": [b.hex() for b in buffers],
            "packed": common_utils.pack_list_of_buffers(buffers).hex(),
        }
    )

# --- KV_STATUS (common/conn.py:736-776) ---
mgr_ns = SimpleNamespace()
mgr_ns._prefill_unique_rank = lambda: 17
for name, tag, carries, status, reason in [
    ("plain_boot", None, False, KVPoll.Bootstrapping, None),
    ("plain_success", None, False, KVPoll.Success, None),
    ("nixl_waiting", b"KV_STATUS", True, KVPoll.WaitingForInput, None),
    ("nixl_failed_reason", b"KV_STATUS", True, KVPoll.Failed, "peer died"),
    ("nixl_failed_unicode", b"KV_STATUS", True, KVPoll.Failed, "dead ☠ péer"),
    ("mori_failed_empty", b"MoriMsgGuard", True, KVPoll.Failed, ""),
]:
    mgr_ns.kv_status_msg_tag = tag
    mgr_ns.kv_status_msg_carries_reason = carries
    frames = common_conn.CommonKVManager._encode_kv_status_message(
        mgr_ns, bootstrap_room=4242, status=status, failure_reason=reason
    )
    parsed = common_conn.CommonKVManager.parse_kv_status_message(mgr_ns, frames)
    V["kv_status"].append(
        {
            "name": name,
            "tag": None if tag is None else tag.decode(),
            "carries_reason": carries,
            "frames": hexs(frames),
            "parsed": [parsed[0], parsed[1], parsed[2], parsed[3]],
        }
    )

# --- ABORT / ABORT_ACK (common/conn.py:182-280) ---
AbortNotification = common_conn.AbortNotification
AbortAck = common_conn.AbortAck
for name, n in [
    ("room_only", AbortNotification(room=99)),
    ("addr", AbortNotification(room=99, decode_ip="10.0.0.3", decode_port=30001)),
    (
        "addr_gen",
        AbortNotification(
            room=99, decode_ip="10.0.0.3", decode_port=30001, generation=7
        ),
    ),
    ("gen_no_addr", AbortNotification(room=99, generation=7)),
]:
    frames = n.to_zmq()
    back = AbortNotification.from_zmq(frames)
    V["abort"].append(
        {
            "name": name,
            "frames": hexs(frames),
            "parsed": {
                "room": back.room,
                "decode_ip": back.decode_ip,
                "decode_port": back.decode_port,
                "generation": back.generation,
            },
        }
    )
# Degrade-not-fail parses: bad return address -> room only.
bad_addr = [b"ABORT", b"99", b"10.0.0.3", b"not-a-port"]
V["abort"].append(
    {
        "name": "bad_addr_degrades",
        "frames": hexs(bad_addr),
        "parsed": {
            "room": AbortNotification.from_zmq(bad_addr).room,
            "decode_ip": None,
            "decode_port": None,
            "generation": None,
        },
    }
)

for name, a in [
    ("plain", AbortAck(room=99, prefill_rank=3)),
    ("gen", AbortAck(room=99, prefill_rank=3, generation=7)),
]:
    frames = a.to_zmq()
    back = AbortAck.from_zmq(frames)
    V["abort_ack"].append(
        {
            "name": name,
            "frames": hexs(frames),
            "parsed": {
                "room": back.room,
                "prefill_rank": back.prefill_rank,
                "generation": back.generation,
            },
        }
    )

# --- BOOTSTRAP (common/conn.py:2573-2577 build, :692-712 parse) ---
boot = [
    b"BOOTSTRAP",
    str(4242).encode("ascii"),
    b"192.168.1.9",
    str(30002).encode("ascii"),
]
V["bootstrap"].append(
    {
        "frames": hexs(boot),
        "parsed": {"room": 4242, "ip": "192.168.1.9", "port": 30002},
    }
)

# --- CHUNK_READY (common/conn.py:813-830 build, :832-849 parse) ---
chunk = [
    b"CHUNK_READY",
    str(4242).encode("ascii"),
    str(3).encode("ascii"),
    str(140).encode("ascii"),
    str(16).encode("ascii"),
    b"writer-0",
    str(5).encode("ascii"),
]
V["chunk_ready"].append(
    {
        "frames": hexs(chunk),
        "parsed": {
            "room": 4242,
            "chunk_idx": 3,
            "page_start": 140,
            "num_pages": 16,
            "writer_id": "writer-0",
        },
    }
)


# --- NIXL registration (nixl/conn.py:3428-3492 build, :376-467 parse) ---
def reg_ns(state_entries, staging):
    kv_args = SimpleNamespace(
        kv_data_ptrs=[0x7F00_0000_1000, 0x7F00_0001_2000],
        kv_data_lens=[1 << 20, 2 << 20],
        kv_data_mem_kinds=["VRAM", "DRAM"],
        kv_item_lens=[4096, 4096],
        kv_layer_ids=[3, 7, 11, 59],
        aux_data_ptrs=[0x7F00_0002_0000],
        state_data_ptrs=[[0x7F00_0010_0000 + 0x1000 * i] for i in range(state_entries)],
        state_item_lens=[[4096 * 4] for _ in range(state_entries)],
        state_data_lens=[[1 << 16] for _ in range(state_entries)],
        state_dim_per_tensor=[[2, 128] for _ in range(state_entries)],
        state_layer_ids=[[i] for i in range(state_entries)],
        gpu_id=5,
        engine_rank=3,
    )
    agent = SimpleNamespace(
        name="agent-X", get_agent_metadata=lambda: b"\x01\x02agent-meta\xff"
    )
    staging_ctx = SimpleNamespace(allocator=None)
    if staging:
        staging_ctx.allocator = SimpleNamespace(
            get_base_ptr=lambda: 0x7F00_8000_0000, get_total_size=lambda: 1 << 24
        )
    return SimpleNamespace(
        kv_args=kv_args,
        agent=agent,
        local_ip="192.168.1.37",
        rank_port=30001,
        attn_tp_size=16,
        dcp_size=1,
        dcp_rank=0,
        enable_staging=staging,
        _staging_ctx=staging_ctx,
    )


build_reg = nixl_conn.NixlKVManager.build_receiver_registration_message
for name, state_entries, staging in [
    ("basic", 4, False),
    ("staging", 4, True),
    ("empty_state", 0, False),
]:
    frames = build_reg(reg_ns(state_entries, staging))
    assert len(frames) == 25, (
        f"upstream registration frame count changed: {len(frames)}"
    )
    info = nixl_conn.KVArgsRegisterInfo.from_zmq(frames[1:])
    V["registration"].append(
        {
            "name": name,
            "frames": hexs(frames),
            "parsed": {
                "room": info.room,
                "endpoint": info.endpoint,
                "dst_port": info.dst_port,
                "agent_name": info.agent_name,
                "agent_metadata": info.agent_metadata.hex(),
                "dst_kv_ptrs": s64s(info.dst_kv_ptrs),
                "dst_kv_mem_kinds": list(info.dst_kv_mem_kinds),
                "dst_aux_ptrs": s64s(info.dst_aux_ptrs),
                "dst_state_data_ptrs": [s64s(l) for l in info.dst_state_data_ptrs],
                "gpu_id": info.gpu_id,
                "decode_tp_size": info.decode_tp_size,
                "decode_tp_rank": info.decode_tp_rank,
                "dst_kv_item_len": str(info.dst_kv_item_len),
                "dst_kv_item_lens": s64s(info.dst_kv_item_lens),
                "dst_kv_layer_ids": list(info.dst_kv_layer_ids),
                "dst_dcp_size": info.dst_dcp_size,
                "dst_dcp_rank": info.dst_dcp_rank,
                "dst_num_slots": None
                if info.dst_num_slots is None
                else str(info.dst_num_slots),
                "dst_state_item_lens": [list(l) for l in info.dst_state_item_lens],
                "dst_state_dim_per_tensor": [
                    list(l) for l in info.dst_state_dim_per_tensor
                ],
                "dst_state_layer_ids": [list(l) for l in info.dst_state_layer_ids],
                "staging_base_ptr": str(info.staging_base_ptr),
                "staging_total_size": str(info.staging_total_size),
                "dst_state_data_lens": [s64s(l) for l in info.dst_state_data_lens],
            },
        }
    )

# --- NIXL metadata (nixl/conn.py:3495-3521 build, :303-339 parse) ---
build_meta = nixl_conn.NixlKVManager.build_receiver_metadata_message
meta_ns = SimpleNamespace(
    local_ip="10.0.0.9", rank_port=30002, agent=SimpleNamespace(name="agent-Y")
)


def call_meta(room, kv, aux, req, states, prefix, dummy):
    return build_meta(
        meta_ns,
        bootstrap_room=room,
        required_dst_info_num=req,
        kv_indices=kv,
        aux_index=aux,
        state_indices=states,
        decode_prefix_len=prefix,
        is_dummy=dummy,
    )


for name, room, kv, aux, states, prefix, dummy in [
    ("kv_run", 777, list(range(100, 106)), 2, None, None, False),
    ("state_only", 778, [], 2, [[1, 2], [3]], None, False),
    ("prefix_hit", 779, [], 2, [[9], [10]], 7, False),
    ("aux_only", 780, [], 2, None, None, False),
    ("kv_and_state", 781, list(range(5, 9)), 2, [[9], None], None, False),
    ("dummy", 782, [], 2, None, None, True),
]:
    frames = call_meta(
        room, np.array(kv, dtype=np.int32), aux, 8, states, prefix, dummy
    )
    assert len(frames) == 11, f"upstream metadata frame count changed: {len(frames)}"
    info = nixl_conn.TransferInfo.from_zmq(frames[1:])
    V["metadata"].append(
        {
            "name": name,
            "frames": hexs(frames),
            "parsed": {
                "room": info.room,
                "endpoint": info.endpoint,
                "dst_port": info.dst_port,
                "agent_name": info.agent_name,
                "dst_kv_indices": [int(x) for x in info.dst_kv_indices],
                "dst_aux_index": info.dst_aux_index,
                "required_dst_info_num": info.required_dst_info_num,
                "dst_state_indices": [list(l) for l in info.dst_state_indices],
                "decode_prefix_len": info.decode_prefix_len,
                "is_dummy": info.is_dummy,
            },
        }
    )

# --- /route HTTP bodies (common/conn.py:1559-1650, :3086-3236) ---
# PUT body: register_to_bootstrap's dict in insertion order,
# decode_allocation_policy appended last; json.dumps default separators.
put = {
    "attn_tp_size": 16,
    "attn_tp_rank": 3,
    "attn_cp_size": 1,
    "attn_cp_rank": 0,
    "attn_dp_size": 2,
    "attn_dp_rank": 1,
    "pp_size": 1,
    "pp_rank": 0,
    "system_dp_size": 1,
    "system_dp_rank": 0,
    "rank_ip": "10.0.0.5",
    "rank_port": 30011,
    "page_size": 1,
    "kv_cache_dtype": "auto",
    "kv_layout": None,
    "dsv41_spec_layout": None,
    "load_balance_method": "round_robin",
    "enable_dsa_cache_layer_split": False,
    "enable_kv_cache_sharding": False,
    "prefill_http_port": 8100,
}
put["decode_allocation_policy"] = "early"
V["http"]["put_row"] = json.dumps(put)

PrefillServerInfo = common_conn.PrefillServerInfo
PrefillRankInfo = common_conn.PrefillRankInfo

V["http"]["rank_row"] = json.dumps(
    dataclasses.asdict(PrefillRankInfo(rank_ip="10.0.0.5", rank_port=30011))
)

psi = PrefillServerInfo(
    attn_tp_size=8,
    attn_cp_size=1,
    dp_size=16,
    pp_size=1,
    page_size=1,
    kv_cache_dtype="auto",
    follow_bootstrap_room=True,
    enable_dsa_cache_layer_split=False,
    dsv41_spec_layout=None,
    kv_layout=None,
    decode_allocation_policy="early",
    enable_kv_cache_sharding=False,
    prefill_http_port=8100,
)
resp = dataclasses.asdict(psi)
for k in ("dsv41_spec_layout", "kv_layout"):
    if resp[k] is None:
        resp.pop(k)
V["http"]["get_row"] = json.dumps(resp)

resp2 = dataclasses.asdict(dataclasses.replace(psi, kv_layout="layer_first"))
resp2.pop("dsv41_spec_layout")
V["http"]["get_row_kv_layout"] = json.dumps(resp2)

resp3 = dataclasses.asdict(
    dataclasses.replace(
        psi, dsv41_spec_layout={"a": 1, "b": "x"}, kv_layout="layer_first"
    )
)
V["http"]["get_row_dsv41"] = json.dumps(resp3)

# --------------------------------------------------------------------------- //
# Room state machines (common/conn.py CommonKVManager methods, driven for real)
# --------------------------------------------------------------------------- //

import threading  # noqa: E402

M = common_conn.CommonKVManager
B, W, T, S, F = (
    KVPoll.Bootstrapping,
    KVPoll.WaitingForInput,
    KVPoll.Transferring,
    KVPoll.Success,
    KVPoll.Failed,
)


def make_mgr_ns(defer=False):
    """A pp_size==1 manager carrying every attribute the real methods touch."""
    ns = SimpleNamespace(
        request_status={},
        _deferred_ack_targets={},
        _deferred_ack_poisoned_rooms=set(),
        _staging_outstanding={},
        deferred_bootstrap=None,
        failure_lock=threading.Lock(),
        failure_records={},
        defer_decode_allocation=defer,
        prefill_response_tracker={},
        required_prefill_response_num_table={},
        enable_staging=False,
        _staging_handler=None,
        _deferred_abort_ack_tracker={},
        _deferred_abort_generation=0,
        DEFAULT_PREFILL_FAILURE_REASON=M.DEFAULT_PREFILL_FAILURE_REASON,
    )
    ns.update_status = lambda room, st: M.update_status(ns, room, st)
    ns.record_failure = lambda room, r: M.record_failure(ns, room, r)
    return ns


V["room"] = {"update_status": [], "decode": [], "deferred_abort": [], "prefill": []}

# --- update_status (common/conn.py:650-677) ---
for name, steps in [
    ("monotone", [(1, B), (1, W), (1, T), (1, S)]),
    ("open_waiting", [(2, W)]),
    ("no_create_on_terminal", [(3, S), (3, F), (3, T)]),
    ("failed_over_success", [(4, B), (4, S), (4, F)]),
    ("failed_terminal", [(5, B), (5, F), (5, S), (5, T)]),
    ("lower_ignored", [(6, B), (6, T), (6, W), (6, B)]),
]:
    ns = make_mgr_ns()
    for room, st in steps:
        ns.update_status(room, st)
    V["room"]["update_status"].append(
        {
            "name": name,
            "steps": [[r, int(s)] for r, s in steps],
            "final": {str(r): int(s) for r, s in ns.request_status.items()},
        }
    )


# --- decode side (common/conn.py:918-970 apply, :2506-2507 open,
#     :2932-2938 clear, :2952-2959 abort, :710-712 abort-message) ---
def decode_scenario(name, defer, ops):
    ns = make_mgr_ns(defer)
    for op in ops:
        k = op[0]
        if k == "open":
            _, room, required = op
            ns.prefill_response_tracker.setdefault(room, set())
            ns.required_prefill_response_num_table[room] = required
            ns.update_status(room, KVPoll.Bootstrapping)
        elif k == "open_tracker_only":
            # Surgery to reach the "tracker present, required missing"
            # warning path (a real race leaves tables exactly like this).
            _, room = op
            ns.prefill_response_tracker[room] = set()
            ns.update_status(room, KVPoll.Bootstrapping)
        elif k == "apply":
            _, room, status, rank, reason = op
            M.apply_prefill_status(
                ns,
                bootstrap_room=room,
                status=status,
                prefill_rank=rank,
                failure_reason=reason,
            )
        elif k == "abort":
            _, room = op
            ns.record_failure(room, "Aborted by AbortReq.")
            ns.update_status(room, KVPoll.Failed)
        elif k == "abort_msg":
            _, room = op
            if room in ns.request_status and ns.request_status[room] != KVPoll.Success:
                ns.update_status(room, KVPoll.Failed)
        elif k == "clear":
            _, room = op
            ns.request_status.pop(room, None)
            ns.required_prefill_response_num_table.pop(room, None)
            ns.prefill_response_tracker.pop(room, None)
    return {
        "name": name,
        "defer": defer,
        "ops": [
            [o[0]] + [int(x) if isinstance(x, KVPoll) else x for x in o[1:]]
            for o in ops
        ],
        "final_status": {str(r): int(s) for r, s in ns.request_status.items()},
        "final_failures": dict(ns.failure_records),
        "final_tracker": {
            str(r): sorted(t) for r, t in ns.prefill_response_tracker.items()
        },
    }


for sc in [
    decode_scenario(
        "quorum_two",
        False,
        [
            ("open", 1, 2),
            ("apply", 1, S, 0, None),
            ("apply", 1, S, 1, None),
        ],
    ),
    decode_scenario(
        "duplicate_rank_no_quorum",
        False,
        [
            ("open", 2, 2),
            ("apply", 2, S, 0, None),
            ("apply", 2, S, 0, None),
        ],
    ),
    decode_scenario(
        "missing_required_warns",
        False,
        [
            ("open_tracker_only", 3),
            ("apply", 3, S, 0, None),
        ],
    ),
    decode_scenario(
        "failed_reason",
        False,
        [
            ("open", 4, 1),
            ("apply", 4, F, 0, "peer died"),
        ],
    ),
    decode_scenario(
        "failed_default_reason",
        False,
        [
            ("open", 5, 1),
            ("apply", 5, F, 0, None),
        ],
    ),
    decode_scenario(
        "late_status_dropped",
        False,
        [
            ("open", 6, 1),
            ("clear", 6),
            ("apply", 6, F, 0, "late"),
        ],
    ),
    decode_scenario(
        "waiting_defer_off",
        False,
        [
            ("open", 7, 1),
            ("apply", 7, W, 0, None),
        ],
    ),
    decode_scenario(
        "waiting_defer_on",
        True,
        [
            ("open", 8, 1),
            ("apply", 8, W, 0, None),
        ],
    ),
    decode_scenario(
        "abort_active",
        False,
        [
            ("open", 9, 1),
            ("abort", 9),
        ],
    ),
    decode_scenario(
        "abort_msg_success_room",
        False,
        [
            ("open", 10, 1),
            ("apply", 10, S, 0, None),
            ("abort_msg", 10),
        ],
    ),
    decode_scenario(
        "abort_msg_failed_room",
        False,
        [
            ("open", 11, 1),
            ("apply", 11, F, 0, None),
            ("abort_msg", 11),
        ],
    ),
    decode_scenario(
        "clear_keeps_failure",
        False,
        [
            ("open", 12, 1),
            ("apply", 12, F, 0, "x"),
            ("clear", 12),
        ],
    ),
]:
    V["room"]["decode"].append(sc)


# --- deferred abort ACK tracker (common/conn.py:973-1019) ---
ns = make_mgr_ns()
abort_ops = []
gen1 = M.register_deferred_abort_room(ns, 1)
abort_ops.append(["register", 1, gen1])
gen2 = M.register_deferred_abort_room(ns, 2)
abort_ops.append(["register", 2, gen2])
M.note_abort_ack(ns, 1, 0, gen1)
abort_ops.append(["note_ack", 1, 0, gen1])
M.note_abort_ack(ns, 1, 1, 99)  # wrong generation: dropped
abort_ops.append(["note_ack", 1, 1, 99])
abort_ops.append(["is_safe", 1, 1, M.is_abort_release_safe(ns, 1, 1)])
gen3 = M.register_deferred_abort_room(ns, 1)  # re-arm clears prior ACKs
abort_ops.append(["register", 1, gen3])
M.note_abort_ack(ns, 1, 0, gen1)  # stale generation: dropped
abort_ops.append(["note_ack", 1, 0, gen1])
abort_ops.append(["is_safe", 1, 1, M.is_abort_release_safe(ns, 1, 1)])
M.note_abort_ack(ns, 1, 0, gen3)
abort_ops.append(["note_ack", 1, 0, gen3])
abort_ops.append(["is_safe", 1, 1, M.is_abort_release_safe(ns, 1, 1)])
M.clear_deferred_abort_state(ns, 1)
abort_ops.append(["clear", 1])
abort_ops.append(["is_safe", 1, 0, M.is_abort_release_safe(ns, 1, 0)])
V["room"]["deferred_abort"] = abort_ops


# --- prefill add-transfer-info (nixl/conn.py:3392-3413, inline block) ---
def prefill_scenario(name, adds):
    ns = make_mgr_ns()
    ns.transfer_infos = {}
    ns.req_to_decode_prefix_len = {}
    results = []
    for room, agent, required, prefix, dummy in adds:
        # The exact inline block from the nixl bootstrap thread.
        if room not in ns.transfer_infos:
            ns.transfer_infos[room] = {}
        ns.transfer_infos[room][agent] = SimpleNamespace(
            required_dst_info_num=required, decode_prefix_len=prefix, is_dummy=dummy
        )
        required = ns.transfer_infos[room][agent].required_dst_info_num
        result = None
        if len(ns.transfer_infos[room]) == required:
            ns.req_to_decode_prefix_len[room] = next(
                (
                    i.decode_prefix_len
                    for i in ns.transfer_infos[room].values()
                    if i.decode_prefix_len is not None
                ),
                0,
            )
            ns.update_status(room, KVPoll.WaitingForInput)
            result = ns.req_to_decode_prefix_len[room]
        results.append(result)
    return {
        "name": name,
        "adds": [[r, a, req, p, d] for r, a, req, p, d in adds],
        "results": results,
        "final_status": {str(r): int(s) for r, s in ns.request_status.items()},
        "final_prefixes": dict(ns.req_to_decode_prefix_len),
    }


for sc in [
    prefill_scenario(
        "quorum_two_agents",
        [
            (101, "a", 2, None, False),
            (101, "b", 2, 7, False),
        ],
    ),
    prefill_scenario(
        "single_agent_no_prefix",
        [
            (102, "a", 1, None, False),
        ],
    ),
    prefill_scenario(
        "overwrite_then_complete",
        [
            (103, "a", 2, None, False),
            (103, "a", 2, 5, False),
            (103, "b", 2, None, False),
        ],
    ),
]:
    V["room"]["prefill"].append(sc)

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(
    json.dumps(
        V,
        indent=1,
        default=lambda o: o.tolist() if isinstance(o, np.ndarray) else int(o),
    )
)
print(
    f"wrote {OUT} "
    f"({len(V['pack'])} pack, {len(V['pack_buffers'])} pack_buffers, "
    f"{len(V['kv_status'])} kv_status, {len(V['abort'])} abort, "
    f"{len(V['abort_ack'])} abort_ack, {len(V['bootstrap'])} bootstrap, "
    f"{len(V['chunk_ready'])} chunk_ready, {len(V['registration'])} registration, "
    f"{len(V['metadata'])} metadata, {len(V['http'])} http, "
    f"room: {len(V['room']['update_status'])} update_status, "
    f"{len(V['room']['decode'])} decode, "
    f"{len(V['room']['deferred_abort'])} deferred_abort ops, "
    f"{len(V['room']['prefill'])} prefill)"
)
