import ast
import pathlib
import textwrap
import types
from typing import Any, Dict, Tuple

import numpy as np
import pytest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
SCHEDULER_PATH = REPO_ROOT / "python/sglang/srt/managers/scheduler.py"
SERVER_ARGS_PATH = REPO_ROOT / "python/sglang/srt/server_args.py"


def _load_class_method(path, class_name, method_name, extra_namespace=None):
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    class_def = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    method = next(
        (
            node
            for node in class_def.body
            if isinstance(node, ast.FunctionDef) and node.name == method_name
        ),
        None,
    )
    if method is None:
        raise AttributeError(f"{class_name} has no method {method_name}")
    namespace = {"Any": Any, "Dict": Dict, "Tuple": Tuple}
    namespace.update(extra_namespace or {})
    exec(textwrap.dedent(ast.get_source_segment(source, method)), namespace)
    return namespace[method_name]


@pytest.mark.parametrize(
    "storage_hit,prompt_len,page_size,expected_h,expected_mode",
    [
        (1024, 1024, 16, 1024, "full_prefix_stitch"),
        (768, 1024, 16, 768, "partial_prefix_stitch"),
        (0, 1024, 16, 0, "source_decode_full_fallback"),
        (1024, 1027, 16, 1024, "full_prefix_stitch"),
    ],
)
def test_stitch_boundary(
    storage_hit, prompt_len, page_size, expected_h, expected_mode
):
    boundary = _load_class_method(
        SCHEDULER_PATH, "Scheduler", "_pd_flip_stitch_boundary"
    )

    assert boundary(storage_hit, prompt_len, page_size) == (expected_h, expected_mode)


def test_stitch_flag_requires_runtime_switch_radix_cache_and_storage_backend():
    validate = _load_class_method(
        SERVER_ARGS_PATH, "ServerArgs", "_validate_pd_flip_hicache_stitch"
    )
    args = types.SimpleNamespace(
        enable_pd_flip_hicache_stitch=True,
        enable_pd_runtime_role_switch=False,
        disaggregation_decode_enable_radix_cache=False,
        hicache_storage_backend=None,
    )

    with pytest.raises(ValueError) as exc_info:
        validate(args)

    message = str(exc_info.value)
    assert "--enable-pd-runtime-role-switch" in message
    assert "--disaggregation-decode-enable-radix-cache" in message
    assert "--hicache-storage-backend" in message


def test_stitch_flag_validation_accepts_complete_configuration():
    validate = _load_class_method(
        SERVER_ARGS_PATH, "ServerArgs", "_validate_pd_flip_hicache_stitch"
    )
    args = types.SimpleNamespace(
        enable_pd_flip_hicache_stitch=True,
        enable_pd_runtime_role_switch=True,
        disaggregation_decode_enable_radix_cache=True,
        hicache_storage_backend="mooncake",
    )

    validate(args)


class FakeSender:
    def __init__(self, prefix_len):
        self.prefix_len = prefix_len
        self.calls = []
        self.init_args = None
        self.sent_indices = None
        self.kv_mgr = object()

    def pop_decode_prefix_len(self):
        self.calls.append("pop")
        return self.prefix_len

    def init(self, num_pages, metadata_index):
        self.calls.append("init")
        self.init_args = (num_pages, metadata_index)

    def send(self, page_indices, state_indices):
        self.calls.append("send")
        self.sent_indices = page_indices


def _make_source_scheduler(*, enabled):
    send_initial = _load_class_method(
        SCHEDULER_PATH, "Scheduler", "_pd_flip_source_send_initial"
    )
    scheduler = types.SimpleNamespace(
        server_args=types.SimpleNamespace(enable_pd_flip_hicache_stitch=enabled),
        _pd_flip_source_page_indices_range=lambda req, start, end: list(
            range(start, end)
        ),
        _pd_flip_source_state_indices=lambda req, end, kv_mgr: [],
    )
    scheduler._pd_flip_source_send_initial = types.MethodType(send_initial, scheduler)
    return scheduler


def test_source_consumes_decode_prefix_before_sender_init():
    sender = FakeSender(prefix_len=32)
    scheduler = _make_source_scheduler(enabled=True)
    entry = {
        "sender": sender,
        "req": object(),
        "committed_len": 100,
        "metadata_index": 7,
    }

    scheduler._pd_flip_source_send_initial(entry)

    assert sender.calls == ["pop", "init", "send"]
    assert sender.init_args == (68, 7)
    assert sender.sent_indices == list(range(32, 100))
    assert entry["mooncake_hit_len"] == 32
    assert entry["source_transfer_start"] == 32
    assert entry["source_transfer_end"] == 100
    assert entry["stitch_mode"] == "prefix_stitch"


def test_source_full_fallback_preserves_fake_decode_behavior():
    sender = FakeSender(prefix_len=32)
    scheduler = _make_source_scheduler(enabled=False)
    entry = {
        "sender": sender,
        "req": object(),
        "committed_len": 4,
        "metadata_index": 3,
    }

    scheduler._pd_flip_source_send_initial(entry)

    assert sender.calls == ["init", "send"]
    assert sender.init_args == (4, 3)
    assert sender.sent_indices == [0, 1, 2, 3]
    assert entry["mooncake_hit_len"] == 0
    assert entry["stitch_mode"] == "source_decode_full_fallback"


def _target_stitch_ready(entry, *, enabled=True):
    method = _load_class_method(
        SCHEDULER_PATH, "Scheduler", "_pd_flip_target_stitch_ready"
    )
    scheduler = types.SimpleNamespace(
        server_args=types.SimpleNamespace(enable_pd_flip_hicache_stitch=enabled)
    )
    return method(scheduler, entry)


def _target_entry(*, h=32, p=64, c0=100, start=32, end=100, restore="ready"):
    prefix_match = types.SimpleNamespace(needs_local_restore=h > 0)
    decode_req = types.SimpleNamespace(
        prefix_match=prefix_match,
        hicache_restore_status=types.SimpleNamespace(value=restore),
    )
    return {
        "decode_req": decode_req,
        "mooncake_hit_len": h,
        "target_prompt_len": p,
        "target_committed_len": c0,
        "target_received_suffix_start": start,
        "target_received_suffix_end": end,
    }


def test_target_stitch_completion_requires_valid_boundaries_and_full_coverage():
    assert _target_stitch_ready(_target_entry())
    assert not _target_stitch_ready(_target_entry(h=65, p=64))
    assert not _target_stitch_ready(_target_entry(p=101, c0=100))
    assert not _target_stitch_ready(_target_entry(start=33))
    assert not _target_stitch_ready(_target_entry(end=99))


def test_target_stitch_completion_requires_hicache_restore_ready():
    assert not _target_stitch_ready(_target_entry(restore="pending"))
    assert not _target_stitch_ready(_target_entry(restore="failed"))


def test_target_full_fallback_does_not_require_hicache_restore():
    entry = _target_entry(h=0, start=0, restore="pending")
    entry["decode_req"].prefix_match.needs_local_restore = False

    assert _target_stitch_ready(entry)
    assert _target_stitch_ready({}, enabled=False)


def test_target_metadata_advertises_bounded_prefix_and_complete_suffix():
    class FakeTensor:
        def __init__(self, values):
            self.values = np.asarray(values)

        def __len__(self):
            return len(self.values)

        def __getitem__(self, item):
            value = self.values[item]
            return FakeTensor(value) if isinstance(value, np.ndarray) else value

        def cpu(self):
            return self

        def numpy(self):
            return self.values

        def tolist(self):
            return self.values.tolist()

    class PrefixMatch:
        def __init__(self):
            self.prefix_indices = FakeTensor([10, 20, 30, 40])
            self.l2_host_hit_length = 0
            self.l3_storage_hit_length = 8

        @property
        def l1_prefix_len(self):
            return len(self.prefix_indices)

        @property
        def decode_prefix_len(self):
            return (
                self.l1_prefix_len
                + self.l2_host_hit_length
                + self.l3_storage_hit_length
            )

        @property
        def needs_local_restore(self):
            return self.decode_prefix_len > self.l1_prefix_len

    class Receiver:
        def send_metadata(
            self, page_indices, metadata_index, state_indices, decode_prefix_len
        ):
            self.sent = (page_indices.tolist(), decode_prefix_len)

    class Queue:
        def __init__(self):
            self.match = PrefixMatch()

        def _match_prefix_and_lock(self, req):
            return self.match

        def _pre_alloc(self, req, **kwargs):
            return np.array([80, 90, 100, 110])

        def _start_hicache_prefetch(self, req, prefix_match):
            self.prefetched = prefix_match.decode_prefix_len

    method = _load_class_method(
        SCHEDULER_PATH,
        "Scheduler",
        "_pd_flip_target_prealloc_and_send_metadata",
        {
            "DecodeRequest": object,
            "time": types.SimpleNamespace(monotonic=lambda: 1.0),
            "TransferBackend": types.SimpleNamespace(FAKE="fake"),
            "kv_to_page_indices": lambda indices, page_size: np.asarray(indices),
        },
    )
    boundary = _load_class_method(
        SCHEDULER_PATH, "Scheduler", "_pd_flip_stitch_boundary"
    )
    req = types.SimpleNamespace(
        rid="req",
        origin_input_ids=list(range(10)),
        kv_committed_len=12,
        req_pool_idx=0,
        bootstrap_room=123,
    )
    receiver = Receiver()
    entry = {
        "decode_req": types.SimpleNamespace(req=req, kv_receiver=receiver),
        "metadata_index": -1,
    }
    queue = Queue()
    scheduler = types.SimpleNamespace(
        server_args=types.SimpleNamespace(
            enable_pd_flip_hicache_stitch=True,
            disaggregation_decode_enable_radix_cache=True,
        ),
        disagg_decode_prealloc_queue=queue,
        req_to_metadata_buffer_idx_allocator=types.SimpleNamespace(alloc=lambda: 0),
        token_to_kv_pool_allocator=types.SimpleNamespace(page_size=4),
        req_to_token_pool=types.SimpleNamespace(
            req_to_token=FakeTensor([list(range(12))])
        ),
        disagg_metadata_buffers=types.SimpleNamespace(
            bootstrap_room=np.zeros((1, 1), dtype=np.int64)
        ),
        enable_decode_hicache=True,
        transfer_backend="fake",
        _pd_flip_stitch_boundary=boundary,
        _pd_flip_target_state_indices=lambda req, committed_len: [],
    )

    method(scheduler, entry)

    assert receiver.sent == ([8, 9, 10, 11], 8)
    assert queue.prefetched == 8
    assert entry["mooncake_hit_len"] == 8
    assert entry["target_prompt_len"] == 10
    assert entry["target_committed_len"] == 12
    assert entry["target_received_suffix_start"] == 8
    assert entry["target_received_suffix_end"] == 12
    assert entry["stitch_mode"] == "full_prefix_stitch"


def test_stitch_uses_explicit_server_and_worker_flags_only():
    scheduler_source = SCHEDULER_PATH.read_text(encoding="utf-8")
    server_args_source = SERVER_ARGS_PATH.read_text(encoding="utf-8")
    worker_source = (
        REPO_ROOT
        / "scripts/playground/disaggregation/pd_flip_docker/run_worker.sh"
    ).read_text(encoding="utf-8")

    assert "SGLANG_PD_FLIP_HICACHE_STITCH" not in scheduler_source
    assert '"--enable-pd-flip-hicache-stitch"' in server_args_source
    assert "--enable-pd-flip-hicache-stitch" in worker_source


def test_stitch_completion_gate_applies_only_to_initial_transfer():
    source = SCHEDULER_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    scheduler = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Scheduler"
    )

    def method_source(name):
        method = next(
            node
            for node in scheduler.body
            if isinstance(node, ast.FunctionDef) and node.name == name
        )
        return ast.get_source_segment(source, method)

    assert "_pd_flip_target_stitch_ready" in method_source(
        "_pd_flip_target_pump_transfer"
    )
    assert "_pd_flip_target_stitch_ready" not in method_source(
        "_pd_flip_target_pump_delta_transfer"
    )
