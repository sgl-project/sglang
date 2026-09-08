"""Contract tests for the cache-neutral, versioned KV-hint transport."""

import ast
import asyncio
import inspect
import re
import subprocess
from array import array
from pathlib import Path
from types import SimpleNamespace

import msgspec
import pytest
from pydantic import TypeAdapter

from sglang.srt.managers.io_struct import (
    GenerateReqInput,
    SessionParams,
    TokenizedGenerateReqInput,
    msgpack_decode,
    msgpack_encode,
)
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="stage-b-test-cpu-intel")

REPO_ROOT = Path(__file__).resolve().parents[4]
PROTO_PATH = REPO_ROOT / "proto/sglang/runtime/v1/sglang.proto"
RUST_REQUEST_UTILS = REPO_ROOT / "rust/sglang-grpc/src/utils/request_utils.rs"
RUST_SERVER = REPO_ROOT / "rust/sglang-grpc/src/server.rs"


def _kv_types():
    # Kept local so a pre-implementation run collects one useful failure per test.
    from sglang.srt.kv_hints import (  # noqa: PLC0415
        KvHintAction,
        KvHints,
        normalize_kv_hints,
    )

    return KvHintAction, KvHints, normalize_kv_hints


def _raw_hints(message_id="message-1", payload=None):
    return {
        "protocol_version": "0.1",
        "message_id": message_id,
        "actions": [
            {
                "action_id": f"action-{message_id}",
                "action_type": "future.action",
                "action_version": "87.4",
                "payload": (
                    {"nested": {"items": [1, 2]}, "unsigned": 2**64 - 1}
                    if payload is None
                    else payload
                ),
            }
        ],
    }


def _normalize_request(**kwargs):
    req = GenerateReqInput(**kwargs)
    req.normalize_batch_and_arguments()
    return req


def _minimal_tokenized(**kwargs):
    values = {
        "rid": "request-1",
        "input_text": "hello",
        # None avoids coupling this metadata test to the separately covered
        # array-extension codec; TokenizedGenerateReqInput explicitly permits it.
        "input_ids": None,
        "input_embeds": None,
        "mm_inputs": None,
        "token_type_ids": None,
        "sampling_params": SamplingParams(),
        "return_logprob": False,
        "logprob_start_len": -1,
        "top_logprobs_num": 0,
        "token_ids_logprob": None,
        "stream": False,
    }
    values.update(kwargs)
    return TokenizedGenerateReqInput(**values)


def _proto_descriptor(tmp_path):
    descriptor_path = tmp_path / "sglang.pb"
    subprocess.run(
        [
            "protoc",
            f"--proto_path={REPO_ROOT / 'proto'}",
            "--include_imports",
            f"--descriptor_set_out={descriptor_path}",
            str(PROTO_PATH.relative_to(REPO_ROOT / "proto")),
        ],
        check=True,
        cwd=REPO_ROOT,
    )
    from google.protobuf import descriptor_pb2  # noqa: PLC0415

    file_set = descriptor_pb2.FileDescriptorSet()
    file_set.ParseFromString(descriptor_path.read_bytes())
    return next(item for item in file_set.file if item.package == "sglang.runtime.v1")


def _message(descriptor, name):
    return next(item for item in descriptor.message_type if item.name == name)


def _field(message, name):
    return next(item for item in message.field if item.name == name)


def _function_source(source, function_name, next_function_name):
    start = source.index(f"fn {function_name}")
    end = source.index(f"fn {next_function_name}", start)
    return source[start:end]


def test_kh001_canonical_types_exist_with_exact_fields():
    KvHintAction, KvHints, _ = _kv_types()

    assert issubclass(KvHintAction, msgspec.Struct)
    assert issubclass(KvHints, msgspec.Struct)
    assert KvHintAction.__struct_fields__ == (
        "action_id",
        "action_type",
        "action_version",
        "payload",
    )
    assert KvHints.__struct_fields__ == (
        "protocol_version",
        "message_id",
        "actions",
    )


def test_kh002_legal_mapping_becomes_nested_canonical_objects():
    KvHintAction, KvHints, normalize = _kv_types()

    hints = normalize(_raw_hints())

    assert isinstance(hints, KvHints)
    assert isinstance(hints.actions[0], KvHintAction)
    assert hints.actions[0].payload["unsigned"] == 2**64 - 1


def test_kh003_unsupported_envelope_version_is_rejected():
    _, _, normalize = _kv_types()
    raw = _raw_hints()
    raw["protocol_version"] = "0.2"

    with pytest.raises(ValueError, match="protocol_version"):
        normalize(raw)


@pytest.mark.parametrize(
    "malformed",
    [
        {},
        {"protocol_version": "0.1", "message_id": "missing-actions"},
        {"protocol_version": 1, "message_id": "m", "actions": []},
        {"protocol_version": "0.1", "message_id": "m", "actions": {}},
        {"protocol_version": "0.1", "message_id": "m", "actions": [None]},
        {
            "protocol_version": "0.1",
            "message_id": "m",
            "actions": [
                {
                    "action_id": "a",
                    "action_type": "future.action",
                    "action_version": "1",
                    "payload": [],
                }
            ],
        },
    ],
)
def test_kh004_malformed_structure_is_rejected(malformed):
    _, _, normalize = _kv_types()

    with pytest.raises(ValueError):
        normalize(malformed)


def test_kh005_unknown_action_type_and_version_are_preserved():
    _, _, normalize = _kv_types()

    hints = normalize(_raw_hints())

    assert hints.actions[0].action_type == "future.action"
    assert hints.actions[0].action_version == "87.4"


def test_kh006_normalization_breaks_all_mutable_aliases():
    _, _, normalize = _kv_types()
    raw = _raw_hints()

    hints = normalize(raw)
    cloned = normalize(hints)
    raw["actions"][0]["payload"]["nested"]["items"].append(3)
    raw["actions"][0]["action_type"] = "mutated"

    assert hints.actions[0].action_type == "future.action"
    assert hints.actions[0].payload["nested"]["items"] == [1, 2]
    assert cloned == hints
    assert cloned is not hints
    assert cloned.actions is not hints.actions
    assert cloned.actions[0].payload is not hints.actions[0].payload
    assert (
        cloned.actions[0].payload["nested"]["items"]
        is not hints.actions[0].payload["nested"]["items"]
    )


@pytest.mark.parametrize(
    ("input_field", "input_value"),
    [("text", "hello"), ("input_ids", [1, 2])],
    ids=["KH010-text", "KH011-input-ids"],
)
def test_kh010_kh011_native_request_parsing_and_normalization(input_field, input_value):
    _, KvHints, _ = _kv_types()
    parsed = TypeAdapter(GenerateReqInput).validate_python(
        {input_field: input_value, "kv_hints": _raw_hints()}
    )

    parsed.normalize_batch_and_arguments()

    assert isinstance(parsed.kv_hints, KvHints)


def test_kh012_per_item_batch_hints_keep_item_alignment():
    _, KvHints, _ = _kv_types()

    req = _normalize_request(
        text=["a", "b", "c"],
        kv_hints=[_raw_hints("m0"), None, _raw_hints("m2")],
    )

    assert [item.message_id if item else None for item in req.kv_hints] == [
        "m0",
        None,
        "m2",
    ]
    assert isinstance(req.kv_hints[0], KvHints)
    assert isinstance(req.kv_hints[2], KvHints)


def test_kh013_batch_length_mismatch_is_rejected():
    with pytest.raises(ValueError, match="kv_hints"):
        _normalize_request(
            text=["a", "b"],
            kv_hints=[_raw_hints("only-one")],
        )


def test_kh014_scalar_hint_is_rejected_for_true_batch():
    with pytest.raises(ValueError, match="kv_hints"):
        _normalize_request(text=["a", "b"], kv_hints=_raw_hints())


def test_kh015_single_request_parallel_sampling_clones_without_aliases():
    req = _normalize_request(
        text="hello", sampling_params={"n": 3}, kv_hints=_raw_hints("one")
    )

    assert [item.message_id for item in req.kv_hints] == ["one"] * 3
    assert len({id(item) for item in req.kv_hints}) == 3
    assert len({id(item.actions) for item in req.kv_hints}) == 3
    assert len({id(item.actions[0].payload) for item in req.kv_hints}) == 3
    assert (
        len({id(item.actions[0].payload["nested"]["items"]) for item in req.kv_hints})
        == 3
    )


def test_kh016_batch_parallel_sampling_keeps_current_main_order_and_no_aliases():
    req = _normalize_request(
        text=["a", "b"],
        sampling_params={"n": 2},
        kv_hints=[_raw_hints("m0"), _raw_hints("m1")],
    )

    assert req.text == ["a", "b", "a", "b"]
    assert [item.message_id for item in req.kv_hints] == ["m0", "m1", "m0", "m1"]
    assert len({id(item.actions[0].payload) for item in req.kv_hints}) == 4


def test_kh020_getitem_keeps_the_correct_canonical_item():
    req = _normalize_request(
        text=["a", "b"],
        kv_hints=[_raw_hints("m0"), _raw_hints("m1")],
    )

    assert req[0].kv_hints.message_id == "m0"
    assert req[1].kv_hints.message_id == "m1"


def test_kh021_tokenizer_manager_forwards_only_canonical_hints():
    _, KvHints, _ = _kv_types()
    from sglang.srt.managers.tokenizer_manager import TokenizerManager  # noqa: PLC0415

    class FakeSamplingParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def normalize(self, tokenizer):
            return None

        def verify(self, vocab_size):
            return None

    manager = object.__new__(TokenizerManager)
    manager.preferred_sampling_params = {}
    manager.sampling_params_class = FakeSamplingParams
    manager.tokenizer = None
    manager.model_config = SimpleNamespace(vocab_size=128)
    manager.fake_bootstrap_room_counter = 0
    req = _normalize_request(
        text="hello", bootstrap_room=7, kv_hints=_raw_hints("tokenizer")
    )
    manager.rid_to_state = {
        req.rid: SimpleNamespace(
            time_stats=SimpleNamespace(set_tokenize_finish_time=lambda: None)
        )
    }

    tokenized = manager._create_tokenized_object(req, "hello", [1, 2])

    assert isinstance(tokenized.kv_hints, KvHints)
    assert tokenized.kv_hints == req.kv_hints


def test_kh022_real_ipc_codec_preserves_nested_canonical_structure():
    KvHintAction, KvHints, normalize = _kv_types()
    hints = normalize(_raw_hints("ipc"))
    req = _minimal_tokenized(kv_hints=hints)

    req.wrap_pickle_fields()
    decoded = msgpack_decode(msgpack_encode(req))
    decoded.unwrap_pickle_fields()

    assert isinstance(decoded.kv_hints, KvHints)
    assert isinstance(decoded.kv_hints.actions[0], KvHintAction)
    assert decoded.kv_hints == hints


def test_kh023_session_reconstruction_preserves_canonical_hints():
    _, KvHints, normalize = _kv_types()
    from sglang.srt.session.session_controller import Session  # noqa: PLC0415

    tokenized = _minimal_tokenized(
        session_params=SessionParams(id="session-1"),
        kv_hints=normalize(_raw_hints("session")),
    )
    session = Session(0, session_id="session-1")

    req = session.create_req(tokenized, tokenizer=None, vocab_size=128)

    assert isinstance(req.kv_hints, KvHints)
    assert req.kv_hints == tokenized.kv_hints


def test_kh024_scheduler_constructs_req_with_canonical_hints(monkeypatch):
    _, KvHints, normalize = _kv_types()
    from sglang.srt.disaggregation.utils import DisaggregationMode  # noqa: PLC0415
    from sglang.srt.managers import scheduler as scheduler_module  # noqa: PLC0415

    real_req_type = scheduler_module.Req
    with pytest.raises(TypeError, match="KvHints or None"):
        real_req_type(
            "unvalidated",
            "hello",
            array("q", [1]),
            SamplingParams(),
            kv_hints=_raw_hints("raw-is-not-internal"),
        )

    captured = {}

    class FakeReq:
        def __init__(self, *args, **kwargs):
            captured.update(kwargs)
            self.rid = args[0]
            self.return_logprob = kwargs["return_logprob"]
            self.finished_reason = None

        def set_finish_with_abort(self, *args, **kwargs):
            self.finished_reason = "aborted"

    fake_scheduler = SimpleNamespace(
        enable_session_radix_cache=False,
        model_config=SimpleNamespace(hf_eos_token_id={2}, vocab_size=128),
        disaggregation_mode=DisaggregationMode.NULL,
        metrics_reporter=SimpleNamespace(enable_metrics=False),
        metrics_collector=None,
        dllm_config=None,
        tokenizer=None,
        _maybe_namespace_elastic_radix_cache=lambda req: None,
        init_req_max_new_tokens=lambda req: None,
        _add_request_to_queue=lambda req: None,
    )
    monkeypatch.setattr(scheduler_module, "Req", FakeReq)
    hints = normalize(_raw_hints("scheduler"))
    tokenized = _minimal_tokenized(kv_hints=hints, bootstrap_port=1)

    scheduler_module.Scheduler.handle_generate_request(
        fake_scheduler, tokenized, mm_input_error="expected stop"
    )

    assert isinstance(captured["kv_hints"], KvHints)
    assert captured["kv_hints"] == hints


class _CapturingTokenizerManager:
    def __init__(self):
        self.request = None

    async def generate_request(self, request, _raw_request):
        request.normalize_batch_and_arguments()
        self.request = request
        yield {"ok": True}


def _bare_engine():
    from sglang.srt.entrypoints.engine import Engine  # noqa: PLC0415

    engine = object.__new__(Engine)
    engine.tokenizer_manager = _CapturingTokenizerManager()
    return engine


def test_kh030_engine_generate_constructs_and_forwards_hints():
    _, KvHints, _ = _kv_types()
    engine = _bare_engine()
    engine.loop = asyncio.new_event_loop()
    try:
        result = engine.generate(prompt="hello", kv_hints=_raw_hints("sync-engine"))
    finally:
        engine.loop.close()

    assert result == {"ok": True}
    assert isinstance(engine.tokenizer_manager.request.kv_hints, KvHints)
    assert engine.tokenizer_manager.request.kv_hints.message_id == "sync-engine"


def test_kh031_engine_async_generate_constructs_and_forwards_hints():
    _, KvHints, _ = _kv_types()
    engine = _bare_engine()

    result = asyncio.run(
        engine.async_generate(prompt="hello", kv_hints=_raw_hints("async-engine"))
    )

    assert result == {"ok": True}
    assert isinstance(engine.tokenizer_manager.request.kv_hints, KvHints)
    assert engine.tokenizer_manager.request.kv_hints.message_id == "async-engine"


def test_kh040_grpc_text_has_typed_appended_field_and_lowering(tmp_path):
    descriptor = _proto_descriptor(tmp_path)
    field = _field(_message(descriptor, "TextGenerateRequest"), "kv_hints")
    source = RUST_REQUEST_UTILS.read_text()
    body = _function_source(source, "build_text_generate_dict", "build_generate_dict")

    assert field.number == 18
    assert field.type_name == ".sglang.runtime.v1.KvHints"
    assert "req.kv_hints" in body


def test_kh041_grpc_tokenized_has_typed_appended_field_and_lowering(tmp_path):
    descriptor = _proto_descriptor(tmp_path)
    field = _field(_message(descriptor, "GenerateRequest"), "kv_hints")
    source = RUST_REQUEST_UTILS.read_text()
    body = _function_source(source, "build_generate_dict", "build_text_embed_dict")

    assert field.number == 17
    assert field.type_name == ".sglang.runtime.v1.KvHints"
    assert "req.kv_hints" in body


def test_kh042_grpc_paths_share_json_equivalent_lowering(tmp_path):
    descriptor = _proto_descriptor(tmp_path)
    action = _message(descriptor, "KvHintAction")
    payload = _field(action, "payload_json")
    source = RUST_REQUEST_UTILS.read_text()

    assert payload.number == 4
    assert payload.type == payload.TYPE_BYTES
    assert source.count("kv_hints_to_json(&req.kv_hints)") == 2
    assert source.count('d.insert("kv_hints"') == 2
    assert (
        "n.as_u64()"
        in (REPO_ROOT / "rust/sglang-grpc/src/utils/py_utils.rs").read_text()
    )


def test_kh043_grpc_malformed_input_maps_to_invalid_argument():
    from sglang.srt.kv_hints import (  # noqa: PLC0415
        SUPPORTED_KV_HINTS_PROTOCOL_VERSION,
    )

    source = RUST_REQUEST_UTILS.read_text()
    server = RUST_SERVER.read_text()
    rust_version = re.search(
        r'SUPPORTED_KV_HINTS_PROTOCOL_VERSION: &str = "([^"]+)"', source
    )

    assert rust_version is not None
    assert rust_version.group(1) == SUPPORTED_KV_HINTS_PROTOCOL_VERSION
    assert "payload_json" in source
    assert "JSON object" in source
    assert source.count("kv_hints_to_json(&req.kv_hints)") == 2
    assert server.count("map_err(Status::invalid_argument)") >= 2


def test_kh050_absent_hints_remain_none_through_normalization_and_ipc():
    req = _normalize_request(text="hello")
    batch_req = _normalize_request(text=["hello", "world"])
    tokenized = _minimal_tokenized(kv_hints=None)
    tokenized.wrap_pickle_fields()
    decoded = msgpack_decode(msgpack_encode(tokenized))
    decoded.unwrap_pickle_fields()

    assert req.kv_hints is None
    assert batch_req.kv_hints is None
    assert decoded.kv_hints is None


def test_kh051_hints_never_enter_sampling_parameters():
    req = _normalize_request(
        text="hello",
        sampling_params={"temperature": 0.25},
        kv_hints=_raw_hints("isolated"),
    )

    assert "kv_hints" not in req.sampling_params
    assert req.sampling_params == {"temperature": 0.25}
    assert "kv_hints" not in inspect.signature(SamplingParams).parameters


def test_kh052_transport_is_inert_and_has_no_backend_or_handler_coupling():
    _, _, _ = _kv_types()
    module_source = (REPO_ROOT / "python/sglang/srt/kv_hints.py").read_text()
    prohibited_roots = [
        REPO_ROOT / "python/sglang/srt/mem_cache",
        REPO_ROOT / "python/sglang/srt/managers/cache_controller.py",
        REPO_ROOT / "python/sglang/srt/disaggregation",
    ]

    tree = ast.parse(module_source)
    imported_modules = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    }
    assert all(
        not module.startswith(
            (
                "sglang.srt.mem_cache",
                "sglang.srt.disaggregation",
                "sglang.srt.managers.cache_controller",
            )
        )
        for module in imported_modules
    )
    assert not any(
        isinstance(node, (ast.FunctionDef, ast.ClassDef))
        and re.search(r"(manager|dispatcher|execute)", node.name, re.IGNORECASE)
        for node in ast.walk(tree)
    )
    for root in prohibited_roots:
        paths = [root] if root.is_file() else root.rglob("*.py")
        assert all("kv_hints" not in path.read_text() for path in paths)
