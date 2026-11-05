"""Round-trip coverage for the IPC structs that used to be pickle-wrapped.

Issue #29465 Task 4 tightened the 13 types in `_REQ_TYPES_WITH_OPAQUE_FIELDS` to
precise msgspec-native annotations and deleted the registry. This test proves
each type now encodes natively over the msgpack IPC path (no `PickleWrapper`
frame) by asserting `msgpack_decode(msgpack_encode(x)) == x`, and guards the
type-specific decisions (the `ExpertWeightPointer` narrowing, the
`CheckWeightsReqOutput` struct mirrors, and the internal-state sanitization).
"""

import dataclasses
import unittest
from array import array

import msgspec

from sglang.srt.managers import io_struct
from sglang.srt.managers.incremental_detokenize import DetokenizeHeader
from sglang.srt.managers.io_struct import (
    AbortReq,
    BackupDramReq,
    BatchEmbeddingOutput,
    BatchStrOutput,
    BatchTokenIDOutput,
    ChecksumInfo,
    CheckWeightsReqOutput,
    DumperControlReqInput,
    DumperControlReqOutput,
    ExpertWeightPointer,
    GetInternalStateReqOutput,
    GetWeightsByNameReqOutput,
    LoadLoRAAdapterFromTensorsReqInput,
    ParallelismInfo,
    RequestOutputStats,
    RpcReqInput,
    SetInternalStateReq,
    SetInternalStateReqOutput,
    UpdateWeightFromDiskReqInput,
    VertexGenerateReqInput,
    flatten_output_ids,
    msgpack_decode,
    msgpack_encode,
    row_output_ids,
)
from sglang.srt.managers.scheduler_components.weight_updater import (
    _merge_checksum_payloads,
)
from sglang.srt.model_executor.cuda_graph_config import CudaGraphConfig
from sglang.srt.observability.req_time_stats import SchedulerOutputTimeStats
from sglang.srt.utils.msgspec_utils import msgspec_to_builtins
from sglang.srt.utils.weight_checker import ChecksumInfo as PydanticChecksumInfo
from sglang.srt.utils.weight_checker import ParallelismInfo as PydanticParallelismInfo
from sglang.srt.utils.weight_versions import WeightVersionSpan
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="stage-b-test-cpu-intel")


def _round_trip(obj):
    return msgpack_decode(msgpack_encode(obj))


def _double_hop(obj):
    # MultiTokenizerRouter and the DP controller re-encode already-decoded
    # structs, so a single hop cannot catch re-encode bugs.
    return msgpack_decode(msgpack_encode(msgpack_decode(msgpack_encode(obj))))


def _batch_output(cls, **fields):
    """A cls with every required field empty, then fields."""
    empty = {f.name: [] for f in msgspec.structs.fields(cls) if f.required}
    if "output_ids_flat" in empty:
        empty["output_ids_flat"] = array("q")
    return cls(**{**empty, **fields})


def _contains_dataclass(obj) -> bool:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return True
    if isinstance(obj, dict):
        return any(_contains_dataclass(v) for v in obj.values())
    if isinstance(obj, (list, tuple, set)):
        return any(_contains_dataclass(v) for v in obj)
    return False


def _parallelism_info(role: str = "target") -> ParallelismInfo:
    return ParallelismInfo(
        role=role,
        tp_rank=0,
        tp_size=2,
        dp_rank=0,
        dp_size=1,
        pp_rank=0,
        pp_size=1,
        rank=0,
        size=2,
    )


def _checksum_info(tag: str) -> ChecksumInfo:
    return ChecksumInfo(
        checksums={f"model.layers.{tag}": "deadbeef"},
        per_gpu_checksum="cafef00d",
        parallelism_info=[_parallelism_info("target"), _parallelism_info("draft")],
    )


# One representative instance per (now-tightened) ex-registry type. The 13th
# entry, SetInjectDumpMetadataReqInput, was deleted as dead code, leaving 12.
REGISTRY_TYPE_INSTANCES = {
    "UpdateWeightFromDiskReqInput": UpdateWeightFromDiskReqInput(
        model_path="dummy", manifest={"w": [1, 2], "meta": {"k": "v"}}
    ),
    "BackupDramReq": BackupDramReq(
        rank=0,
        weight_pointer_map={
            "experts.0.gate_proj": ExpertWeightPointer(weight_ptr=8, byte_size=4),
            "experts.1.up_proj": ExpertWeightPointer(weight_ptr=16, byte_size=8),
        },
        session_id="session",
        buffer_size=1024,
    ),
    "GetWeightsByNameReqOutput/flat": GetWeightsByNameReqOutput(
        parameter=[1.0, 2.5, 3.0]
    ),
    "GetWeightsByNameReqOutput/nested": GetWeightsByNameReqOutput(
        parameter=[[1.0, 2.0], [3.0]]
    ),
    "GetWeightsByNameReqOutput/none": GetWeightsByNameReqOutput(parameter=None),
    "CheckWeightsReqOutput": CheckWeightsReqOutput(
        success=True,
        message="Success.",
        payload=[_checksum_info("0"), _checksum_info("1")],
    ),
    "GetInternalStateReqOutput": GetInternalStateReqOutput(
        internal_state={"a": 1, "b": [1, 2], "c": {"d": "e"}, "f": None}
    ),
    "SetInternalStateReq": SetInternalStateReq(
        server_args={
            "pp_max_micro_batch_size": 4,
            "speculative_accept_threshold_acc": 0.5,
        }
    ),
    "SetInternalStateReqOutput": SetInternalStateReqOutput(updated=True),
    "VertexGenerateReqInput": VertexGenerateReqInput(
        instances=[{"prompt": "hi"}], parameters={"max_tokens": 8}
    ),
    "RpcReqInput/empty": RpcReqInput(method="collective_rpc", parameters={}),
    "RpcReqInput/scalars": RpcReqInput(
        method="collective_rpc",
        parameters={"flag": True, "n": 1, "ratio": 2.0, "name": "x", "opt": None},
    ),
    "RpcReqInput/none": RpcReqInput(method="collective_rpc", parameters=None),
    "LoadLoRAAdapterFromTensorsReqInput": LoadLoRAAdapterFromTensorsReqInput(
        lora_name="adapter",
        config_dict={"r": 8, "lora_alpha": 16, "target_modules": ["q_proj", "v_proj"]},
        serialized_named_tensors=[b"tp0-bytes", b"tp1-bytes"],
        added_tokens_config={"<extra>": 32000},
    ),
    "DumperControlReqInput": DumperControlReqInput(method="start", body={"k": "v"}),
    "DumperControlReqOutput": DumperControlReqOutput(
        success=True, response=[{"worker": 0, "ok": True}]
    ),
}

NARROWED_BACKUP_KEYS = ("name", "shape", "numel", "dtype", "element_size")


class TestMsgpackIpcRoundtrip(CustomTestCase):
    def test_registry_is_empty(self):
        # `getattr(..., ())` is deliberate: the acceptance criterion for Task 4 is
        # that the symbol is *deleted*, so this asserts its absence rather than
        # defensively reading a field. It must survive the symbol removal.
        self.assertEqual(
            getattr(io_struct, "_REQ_TYPES_WITH_OPAQUE_FIELDS", ()),
            (),
        )

    def test_each_type_round_trips_natively(self):
        for name, instance in REGISTRY_TYPE_INSTANCES.items():
            with self.subTest(type=name):
                encoded = msgpack_encode(instance)
                # Natively encoded structs are never wrapped: a PickleWrapper
                # frame would decode back to a PickleWrapper, not the type.
                self.assertNotIsInstance(
                    msgpack_decode(encoded), io_struct.PickleWrapper
                )
                self.assertEqual(_round_trip(instance), instance)
                self.assertEqual(_double_hop(instance), instance)

    def test_backup_dram_req_is_narrowed(self):
        # ExpertWeightPointer carries only the two fields the consumer reads; the
        # five torch-metadata keys the producer used to send are gone.
        field_names = {f.name for f in msgspec.structs.fields(ExpertWeightPointer)}
        self.assertEqual(field_names, {"weight_ptr", "byte_size"})
        for dropped in NARROWED_BACKUP_KEYS:
            self.assertNotIn(dropped, field_names)

        decoded = _round_trip(REGISTRY_TYPE_INSTANCES["BackupDramReq"])
        pointer = decoded.weight_pointer_map["experts.0.gate_proj"]
        self.assertEqual((pointer.weight_ptr, pointer.byte_size), (8, 4))

    def test_check_weights_mirrors_match_pydantic_models(self):
        # Field-parity guard: the msgspec wire structs must not drift from the
        # pydantic source of truth in weight_checker.
        self.assertEqual(
            {f.name for f in msgspec.structs.fields(ParallelismInfo)},
            set(PydanticParallelismInfo.model_fields),
        )
        self.assertEqual(
            {f.name for f in msgspec.structs.fields(ChecksumInfo)},
            set(PydanticChecksumInfo.model_fields),
        )

    def test_check_weights_multi_rank_payload(self):
        # tp>1 sends one ChecksumInfo per rank; the list round-trips and stays a
        # {field: value} dict once converted back to builtins for the HTTP body.
        instance = REGISTRY_TYPE_INSTANCES["CheckWeightsReqOutput"]
        decoded = _round_trip(instance)
        self.assertEqual(len(decoded.payload), 2)
        as_dict = msgspec_to_builtins(decoded.payload[0])
        self.assertEqual(as_dict["per_gpu_checksum"], "cafef00d")
        self.assertEqual(
            [pi["role"] for pi in as_dict["parallelism_info"]], ["target", "draft"]
        )
        self.assertIn("tp_rank", as_dict["parallelism_info"][0])

    def test_check_weights_producer_conversion(self):
        # the merged per-role payload is what msgspec.convert has to accept
        def _dump(role):
            return PydanticChecksumInfo(
                checksums={"model.layers.0": "deadbeef"},
                per_gpu_checksum="cafef00d",
                parallelism_info=PydanticParallelismInfo(
                    role=role,
                    tp_rank=0,
                    tp_size=2,
                    dp_rank=0,
                    dp_size=1,
                    pp_rank=0,
                    pp_size=1,
                    rank=0,
                    size=2,
                ),
            ).model_dump()

        merged = _merge_checksum_payloads(
            [("target", _dump("target")), ("draft", _dump("draft"))]
        )
        converted = msgspec.convert(merged, ChecksumInfo)
        self.assertEqual(
            sorted(converted.checksums), ["draft.model.layers.0", "model.layers.0"]
        )
        self.assertEqual(
            [pi.role for pi in converted.parallelism_info], ["target", "draft"]
        )
        self.assertEqual(converted.parallelism_info[0].tp_rank, 0)
        output = CheckWeightsReqOutput(success=True, message="ok", payload=[converted])
        self.assertEqual(_round_trip(output), output)

    def test_get_internal_state_sanitizes_dataclass(self):
        # A live vars(ServerArgs) dump holds a dataclass: cuda_graph_config is an
        # Optional[CudaGraphConfig]. The producer sanitizes via msgspec_to_builtins
        # so it does not survive onto the wire; materialize CudaGraphConfig
        # explicitly.
        raw = {
            "cuda_graph_config": CudaGraphConfig(),
            "max_running_requests": 256,
        }
        self.assertTrue(_contains_dataclass(raw))

        sanitized = msgspec_to_builtins(raw)
        self.assertFalse(_contains_dataclass(sanitized))
        self.assertIsInstance(sanitized["cuda_graph_config"], dict)

        output = GetInternalStateReqOutput(internal_state=sanitized)
        self.assertEqual(_round_trip(output), output)


class TestWeightVersionSpansRoundTrip(CustomTestCase):
    """The per-request weight-version spans ride the same msgpack IPC path."""

    def test_abort_req_carries_spans(self):
        """A scheduler-side abort keeps its spans across the wire."""
        obj = AbortReq(
            rid="r0",
            weight_versions=[
                WeightVersionSpan(version="v1", start=0, end=3),
                WeightVersionSpan(version="v2", start=3, end=7),
            ],
        )

        decoded = _double_hop(obj)

        self.assertEqual(
            decoded.weight_versions,
            [
                WeightVersionSpan(version="v1", start=0, end=3),
                WeightVersionSpan(version="v2", start=3, end=7),
            ],
        )
        self.assertIsInstance(decoded.weight_versions[0], WeightVersionSpan)

    def test_abort_req_defaults_to_no_spans(self):
        """The field is optional on the wire, so an abort without spans decodes to None."""
        self.assertIsNone(_round_trip(AbortReq(rid="r0")).weight_versions)


class TestRequestStatsRoundTrip(CustomTestCase):
    """Per-row request stats, with their time stats, decode back to typed
    structs, not plain lists."""

    def test_batch_outputs_carry_request_stats(self):
        time_stats = SchedulerOutputTimeStats(
            queue_time=0.5, forward_entry_time=1.7e9, storage_staging_wait=0.0
        )
        stats = RequestOutputStats(
            prompt_tokens=7,
            cached_tokens=3,
            cached_tokens_details={"device": 3, "host": 0, "storage_backend": "X"},
            retraction_count=1,
            image_tokens=4,
            audio_tokens=0,
            video_tokens=0,
            time_stats=time_stats,
        )
        for cls in (BatchTokenIDOutput, BatchStrOutput, BatchEmbeddingOutput):
            # Embedding rows always carry their stats.
            request_stats = [stats, stats if cls is BatchEmbeddingOutput else None]
            with self.subTest(type=cls.__name__):
                obj = _batch_output(cls, rids=["a", "b"])
                obj.request_stats = request_stats

                decoded = _double_hop(obj)

                self.assertEqual(decoded.request_stats, request_stats)
                self.assertIsInstance(decoded.request_stats[0], RequestOutputStats)
                self.assertIsInstance(
                    decoded.request_stats[0].time_stats, SchedulerOutputTimeStats
                )


class TestDetokenizeRoundTrip(CustomTestCase):
    def test_header_and_row_routes_round_trip(self):
        header = DetokenizeHeader(
            prompt_tail_ids=array("q", [151644, 872, 198]),
            skip_special_tokens=False,
            spaces_between_special_tokens=True,
            no_stop_trim=True,
            http_worker_ipc="ipc:///tmp/tokenizer-0",
        )
        obj = _batch_output(BatchTokenIDOutput, rids=["a", "b"])
        obj.detokenize_headers = [header, None]
        obj.http_worker_ipcs = [None, "ipc:///tmp/tokenizer-1"]

        decoded = _double_hop(obj)

        self.assertEqual(decoded.detokenize_headers, [header, None])
        self.assertIsInstance(decoded.detokenize_headers[0], DetokenizeHeader)
        self.assertIsInstance(decoded.detokenize_headers[0].prompt_tail_ids, array)
        self.assertEqual(decoded.http_worker_ipcs, [None, "ipc:///tmp/tokenizer-1"])

    def test_output_ids_columns_round_trip(self):
        rows = [[151644, 872], [], [-101, 2**40]]
        flat, offsets = flatten_output_ids(rows)
        for cls in (BatchTokenIDOutput, BatchStrOutput):
            with self.subTest(type=cls.__name__):
                obj = _batch_output(
                    cls,
                    rids=["a", "b", "c"],
                    output_ids_flat=flat,
                    output_ids_offsets=offsets,
                )

                decoded = _double_hop(obj)

                self.assertIsInstance(decoded.output_ids_flat, array)
                self.assertEqual(
                    [list(row_output_ids(decoded, i)) for i in range(3)], rows
                )


if __name__ == "__main__":
    unittest.main()
