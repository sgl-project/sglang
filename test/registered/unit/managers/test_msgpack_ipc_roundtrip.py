"""Round-trip coverage for the IPC structs that used to be pickle-wrapped.

Issue #29465 Task 4 tightened the 13 types in `_REQ_TYPES_WITH_OPAQUE_FIELDS` to
precise msgspec-native annotations and deleted the registry. This test proves
each type now encodes natively over the msgpack IPC path (no `PickleWrapper`
frame) by asserting `msgpack_decode(msgpack_encode(x)) == x`, and guards the
type-specific decisions (the `ExpertWeightPointer` narrowing, the
`CheckWeightsReqOutput` struct mirrors, and the internal-state sanitization).
"""

import dataclasses
import enum
import unittest

import msgspec

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers import io_struct
from sglang.srt.managers.io_struct import (
    BackupDramReq,
    ChecksumInfo,
    CheckWeightsReqOutput,
    DumperControlReqInput,
    DumperControlReqOutput,
    ExpertWeightPointer,
    GetInternalStateReqOutput,
    GetWeightsByNameReqOutput,
    LoadLoRAAdapterFromTensorsReqInput,
    ParallelismInfo,
    RpcReqInput,
    SetInternalStateReq,
    SetInternalStateReqOutput,
    UpdateWeightFromDiskReqInput,
    VertexGenerateReqInput,
    msgpack_decode,
    msgpack_encode,
)
from sglang.srt.model_executor.cuda_graph_config import CudaGraphConfig
from sglang.srt.utils.msgspec_utils import msgspec_to_builtins
from sglang.srt.utils.weight_checker import ChecksumInfo as PydanticChecksumInfo
from sglang.srt.utils.weight_checker import ParallelismInfo as PydanticParallelismInfo
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-c-test-cpu")


def _round_trip(obj):
    return msgpack_decode(msgpack_encode(obj))


def _double_hop(obj):
    # MultiTokenizerRouter and the DP controller re-encode already-decoded
    # structs, so a single hop cannot catch re-encode bugs.
    return msgpack_decode(msgpack_encode(msgpack_decode(msgpack_encode(obj))))


def _contains_dataclass_or_enum(obj) -> bool:
    if isinstance(obj, enum.Enum):
        return True
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return True
    if isinstance(obj, dict):
        return any(_contains_dataclass_or_enum(v) for v in obj.values())
    if isinstance(obj, (list, tuple, set)):
        return any(_contains_dataclass_or_enum(v) for v in obj)
    return False


def _parallelism_info() -> ParallelismInfo:
    return ParallelismInfo(
        tp_rank=0, tp_size=2, dp_rank=0, dp_size=1, pp_rank=0, pp_size=1, rank=0, size=2
    )


def _checksum_info(tag: str) -> ChecksumInfo:
    return ChecksumInfo(
        checksums={f"model.layers.{tag}": "deadbeef"},
        per_gpu_checksum="cafef00d",
        parallelism_info=_parallelism_info(),
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
        serialized_tensors="",
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
        self.assertIn("tp_rank", as_dict["parallelism_info"])

    def test_get_internal_state_sanitizes_dataclass_and_enum(self):
        # A live vars() dump can hold a dataclass (CudaGraphConfig) and an enum
        # (DisaggregationMode). The producer sanitizes via msgspec_to_builtins so
        # neither survives onto the wire; materialize CudaGraphConfig explicitly.
        raw = {
            "cuda_graph_config": CudaGraphConfig(),
            "disaggregation_mode": DisaggregationMode.NULL,
            "max_running_requests": 256,
        }
        self.assertTrue(_contains_dataclass_or_enum(raw))

        sanitized = msgspec_to_builtins(raw)
        self.assertFalse(_contains_dataclass_or_enum(sanitized))
        self.assertEqual(sanitized["disaggregation_mode"], "null")
        self.assertIsInstance(sanitized["cuda_graph_config"], dict)

        output = GetInternalStateReqOutput(internal_state=sanitized)
        self.assertEqual(_round_trip(output), output)


if __name__ == "__main__":
    unittest.main()
