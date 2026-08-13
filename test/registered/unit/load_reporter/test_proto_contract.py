"""Descriptor-level contract test for the canonical LoadMonitorService proto."""

import ast
import re
import sys
from pathlib import Path

import pytest

from sglang.srt.load_reporter.proto import load_monitor_pb2
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_REPO_ROOT = Path(__file__).resolve().parents[4]
_PROTO_PACKAGE = _REPO_ROOT / "python/sglang/srt/load_reporter/proto"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _service():
    return load_monitor_pb2.DESCRIPTOR.services_by_name["LoadMonitorService"]


def _message(name):
    return load_monitor_pb2.DESCRIPTOR.message_types_by_name[name]


def _oneof_fields_by_number(msg_name, oneof_name):
    """Return {field_number: field_name} for the named oneof in msg_name."""
    msg = _message(msg_name)
    oneof = msg.oneofs_by_name[oneof_name]
    return {f.number: f.name for f in oneof.fields}


def _fields_by_number(msg_name):
    return {f.number: f.name for f in _message(msg_name).fields}


class TestGeneratedRuntimeCompatibility:
    def test_protobuf_gencode_targets_declared_minimum(self):
        source = (_PROTO_PACKAGE / "load_monitor_pb2.py").read_text()
        match = re.search(r"^# Protobuf Python Version: (\S+)$", source, re.MULTILINE)

        assert match is not None
        assert match.group(1) == "6.31.1"

    def test_grpc_gencode_targets_declared_minimum(self):
        source = (_PROTO_PACKAGE / "load_monitor_pb2_grpc.py").read_text()
        module = ast.parse(source)
        generated_version = next(
            node.value.value
            for node in module.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "GRPC_GENERATED_VERSION"
                for target in node.targets
            )
            and isinstance(node.value, ast.Constant)
        )

        assert generated_version == "1.78.0"


# ---------------------------------------------------------------------------
# Service contract
# ---------------------------------------------------------------------------


class TestServiceDescriptor:
    def test_full_service_name(self):
        assert _service().full_name == "sglang.router.loadmonitor.v1.LoadMonitorService"

    def test_monitor_method_exists(self):
        assert "Monitor" in _service().methods_by_name

    def test_monitor_is_client_streaming(self):
        method = _service().methods_by_name["Monitor"]
        assert (
            method.client_streaming is True
        ), "Monitor must be client-streaming (bidi)"

    def test_monitor_is_server_streaming(self):
        method = _service().methods_by_name["Monitor"]
        assert (
            method.server_streaming is True
        ), "Monitor must be server-streaming (bidi)"

    def test_report_rpc_removed(self):
        assert (
            "Report" not in _service().methods_by_name
        ), "Old Report RPC must be removed; only Monitor should exist"

    def test_only_monitor_method_exists(self):
        assert list(_service().methods_by_name.keys()) == ["Monitor"]


# ---------------------------------------------------------------------------
# RouterFrame oneof
# ---------------------------------------------------------------------------


class TestRouterFrame:
    def test_has_payload_oneof(self):
        assert "payload" in _message("RouterFrame").oneofs_by_name

    def test_register_field_number(self):
        fields = _oneof_fields_by_number("RouterFrame", "payload")
        assert fields[1] == "register"

    def test_update_config_field_number(self):
        fields = _oneof_fields_by_number("RouterFrame", "payload")
        assert fields[2] == "update_config"

    def test_keep_alive_field_number(self):
        fields = _oneof_fields_by_number("RouterFrame", "payload")
        assert fields[3] == "keep_alive"

    def test_stop_field_number(self):
        fields = _oneof_fields_by_number("RouterFrame", "payload")
        assert fields[4] == "stop"


# ---------------------------------------------------------------------------
# WorkerFrame oneof
# ---------------------------------------------------------------------------


class TestWorkerFrame:
    def test_has_payload_oneof(self):
        assert "payload" in _message("WorkerFrame").oneofs_by_name

    def test_registered_field_number(self):
        fields = _oneof_fields_by_number("WorkerFrame", "payload")
        assert fields[1] == "registered"

    def test_report_field_number(self):
        fields = _oneof_fields_by_number("WorkerFrame", "payload")
        assert fields[2] == "report"

    def test_error_field_number(self):
        fields = _oneof_fields_by_number("WorkerFrame", "payload")
        assert fields[3] == "error"


# ---------------------------------------------------------------------------
# Preserved LoadReport field numbers
# ---------------------------------------------------------------------------


class TestLoadReportFieldNumbers:
    def test_source_instance_id_is_1(self):
        assert _fields_by_number("LoadReport")[1] == "source_instance_id"

    def test_sequence_id_is_2(self):
        assert _fields_by_number("LoadReport")[2] == "sequence_id"

    def test_report_time_unix_ms_is_3(self):
        assert _fields_by_number("LoadReport")[3] == "report_time_unix_ms"

    def test_worker_is_4(self):
        assert _fields_by_number("LoadReport")[4] == "worker"

    def test_status_is_5(self):
        assert _fields_by_number("LoadReport")[5] == "status"

    def test_ranks_is_7(self):
        assert _fields_by_number("LoadReport")[7] == "ranks"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
