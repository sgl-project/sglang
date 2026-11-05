"""Bounded CPU AFD task service: no inference HTTP endpoint or device runtime.

The F process owns a central virtual event graph for all modeled A/F ranks.
The A process keeps the real Scheduler/request lifecycle. Loopback control RPC
wall time is not the modeled A2F/F2A link latency; those costs stay in the graph.
"""

import hashlib
import json
import socket
import socketserver
from pathlib import Path

from .adapters import create_adapter
from .native_config import NativeAFDConfig
from .runner import simulate

MAX_MESSAGE = 4 * 1024 * 1024


def _signature(native):
    return dict(
        routing=native.routing,
        attention_ranks=native.attention_ranks,
        ffn_ranks=native.ffn_ranks,
        stages=native.stages,
        stage_capacity=native.stage_capacity,
        attention_backend=native.raw.get("attention_backend", "trtllm_mha"),
    )


def _read_message(stream):
    line = stream.readline(MAX_MESSAGE + 1)
    if not line or len(line) > MAX_MESSAGE or not line.endswith(b"\n"):
        raise ValueError("AFD_TASK_MESSAGE_INVALID_OR_TOO_LARGE")
    value = json.loads(line)
    if not isinstance(value, dict):
        raise ValueError("AFD_TASK_MESSAGE_MUST_BE_AN_OBJECT")
    return value


def _write_message(stream, value):
    wire = json.dumps(value, allow_nan=False).encode() + b"\n"
    if len(wire) > MAX_MESSAGE:
        raise ValueError("AFD_TASK_RESPONSE_TOO_LARGE")
    stream.write(wire)
    stream.flush()


class CostCatalog:
    """Exact workload lookup; no nearest-neighbor, zero or legacy fallback."""

    def __init__(self, path, source_root, native, *, allow_fixture=False):
        path = Path(path)
        data = json.loads(path.read_text())
        if data.get("schema_version") != 1:
            raise ValueError("AFD_CATALOG_SCHEMA_UNSUPPORTED")
        self.adapter = create_adapter(source_root, "afd", routing=native.routing)
        self.adapter.verify()
        self.native = native
        self.decode = {}
        self.prefill = {}
        self.fixture = False
        for relative in data.get("decode", []):
            p = (path.parent / relative).resolve()
            document = json.loads(p.read_text())
            native.check(document["deployment"])
            from .costs import CostTable

            CostTable(document, self.adapter.cost_mapping(document["deployment"]))
            if document["fixture"] and not allow_fixture:
                raise ValueError("AFD_SERVING_FIXTURE_REQUIRES_EXPLICIT_OPT_IN")
            self.fixture |= document["fixture"]
            c = document["deployment"]
            key = c["batch_size"], c["effective_kv"]
            if key in self.decode:
                raise ValueError("AFD_CATALOG_DUPLICATE_DECODE_KEY")
            self.decode[key] = document
        for row in data.get("prefill", []):
            key = tuple(sorted(tuple(x) for x in row["requests"]))
            ns = row["duration_ns"]
            basis = row["basis"]
            if not key or any(
                len(x) != 2 or any(type(v) is not int or v < 0 for v in x) or x[0] < 1
                for x in key
            ):
                raise ValueError("AFD_PREFILL_REQUEST_SHAPE_INVALID")
            if type(ns) is not int or ns < 1 or not row.get("source"):
                raise ValueError("AFD_PREFILL_COST_INVALID")
            if basis not in ("exact", "interpolation", "formula", "fixture"):
                raise ValueError("AFD_PREFILL_COST_BASIS_INVALID")
            if basis == "formula" and not row.get("equation"):
                raise ValueError("AFD_PREFILL_FORMULA_MISSING")
            if basis == "interpolation" and not row.get("neighbors"):
                raise ValueError("AFD_PREFILL_NEIGHBORS_MISSING")
            if basis == "fixture" and not allow_fixture:
                raise ValueError("AFD_SERVING_FIXTURE_REQUIRES_EXPLICIT_OPT_IN")
            self.fixture |= basis == "fixture"
            if key in self.prefill:
                raise ValueError("AFD_CATALOG_DUPLICATE_PREFILL_KEY")
            self.prefill[key] = row
        if not self.decode:
            raise ValueError("AFD_CATALOG_DECODE_COSTS_REQUIRED")
        self.digest = hashlib.sha256(
            json.dumps(data, sort_keys=True).encode()
        ).hexdigest()

    def execute(self, requests, phase):
        if not requests or len({r["id"] for r in requests}) != len(requests):
            raise ValueError("AFD_REQUEST_IDENTITIES_REQUIRED_AND_UNIQUE")
        for r in requests:
            if not isinstance(r["id"], str) or not r["id"]:
                raise ValueError("AFD_REQUEST_ID_INVALID")
            if (
                any(type(r[k]) is not int or r[k] < 0 for k in ("extend", "past"))
                or r["extend"] < 1
            ):
                raise ValueError("AFD_REQUEST_LENGTH_INVALID")
        if phase == "prefill":
            key = tuple(sorted((r["extend"], r["past"]) for r in requests))
            if key not in self.prefill:
                raise ValueError(
                    "AFD_PREFILL_COST_MISSING: exact request composition required"
                )
            row = self.prefill[key]
            return dict(
                duration_ns=row["duration_ns"],
                request_finish_ns={r["id"]: row["duration_ns"] for r in requests},
                placement={},
                method="declared_prefill_batch_cost",
                basis=row["basis"],
                source=row["source"],
            )
        if phase != "decode" or any(r["extend"] != 1 for r in requests):
            raise ValueError("AFD_BATCH_PHASE_INVALID")
        kvs = {r["past"] + 1 for r in requests}
        if len(kvs) != 1:
            raise ValueError("AFD_NONUNIFORM_KV_REQUIRES_REQUEST_AWARE_ADAPTER")
        key = len(requests), next(iter(kvs))
        if key not in self.decode:
            raise ValueError(
                f"AFD_DECODE_COST_MISSING: batch={key[0]}, effective_kv={key[1]}"
            )
        document = self.decode[key]
        result = simulate(document, adapter=self.adapter)
        rows = key[0] // 16
        ready = {
            e["actor"]: e["finish_ns"]
            for e in result["events"]
            if e["kind"] == "sampling"
        }
        placement = {
            r["id"]: dict(
                attention_rank=i // (2 * rows), stage=(i // rows) % 2, slot=i % rows
            )
            for i, r in enumerate(requests)
        }
        finishes = {
            rid: ready["A" + str(pos["attention_rank"])]
            for rid, pos in placement.items()
        }
        return dict(
            duration_ns=result["control_completion_ns"],
            request_finish_ns=finishes,
            placement=placement,
            method="source_bound_afd_event_graph",
            critical_path=result["critical_path"],
            source_files=result["source_files"],
            events=result["events"],
            deployment=result["deployment"],
        )


class AFDTaskServer(socketserver.TCPServer):
    allow_reuse_address = False

    def __init__(self, native, catalog, output_dir):
        host = native.raw.get("rendezvous_host", "127.0.0.1")
        if host not in ("127.0.0.1", "localhost"):
            raise ValueError("AFD_TASK_SERVICE_LOOPBACK_ONLY")
        self.native = native
        self.catalog = catalog
        self.output = Path(output_dir)
        self.output.mkdir(parents=True, exist_ok=False)
        self.sequence = 0

        class Handler(socketserver.StreamRequestHandler):
            def handle(inner):
                inner.connection.settimeout(120)
                try:
                    request = _read_message(inner.rfile)
                    response = self.dispatch(request)
                    _write_message(inner.wfile, dict(ok=True, **response))
                except (ValueError, KeyError, TypeError, OSError) as exc:
                    _write_message(
                        inner.wfile,
                        dict(ok=False, error=f"{type(exc).__name__}: {exc}"),
                    )

        super().__init__((host, native.raw.get("rendezvous_port", 1239)), Handler)
        manifest = dict(
            kind="afd_cpu_internal_task_service",
            address=list(self.server_address),
            native_afd_config=native.raw,
            signature=_signature(native),
            catalog_sha256=catalog.digest,
            fixture=catalog.fixture,
            capacity_fit="unknown",
            service_accuracy_verified=False,
        )
        (self.output / "SERVICE.json").write_text(json.dumps(manifest, indent=2) + "\n")

    def dispatch(self, request):
        if request.get("signature") != _signature(self.native):
            raise ValueError("AFD_PEER_CONFIG_MISMATCH")
        if request.get("op") == "handshake":
            return dict(
                signature=_signature(self.native),
                catalog_sha256=self.catalog.digest,
                fixture=self.catalog.fixture,
                kind="afd_cpu_internal_task_service",
            )
        if request.get("op") != "batch":
            raise ValueError("AFD_TASK_OPERATION_UNSUPPORTED")
        result = self.catalog.execute(request["requests"], request["phase"])
        sequence = self.sequence
        self.sequence += 1
        record = dict(
            sequence=sequence,
            requests=request["requests"],
            phase=request["phase"],
            **result,
        )
        (self.output / f"step-{sequence:06d}.json").write_text(
            json.dumps(record, indent=2) + "\n"
        )
        result.pop("events", None)
        return dict(sequence=sequence, fixture=self.catalog.fixture, **result)


class AFDTaskClient:
    def __init__(self, native, *, timeout=120):
        self.native = native
        self.timeout = timeout
        self.request("handshake")

    def request(self, op, **payload):
        host = self.native.raw.get("rendezvous_host", "127.0.0.1")
        if host not in ("localhost", "127.0.0.1"):
            raise ValueError("AFD_TASK_SERVICE_LOOPBACK_ONLY")
        with socket.create_connection(
            (host, self.native.raw.get("rendezvous_port", 1239)), timeout=self.timeout
        ) as sock:
            with sock.makefile("rwb") as stream:
                _write_message(
                    stream, dict(op=op, signature=_signature(self.native), **payload)
                )
                result = _read_message(stream)
        if not result.get("ok"):
            raise ValueError("AFD_F_TASK_FAILED: " + result.get("error", "unknown"))
        return result


class AFDRequestPredictor:
    """Scheduler predictor bridge; returns per-request graph completion offsets."""

    uses_virtual_event_time = True
    requires_exact_request_context = True

    def __init__(self, model=None, hw=None, config=None, *, afd_config):
        self.model, self.hw, self.config = model, hw, config
        if model is not None:
            expected = dict(
                hidden_size=4096,
                num_hidden_layers=94,
                num_attention_heads=64,
                num_key_value_heads=4,
                head_dim=128,
                vocab_size=151936,
            )
            if any(getattr(model, k, None) != v for k, v in expected.items()):
                raise ValueError("AFD_REQUEST_MODEL_PROFILE_MISMATCH")
            hf = model.hf_config

            def get(name):
                return hf.get(name) if isinstance(hf, dict) else getattr(hf, name, None)

            if get("num_experts") != 128 or get("num_experts_per_tok") != 8:
                raise ValueError("AFD_REQUEST_EXPERT_PROFILE_MISMATCH")
        self.native = NativeAFDConfig.parse(afd_config)
        self.client = AFDTaskClient(self.native)
        self.last = None
        self.placements = {}
        self.steps = []

    def predict_infer_time(self, batch):
        phase = getattr(batch, "forward_phase", None) or (
            "decode" if batch.is_decode() else "prefill"
        )
        requests = [
            dict(id=r.request_id, extend=r.extend_length, past=r.past_kv_length)
            for r in batch.reqs
        ]
        result = self.client.request("batch", requests=requests, phase=phase)
        for rid, where in result["placement"].items():
            if rid in self.placements and self.placements[rid] != where:
                raise ValueError("AFD_REQUEST_MIGRATION_REQUIRES_EXPLICIT_MODEL")
        self.placements = {
            rid: p
            for rid, p in self.placements.items()
            if rid in result["request_finish_ns"]
        }
        self.placements.update(result["placement"])
        self.last = result
        self.steps.append(
            dict(
                sequence=result["sequence"],
                phase=phase,
                method=result["method"],
                fixture=result["fixture"],
            )
        )
        return result["duration_ns"] / 1e9

    def completion_offsets(self):
        return (
            {rid: ns / 1e9 for rid, ns in self.last["request_finish_ns"].items()}
            if self.last
            else {}
        )

    def get_metrics(self):
        return dict(
            afd_steps=self.steps,
            afd_fixture=bool(self.last and self.last["fixture"]),
            afd_capacity_fit="unknown",
            afd_service_accuracy_verified=False,
            afd_host_overhead="cost-ledger events only; CPU interpreter and control-RPC wall time excluded",
        )

    def reset_metrics(self):
        self.steps = []
