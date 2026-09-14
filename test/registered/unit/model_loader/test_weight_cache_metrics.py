"""CPU-only unit tests for the weight cache daemon Prometheus exporter."""

import socket
import threading
import unittest
import urllib.request

import torch
from prometheus_client import generate_latest
from prometheus_client.parser import text_string_to_metric_families

from sglang.srt.weight_cache import metrics as weight_cache_metrics
from sglang.srt.weight_cache.daemon import WeightCacheDaemon
from sglang.srt.weight_cache.protocol import CacheConfig, recv_msg, send_msg
from sglang.srt.weight_cache.transport import TorchIpcTransportBackend
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=12, suite="base-a-test-cpu")

_PREFIX = "sglang:weight_cache_daemon_"


def _cache_config(**overrides) -> CacheConfig:
    base = dict(
        model_path="/models/demo",
        model_arch="LlamaForCausalLM",
        tp_size=2,
        tp_rank=1,
        pp_size=1,
        pp_rank=0,
        dp_size=1,
        ep_size=1,
        moe_dp_size=1,
        moe_dp_rank=0,
        moe_ep_rank=0,
        enable_dp_attention=False,
        enable_dp_lm_head=False,
        attn_cp_size=1,
        moe_dense_tp_size=None,
        moe_a2a_backend="none",
        quant_method="",
        quant_config_hash="",
        dtype="torch.float16",
        revision="",
        device_capability="8.0",
        torch_version="2.5.1",
    )
    base.update(overrides)
    return CacheConfig(**base)


def _make_daemon(**overrides) -> WeightCacheDaemon:
    """A WeightCacheDaemon with just the attributes the snapshot touches,
    built without __init__ (which needs a resolved ServerArgs + CUDA)."""
    daemon = object.__new__(WeightCacheDaemon)
    for key, value in {
        "gpu_id": 3,
        "tp_rank": 1,
        "socket_path": "/tmp/sglang_weight_cache_GPU-test.sock",
        "ready_path": "/tmp/sglang_weight_cache_GPU-test.ready",
        "config": _cache_config(),
        "state_entries": {"a": {}, "b": {}},
        "transport_backend": None,
        "preloaded_weights_bytes": 4096,
        "_started_at": 1_000.0,
        "_loaded_at": 1_010.0,
        "_load_seconds": 9.5,
        "_serve_count": 3,
        "_mismatch_count": 1,
        "_last_served_at": 1_020.0,
        "_served_client_pids": set(),
        "_status_lock": threading.Lock(),
        **overrides,
    }.items():
        setattr(daemon, key, value)
    return daemon


def _scrape(registry) -> dict:
    """{metric name: {frozenset(labels): value}} for every sample in the registry."""
    out = {}
    for family in text_string_to_metric_families(generate_latest(registry).decode()):
        for sample in family.samples:
            out.setdefault(sample.name, {})[frozenset(sample.labels.items())] = (
                sample.value
            )
    return out


class TestSnapshotToMetrics(CustomTestCase):
    def test_snapshot_fields_become_labeled_series(self):
        daemon = _make_daemon()
        samples = _scrape(weight_cache_metrics.build_registry(daemon._status_snapshot))

        labels = frozenset({("gpu_id", "3"), ("tp_rank", "1"), ("pp_rank", "0")})

        # Each gauge is exactly one series, labeled by daemon identity only.
        self.assertEqual(samples[_PREFIX + "loaded"], {labels: 1})
        self.assertEqual(samples[_PREFIX + "preloaded_weights_bytes"], {labels: 4096})
        self.assertEqual(samples[_PREFIX + "num_tensors"], {labels: 2})
        self.assertEqual(samples[_PREFIX + "load_seconds"], {labels: 9.5})
        self.assertEqual(samples[_PREFIX + "live_clients"], {labels: 0})
        self.assertEqual(
            samples[_PREFIX + "last_served_timestamp_seconds"], {labels: 1_020.0}
        )
        self.assertGreater(samples[_PREFIX + "uptime_seconds"][labels], 0)

        self.assertEqual(
            samples[_PREFIX + "fetch_state_total"],
            {labels | {("result", "hit")}: 3, labels | {("result", "mismatch")}: 1},
        )

        (info_labels,) = samples[_PREFIX + "info"].keys()
        info_labels = dict(info_labels)
        self.assertEqual(info_labels["model_path"], "/models/demo")
        self.assertEqual(info_labels["model_arch"], "LlamaForCausalLM")
        self.assertEqual(info_labels["tp_size"], "2")

    def test_unloaded_daemon_reports_loaded_zero_and_omits_unknowns(self):
        daemon = _make_daemon(
            _loaded_at=None,
            _load_seconds=None,
            _last_served_at=None,
            preloaded_weights_bytes=0,
            state_entries={},
        )
        samples = _scrape(weight_cache_metrics.build_registry(daemon._status_snapshot))
        (value,) = samples["sglang:weight_cache_daemon_loaded"].values()
        self.assertEqual(value, 0)
        # Unknown-yet fields yield no sample rather than a fake 0.
        self.assertNotIn("sglang:weight_cache_daemon_load_seconds", samples)
        self.assertNotIn(
            "sglang:weight_cache_daemon_last_served_timestamp_seconds", samples
        )

    def test_counters_track_fetch_state_outcomes(self):
        backend = TorchIpcTransportBackend()
        entries = backend.prepare_export({"x": (torch.arange(4), True)})
        daemon = _make_daemon(
            _serve_count=0,
            _mismatch_count=0,
            transport_backend=backend,
            state_entries=entries,
        )
        registry = weight_cache_metrics.build_registry(daemon._status_snapshot)

        def exchange(request):
            server_sock, client_sock = socket.socketpair(socket.AF_UNIX)
            try:
                send_msg(client_sock, request)
                daemon._handle_connection(server_sock)
                return recv_msg(client_sock)
            finally:
                server_sock.close()
                client_sock.close()

        hit = exchange({"type": "fetch_state", "config": daemon.config.to_dict()})
        self.assertEqual(hit["status"], "ok")
        miss = exchange(
            {
                "type": "fetch_state",
                "config": _cache_config(model_path="/models/other").to_dict(),
            }
        )
        self.assertEqual(miss["status"], "mismatch")

        fetch = _scrape(registry)["sglang:weight_cache_daemon_fetch_state_total"]
        by_result = {dict(k)["result"]: v for k, v in fetch.items()}
        self.assertEqual(by_result, {"hit": 1, "mismatch": 1})


class TestMetricsHttpServer(CustomTestCase):
    def test_metrics_endpoint_serves_scrape(self):
        daemon = _make_daemon()
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        weight_cache_metrics.start_metrics_server(
            daemon._status_snapshot, port, addr="127.0.0.1"
        )

        body = (
            urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=5)
            .read()
            .decode()
        )
        self.assertIn('sglang:weight_cache_daemon_loaded{gpu_id="3"', body)
        self.assertIn("sglang:weight_cache_daemon_fetch_state_total", body)


if __name__ == "__main__":
    unittest.main()
