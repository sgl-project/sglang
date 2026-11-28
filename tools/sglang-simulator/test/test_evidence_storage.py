"""Lossless storage controls; synthetic CPU evidence, never accuracy admission."""

import builtins
import gzip
import hashlib
import io
import json

import pytest
from sglang_simulator.simulation.iteration_stats import IterationStats
from sglang_simulator.utils.evidence import (
    ZstdCodec,
    evidence_level,
    open_evidence_jsonl,
)


def rows():
    return [
        {"forward_latency": 0.1, "native": {"source": "x" * 100000}},
        {"forward_latency": 0.2, "cpu_overhead": 0.03, "name": "证据"},
    ]


@pytest.mark.parametrize("codec", ["none", "gzip", "zstd"])
def test_export_recovers_identical_original_jsonl(tmp_path, codec):
    if codec == "zstd":
        zstd = pytest.importorskip("zstandard")
    expected = "".join(json.dumps(row) + "\n" for row in rows())
    p = tmp_path / "output"
    with open_evidence_jsonl(p, codec) as stream:
        assert stream.write(expected) == len(expected)
    if codec == "zstd":
        with zstd.ZstdDecompressor().stream_reader(p.open("rb")) as stream:
            raw = stream.read()
    elif codec == "gzip":
        raw = gzip.decompress(p.read_bytes())
    else:
        raw = p.read_bytes()
    assert raw.decode() == expected
    assert hashlib.sha256(raw).digest() == hashlib.sha256(expected.encode()).digest()


def test_zstd_spool_preserves_order_metrics_random_access_and_reset(tmp_path):
    pytest.importorskip("zstandard")
    store = IterationStats(lambda: tmp_path, compression="zstd")
    original = rows()
    for row in original:
        store.append(row)
    assert list(store) == original and store[-1] == original[-1]
    assert store.metric_rows() == [
        {"forward_latency": 0.1, "l2_load_latency": 0, "cpu_overhead": 0},
        {"forward_latency": 0.2, "l2_load_latency": 0, "cpu_overhead": 0.03},
    ]
    before = store.path
    store.clear()
    assert not before.exists()
    store.append(original[1])
    assert list(store) == original[1:]
    store.clear()


@pytest.mark.parametrize("mutation", ["truncate", "checksum", "extra"])
def test_zstd_row_frames_reject_incomplete_corrupt_or_extra_bytes(mutation):
    pytest.importorskip("zstandard")
    codec = ZstdCodec()
    encoded = codec.compress(b'{"event":"original"}\n')
    if mutation == "truncate":
        encoded = encoded[:-1]
    elif mutation == "checksum":
        encoded = encoded[:-1] + bytes([encoded[-1] ^ 1])
    else:
        encoded += b"unexpected"
    with pytest.raises(ValueError, match="invalid or incomplete"):
        codec.decompress(encoded)


def test_zstd_failed_export_keeps_complete_spool(tmp_path):
    pytest.importorskip("zstandard")
    store = IterationStats(lambda: tmp_path, compression="zstd")
    store.append(rows()[0])

    class Broken:
        def write(self, _):
            raise OSError("disk full")

    with pytest.raises(OSError, match="disk full"):
        store.write_jsonl(Broken())
    assert store.path.is_file() and list(store) == rows()[:1]
    store.clear()


def test_metrics_only_keeps_numeric_reduction_without_creating_verbose_files(tmp_path):
    store = IterationStats(lambda: tmp_path / "not-created", metrics_only=True)
    full = IterationStats(lambda: tmp_path / "full")
    for row in rows():
        store.append(row)
        full.append(row)
    assert store.metric_rows() == full.metric_rows()
    assert len(store) == len(full) == 2
    assert store.path is None and not (tmp_path / "not-created").exists()
    with pytest.raises(RuntimeError, match="disabled by metrics policy"):
        store.write_jsonl(io.StringIO())
    store.clear()
    assert len(store) == 0
    full.clear()


def test_optional_dependency_is_only_required_for_full_zstd(tmp_path, monkeypatch):
    original = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "zstandard":
            raise ImportError("deliberately absent optional dependency")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(ImportError, match=r"sglang-simulator\[zstd\]"):
        IterationStats(lambda: tmp_path, compression="zstd")
    store = IterationStats(lambda: tmp_path, compression="zstd", metrics_only=True)
    store.append(rows()[0])
    assert store.path is None


def test_retention_level_is_explicit_and_validated(monkeypatch):
    monkeypatch.delenv("SGLANG_SIMULATOR_EVIDENCE_LEVEL", raising=False)
    assert evidence_level() == "full"
    monkeypatch.setenv("SGLANG_SIMULATOR_EVIDENCE_LEVEL", "metrics")
    assert evidence_level() == "metrics"
    monkeypatch.setenv("SGLANG_SIMULATOR_EVIDENCE_LEVEL", "unknown")
    with pytest.raises(ValueError, match="full or metrics"):
        evidence_level()
