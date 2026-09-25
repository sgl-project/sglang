"""CPU coverage for byte-exact pulls of serialized NVFP4 checkpoints."""

import importlib.util
import json
import shutil
import struct
from pathlib import Path

import numpy as np
import pytest
import safetensors.numpy
import xxhash
import zstandard

# The filesystem receiver does not need the serving stack or torch. Load it
# directly so these tests also run in a minimal CPU environment.
REPO_ROOT = Path(__file__).resolve().parents[4]


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


register_cpu_ci = _load_module(
    "ci_register", REPO_ROOT / "python/sglang/test/ci/ci_register.py"
).register_cpu_ci
register_cpu_ci(est_time=2, suite="base-a-test-cpu")

local_checkpoint = _load_module(
    "local_checkpoint", REPO_ROOT / "python/sglang/srt/weight_sync/local_checkpoint.py"
)

LAYOUT = {
    "expert.weight": ("U8", [2, 16]),
    "expert.weight_scale": ("F8_E4M3", [2, 2]),
    "expert.weight_scale_2": ("F32", []),
}


def _checkpoint_bytes(tensors):
    # Write native packed/FP8/scalar headers without converting their payloads
    # through numpy dtypes (which cannot represent FP8).
    header, payload = {}, bytearray()
    for name, (dtype, shape) in LAYOUT.items():
        begin = len(payload)
        payload.extend(tensors[name])
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [begin, len(payload)],
        }
    encoded = json.dumps(header).encode()
    encoded += b" " * (-len(encoded) % 8)
    return struct.pack("<Q", len(encoded)) + encoded + payload


def _publish(source, version, old, new, encoding="xor", bad_checksum=False):
    directory = source / f"weight_v{version:06d}"
    directory.mkdir(exist_ok=True)
    deltas, checksums = {}, {}
    for name in old:
        before = np.frombuffer(old[name], dtype=np.uint8)
        after = np.frombuffer(new[name], dtype=np.uint8)
        if np.array_equal(before, after):
            continue
        if encoding == "xor":
            payload = (before ^ after).tobytes()
        else:
            changed = np.flatnonzero(before != after).astype("<u4")
            payload = (
                struct.pack("<I", changed.size)
                + changed.tobytes()
                + after[changed].tobytes()
            )
        deltas[name] = np.frombuffer(
            zstandard.ZstdCompressor().compress(payload), dtype=np.uint8
        )
        checksums[name] = (
            "bad-checksum" if bad_checksum else xxhash.xxh3_128_hexdigest(new[name])
        )
    filename = "model-00000-of-00001.safetensors"
    if deltas:
        safetensors.numpy.save_file(deltas, directory / filename, metadata=checksums)
    (directory / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {
                    "version": str(version),
                    "base_version": str(version - 1),
                    "delta_encoding": encoding,
                    "compression_format": "zstd",
                    "checksum_format": "xxh3-128",
                },
                "weight_map": {name: filename for name in deltas},
            }
        )
    )


@pytest.fixture
def stream(tmp_path):
    base, source, local = (tmp_path / name for name in ("base", "source", "local"))
    base.mkdir()
    source.mkdir()
    initial = {
        "expert.weight": bytes(range(32)),
        "expert.weight_scale": bytes([0x30, 0x38, 0x40, 0x48]),
        "expert.weight_scale_2": struct.pack("<f", 0.125),
    }
    updated = {
        "expert.weight": bytes(reversed(range(32))),
        "expert.weight_scale": bytes([0x38, 0x40, 0x48, 0x50]),
        "expert.weight_scale_2": struct.pack("<f", 0.25),
    }
    (base / "model.safetensors").write_bytes(_checkpoint_bytes(initial))
    (base / "config.json").write_text(
        '{"quantization_config": {"quant_method": "modelopt"}}'
    )
    return base, source, local, initial, updated


def _pull(stream, version):
    base, source, local, _, _ = stream
    local_checkpoint.pull(str(local), str(base), str(source), version)


def _assert_checkpoint(stream, tensors, version, filename="model.safetensors"):
    base, _, local, _, _ = stream
    assert (local / filename).read_bytes() == _checkpoint_bytes(tensors)
    assert (local / "config.json").read_bytes() == (base / "config.json").read_bytes()
    assert local_checkpoint._read_applied_version(str(local)) == version


@pytest.mark.parametrize("encoding", ["xor", "overwrite"])
def test_quantized_delta_chain_and_repeated_pull(stream, encoding):
    _, source, _, initial, updated = stream
    _publish(source, 1, initial, updated, encoding)
    # An all-unchanged sync has no safetensors shards but still advances version.
    _publish(source, 2, updated, updated, encoding)
    _pull(stream, 2)
    _assert_checkpoint(stream, updated, 2)
    _pull(stream, 2)
    _assert_checkpoint(stream, updated, 2)


def test_noop_delta_skips_tensor_scan_and_preserves_marker_on_write_failure(
    stream, monkeypatch
):
    _, source, _, initial, _ = stream
    _pull(stream, 0)
    _publish(source, 1, initial, initial)

    def unexpected_tensor_scan(*args):
        raise AssertionError("An unchanged version must not scan checkpoint tensors")

    def failed_replace(*args):
        raise OSError("interrupted state replacement")

    monkeypatch.setattr(local_checkpoint, "_tensor_locations", unexpected_tensor_scan)
    with monkeypatch.context() as context:
        context.setattr(local_checkpoint.os, "replace", failed_replace)
        with pytest.raises(OSError, match="interrupted state replacement"):
            _pull(stream, 1)
    _assert_checkpoint(stream, initial, 0)
    _pull(stream, 1)
    _assert_checkpoint(stream, initial, 1)


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("base_version", "2", RuntimeError),
        ("compression_format", "unsupported", NotImplementedError),
        ("delta_encoding", "unsupported", NotImplementedError),
        ("checksum_format", None, KeyError),
    ],
)
def test_noop_delta_keeps_metadata_validation(stream, field, value, error):
    _, source, _, initial, _ = stream
    _pull(stream, 0)
    _publish(source, 1, initial, initial)
    index_path = source / "weight_v000001/model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    if value is None:
        del index["metadata"][field]
    else:
        index["metadata"][field] = value
    index_path.write_text(json.dumps(index))
    with pytest.raises(error):
        _pull(stream, 1)
    _assert_checkpoint(stream, initial, 0)


def test_new_trainer_stream_reseeds_reused_local_checkpoint(stream):
    _, source, _, initial, updated = stream
    _publish(source, 1, initial, updated)
    _pull(stream, 1)
    shutil.rmtree(source)
    source.mkdir()
    _pull(stream, 0)
    _assert_checkpoint(stream, initial, 0)
    next_run = {**initial, "expert.weight_scale_2": struct.pack("<f", 0.5)}
    _publish(source, 1, initial, next_run)
    _pull(stream, 1)
    _assert_checkpoint(stream, next_run, 1)


def test_stale_positive_target_does_not_revert_checkpoint(stream):
    _, source, _, initial, updated = stream
    latest = {**updated, "expert.weight_scale_2": struct.pack("<f", 0.5)}
    _publish(source, 1, initial, updated)
    _publish(source, 2, updated, latest)
    _pull(stream, 2)
    _pull(stream, 1)
    _assert_checkpoint(stream, latest, 2)


@pytest.mark.parametrize(
    "damage", ["missing_shard", "missing_tensor", "missing_index", "empty_publication"]
)
def test_incomplete_publication_preserves_checkpoint_and_version(stream, damage):
    _, source, _, initial, updated = stream
    _pull(stream, 0)
    _publish(source, 1, initial, updated)
    shard = source / "weight_v000001/model-00000-of-00001.safetensors"
    if damage in {"missing_index", "empty_publication"}:
        (shard.parent / "model.safetensors.index.json").unlink()
        if damage == "empty_publication":
            shard.unlink()
        expected_error = FileNotFoundError
        expected_message = "published weight index missing"
    elif damage == "missing_shard":
        shard.unlink()
        expected_error = FileNotFoundError
        expected_message = "model-00000-of-00001.safetensors"
    else:
        tensors = safetensors.numpy.load_file(shard)
        del tensors["expert.weight_scale"]
        safetensors.numpy.save_file(
            tensors,
            shard,
            metadata={
                name: xxhash.xxh3_128_hexdigest(updated[name]) for name in tensors
            },
        )
        expected_error = RuntimeError
        expected_message = "tensor mapping mismatch"
    with pytest.raises(expected_error, match=expected_message):
        _pull(stream, 1)
    _assert_checkpoint(stream, initial, 0)


@pytest.mark.parametrize("indexed", [False, True])
def test_full_checkpoint_publication_remains_supported(stream, indexed):
    base, source, _, _, updated = stream
    directory = source / "weight_v000001"
    directory.mkdir()
    filename = "model-00000-of-00001.safetensors" if indexed else "model.safetensors"
    (directory / filename).write_bytes(_checkpoint_bytes(updated))
    shutil.copy2(base / "config.json", directory / "config.json")
    if indexed:
        (directory / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "metadata": {},
                    "weight_map": {name: filename for name in updated},
                }
            )
        )
    _pull(stream, 1)
    _assert_checkpoint(stream, updated, 1, filename)


def test_failed_xor_apply_invalidates_state_and_retry_reseeds(stream):
    _, source, local, initial, updated = stream
    _pull(stream, 0)
    _publish(source, 1, initial, updated, bad_checksum=True)
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        _pull(stream, 1)
    assert local_checkpoint._read_applied_version(str(local)) is None
    # The failed apply already modified bytes. Reapplying XOR directly would
    # revert them; fixing the publication and retrying must rebuild the base.
    _publish(source, 1, initial, updated)
    _pull(stream, 1)
    _assert_checkpoint(stream, updated, 1)


def test_failed_seed_copy_invalidates_previous_version(stream, monkeypatch):
    _, source, local, initial, updated = stream
    _publish(source, 1, initial, updated)
    _pull(stream, 1)
    original_copy = shutil.copy2

    def failed_copy(src, dst):
        original_copy(src, dst)
        raise OSError("interrupted checkpoint copy")

    with monkeypatch.context() as context:
        context.setattr(local_checkpoint.shutil, "copy2", failed_copy)
        with pytest.raises(OSError, match="interrupted checkpoint copy"):
            _pull(stream, 0)
    assert local_checkpoint._read_applied_version(str(local)) is None
    _pull(stream, 0)
    _assert_checkpoint(stream, initial, 0)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
