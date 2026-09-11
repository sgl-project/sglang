"""CPU unit tests for the artifact store layout and atomic writes (design
section 11). Everything runs in ``tmp_path``; no GPU, no torch."""

import hashlib
import sys

import msgspec
import pytest

from sglang.srt.model_executor.graph_serialization.fingerprint import (
    GeometryFingerprint,
    GraphArtifactFingerprint,
    ParallelFingerprint,
    fingerprint_digest,
)
from sglang.srt.model_executor.graph_serialization.format import (
    FORMAT_VERSION,
    ArtifactManifest,
    CommStateBlob,
    KernelIdentity,
    KernelImage,
    KernelNode,
    OutputSchema,
    PointerSlot,
    RankManifest,
    RegionKind,
    RegionRef,
    RegionSpec,
    RunnerBundle,
    SerializedGraph,
    ShapeArtifact,
    ShapeKeyRecord,
)
from sglang.srt.model_executor.graph_serialization.store import (
    BUNDLE_SUFFIX,
    DIAGNOSTICS_FILE,
    MANIFEST_FILE,
    RANK_MANIFEST_FILE,
    GraphArtifactStore,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _fingerprint():
    return GraphArtifactFingerprint(
        parallel=ParallelFingerprint(tp_size=2, tp_rank=1),
        geometry=(GeometryFingerprint(runner="decode", capture_sizes=(1, 2, 4)),),
        weight_layout_digest="w" * 8,
    )


def _artifact(size, label_extra=None):
    node = KernelNode(
        identity=0,
        grid=(4, 1, 1),
        block=(128, 1, 1),
        smem=0,
        launch_form="kernel_params",
        params_off=0,
        params_len=16,
        launch_attrs=((4, b"\x02\x00\x00\x00"),),
    )
    graph = SerializedGraph(
        nodes=(node,),
        edges=(),
        param_bytes=b"\x00" * 16,
        slots=(
            PointerSlot(
                node=0, param=0, byte_offset=8, ref=RegionRef("kv:k_buffer:0", 4096)
            ),
        ),
        signature=f"sig-{size}",
    )
    return ShapeArtifact(
        shape_key=ShapeKeyRecord(size=size, variant_label=label_extra),
        backend="full",
        graphs=(graph,),
        output=OutputSchema(
            kind="tensor",
            tensor=RegionRef("pool:seg:0", 0),
            shape=(size, 4096),
            stride=(4096, 1),
            dtype="bfloat16",
        ),
        kernels=(KernelIdentity(name="k", image_sha256="a" * 64),),
    )


def _bundle(runner="decode", *sizes):
    return RunnerBundle(
        runner=runner,
        backend="full",
        shapes=tuple(_artifact(s) for s in (sizes or (1, 2))),
        comm=(
            CommStateBlob(
                impl="ca_v2",
                group="tp",
                rows=((3, RegionRef("static:x", 0), 4096),),
                max_row=3,
            ),
        ),
    )


def _rank_manifest(**overrides):
    fields = dict(
        fingerprint=_fingerprint(),
        regions=(
            RegionSpec(
                region_id="kv:k_buffer:0",
                kind=RegionKind.KV.value,
                nbytes=1 << 20,
                base_at_save=0x7F00_0000_0000,
            ),
        ),
        placement="relocate",
        runners=("decode",),
        images=(
            KernelImage(
                image_sha256="a" * 64,
                provider="elf_section",
                source="libsgl_kernel.so",
                kernel_names=("k",),
                nbytes=1234,
            ),
        ),
        verdicts={"size=1": "serializable", "size=2": "needs_recapture"},
    )
    fields.update(overrides)
    return RankManifest(**fields)


def _no_staging_left(path):
    return not any(p.name.startswith(".tmp-") for p in path.rglob("*"))


# -- construction --------------------------------------------------------------


def test_constructor_has_no_side_effects(tmp_path):
    root = tmp_path / "cuda_graphs"
    digest = fingerprint_digest(_fingerprint())
    store = GraphArtifactStore(root, digest)

    assert not root.exists()
    assert store.root == root
    assert store.digest == digest
    assert store.artifact_dir == root / digest
    assert store.images_dir == root / digest / "images"
    assert store.manifest_path == root / digest / MANIFEST_FILE
    assert store.rank_dir(3) == root / digest / "rank_3"
    assert not store.has_rank(0)
    assert not store.has_manifest()
    assert not root.exists()


def test_constructor_accepts_placeholder_digest_and_rejects_paths(tmp_path):
    assert GraphArtifactStore(str(tmp_path), "pending").digest == "pending"
    with pytest.raises(ValueError):
        GraphArtifactStore(tmp_path, "../escape")
    with pytest.raises(ValueError):
        GraphArtifactStore(tmp_path, "")


# -- rank round trip -----------------------------------------------------------


def test_write_rank_read_rank_round_trip(tmp_path):
    store = GraphArtifactStore(tmp_path / "cg", "d1")
    manifest = _rank_manifest()
    bundles = {"decode": _bundle("decode", 1, 2)}
    diagnostics = {"graphs_by_verdict": {"serializable": 1}, "harvest_forwards": 0}

    assert not store.has_rank(1)
    rank_dir = store.write_rank(
        1, manifest=manifest, bundles=bundles, diagnostics=diagnostics
    )

    assert rank_dir == store.rank_dir(1)
    assert store.has_rank(1)
    assert not store.has_rank(0)
    assert sorted(p.name for p in rank_dir.iterdir()) == [
        f"decode{BUNDLE_SUFFIX}",
        DIAGNOSTICS_FILE,
        RANK_MANIFEST_FILE,
    ]
    assert _no_staging_left(tmp_path)

    read_manifest, read_bundles = store.read_rank(1)
    assert read_manifest == manifest
    assert read_manifest.fingerprint == _fingerprint()
    assert fingerprint_digest(read_manifest.fingerprint) == fingerprint_digest(
        _fingerprint()
    )
    assert read_bundles == bundles
    assert read_bundles["decode"].shapes[0].shape_key.to_shape_key().size == 1

    # rank.json is human-readable JSON and diagnostics.json holds the mapping.
    text = (rank_dir / RANK_MANIFEST_FILE).read_text()
    assert '\n  "format_version": 1' in text
    assert msgspec.json.decode((rank_dir / DIAGNOSTICS_FILE).read_bytes()) == (
        diagnostics
    )


def test_write_rank_fills_runners_and_rejects_mismatch(tmp_path):
    store = GraphArtifactStore(tmp_path, "d")
    bundles = {"prefill": _bundle("prefill", 64), "decode": _bundle("decode", 1)}

    store.write_rank(0, manifest=_rank_manifest(runners=()), bundles=bundles)
    manifest, read = store.read_rank(0)
    assert manifest.runners == ("decode", "prefill")
    assert set(read) == {"decode", "prefill"}

    with pytest.raises(ValueError, match="does not match"):
        store.write_rank(
            0, manifest=_rank_manifest(runners=("decode",)), bundles=bundles
        )
    with pytest.raises(ValueError, match="bare file-name component"):
        store.write_rank(
            0, manifest=_rank_manifest(runners=()), bundles={"../x": _bundle("x")}
        )
    # The failed writes left the earlier rank intact and no staging behind.
    assert store.has_rank(0)
    assert _no_staging_left(tmp_path)


def test_write_rank_replaces_a_stale_rank_wholesale(tmp_path):
    store = GraphArtifactStore(tmp_path, "d")
    store.write_rank(
        2,
        manifest=_rank_manifest(runners=("decode", "prefill")),
        bundles={"decode": _bundle("decode", 1), "prefill": _bundle("prefill", 64)},
        diagnostics={"old": True},
    )
    store.write_rank(
        2, manifest=_rank_manifest(runners=("decode",)), bundles={"decode": _bundle()}
    )
    manifest, bundles = store.read_rank(2)
    assert manifest.runners == ("decode",)
    assert set(bundles) == {"decode"}
    assert not (store.rank_dir(2) / f"prefill{BUNDLE_SUFFIX}").exists()
    assert not (store.rank_dir(2) / DIAGNOSTICS_FILE).exists()
    assert _no_staging_left(tmp_path)


def test_read_rank_rejects_other_format_version(tmp_path):
    store = GraphArtifactStore(tmp_path, "d")
    newer = _rank_manifest(format_version=FORMAT_VERSION + 1)
    store.write_rank(0, manifest=newer, bundles={"decode": _bundle()})

    with pytest.raises(ValueError) as info:
        store.read_rank(0)
    assert f"format_version {FORMAT_VERSION + 1}" in str(info.value)
    assert f"format_version {FORMAT_VERSION}" in str(info.value)
    # Existence is unaffected by readability.
    assert store.has_rank(0)


def test_read_rank_missing(tmp_path):
    store = GraphArtifactStore(tmp_path, "d")
    with pytest.raises(FileNotFoundError):
        store.read_rank(0)
    store.write_rank(0, manifest=_rank_manifest(), bundles={"decode": _bundle()})
    (store.rank_dir(0) / f"decode{BUNDLE_SUFFIX}").unlink()
    with pytest.raises(FileNotFoundError, match="decode"):
        store.read_rank(0)


# -- top-level manifest --------------------------------------------------------


def test_manifest_round_trip(tmp_path):
    store = GraphArtifactStore(tmp_path / "cg", "d")
    manifest = ArtifactManifest(
        fingerprint_digest="d",
        world_size=2,
        ranks=(0, 1),
        placement="relocate",
        sglang_version="0.5.0",
    )
    assert not store.has_manifest()
    path = store.write_manifest(manifest)
    assert path == store.manifest_path
    assert store.has_manifest()
    assert store.read_manifest() == manifest
    assert _no_staging_left(tmp_path)

    store.write_manifest(
        msgspec.structs.replace(manifest, format_version=FORMAT_VERSION + 5)
    )
    with pytest.raises(ValueError, match=f"format_version {FORMAT_VERSION + 5}"):
        store.read_manifest()


def test_read_manifest_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        GraphArtifactStore(tmp_path, "d").read_manifest()


# -- kernel images -------------------------------------------------------------


def test_image_put_get_and_corruption(tmp_path):
    store = GraphArtifactStore(tmp_path / "cg", "d")
    data = b"\x7fELF" + bytes(range(64))
    sha = hashlib.sha256(data).hexdigest()

    path = store.put_image(sha, data)
    assert path == store.images_dir / f"{sha}.img"
    assert path.read_bytes() == data
    assert store.get_image(sha) == data
    assert _no_staging_left(tmp_path)

    # Idempotent for identical bytes.
    assert store.put_image(sha, data) == path

    # A mislabelled put is refused before anything is written.
    other = hashlib.sha256(b"other").hexdigest()
    with pytest.raises(ValueError, match="not the sha256"):
        store.put_image(other, data)
    assert not (store.images_dir / f"{other}.img").exists()
    with pytest.raises(ValueError):
        store.put_image("../x", data)

    # Corruption on disk is detected on read.
    path.write_bytes(data[:-1] + b"\x00")
    with pytest.raises(ValueError, match="corrupt"):
        store.get_image(sha)
    with pytest.raises(FileNotFoundError):
        store.get_image("f" * 64)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
