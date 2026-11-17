import hashlib
import json
import os
import sys

import pytest

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, stage="extra-a", runner_config="1-gpu-small")

packages = pytest.importorskip("flashinfer.trace_apply.packages")

from sglang import Engine  # noqa: E402
from sglang.srt import runtime_context as rc  # noqa: E402
from sglang.srt.kernel_plugins import flashinfer_packages  # noqa: E402
from sglang.srt.server_args import ServerArgs  # noqa: E402

SITE = "sglang.kernels.ops.mamba.triton_ops.ssd_chunk_scan._chunk_scan_fwd"
# A pass-through Solution with a setup step, fitting one-chunk prefills (K == 1). It
# records the process it ran in and FlashInfer's counts there.
KERNEL = """\
import json, os


def setup(context):
    return None


def run(state, **kwargs):
    from flashinfer.trace_apply import stats
    from sglang.kernels.ops.mamba.triton_ops import ssd_chunk_scan

    with open(os.environ["KERNEL_PACKAGE_PROBE"], "w") as f:
        json.dump({"pid": os.getpid(), "stats": stats()}, f)
    return ssd_chunk_scan._chunk_scan_fwd.__wrapped__(**kwargs)
"""
SHAPES = {"cb": "BKGSS", "x": "BLHP", "dt": "BHKS", "dA_cumsum": "BHKS", "C": "BLGN"}
SHAPES |= {"states": "BKHPN", "D": "H", "seq_idx": "BL", "out": "BLHP"}
MAMBA2 = {
    "architectures": ["Mamba2ForCausalLM"],
    "model_type": "mamba2",
    "hidden_size": 256,
    "intermediate_size": 512,
    "rms_norm": True,
    "num_hidden_layers": 1,
    "num_heads": 8,
    "head_dim": 64,
    "n_groups": 1,
    "state_size": 64,
    "vocab_size": 1024,
    "torch_dtype": "bfloat16",
}


def seal(root):
    definition = {
        "name": "chunk_scan",
        "op_type": "mamba",
        "tags": [f"fi_api:{SITE}"],
        "axes": {"K": {"type": "const", "value": 1}},
        "inputs": {name: {"shape": list(axes)} for name, axes in SHAPES.items()},
        "outputs": {},
    }
    solution = {
        "name": "pass_through",
        "definition": "chunk_scan",
        "author": "test",
        "spec": {
            "language": "python",
            "target_hardware": ["any"],
            "entry_point": "kernel.py::run",
            "setup_entry_point": "kernel.py::setup",
            "destination_passing_style": True,
        },
        "sources": [{"path": "kernel.py", "content": KERNEL}],
    }
    files = {"definition.json": definition, "solution.json": solution}
    files = {name: json.dumps(doc).encode() for name, doc in files.items()}
    files["kernel.py"] = KERNEL.encode()
    digests = {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}
    listing = "".join(f"{name}\0{d}\n" for name, d in sorted(digests.items()))
    bundle_sha256 = hashlib.sha256(listing.encode()).hexdigest()
    manifest = json.dumps({"files": digests, "bundle_sha256": bundle_sha256}).encode()
    digest = hashlib.sha256(manifest).hexdigest()
    (root / digest / "bundle").mkdir(parents=True)
    (root / digest / "manifest.json").write_bytes(manifest)
    for name, data in files.items():
        (root / digest / "bundle" / name).write_bytes(data)
    return digest


@pytest.fixture
def package(tmp_path, monkeypatch):
    root = tmp_path / "packages"
    digest = seal(root)
    monkeypatch.setenv("FLASHINFER_KERNEL_PACKAGE_ROOT", str(root))
    monkeypatch.setenv("FLASHINFER_KERNEL_PACKAGES", digest)
    monkeypatch.setenv("KERNEL_PACKAGE_PROBE", str(tmp_path / "probe.json"))
    return root, digest


def test_package_serves_chunk_scan_in_scheduler(package, tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps(MAMBA2))
    engine = Engine(
        model_path=str(model),
        load_format="dummy",
        skip_tokenizer_init=True,
        disable_cuda_graph=True,
        mem_fraction_static=0.3,
    )
    try:
        sampling = {"max_new_tokens": 1}
        engine.generate(
            input_ids=list(range(300)), sampling_params=sampling
        )  # 2 chunks
        engine.generate(input_ids=list(range(500, 508)), sampling_params=sampling)
        updated = engine.update_weights_from_disk(str(model))
    finally:
        engine.shutdown()
    served = json.loads((tmp_path / "probe.json").read_text())  # written by a hit
    assert served["pid"] != os.getpid()
    assert served["stats"][SITE]["fallback_axis_K"] >= 1
    assert not updated[0] and "online weight updates would rewrite" in updated[1]


def test_setup_package_refuses_memory_saver(package):
    root, digest = package
    packages.load_packages(root, [digest])  # FlashInfer was imported before the env
    rc.reset_context()
    rc.publish(ServerArgs(model_path="dummy", enable_memory_saver=True), role="test")
    try:
        with pytest.raises(ValueError, match="--enable-memory-saver would rewrite"):
            flashinfer_packages.activate(rc.get_exec())
    finally:
        rc.reset_context()
        packages.release_packages()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
