import hashlib
import json
import os
import sys

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from sglang import Engine
from sglang.srt import runtime_context as rc
from sglang.srt.kernel_plugins import flashinfer_packages
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, stage="extra-a", runner_config="1-gpu-small")

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


def seal(root, kernel=KERNEL, chunks=1):
    definition = {
        "name": "chunk_scan",
        "op_type": "mamba",
        "tags": [f"fi_api:{SITE}"],
        "axes": {"K": {"type": "const", "value": chunks}},
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
        "sources": [{"path": "kernel.py", "content": kernel}],
    }
    files = {"definition.json": definition, "solution.json": solution}
    files = {name: json.dumps(doc).encode() for name, doc in files.items()}
    files["kernel.py"] = kernel.encode()
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


class Doubler:
    @flashinfer_packages.kernel_site
    def forward(self, x, k=None, scale=2.0):
        return x * scale


def test_site_fit_rules(monkeypatch):
    stats = pytest.importorskip("flashinfer.trace_apply").stats
    site = f"{__name__}.Doubler.forward"
    inputs = {"x": {"shape": ["T", "H"]}, "k": {"optional": True}, "scale": {}}
    pinned = {"inputs": inputs, "axes": {"T": {"type": "const", "value": 2}}}
    routes = [
        (pinned, lambda x, k, scale: x + 1),
        ({"inputs": inputs}, lambda x, k, scale: x - 1),
    ]
    monkeypatch.setitem(flashinfer_packages._routes, site, routes)
    forward = Doubler().forward

    # A method, a parameter left at its default and an optional input at None fit.
    assert forward(torch.zeros(2, 4)).sum() == 8
    assert forward(torch.zeros(3, 4)).sum() == -12  # T is not 2: the next package
    assert forward(torch.ones(4)).sum() == 8  # no package fits: the stock function
    counts = stats()[site]
    assert counts == {"hit": 2, "fallback_rank_x": 1}

    # A compiler tracing the call: a pinned size is not matched against a symbolic
    # one, the package that fits is compiled in, and no request is counted.
    trace = make_fx(lambda x: forward(x), tracing_mode="symbolic")
    traced = trace(torch.zeros(2, 4))
    assert "sub" in traced.code and "add" not in traced.code
    assert stats()[site] == counts


@pytest.fixture
def package(tmp_path, monkeypatch):
    # Skipped here, not at import: CI runs this file as a script, where a
    # module-level skip is an uncaught exception.
    packages = pytest.importorskip("flashinfer.trace_apply.packages")
    if not hasattr(packages, "installed_packages"):
        pytest.skip("needs FlashInfer's installed_packages()")
    root = tmp_path / "packages"
    # A package that fails to import, selected first under the same names: its
    # Definition (K == 3) must not be routed to the next package's kernel.
    digests = [seal(root, "raise RuntimeError\n", 3), seal(root)]
    monkeypatch.setenv("FLASHINFER_KERNEL_PACKAGE_ROOT", str(root))
    monkeypatch.setenv("FLASHINFER_KERNEL_PACKAGES", ",".join(digests))
    monkeypatch.setenv("KERNEL_PACKAGE_PROBE", str(tmp_path / "probe.json"))
    return packages, root, digests


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
    packages, root, digests = package
    # FlashInfer was imported before the env; load as its import does.
    packages.load_packages(root, digests, strict=False)
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
