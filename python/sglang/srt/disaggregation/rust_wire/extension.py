"""Load the bundled Rust disaggregation wire-codec extension or a
fingerprinted local build."""

import os
import sys

from sglang.srt.rust_extensions import load_rust_extension

_PYTHON_MODULE = "sglang.srt.disaggregation.rust_wire.disagg_wire"


def load_disagg_wire_extension():
    # build_environment REPLACES os.environ for the cargo child, so start
    # from a full copy.
    env = dict(os.environ)
    # Build against the running interpreter — pyo3 otherwise probes PATH and
    # may pick a different python (wrong ABI, missing symbols at import).
    env["PYO3_PYTHON"] = sys.executable
    if sys.platform == "darwin":
        # pyo3's extension-module build leaves Python symbols undefined;
        # macOS ld requires the explicit dynamic-lookup directive that
        # maturin adds for wheel builds. Linux shared objects resolve them
        # at load time without any flag.
        flag = "-Clink-args=-undefined dynamic_lookup"
        existing = env.get("CARGO_ENCODED_RUSTFLAGS")
        env["CARGO_ENCODED_RUSTFLAGS"] = f"{existing}\x1f{flag}" if existing else flag
    return load_rust_extension(_PYTHON_MODULE, build_environment=env)


bindings = load_disagg_wire_extension()
