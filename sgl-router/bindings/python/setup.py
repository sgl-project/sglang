import os
import warnings

from setuptools import setup

with_rust = os.environ.get("SGLANG_ROUTER_BUILD_WITH_RUST", None)
with_rust = with_rust is None or (not with_rust.lower() in ["0", "false", "no"])

rust_extensions = []
if with_rust:
    from setuptools_rust import Binding, RustExtension

    rust_extensions.append(
        RustExtension(
            target="sglang_router_rs",
            path="Cargo.toml",
            binding=Binding.PyO3,
        )
    )
else:
    warnings.warn(
        "Building 'sglang-router' without Rust support. Performance may be degraded."
    )

setup(
    rust_extensions=rust_extensions,
    zip_safe=False,
)
