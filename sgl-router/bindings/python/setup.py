import os

from setuptools import setup

no_rust = os.environ.get("SGLANG_ROUTER_BUILD_NO_RUST") == "1"

rust_extensions = []
if not no_rust:
    from setuptools_rust import Binding, RustExtension

    rust_extensions.append(
        RustExtension(
            target="sglang_router_rs",
            path="Cargo.toml",
            binding=Binding.PyO3,
        )
    )

setup(
    rust_extensions=rust_extensions,
    zip_safe=False,
)
