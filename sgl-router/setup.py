import os

from setuptools import setup

no_rust = os.environ.get("SGLANG_ROUTER_BUILD_NO_RUST") == "1"

# Define dependencies
install_requires = [
    "setproctitle",
    "aiohttp",
    "orjson",
    "uvicorn",
    "fastapi",
]

# Add kubernetes dependency for non-Rust builds (Python-only mini_lb)
if no_rust:
    install_requires.append("kubernetes>=18.0.0")

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
    name="sglang-router",
    version="0.1.9",
    description="SGLang Router - High-performance request distribution across worker nodes",
    author="SGLang Team",
    python_requires=">=3.8",
    install_requires=install_requires,
    rust_extensions=rust_extensions,
    zip_safe=False,
)
