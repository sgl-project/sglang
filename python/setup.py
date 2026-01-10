"""
Custom setup.py for SGLang that compiles protobuf files during build.

This file works alongside pyproject.toml. It hooks into the build process
to automatically generate gRPC/protobuf Python files from .proto sources
when building the wheel or doing editable installs.
"""

import os
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py
from setuptools.command.develop import develop
from setuptools.command.egg_info import egg_info
from setuptools.errors import SetupError

PROTO_SOURCE = "sglang/srt/grpc/sglang_scheduler.proto"


def compile_proto():
    """Compile the protobuf file to Python using grpc_tools.protoc."""
    proto_path = Path(__file__).parent / PROTO_SOURCE

    if not proto_path.exists():
        print(f"Warning: Proto file not found at {proto_path}, skipping generation")
        return

    print(f"Generating gRPC files from {PROTO_SOURCE}")

    output_dir = proto_path.parent
    proto_dir = proto_path.parent

    # Import grpc_tools.protoc directly instead of running as subprocess.
    # This ensures we use the grpcio-tools installed in the build environment,
    # since sys.executable may point to the main Python interpreter in
    # pip's isolated build environments.
    try:
        import grpc_tools
        from grpc_tools import protoc
    except ImportError as e:
        raise SetupError(
            f"Failed to import grpc_tools: {e}. "
            "Ensure grpcio-tools is listed in build-system.requires in pyproject.toml"
        )

    # Get the path to well-known proto files bundled with grpcio-tools
    # (e.g., google/protobuf/timestamp.proto, google/protobuf/struct.proto)
    grpc_tools_proto_path = Path(grpc_tools.__file__).parent / "_proto"

    # Build the protoc arguments (protoc.main expects argv-style list)
    args = [
        "protoc",  # argv[0] is the program name
        f"-I{proto_dir}",
        f"-I{grpc_tools_proto_path}",  # Include path for well-known protos
        f"--python_out={output_dir}",
        f"--grpc_python_out={output_dir}",
        f"--pyi_out={output_dir}",
        str(proto_dir / proto_path.name),
    ]

    print(f"Running protoc with args: {args[1:]}")

    # Save and restore cwd since protoc may change it
    original_cwd = os.getcwd()
    try:
        result = protoc.main(args)
        if result != 0:
            raise SetupError(f"protoc failed with exit code {result}")
    finally:
        os.chdir(original_cwd)

    # Fix imports in generated grpc file (change absolute to relative imports)
    _fix_imports(output_dir, proto_path.stem)

    print(f"Successfully generated gRPC files in {output_dir}")


def _fix_imports(output_dir: Path, proto_stem: str):
    """Fix imports in generated files to use relative imports."""
    grpc_file = output_dir / f"{proto_stem}_pb2_grpc.py"

    if grpc_file.exists():
        content = grpc_file.read_text()
        # Change absolute import to relative import
        old_import = f"import {proto_stem}_pb2"
        new_import = f"from . import {proto_stem}_pb2"

        if old_import in content:
            content = content.replace(old_import, new_import)
            grpc_file.write_text(content)
            print("Fixed imports in generated gRPC file")


class BuildPyWithProto(build_py):
    """Build Python modules, generating gRPC files from .proto sources first."""

    def run(self):
        compile_proto()
        super().run()


class DevelopWithProto(develop):
    """Editable install with gRPC file generation."""

    def run(self):
        compile_proto()
        super().run()


class EggInfoWithProto(egg_info):
    """Egg info generation with gRPC file generation."""

    def run(self):
        compile_proto()
        super().run()


setup(
    cmdclass={
        "build_py": BuildPyWithProto,
        "develop": DevelopWithProto,
        "egg_info": EggInfoWithProto,
    },
)
