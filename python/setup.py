"""
Custom setup.py for SGLang that compiles protobuf files during build.

This file works alongside pyproject.toml. It hooks into the build process
to automatically generate gRPC/protobuf Python files from .proto sources
when building the wheel or doing editable installs.
"""

import subprocess
import sys
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

    # Build the protoc command
    cmd = [
        sys.executable,
        "-m",
        "grpc_tools.protoc",
        f"-I{proto_dir}",
        f"--python_out={output_dir}",
        f"--grpc_python_out={output_dir}",
        f"--pyi_out={output_dir}",
        proto_path.name,
    ]

    print(f"Running: {' '.join(cmd)}")

    try:
        subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=proto_dir,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr or e.stdout or "Unknown error"
        raise SetupError(f"protoc failed with exit code {e.returncode}: {error_msg}")

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
