#!/usr/bin/env python3
"""
Compile protobuf files for SGLang gRPC server.

This script compiles .proto files to Python code using grpc_tools.protoc.
It generates:
- *_pb2.py (protobuf message classes)
- *_pb2_grpc.py (gRPC service classes)
- *_pb2.pyi (type hints for mypy/IDEs)

Usage:
    python compile_proto.py [--check] [--proto-file PROTO_FILE]

Options:
    --check         Check if regeneration is needed (exit 1 if needed)
    --proto-file    Specify proto file (default: sglang_scheduler.proto)

### Install Dependencies
pip install "grpcio==1.75.1" "grpcio-tools==1.75.1"

### Run Script
cd python/sglang/srt/grpc
python compile_proto.py
"""


import argparse
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

GRPC_VERSION = "1.75.1"


def get_file_mtime(path: Path) -> float:
    """Get file modification time, return 0 if file doesn't exist."""
    try:
        return path.stat().st_mtime
    except FileNotFoundError:
        return 0.0


def check_regeneration_needed(proto_file: Path, output_dir: Path) -> bool:
    """Check if proto files are newer than generated files."""
    proto_mtime = get_file_mtime(proto_file)

    generated_files = [
        output_dir / f"{proto_file.stem}_pb2.py",
        output_dir / f"{proto_file.stem}_pb2_grpc.py",
        output_dir / f"{proto_file.stem}_pb2.pyi",
    ]

    for gen_file in generated_files:
        if get_file_mtime(gen_file) < proto_mtime:
            return True

    return False


def compile_proto(proto_file: Path, output_dir: Path, verbose: bool = True) -> bool:
    """Compile the protobuf file to Python."""

    if not proto_file.exists():
        print(f"Error: Proto file not found: {proto_file}")
        return False

    if verbose:
        print(f"Found proto file: {proto_file}")

    # Check if grpc_tools is available
    try:
        import grpc_tools.protoc  # noqa: F401
    except ImportError:
        print("Error: grpcio-tools not installed")
        print(
            f'Install with: pip install "grpcio-tools=={GRPC_VERSION}" "grpcio=={GRPC_VERSION}"'
        )
        return False

    grpc_tools_version = version("grpcio-tools")
    grpc_version = version("grpcio")
    if grpc_tools_version != GRPC_VERSION or grpc_version != GRPC_VERSION:
        raise RuntimeError(
            f"Error: grpcio-tools version {grpc_tools_version} and grpcio version {grpc_version} detected, but {GRPC_VERSION} is required."
        )

    # Compile command
    cmd = [
        sys.executable,
        "-m",
        "grpc_tools.protoc",
        f"-I{proto_file.parent}",
        f"--python_out={output_dir}",
        f"--grpc_python_out={output_dir}",
        f"--pyi_out={output_dir}",  # Generate type stubs
        str(proto_file.name),
    ]

    if verbose:
        print(f"Running: {' '.join(cmd)}")

    # Run protoc
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=proto_file.parent)

    if result.returncode != 0:
        print(f"Error compiling proto:")
        print(result.stderr)
        if result.stdout:
            print(result.stdout)
        return False

    # Verify generated files exist
    generated_files = [
        f"{proto_file.stem}_pb2.py",
        f"{proto_file.stem}_pb2_grpc.py",
        f"{proto_file.stem}_pb2.pyi",
    ]

    missing_files = []
    for gen_file in generated_files:
        if not (output_dir / gen_file).exists():
            missing_files.append(gen_file)

    if missing_files:
        print(f"Error: Expected generated files not found: {missing_files}")
        return False

    if verbose:
        print("Successfully compiled protobuf files:")
        for gen_file in generated_files:
            print(f"  - {output_dir}/{gen_file}")

    # Fix imports in generated files
    fix_imports(output_dir, proto_file.stem, verbose)

    return True


def fix_imports(output_dir: Path, proto_stem: str, verbose: bool = True) -> None:
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
            if verbose:
                print("Fixed imports in generated files")


def add_generation_header(output_dir: Path, proto_stem: str) -> None:
    """Add header to generated files indicating they are auto-generated."""
    header = """# This file is auto-generated. Do not edit manually.
# Regenerate with: python compile_proto.py

"""

    files_to_update = [f"{proto_stem}_pb2.py", f"{proto_stem}_pb2_grpc.py"]

    for filename in files_to_update:
        file_path = output_dir / filename
        if file_path.exists():
            content = file_path.read_text()
            if not content.startswith("# This file is auto-generated"):
                file_path.write_text(header + content)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Compile protobuf files for SGLang gRPC server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--check",
        action="store_true",
        help="Check if regeneration is needed (exit 1 if needed)",
    )

    parser.add_argument(
        "--proto-file",
        type=str,
        default="sglang_scheduler.proto",
        help="Proto file to compile (default: sglang_scheduler.proto)",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=True,
        help="Verbose output (default: True)",
    )

    parser.add_argument(
        "-q", "--quiet", action="store_true", help="Quiet mode (overrides verbose)"
    )

    args = parser.parse_args()

    # Handle verbosity
    verbose = args.verbose and not args.quiet

    # Get paths
    script_dir = Path(__file__).parent
    proto_file = script_dir / args.proto_file
    output_dir = script_dir

    # Check mode
    if args.check:
        if check_regeneration_needed(proto_file, output_dir):
            if verbose:
                print("Proto files need regeneration")
            sys.exit(1)
        else:
            if verbose:
                print("Generated files are up to date")
            sys.exit(0)

    # Compile mode
    success = compile_proto(proto_file, output_dir, verbose)

    if success:
        # Add generation headers
        add_generation_header(output_dir, proto_file.stem)

        if verbose:
            print("\n✅ Protobuf compilation successful!")
            print("Generated files are ready for use")
    else:
        if verbose:
            print("\n❌ Protobuf compilation failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
