"""Test-only clients compiled from the runtime schema, without Dynamo dependencies."""

import importlib.util
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

from google.protobuf import descriptor_pb2, descriptor_pool, message_factory


def runtime_proto():
    import grpc_tools
    from grpc_tools import protoc

    root = Path(__file__).resolve().parents[3]
    with tempfile.TemporaryDirectory() as directory:
        descriptor = Path(directory) / "runtime.pb"
        result = protoc.main(
            [
                "protoc",
                f"-I{root / 'proto'}",
                f"-I{Path(grpc_tools.__file__).parent / '_proto'}",
                "--include_imports",
                f"--descriptor_set_out={descriptor}",
                str(root / "proto/sglang/runtime/v1/sglang.proto"),
            ]
        )
        if result:
            raise RuntimeError(f"runtime protobuf compilation failed: {result}")
        files = descriptor_pb2.FileDescriptorSet.FromString(descriptor.read_bytes())
    pool = descriptor_pool.DescriptorPool()
    for file in files.file:
        pool.Add(file)
    service = pool.FindServiceByName("sglang.runtime.v1.SglangService")
    types = {
        message.name: message_factory.GetMessageClass(message)
        for message in service.file.message_types_by_name.values()
    }
    return SimpleNamespace(**types), service


def runtime_client(channel, service):
    methods = {}
    for method in service.methods:
        request = message_factory.GetMessageClass(method.input_type)
        response = message_factory.GetMessageClass(method.output_type)
        factory = (
            channel.unary_stream if method.server_streaming else channel.unary_unary
        )
        methods[method.name] = factory(
            f"/{service.full_name}/{method.name}",
            request_serializer=request.SerializeToString,
            response_deserializer=response.FromString,
        )
    return SimpleNamespace(**methods)


def load_test_grpc_extension():
    """Permit testing a specific cargo-built artifact without installing a wheel."""
    module_name = "sglang.srt.rust_extensions._grpc"
    if artifact := os.environ.get("SGLANG_TEST_GRPC_EXTENSION"):
        spec = importlib.util.spec_from_file_location(module_name, artifact)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    from sglang.srt.rust_extensions import load_rust_extension

    return load_rust_extension(module_name, mode="never")
