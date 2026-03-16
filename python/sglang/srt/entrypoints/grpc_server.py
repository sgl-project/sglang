"""
Thin gRPC server wrapper — delegates to smg-grpc-servicer package.
"""


async def serve_grpc(server_args, model_info=None):
    """Start the standalone gRPC server with integrated scheduler."""
    try:
        from smg_grpc_servicer.sglang.server import serve_grpc as _serve_grpc
    except ImportError:
        raise ImportError(
            "gRPC mode requires the smg-grpc-servicer package. "
            "Install it with: pip install smg-grpc-servicer[sglang]"
        ) from None
    await _serve_grpc(server_args, model_info)
