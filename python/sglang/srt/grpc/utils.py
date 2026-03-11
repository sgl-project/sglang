"""gRPC utility functions."""

from http import HTTPStatus

import grpc

_HTTP_TO_GRPC_CODE = {
    HTTPStatus.BAD_REQUEST: grpc.StatusCode.INVALID_ARGUMENT,
    HTTPStatus.SERVICE_UNAVAILABLE: grpc.StatusCode.UNAVAILABLE,
    HTTPStatus.INTERNAL_SERVER_ERROR: grpc.StatusCode.INTERNAL,
}


def abort_code_from_output(output: dict) -> grpc.StatusCode:
    """Map a scheduler error output to the appropriate gRPC status code."""
    finish_reason = output.get("meta_info", {}).get("finish_reason")
    if isinstance(finish_reason, dict):
        status_code = finish_reason.get("status_code")
        if status_code is not None:
            return _HTTP_TO_GRPC_CODE.get(status_code, grpc.StatusCode.INTERNAL)
    return grpc.StatusCode.INTERNAL
