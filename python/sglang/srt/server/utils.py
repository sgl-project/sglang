from http import HTTPStatus

from fastapi.responses import ORJSONResponse


def _create_error_response(e):
    return ORJSONResponse(
        {"error": {"message": str(e)}}, status_code=HTTPStatus.BAD_REQUEST
    )

