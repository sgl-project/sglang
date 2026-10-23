"""Regression test for sglang #43409: /v1/images/edits silently ignores ``mask``.

The edits endpoint declares ``mask`` for OpenAI API compatibility, but the
multimodal pipelines never consume a mask image. An uploaded mask therefore
used to produce a successful response that looked as if the mask had been
applied. It must now fail loudly with 501 instead.
"""

from io import BytesIO

import pytest
from fastapi import HTTPException, UploadFile

from sglang.multimodal_gen.runtime.entrypoints.openai.image_api import (
    _reject_unsupported_mask,
)


def _upload_file(filename: str = "mask.png") -> UploadFile:
    return UploadFile(filename=filename, file=BytesIO(b"fake-png-bytes"))


def test_mask_is_rejected_with_501():
    with pytest.raises(HTTPException) as exc_info:
        _reject_unsupported_mask(_upload_file())
    assert exc_info.value.status_code == 501
    assert "mask" in exc_info.value.detail.lower()


def test_uploaded_mask_filename_is_not_special_cased():
    # The rejection must not depend on the file name, only on a mask being sent.
    with pytest.raises(HTTPException) as exc_info:
        _reject_unsupported_mask(_upload_file(filename="anything.png"))
    assert exc_info.value.status_code == 501


def test_no_mask_passes_through():
    # The guard must be a no-op when no mask is uploaded (the common case).
    assert _reject_unsupported_mask(None) is None
