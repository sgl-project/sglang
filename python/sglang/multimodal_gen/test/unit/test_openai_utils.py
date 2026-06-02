# SPDX-License-Identifier: Apache-2.0

import asyncio
import io

from starlette.datastructures import UploadFile as StarletteUploadFile

from sglang.multimodal_gen.runtime.entrypoints.openai.utils import (
    _save_upload_to_path,
)


def test_save_upload_to_path_accepts_starlette_upload_file(tmp_path):
    upload = StarletteUploadFile(
        io.BytesIO(b"image-bytes"),
        filename="input.png",
    )
    target_path = tmp_path / "input.png"

    saved_path = asyncio.run(_save_upload_to_path(upload, str(target_path)))

    assert saved_path == str(target_path)
    assert target_path.read_bytes() == b"image-bytes"
