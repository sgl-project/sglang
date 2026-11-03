# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Serialize a single inference result into the requested embedding formats."""

import base64
import struct
from collections.abc import Sequence
from typing import Optional

import numpy as np

from sglang.srt.entrypoints.cohere.protocol import (
    CohereBilledUnits,
    CohereEmbedByTypeEmbeddings,
    CohereEmbedRequest,
    CohereEmbedResponse,
    CohereMeta,
)


def build_typed_embeddings(
    float_embeddings: list[list[float]], embedding_types: Sequence[str]
) -> CohereEmbedByTypeEmbeddings:
    result = CohereEmbedByTypeEmbeddings()
    # Compute the bit packing once even when both signed forms are requested.
    packed = None
    for embedding_type in embedding_types:
        if embedding_type == "float":
            result.float = float_embeddings
        elif embedding_type == "base64":
            result.base64 = [
                base64.b64encode(struct.pack(f"<{len(row)}f", *row)).decode("ascii")
                for row in float_embeddings
            ]
        elif embedding_type in ("binary", "ubinary"):
            if not float_embeddings:
                setattr(result, embedding_type, [])
                continue
            if packed is None:
                array = np.asarray(float_embeddings, dtype=np.float64)
                if array.ndim != 2:
                    raise ValueError("Expected a 2D batch of embeddings")
                if array.shape[1] % 8:
                    raise ValueError(
                        "Embedding dimension must be a multiple of 8 for binary "
                        f"embedding types, but got {array.shape[1]}"
                    )
                # Zero (including -0.0) maps to 1; the first component is the MSB.
                packed = np.packbits(array >= 0, axis=-1, bitorder="big")
            values = (
                packed.astype(np.int16) - 128 if embedding_type == "binary" else packed
            )
            setattr(result, embedding_type, values.tolist())
        else:
            raise ValueError(f"Unsupported embedding type: {embedding_type}")
    return result


def build_embed_response(
    request: CohereEmbedRequest,
    results: list[dict],
    request_id: Optional[str] = None,
) -> CohereEmbedResponse:
    embeddings = build_typed_embeddings(
        [item["embedding"] for item in results], request.embedding_types or ["float"]
    )
    token_count = sum(item["meta_info"]["prompt_tokens"] for item in results)
    # Match vLLM v0.32.0: any image in the request puts ALL tokens in image_tokens.
    # This is request-level accounting, not an exact count of vision tokens.
    billed_units = CohereBilledUnits(
        input_tokens=0 if request.has_images else token_count,
        image_tokens=token_count if request.has_images else 0,
    )
    kwargs = {"id": request_id} if request_id is not None else {}
    return CohereEmbedResponse(
        embeddings=embeddings,
        texts=request.texts,
        meta=CohereMeta(billed_units=billed_units),
        **kwargs,
    )
