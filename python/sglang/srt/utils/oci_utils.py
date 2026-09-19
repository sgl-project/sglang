# SPDX-License-Identifier: Apache-2.0
"""Resolve `oci://` model references to a local path.

Models published as CNCF ModelPack (https://github.com/modelpack/model-spec)
OCI artifacts live in ordinary container registries, so they reuse the registry,
credentials, mirroring and air-gap tooling a deployment already has for
container images.

Acquisition is delegated to a running `llmman serve`
(https://github.com/llmmanorg/llmman), which already implements the ModelPack
media types, registry auth, resumable blob download and a content-addressed
store. The daemon does the pull (POST /api/pull, streamed so a multi-gigabyte
fetch is not silent) but deliberately exposes no local path, so
`llmman resolve --no-pull` reports where the bytes landed. That directory is
handed to the ordinary HuggingFace loading path.

An explicit `oci://` scheme is required rather than sniffing a bare
`registry/name:tag`: that shape is indistinguishable from a HuggingFace repo id
(`org/model`), so guessing would silently hijack existing `--model-path
org/model` deployments.
"""

from __future__ import annotations

import logging
from pathlib import Path

from sglang.srt.utils import llmman

logger = logging.getLogger(__name__)

SUPPORTED_SCHEMES = ["oci://"]


def is_oci_uri(model_or_path: str | Path | None) -> bool:
    """True if the reference carries the `oci://` scheme.

    Cast to str to handle pathlib.Path inputs, mirroring is_runai_obj_uri.
    """
    if model_or_path is None:
        return False
    return str(model_or_path).lower().startswith(tuple(SUPPORTED_SCHEMES))


def strip_oci_scheme(reference: str | Path) -> str:
    """Drop the `oci://` prefix, leaving the reference llmman understands."""
    text = str(reference)
    if is_oci_uri(text):
        return text[len(SUPPORTED_SCHEMES[0]) :]
    return text


def resolve_oci_model(reference: str | Path) -> str:
    """Pull an `oci://` reference through llmman and return the local path."""
    bare = strip_oci_scheme(reference)
    if not bare.strip():
        raise ValueError(f"empty OCI model reference: '{reference}'")

    def _progress(status, completed, total):
        if total:
            logger.info("llmman: %s (%s/%s bytes)", status, completed, total)
        else:
            logger.info("llmman: %s", status)

    return llmman.pull_and_resolve(bare.strip(), progress=_progress)
