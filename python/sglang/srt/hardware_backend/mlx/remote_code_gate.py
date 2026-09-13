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
"""Pre-execution gate for checkpoint-shipped model code on the MLX backend.

mlx-lm's loader executes ``config.json``'s ``model_file`` unconditionally:
``mlx_lm.utils.load_model`` imports that Python file straight out of the
checkpoint directory, and ``mlx_lm.load()`` exposes no ``trust_remote_code``
parameter to refuse it. The gate therefore lives on the SGLang side:

1. Resolve the model path (local directory or HF repo id + revision) to a
   local directory exactly once, with mlx-lm's own resolver.
2. Inspect THAT directory's ``config.json``. If it declares ``model_file``
   and the server was not started with ``--trust-remote-code``, refuse
   before any checkpoint Python can execute.
3. Hand the same resolved directory to ``mlx_lm.load`` (for which an
   existing local directory is a no-op resolution), so the inspected and
   executed snapshots cannot diverge.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional


class RemoteCodeGateError(RuntimeError):
    """A checkpoint failed the remote-code gate (refusal or bad metadata)."""


def resolve_model_directory(model_path: str, revision: Optional[str] = None) -> Path:
    """Resolve a model path or HF repo id to a local snapshot directory.

    Uses mlx-lm's resolver so the directory is byte-identical to what a
    direct ``mlx_lm.load`` call would consume; existing local paths are
    returned as-is (no network access). mlx-lm 0.31.x exposes this as
    ``mlx_lm.utils._download`` (formerly ``get_model_path``); mlx-lm is
    unpinned, so accept either name.
    """
    from mlx_lm import utils as mlx_lm_utils

    resolver = getattr(mlx_lm_utils, "_download", None) or getattr(
        mlx_lm_utils, "get_model_path", None
    )
    if resolver is None:
        raise RemoteCodeGateError(
            "this mlx-lm exposes neither mlx_lm.utils._download nor "
            "mlx_lm.utils.get_model_path, so the checkpoint directory cannot "
            "be resolved for inspection before mlx-lm loads it"
        )
    resolved = resolver(model_path, revision=revision)
    # get_model_path returned (path, config) in some releases.
    if isinstance(resolved, tuple):
        resolved = resolved[0]
    return Path(resolved)


def ensure_remote_code_allowed(model_dir: Path, trust_remote_code: bool) -> None:
    """Refuse ``model_file`` checkpoints unless remote code is trusted.

    Must be called with the SAME resolved directory that is subsequently
    passed to ``mlx_lm.load``. Raises :class:`RemoteCodeGateError` before
    any checkpoint Python executes when the checkpoint declares
    ``model_file`` without ``--trust-remote-code``, when its config is
    unreadable, or when the ``model_file`` value is malformed.
    """
    config_path = model_dir / "config.json"
    try:
        config = json.loads(config_path.read_text())
    except FileNotFoundError:
        raise RemoteCodeGateError(
            f"no config.json in resolved model directory {model_dir}; "
            "not a loadable MLX checkpoint"
        ) from None
    except json.JSONDecodeError as e:
        raise RemoteCodeGateError(
            f"config.json in {model_dir} is not valid JSON ({e}); refusing "
            "to load a checkpoint whose metadata cannot be inspected"
        ) from None
    if not isinstance(config, dict):
        raise RemoteCodeGateError(
            f"config.json in {model_dir} must contain a JSON object, "
            f"found {type(config).__name__}"
        )

    model_file = config.get("model_file")
    if model_file is None:
        return

    if not isinstance(model_file, str) or not model_file:
        raise RemoteCodeGateError(
            f"config.json in {model_dir} has a non-string or empty "
            f"model_file entry ({model_file!r})"
        )
    candidate = Path(model_file)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise RemoteCodeGateError(
            f"model_file {model_file!r} in {model_dir} must be a relative "
            "path inside the checkpoint directory (no absolute paths, no "
            "'..' traversal)"
        )
    if not (model_dir / candidate).is_file():
        raise RemoteCodeGateError(
            f"config.json in {model_dir} declares model_file "
            f"{model_file!r} but that file does not exist in the "
            "checkpoint directory"
        )

    if not trust_remote_code:
        raise RemoteCodeGateError(
            f"checkpoint {model_dir} ships custom model code "
            f"(model_file={model_file!r} in config.json), which mlx-lm "
            "would execute at load time. Refusing to load it: restart the "
            "server with --trust-remote-code if you trust this checkpoint."
        )
