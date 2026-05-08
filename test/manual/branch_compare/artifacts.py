"""Read/write helpers for branch_compare artifacts.

Layout under <artifact-dir>:
    meta.json        - global metadata + per-prompt index
    prompt_<i>.pt    - per-request torch tensors (output_ids + top-K)

The per-request file format is also what the server's
`forced_token_ids_path` loader reads on the verify phase: it accepts a
dict with an "output_ids" key (1-D int tensor) or a bare 1-D int tensor.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

import torch


def prompt_filename(idx: int) -> str:
    return f"prompt_{idx}.pt"


def write_meta(artifact_dir: str, meta: Dict[str, Any]) -> str:
    os.makedirs(artifact_dir, exist_ok=True)
    path = os.path.join(artifact_dir, "meta.json")
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
    return path


def read_meta(artifact_dir: str) -> Dict[str, Any]:
    with open(os.path.join(artifact_dir, "meta.json")) as f:
        return json.load(f)


def write_prompt_artifact(
    artifact_dir: str,
    idx: int,
    output_ids: List[int],
    top_k_token_ids: List[List[int]],
    top_k_logprobs: List[List[float]],
    chosen_logprob: List[float],
) -> str:
    """Write a per-request torch artifact.

    Tensors:
      output_ids:        int64 [N]      (also serves as forced_token_ids on verify)
      top_k_token_ids:   int32 [N, K]
      top_k_logprobs:    bf16  [N, K]
      chosen_logprob:    bf16  [N]
    """
    n = len(output_ids)
    if n == 0:
        k = 0
    else:
        k = len(top_k_token_ids[0]) if top_k_token_ids else 0

    out = {
        "output_ids": torch.tensor(output_ids, dtype=torch.int64),
        "top_k_token_ids": (
            torch.tensor(top_k_token_ids, dtype=torch.int32)
            if k > 0
            else torch.zeros((n, 0), dtype=torch.int32)
        ),
        "top_k_logprobs": (
            torch.tensor(top_k_logprobs, dtype=torch.float32).to(torch.bfloat16)
            if k > 0
            else torch.zeros((n, 0), dtype=torch.bfloat16)
        ),
        "chosen_logprob": torch.tensor(chosen_logprob, dtype=torch.float32).to(
            torch.bfloat16
        ),
    }
    path = os.path.join(artifact_dir, prompt_filename(idx))
    torch.save(out, path)
    return path


def read_prompt_artifact(artifact_dir: str, idx: int) -> Dict[str, torch.Tensor]:
    path = os.path.join(artifact_dir, prompt_filename(idx))
    return torch.load(path, map_location="cpu", weights_only=False)


def absolute_logprobs_path(record_dir: str, idx: int) -> str:
    """Return the absolute path the verify phase passes as
    sampling_params.forced_token_ids_path. The server reads this file at
    request admission and pulls out the `output_ids` 1-D tensor.
    """
    return os.path.abspath(os.path.join(record_dir, prompt_filename(idx)))
