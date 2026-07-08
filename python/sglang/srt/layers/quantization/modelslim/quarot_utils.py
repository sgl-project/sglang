"""QuaRot anti-rotation utilities for Eagle3 speculative decoding.

When the target (main) model is quantized with QuaRot, hidden states between
layers are in a rotated space (h @ Q).  Eagle3 captures those states and feeds
them to the draft model whose ``fc`` layer was trained on original-space hidden
states, so we must undo the rotation at load time.
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional

import torch
from safetensors.torch import load_file

logger = logging.getLogger(__name__)


def _get_quarot_rotation_path(model_path: str) -> Optional[str]:
    """Read ``quant_model_description.json`` and return the QuaRot rotation matrix
    file path, or ``None`` when the model is not a QuaRot model.
    """
    quant_desc_path = os.path.join(model_path, "quant_model_description.json")
    if not os.path.isfile(quant_desc_path):
        return None

    with open(quant_desc_path) as f:
        quant_desc = json.load(f)

    try:
        rotation_relpath = quant_desc["optional"]["quarot"]["rotation_map"][
            "global_rotation"
        ]
    except KeyError:
        return None

    return str(Path(model_path) / rotation_relpath)


def _load_quarot_rotation_matrix(rotation_path: str) -> torch.Tensor:
    """Load the global rotation matrix Q from a safetensors file."""
    return load_file(rotation_path)["global_rotation"]


def maybe_apply_quarot_anti_rotation(model_path: str, draft_model) -> bool:
    """If the target model is QuaRot-quantized, apply inverse rotation to the
    draft model's ``fc`` and ``embed_tokens`` weights.

    Returns ``True`` when the anti-rotation was applied, ``False`` otherwise.
    """
    rotation_path = _get_quarot_rotation_path(model_path)
    if rotation_path is None:
        return False

    draft_model_inner = draft_model.model
    num_aux = draft_model_inner.num_aux_hidden_states
    device = draft_model_inner.fc.weight.device
    Q = _load_quarot_rotation_matrix(rotation_path).to(
        dtype=torch.float32, device=device
    )

    # -- fc anti-rotation ---------------------------------------------------
    # fc maps concat(rotated hiddens) -> hidden_size.
    # Q_k = block_diag(Q, ..., Q)  (num_aux copies along the diagonal).
    # W_fc' = W_fc @ Q_k  so that  fc'(h_rot) = fc(h_orig).
    Q_k = torch.block_diag(*[Q] * num_aux)
    fc = draft_model_inner.fc
    fc_weight = fc.weight.data.to(torch.float32) @ Q_k
    fc.weight.data.copy_(fc_weight.to(fc.weight.dtype))

    # -- embed_tokens anti-rotation -----------------------------------------
    # set_embed() is a no-op when the draft and target hidden sizes differ
    # (the draft keeps its own un-rotated embedding).  Only anti-rotate when
    # the embedding was actually shared from the QuaRot target model.
    config = draft_model.config
    if not (
        hasattr(config, "target_hidden_size")
        and config.target_hidden_size != config.hidden_size
    ):
        # After init_lm_head(), draft and target share the SAME embed_tokens
        # tensor (set_embed assigns target's Parameter directly).  We must
        # clone before rotating so the target model is not affected.
        embed = draft_model_inner.embed_tokens
        embed_weight = embed.weight.data.clone().to(torch.float32) @ Q.T
        embed.weight = torch.nn.Parameter(
            embed_weight.to(embed.weight.dtype), requires_grad=False
        )

    logger.info(
        "Applied QuaRot anti-rotation to draft model: " "fc (Q <%s>, num_aux=%d).",
        "×".join(str(d) for d in Q.shape),
        num_aux,
    )
    return True
