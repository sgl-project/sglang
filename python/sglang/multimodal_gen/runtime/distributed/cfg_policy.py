from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req


@dataclass
class CFGBranch:
    """Immutable specification of one CFG branch forward pass.

    Built once before the denoising loop; read-only across all steps.
    """

    name: str
    is_conditional: bool
    kwargs: dict[str, Any]

    def configure_batch(self, batch: "Req") -> None:
        """Set batch state before this branch's forward pass.

        Override for richer per-branch context (e.g. a branch index instead of
        a single boolean) when a model needs more than two guidance modes.
        """
        batch.is_cfg_negative = not self.is_conditional


@dataclass
class CFGPolicy:
    """Owns the CFG branches for one generation run and combines their predictions.

    Built once before the denoising loop via ``build()``, then used read-only
    across all steps.  Subclass and override ``build()`` / ``combine()`` for
    custom CFG schemes (N-branch, multi-output, etc.).

    The default implementation handles standard 2-branch CFG.  With a single
    branch (CFG disabled) ``combine()`` returns the prediction unchanged.
    """

    branches: list[CFGBranch] = field(default_factory=list)

    def build(
        self,
        batch: "Req",
        image_kwargs: dict[str, Any],
        pos_cond_kwargs: dict[str, Any],
        neg_cond_kwargs: dict[str, Any],
    ) -> "CFGPolicy":
        """Return a new policy with branches populated.

        Called once before the denoising loop.  The returned policy is
        immutable for the lifetime of the run.  Override to declare N branches.
        """
        branches = [CFGBranch("conditional", True, {**image_kwargs, **pos_cond_kwargs})]
        if batch.do_classifier_free_guidance:
            branches.append(
                CFGBranch("unconditional", False, {**image_kwargs, **neg_cond_kwargs})
            )
        return dataclasses.replace(self, branches=branches)

    def combine(
        self,
        predictions: list[torch.Tensor | tuple[torch.Tensor, ...]],
        batch: "Req",
        cfg_scale: float,
        pipeline_config: Any,
        *,
        cfg_parallel: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """Combine branch predictions into the final noise estimate.

        Default: standard 2-branch CFG formula applied element-wise, followed
        by normalization / rescale / model-specific postprocess.
        Single-branch (CFG disabled): returns the prediction unchanged.
        Override for N-branch or multi-output models.
        """
        if len(predictions) == 1:
            return predictions[0]
        pos_t = _wrap(predictions[0])
        neg_t = _wrap(predictions[1])
        if cfg_parallel:
            # Match the old CFG-parallel calculation: multiply the positive
            # prediction by cfg_scale and the negative prediction by
            # (1 - cfg_scale) before adding them. The serial CFG formula is
            # mathematically equivalent, but bf16 rounding changes WAN outputs.
            results = [
                cfg_scale * p + (1 - cfg_scale) * n for p, n in zip(pos_t, neg_t)
            ]
        else:
            results = [n + cfg_scale * (p - n) for p, n in zip(pos_t, neg_t)]
        results[0] = _apply_cfg_postprocess(
            results[0], pos_t[0], batch, pipeline_config
        )
        return _unwrap(tuple(results))


# Helpers used by CFGPolicy and run_cfg_parallel.


def _wrap(
    pred: torch.Tensor | tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, ...]:
    return pred if isinstance(pred, tuple) else (pred,)


def _unwrap(
    pred: tuple[torch.Tensor, ...],
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    return pred[0] if len(pred) == 1 else pred


def _apply_cfg_postprocess(
    noise_pred: torch.Tensor,
    noise_pred_cond: torch.Tensor,
    batch: "Req",
    pipeline_config: Any,
) -> torch.Tensor:
    if batch.cfg_normalization and float(batch.cfg_normalization) > 0:
        noise_pred = _apply_cfg_normalization(
            noise_pred, noise_pred_cond, float(batch.cfg_normalization)
        )
    if batch.guidance_rescale > 0.0:
        noise_pred = _rescale_noise_cfg(
            noise_pred, noise_pred_cond, guidance_rescale=batch.guidance_rescale
        )
    return pipeline_config.postprocess_cfg_noise(batch, noise_pred, noise_pred_cond)


def _apply_cfg_normalization(
    noise_pred: torch.Tensor,
    noise_pred_cond: torch.Tensor,
    cfg_normalization: float,
) -> torch.Tensor:
    cond_f = noise_pred_cond.float()
    pred_f = noise_pred.float()
    ori_norm = torch.linalg.vector_norm(cond_f)
    new_norm = torch.linalg.vector_norm(pred_f)
    max_norm = ori_norm * cfg_normalization
    if new_norm > max_norm:
        noise_pred = noise_pred * (max_norm / new_norm)
    return noise_pred


def _rescale_noise_cfg(
    noise_cfg: torch.Tensor,
    noise_pred_text: torch.Tensor,
    guidance_rescale: float = 0.0,
) -> torch.Tensor:
    std_text = noise_pred_text.std(
        dim=list(range(1, noise_pred_text.ndim)), keepdim=True
    )
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    return guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
