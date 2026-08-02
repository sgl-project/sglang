# SPDX-License-Identifier: Apache-2.0
"""Pipeline config for the Boogu-Image edit (reference-image / TI2I) path.

Additive on top of the text-to-image :class:`BooguImagePipelineConfig`: the DiT,
VAE, encoder, scheduler, tokenizer, and instruction conditioning are all reused
unchanged. The only edit-specific pieces are

* a dual (text + image) classifier-free-guidance scheme, expressed as an
  N-branch :class:`BooguImageEditCFGPolicy`; and
* neutralizing ``InputValidationStage``'s condition-image resizing so the raw
  reference PIL reaches :class:`BooguImageEditEncodingStage`, which performs the
  upstream dual-resolution preprocessing (384px VLM copy, 2048px VAE copy) and
  the ``align_res`` output-size override itself.

The text-to-image pipeline (PR #33182) is untouched.
"""

import dataclasses
from dataclasses import dataclass, field
from typing import Any

import torch

from sglang.multimodal_gen.configs.pipeline_configs.base import ModelTaskType
from sglang.multimodal_gen.configs.pipeline_configs.boogu_image import (
    BooguImagePipelineConfig,
)
from sglang.multimodal_gen.runtime.distributed.cfg_policy import (
    CFGBranch,
    CFGPolicy,
    _apply_cfg_postprocess,
    _unwrap,
    _wrap,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req


class BooguImageEditCFGPolicy(CFGPolicy):
    """Dual (text + image) classifier-free guidance for Boogu-Image edit.

    Reproduces the branch priority of the upstream ``processing`` loop. Let
    ``c`` be the conditional (positive instruction + reference), ``dt`` the
    text-dropped prediction (negative instruction, reference kept), and ``da``
    the fully-dropped prediction (negative instruction, no reference). With text
    guidance ``s_t`` and image guidance ``s_i``:

    * double  (``s_t > 1`` and ``s_i > 1``, reference present): 3 branches
      ``c + (s_t-1)(c-dt) + (s_i-1)(dt-da)``
    * text-only (``s_t > 1``): 2 branches ``c + (s_t-1)(c-dt)``
    * image-only (``s_i > 1``, ``s_t <= 1``): 2 branches ``c + (s_i-1)(c-di)``
      where ``di`` drops only the reference.
    * neither: single branch, returned unchanged.

    Text guidance arrives as the ``cfg_scale`` argument to :meth:`combine`
    (Boogu's single transformer always uses ``batch.guidance_scale``); image
    guidance is read from ``batch.guidance_scale_2``.
    """

    def build(
        self,
        batch: Req,
        image_kwargs: dict[str, Any],
        pos_cond_kwargs: dict[str, Any],
        neg_cond_kwargs: dict[str, Any],
    ) -> "BooguImageEditCFGPolicy":
        ref = batch.ref_image_hidden_states
        text_on = batch.do_classifier_free_guidance and batch.guidance_scale > 1.0
        image_on = ref is not None and batch.guidance_scale_2 > 1.0

        cond = {**image_kwargs, **pos_cond_kwargs, "ref_image_hidden_states": ref}
        branches = [CFGBranch("cond_ref", True, cond)]

        if text_on and image_on:
            branches.append(
                CFGBranch(
                    "drop_text",
                    False,
                    {**image_kwargs, **neg_cond_kwargs, "ref_image_hidden_states": ref},
                )
            )
            branches.append(
                CFGBranch(
                    "drop_all",
                    False,
                    {
                        **image_kwargs,
                        **neg_cond_kwargs,
                        "ref_image_hidden_states": None,
                    },
                )
            )
        elif text_on:
            branches.append(
                CFGBranch(
                    "drop_text",
                    False,
                    {**image_kwargs, **neg_cond_kwargs, "ref_image_hidden_states": ref},
                )
            )
        elif image_on:
            branches.append(
                CFGBranch(
                    "drop_image",
                    False,
                    {
                        **image_kwargs,
                        **pos_cond_kwargs,
                        "ref_image_hidden_states": None,
                    },
                )
            )
        return dataclasses.replace(self, branches=branches)

    def combine(
        self,
        predictions: list[torch.Tensor | tuple[torch.Tensor, ...]],
        batch: Req,
        cfg_scale: float,
        pipeline_config: Any,
        *,
        cfg_parallel: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        if len(predictions) == 1:
            return predictions[0]

        text_gs = cfg_scale
        image_gs = batch.guidance_scale_2
        cond = _wrap(predictions[0])

        if len(predictions) == 3:
            drop_text = _wrap(predictions[1])
            drop_all = _wrap(predictions[2])
            results = [
                c + (text_gs - 1.0) * (c - dt) + (image_gs - 1.0) * (dt - da)
                for c, dt, da in zip(cond, drop_text, drop_all)
            ]
        else:
            uncond = _wrap(predictions[1])
            # Exactly one of text-only / image-only produced this 2-branch set.
            scale = text_gs if text_gs > 1.0 else image_gs
            results = [c + (scale - 1.0) * (c - u) for c, u in zip(cond, uncond)]

        results[0] = _apply_cfg_postprocess(results[0], cond[0], batch, pipeline_config)
        return _unwrap(tuple(results))


@dataclass
class BooguImageEditPipelineConfig(BooguImagePipelineConfig):
    """Edit-path config; inherits all Boogu-Image T2I structure and callbacks."""

    task_type: ModelTaskType = ModelTaskType.TI2I

    cfg_policy: CFGPolicy = field(default_factory=BooguImageEditCFGPolicy)

    def calculate_condition_image_size(self, image, width, height):
        # Skip InputValidationStage resizing; the reference PIL is preprocessed
        # (dual-resolution) inside BooguImageEditEncodingStage instead.
        return None

    def prepare_calculated_size(self, image):
        # Do not derive the output size here; align_res is applied in the edit
        # encoding stage from the VAE-preprocessed reference dimensions.
        return None
