# SPDX-License-Identifier: Apache-2.0

from typing import Any

from sglang.multimodal_gen.configs.sample.sensenova_u1 import (
    SenseNovaU1PixelFlowCFG,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.sensenova_u1_prepare import (
    SenseNovaU1PixelFlowForwardContext,
    SenseNovaU1PixelFlowPrepared,
)


class SenseNovaU1PixelFlowDenoiser:
    def __init__(
        self,
        model: Any,
        *,
        forward_batch_provider: Any,
    ) -> None:
        self.model = model
        self.forward_batch_provider = forward_batch_provider

    def forward(self, prepared: SenseNovaU1PixelFlowPrepared) -> Any:
        import torch

        image_prediction = prepared.image_prediction

        for step_i in range(prepared.steps):
            timestep = prepared.timesteps[step_i]
            next_timestep = prepared.timesteps[step_i + 1]
            z = _patchify(
                image_prediction,
                prepared.patch_size * prepared.merge_size,
            )
            image_input = _patchify(
                image_prediction,
                prepared.patch_size,
                channel_first=True,
            )
            image_embeds = self.model.extract_feature(
                image_input.view(prepared.grid_h * prepared.grid_w, -1),
                gen_model=True,
                grid_hw=prepared.gen_grid_hw,
            ).view(1, prepared.token_h * prepared.token_w, -1)
            timestep_values = timestep.expand(prepared.token_h * prepared.token_w)
            timestep_embeddings = self.model.fm_modules["timestep_embedder"](
                timestep_values
            ).view(1, prepared.token_h * prepared.token_w, -1)
            if getattr(self.model.config, "add_noise_scale_embedding", False):
                noise_values = torch.full_like(
                    timestep_values,
                    prepared.noise_scale
                    / float(self.model.config.noise_scale_max_value),
                )
                timestep_embeddings = timestep_embeddings + self.model.fm_modules[
                    "noise_scale_embedder"
                ](noise_values).view(1, prepared.token_h * prepared.token_w, -1)
            image_embeds = image_embeds + timestep_embeddings

            v_condition = self._predict_v(
                forward_context=prepared.condition,
                image_embeds=image_embeds,
                timestep=timestep,
                z=z,
            )
            use_cfg = _should_apply_cfg(prepared.cfg, timestep)
            v_pred = self._combine_cfg_velocity(
                prepared=prepared,
                image_embeds=image_embeds,
                timestep=timestep,
                z=z,
                v_condition=v_condition,
                use_cfg=use_cfg,
            )
            if prepared.cfg.needs_cfg and use_cfg:
                v_pred = self._apply_cfg_renorm(
                    v_condition=v_condition,
                    v_pred=v_pred,
                    cfg=prepared.cfg,
                )

            z = z + (next_timestep - timestep) * v_pred
            image_prediction = _unpatchify(
                z,
                prepared.patch_size * prepared.merge_size,
                prepared.height,
                prepared.width,
            )
        return image_prediction

    def _combine_cfg_velocity(
        self,
        *,
        prepared: SenseNovaU1PixelFlowPrepared,
        image_embeds: Any,
        timestep: Any,
        z: Any,
        v_condition: Any,
        use_cfg: bool,
    ) -> Any:
        cfg = prepared.cfg
        if not use_cfg or not cfg.needs_cfg:
            return v_condition
        if cfg.img_scale == 1.0:
            v_img_condition = self._predict_v(
                forward_context=_require_forward_context(prepared.img_condition),
                image_embeds=image_embeds,
                timestep=timestep,
                z=z,
            )
            return v_img_condition + cfg.text_scale * (v_condition - v_img_condition)
        if cfg.text_scale == cfg.img_scale:
            v_uncondition = self._predict_v(
                forward_context=_require_forward_context(prepared.uncondition),
                image_embeds=image_embeds,
                timestep=timestep,
                z=z,
            )
            return v_uncondition + cfg.text_scale * (v_condition - v_uncondition)

        v_img_condition = self._predict_v(
            forward_context=_require_forward_context(prepared.img_condition),
            image_embeds=image_embeds,
            timestep=timestep,
            z=z,
        )
        v_uncondition = self._predict_v(
            forward_context=_require_forward_context(prepared.uncondition),
            image_embeds=image_embeds,
            timestep=timestep,
            z=z,
        )
        return (
            v_uncondition
            + cfg.text_scale * (v_condition - v_img_condition)
            + cfg.img_scale * (v_img_condition - v_uncondition)
        )

    def _predict_v(
        self,
        *,
        forward_context: SenseNovaU1PixelFlowForwardContext,
        image_embeds: Any,
        timestep: Any,
        z: Any,
    ) -> Any:
        forward_batch_context = self.forward_batch_provider(
            prepared=forward_context.prepared,
            g_query_embeds=image_embeds,
            timestep=timestep,
        )
        forward_batch = getattr(
            forward_batch_context,
            "forward_batch",
            forward_batch_context,
        )
        try:
            return _predict_pixel_flow_from_srt(
                self.model,
                image_embeds=image_embeds,
                indexes_image=forward_context.indexes_image,
                forward_batch=forward_batch,
                timestep=timestep,
                z=z,
            )
        finally:
            release = getattr(forward_batch_context, "release", None)
            if callable(release):
                release()

    @staticmethod
    def _apply_cfg_renorm(
        *,
        v_condition: Any,
        v_pred: Any,
        cfg: SenseNovaU1PixelFlowCFG,
    ) -> Any:
        cfg_renorm_type = cfg.renorm_type
        if cfg_renorm_type == "none":
            return v_pred
        if cfg_renorm_type == "global":
            norm_v_condition = v_condition.norm(dim=(1, 2), keepdim=True)
            norm_v_cfg = v_pred.norm(dim=(1, 2), keepdim=True)
        elif cfg_renorm_type == "channel":
            norm_v_condition = v_condition.norm(dim=-1, keepdim=True)
            norm_v_cfg = v_pred.norm(dim=-1, keepdim=True)
        else:
            raise ValueError(
                "Unsupported SenseNova U1 pixel-flow CFG renorm type: "
                f"{cfg_renorm_type}"
            )
        scale = (norm_v_condition / (norm_v_cfg + 1e-8)).clamp(
            min=cfg.renorm_min, max=1.0
        )
        return v_pred * scale


def _should_apply_cfg(cfg: SenseNovaU1PixelFlowCFG, timestep: Any) -> bool:
    return (float(timestep) > cfg.start and float(timestep) < cfg.end) or (
        cfg.start == 0.0
    )


def _require_forward_context(
    context: SenseNovaU1PixelFlowForwardContext | None,
) -> SenseNovaU1PixelFlowForwardContext:
    if context is None:
        raise RuntimeError("SenseNova U1 pixel-flow CFG forward context is missing")
    return context


def _patchify(images: Any, patch_size: int, *, channel_first: bool = False) -> Any:
    import torch

    h, w = images.shape[2] // patch_size, images.shape[3] // patch_size
    x = images.reshape(images.shape[0], 3, h, patch_size, w, patch_size)
    if channel_first:
        x = torch.einsum("nchpwq->nhwcpq", x)
    else:
        x = torch.einsum("nchpwq->nhwpqc", x)
    return x.reshape(images.shape[0], h * w, patch_size**2 * 3)


def _unpatchify(
    x: Any,
    patch_size: int,
    h: int | None = None,
    w: int | None = None,
) -> Any:
    import torch

    if h is None or w is None:
        h = w = int(x.shape[1] ** 0.5)
    else:
        h = h // patch_size
        w = w // patch_size
    x = x.reshape(x.shape[0], h, w, patch_size, patch_size, 3)
    x = torch.einsum("nhwpqc->nchpwq", x)
    return x.reshape(x.shape[0], 3, h * patch_size, w * patch_size)


def _predict_pixel_flow_from_srt(
    model: Any,
    *,
    image_embeds: Any,
    indexes_image: Any,
    forward_batch: Any,
    timestep: Any,
    z: Any,
) -> Any:
    batch_size, image_token_num = image_embeds.shape[:2]
    hidden_states = model.language_model.forward_u1_gen_embeds(
        input_embeds=image_embeds.reshape(-1, image_embeds.shape[-1]),
        positions=indexes_image,
        forward_batch=forward_batch,
    ).view(batch_size, image_token_num, -1)
    x_pred = model.fm_modules["fm_head"](hidden_states).view(
        batch_size,
        image_token_num,
        -1,
    )

    t = timestep.to(device=z.device, dtype=z.dtype)
    return (x_pred - z) / (1 - t).clamp_min(float(getattr(model.config, "t_eps", 0.02)))
