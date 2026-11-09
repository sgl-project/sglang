# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import torch

from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.managers.memory_managers.component_manager import (
    ComponentUse,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.condition_encoding import (
    ConditionEncodingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.constants import (
    MINIMAX_H3_KEYFRAME_COND_ROWS_EXTRA_KEY,
    MINIMAX_H3_PREPARED_REFERENCE_VIDEO_EXTRA_KEY,
    MINIMAX_H3_REFERENCE_IMAGE_ROWS_EXTRA_KEY,
    MINIMAX_H3_REFERENCE_VIDEO_ROWS_EXTRA_KEY,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.task_profiles import (
    MINIMAX_H3_FL2VA_KEYFRAME_SIGNATURES,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs


class MiniMaxH3VisualEncodingStage(ConditionEncodingStage):
    deduplicated_extra_output_keys = (
        MINIMAX_H3_KEYFRAME_COND_ROWS_EXTRA_KEY,
        MINIMAX_H3_REFERENCE_IMAGE_ROWS_EXTRA_KEY,
        MINIMAX_H3_REFERENCE_VIDEO_ROWS_EXTRA_KEY,
    )

    def __init__(
        self,
        video_vae,
        vae_arch_config,
    ) -> None:
        super().__init__()
        self.video_vae = video_vae
        self.vae_arch_config = vae_arch_config

    @property
    def role_affinity(self) -> RoleType:
        return RoleType.ENCODER

    def component_uses(
        self, server_args: ServerArgs, stage_name: str | None = None
    ) -> list[ComponentUse]:
        stage_name = self._component_stage_name(stage_name)
        return [ComponentUse(stage_name, "video_vae")]

    def build_dedup_fingerprint(self, batch: Req, server_args: ServerArgs):
        parent_request_id = batch.extra.get("parent_request_id")
        return (
            ("expanded_outputs", parent_request_id)
            if parent_request_id is not None
            else id(batch)
        )

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.material_io import (
            minimax_h3_cleanup_temp_dirs,
        )

        try:
            return self._forward(batch, server_args)
        except Exception:
            # No later stage will run after an encoder failure.
            minimax_h3_cleanup_temp_dirs(batch)
            raise
        finally:
            # Text and visual encoding are the only full-RGB consumers. Release
            # the shared mapping on both success and failure before later stages
            # can retain a several-hundred-MiB request-local view.
            batch.extra.pop(MINIMAX_H3_PREPARED_REFERENCE_VIDEO_EXTRA_KEY, None)

    def _forward(self, batch: Req, server_args: ServerArgs) -> Req:
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.resolved_plan import (
            minimax_h3_plan_from_batch,
        )

        plan = minimax_h3_plan_from_batch(batch)
        if plan is not None:
            routed = plan.encoders.get("visual")
            if not routed:
                return batch
            from .replica_broadcast import (
                minimax_h3_replica_broadcast_error,
                minimax_h3_replica_broadcast_extra,
                minimax_h3_replica_ctx,
            )

            output_keys = self._visual_output_keys(plan, routed)
            replica_world, replica_rank = minimax_h3_replica_ctx()
            parallel_encode = replica_world > 1 and bool(self.video_vae.parallel_tiling)
            owner_exception = None
            owner_error = None
            if (parallel_encode or replica_rank == 0) and any(
                key not in batch.extra for key in output_keys
            ):
                try:
                    with self.use_declared_component(
                        component_name="video_vae",
                        module=self.video_vae,
                    ) as video_vae:
                        assert video_vae is not None
                        self.video_vae = video_vae
                        self._encode_keyframes_from_plan(batch, plan, routed)
                except Exception as exc:
                    owner_exception = exc
                    owner_error = f"{type(exc).__name__}: {exc}"
            if parallel_encode:
                if owner_exception is not None:
                    raise owner_exception
            else:
                owner_error = minimax_h3_replica_broadcast_error(owner_error)
                if owner_error is not None:
                    if owner_exception is not None:
                        raise owner_exception
                    raise RuntimeError(
                        f"MiniMax H3 visual encode failed on rank 0: {owner_error}"
                    )
                for key in output_keys:
                    minimax_h3_replica_broadcast_extra(batch, key)
            return batch
        if (
            batch.sampling_params is not None
            and batch.sampling_params.image_path is not None
        ):
            raise NotImplementedError(
                "MiniMaxH3VisualEncodingStage direct visual tokenizer encode "
                "requires a canonical minimax_h3 request (resolved plan); "
                "legacy image_path-only requests are unsupported."
            )
        return batch

    @staticmethod
    def _visual_output_keys(plan, routed) -> tuple[str, ...]:
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.constants import (
            MINIMAX_H3_REFERENCE_IMAGE_ROWS_EXTRA_KEY,
            MINIMAX_H3_REFERENCE_VIDEO_ROWS_EXTRA_KEY,
        )

        routed_set = set(routed)
        chains = {
            material.material_chain
            for material in plan.materials
            if material.condition_index in routed_set
        }
        keys = []
        if "image.target_canvas" in chains:
            keys.append(MINIMAX_H3_KEYFRAME_COND_ROWS_EXTRA_KEY)
        if "image.reference_preserve" in chains:
            keys.append(MINIMAX_H3_REFERENCE_IMAGE_ROWS_EXTRA_KEY)
        if chains & {
            "video.reference_preserve",
            "video_audio.reference_preserve",
        }:
            keys.append(MINIMAX_H3_REFERENCE_VIDEO_ROWS_EXTRA_KEY)
        if not keys:
            raise ValueError("MiniMax H3 visual routing selected no visual materials")
        return tuple(keys)

    def _encode_keyframes_from_plan(self, batch: Req, plan, routed) -> None:
        """Direct keyframe encode: seeded sampled encode_images ->
        normalized [n,96] cond rows in batch.extra."""
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.keyframe_encoding import (
            minimax_h3_scoped_encode_fp32,
        )

        materials = [m for m in plan.materials if m.condition_index in set(routed)]
        chains = {m.material_chain for m in materials}
        keyframe_materials = [
            material
            for material in materials
            if material.material_chain == "image.target_canvas"
        ]
        if keyframe_materials and str(plan.task) in {"fl2va", "ref2va"}:
            frame_indices = tuple(
                material.frame_index for material in keyframe_materials
            )
            if frame_indices not in MINIMAX_H3_FL2VA_KEYFRAME_SIGNATURES:
                raise ValueError(
                    "MiniMax H3 visual encoding requires an ordered keyframe signature "
                    f"in {MINIMAX_H3_FL2VA_KEYFRAME_SIGNATURES!r}, got "
                    f"{frame_indices!r}"
                )
        elif keyframe_materials:
            raise ValueError(
                f"task {plan.task!r} cannot carry image.target_canvas materials"
            )
        video_chains = {
            "video.reference_preserve",
            "video_audio.reference_preserve",
        }
        supported_chains = {
            "image.target_canvas",
            "image.reference_preserve",
            *video_chains,
        }
        unsupported = sorted(chains - supported_chains)
        if unsupported:
            raise NotImplementedError(
                "MiniMaxH3VisualEncodingStage cannot encode material chains "
                f"{unsupported}"
            )
        # One VAE dtype toggle for every visual condition in the request.
        with minimax_h3_scoped_encode_fp32(self.video_vae):
            if keyframe_materials:
                self._encode_target_keyframes(batch, plan)
            if "image.reference_preserve" in chains:
                self._encode_reference_image(batch, plan)
            if chains & video_chains:
                self._encode_reference_video(batch, plan)

    def _encode_target_keyframes(self, batch: Req, plan) -> None:
        if MINIMAX_H3_KEYFRAME_COND_ROWS_EXTRA_KEY in batch.extra:
            return
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.canvas import (
            minimax_h3_prepared_keyframes,
        )
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.keyframe_encoding import (
            minimax_h3_encode_keyframe_cond_rows,
        )

        # Parallel tiling gives each replicated rank complete tiles, then gathers
        # them before the seeded posterior sample.
        prepared = minimax_h3_prepared_keyframes(batch, plan)
        prepared_indices = tuple(prepared.get("semantic_frame_indices") or ())
        if prepared_indices not in MINIMAX_H3_FL2VA_KEYFRAME_SIGNATURES or len(
            prepared.get("images") or ()
        ) != len(prepared_indices):
            raise ValueError(
                "keyframe visual preparation requires one or two ordered images "
                "with a supported semantic_frame_indices signature"
            )
        encoded = []
        rows_list = []
        for item in prepared["images"]:
            image = item["image"]
            width, height = item["canvas_width"], item["canvas_height"]
            # The encode sampling seed is pinned at 42 (the VAE sample
            # seed is part of the contract), independent of the request seed.
            rows = minimax_h3_encode_keyframe_cond_rows(
                self.video_vae,
                image,
                self.vae_arch_config,
            )
            encoded.append(
                {
                    "rows": rows,
                    "latent_h": height // 16,
                    "latent_w": width // 16,
                    "canvas_height": height,
                    "canvas_width": width,
                    "frame_index": item.get("frame_index"),
                    "resolved_frame_index": item.get("resolved_frame_index"),
                    "condition_index": item.get("condition_index"),
                }
            )
            rows_list.append(rows)
        rows = rows_list[0] if len(rows_list) == 1 else torch.cat(rows_list, dim=0)
        first = encoded[0]
        batch.extra[MINIMAX_H3_KEYFRAME_COND_ROWS_EXTRA_KEY] = {
            "rows": rows,
            "latent_h": first["latent_h"],
            "latent_w": first["latent_w"],
            "canvas_height": first["canvas_height"],
            "canvas_width": first["canvas_width"],
            "keyframes": encoded,
            "semantic_frame_indices": prepared.get("semantic_frame_indices"),
            "pixel_frame_indices": prepared.get("pixel_frame_indices"),
            "frame_count": prepared.get("frame_count"),
        }

    def _encode_reference_video(self, batch: Req, plan) -> None:
        """ref2va video/video_audio encode from shared transformed frames."""
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.constants import (
            MINIMAX_H3_REFERENCE_VIDEO_ROWS_EXTRA_KEY,
        )
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.reference_encoding import (
            minimax_h3_encode_reference_video_rows,
            minimax_h3_prepared_reference_videos,
        )

        if MINIMAX_H3_REFERENCE_VIDEO_ROWS_EXTRA_KEY in batch.extra:
            return
        prepared = minimax_h3_prepared_reference_videos(
            batch,
            plan,
            share_across_replicas=bool(self.video_vae.parallel_tiling),
        )
        videos = prepared.get("videos")
        if not isinstance(videos, list) or not videos:
            raise ValueError(
                "prepared reference videos payload must carry a non-empty "
                f"'videos' list, got {videos!r}"
            )
        entries = []
        for item in videos:
            rows, latent_t, latent_h, latent_w = minimax_h3_encode_reference_video_rows(
                self.video_vae,
                item["frames"],
                self.vae_arch_config,
            )
            entries.append(
                {
                    "rows": rows,
                    "latent_t": latent_t,
                    "latent_h": latent_h,
                    "latent_w": latent_w,
                    "condition_index": int(item["condition_index"]),
                    "material_chain": str(item["material_chain"]),
                }
            )
        for item in videos:
            # Text and visual encoding are the only RGB-frame consumers. Drop
            # the large request-local arrays as soon as both have completed.
            item.pop("frames", None)
        payload = dict(entries[0])
        payload["videos"] = entries
        batch.extra[MINIMAX_H3_REFERENCE_VIDEO_ROWS_EXTRA_KEY] = payload

    def _encode_reference_image(self, batch: Req, plan) -> None:
        """ref2va reference image encode: cap_resize (intrinsic
        geometry) + the verified keyframe recipe; rows use the image's OWN
        latent grid, not the target canvas."""
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.constants import (
            MINIMAX_H3_REFERENCE_IMAGE_ROWS_EXTRA_KEY,
        )
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.keyframe_encoding import (
            minimax_h3_encode_keyframe_cond_rows,
        )
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.reference_encoding import (
            minimax_h3_prepared_reference_image,
        )

        if MINIMAX_H3_REFERENCE_IMAGE_ROWS_EXTRA_KEY in batch.extra:
            return
        prepared = minimax_h3_prepared_reference_image(batch, plan)
        entries = []
        for item in prepared["images"]:
            image = item["image"]
            rows = minimax_h3_encode_keyframe_cond_rows(
                self.video_vae,
                image,
                self.vae_arch_config,
            )
            width, height = image.size
            entries.append(
                {
                    "rows": rows,
                    "latent_h": height // 16,
                    "latent_w": width // 16,
                    "condition_index": int(item["condition_index"]),
                    "material_chain": "image.reference_preserve",
                }
            )
        payload = dict(entries[0])  # single-image consumers keep the keys
        payload["images"] = entries
        batch.extra[MINIMAX_H3_REFERENCE_IMAGE_ROWS_EXTRA_KEY] = payload


__all__ = ["MiniMaxH3VisualEncodingStage"]
