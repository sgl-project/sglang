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
    MINIMAX_H3_REFERENCE_AUDIO_ROWS_EXTRA_KEY,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs


class MiniMaxH3AudioEncodingStage(ConditionEncodingStage):
    deduplicated_extra_output_keys = (MINIMAX_H3_REFERENCE_AUDIO_ROWS_EXTRA_KEY,)

    def __init__(self, audio_vae, vae_arch_config) -> None:
        super().__init__()
        self.audio_vae = audio_vae
        self.vae_arch_config = vae_arch_config

    @property
    def role_affinity(self) -> RoleType:
        return RoleType.ENCODER

    def component_uses(
        self, server_args: ServerArgs, stage_name: str | None = None
    ) -> list[ComponentUse]:
        stage_name = self._component_stage_name(stage_name)
        return [ComponentUse(stage_name, "audio_vae")]

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
        finally:
            # Audio encoding is the final material consumer in the MiniMax H3
            # encoder pipeline, including requests with no routed audio.
            minimax_h3_cleanup_temp_dirs(batch, owners=("material",))

    def _forward(self, batch: Req, server_args: ServerArgs) -> Req:
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.resolved_plan import (
            minimax_h3_plan_from_batch,
        )

        plan = minimax_h3_plan_from_batch(batch)
        if plan is not None:
            routed = plan.encoders.get("audio")
            if not routed:
                return batch
            self._encode_references_from_plan(batch, plan, routed)
            return batch
        if (
            batch.sampling_params is not None
            and batch.sampling_params.audio_path is not None
        ):
            raise NotImplementedError(
                "MiniMaxH3AudioEncodingStage direct audio tokenizer encode "
                "requires a canonical minimax_h3 request (resolved plan); "
                "legacy audio_path-only requests are unsupported."
            )
        return batch

    def _encode_references_from_plan(self, batch: Req, plan, routed) -> None:
        """Direct reference-audio encode: audio VAE posterior mean ->
        normalized channel-major rows in batch.extra."""
        routed_set = set(routed)
        routed_materials = [
            material
            for material in plan.materials
            if material.condition_index in routed_set
        ]
        from .replica_broadcast import (
            minimax_h3_replica_broadcast_error,
            minimax_h3_replica_broadcast_extra,
            minimax_h3_replica_ctx,
        )

        _, replica_rank = minimax_h3_replica_ctx()
        owner_exception = None
        owner_error = None
        if (
            replica_rank == 0
            and MINIMAX_H3_REFERENCE_AUDIO_ROWS_EXTRA_KEY not in batch.extra
        ):
            try:
                with self.use_declared_component(
                    component_name="audio_vae",
                    module=self.audio_vae,
                ) as audio_vae:
                    assert audio_vae is not None
                    self.audio_vae = audio_vae
                    batch.extra[MINIMAX_H3_REFERENCE_AUDIO_ROWS_EXTRA_KEY] = (
                        self._encode_reference_payload(
                            batch,
                            plan,
                            routed_materials,
                        )
                    )
            except Exception as exc:
                owner_exception = exc
                owner_error = f"{type(exc).__name__}: {exc}"
        owner_error = minimax_h3_replica_broadcast_error(owner_error)
        if owner_error is not None:
            if owner_exception is not None:
                raise owner_exception
            raise RuntimeError(
                f"MiniMax H3 audio encode failed on rank 0: {owner_error}"
            )
        minimax_h3_replica_broadcast_extra(
            batch, MINIMAX_H3_REFERENCE_AUDIO_ROWS_EXTRA_KEY
        )

    def _encode_reference_payload(self, batch: Req, plan, materials) -> dict:
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.material_io import (
            minimax_h3_localize_material_uri,
        )
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.prequeue import (
            MINIMAX_H3_PROBE_FACTS_EXTRA_KEY,
        )
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.reference_encoding import (
            _AudioVAEDeterminismContext,
            minimax_h3_encode_reference_audio_rows,
        )

        if not materials:
            raise ValueError("ref2va audio routing selected no reference materials")
        entries = []
        max_duration_seconds = (
            float(plan.shape["frame_count"]) / float(plan.shape["fps"])
            if plan.shape.get("frame_count") is not None
            and plan.shape.get("fps") is not None
            else None
        )
        # One determinism-flag toggle for the whole routed set, not one per
        # material: _AudioVAEDeterminismContext is reentrant, so each
        # material's own nested context (inside
        # minimax_h3_encode_reference_audio_rows) becomes a no-op depth
        # increment/decrement under this outer scope.
        with _AudioVAEDeterminismContext():
            for material in materials:
                audio_path = minimax_h3_localize_material_uri(
                    batch,
                    material.uri,
                    condition_type=material.condition_type,
                    condition_index=int(material.condition_index),
                )
                material_chain = str(material.material_chain)
                source_facts = batch.extra.get(
                    MINIMAX_H3_PROBE_FACTS_EXTRA_KEY, {}
                ).get(int(material.condition_index))
                if not isinstance(source_facts, dict):
                    raise ValueError(
                        "reference-audio encoding requires cached pre-queue probe "
                        f"facts for conditions[{int(material.condition_index)}]"
                    )
                input_has_audio = bool(source_facts.get("has_audio", True))
                if material_chain == "video.reference_preserve" and not input_has_audio:
                    # Keep the visual reference block in request order while
                    # representing the absent soundtrack as a zero-length
                    # audio condition.
                    out = {
                        "rows": torch.empty((0, 32), dtype=torch.float32),
                        "ref_audio_t": 0,
                        "duration_seconds": 0.0,
                    }
                else:
                    out = minimax_h3_encode_reference_audio_rows(
                        self.audio_vae,
                        audio_path,
                        self.vae_arch_config,
                        material_chain=material_chain,
                        max_duration_seconds=max_duration_seconds,
                        start_time_seconds=float(material.start_time_seconds),
                        source_sample_rate=(
                            int(source_facts["audio_sample_rate"])
                            if material_chain == "audio"
                            else None
                        ),
                    )
                entries.append(
                    {
                        **out,
                        "condition_index": int(material.condition_index),
                        "material_chain": material_chain,
                    }
                )
        payload = dict(entries[0]) if len(entries) == 1 else {}
        payload["audios"] = entries
        return payload


__all__ = ["MiniMaxH3AudioEncodingStage"]
