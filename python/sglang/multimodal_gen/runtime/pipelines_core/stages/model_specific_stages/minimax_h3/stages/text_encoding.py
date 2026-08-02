# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import torch

from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.constants import (
    MINIMAX_H3_TEXT_EMBEDDINGS_EXTRA_KEY,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.task_profiles import (
    MINIMAX_H3_FL2VA_KEYFRAME_SIGNATURES,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.text_encoding import (
    TextEncodingStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class MiniMaxH3TextEncodingStage(TextEncodingStage):
    deduplicated_output_fields = ("prompt_embeds", "prompt_seq_lens")
    deduplicated_extra_output_keys = (MINIMAX_H3_TEXT_EMBEDDINGS_EXTRA_KEY,)

    def __init__(self, text_encoder, tokenizer, processor) -> None:
        super().__init__(
            text_encoders=[text_encoder],
            tokenizers=[tokenizer],
        )
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        if processor is None:
            raise ValueError(
                "MiniMaxH3TextEncodingStage requires the pipeline processor "
                "component (model_index.json: processor)"
            )
        self.processor = processor

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.resolved_plan import (
            minimax_h3_plan_from_batch,
        )

        plan = minimax_h3_plan_from_batch(batch)
        if plan is not None:
            try:
                self._encode_from_plan(batch, plan)
                self._publish_native_text_conditioning(batch)
            except Exception:
                from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.material_io import (
                    minimax_h3_cleanup_temp_dirs,
                )

                minimax_h3_cleanup_temp_dirs(batch)
                raise
            return batch
        if batch.sampling_params is not None and (
            batch.sampling_params.prompt is not None
            or batch.sampling_params.prompt_path is not None
        ):
            raise NotImplementedError(
                "MiniMaxH3TextEncodingStage direct Qwen3VL encoder forward requires "
                "a canonical minimax_h3 request (resolved plan); legacy prompt-only "
                "requests are unsupported."
            )
        return batch

    def build_dedup_fingerprint(self, batch: Req, server_args: ServerArgs):
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.resolved_plan import (
            minimax_h3_plan_from_batch,
        )

        plan = minimax_h3_plan_from_batch(batch)
        if plan is None:
            return super().build_dedup_fingerprint(batch, server_args)
        materials = tuple(
            (
                item.condition_index,
                item.role,
                item.condition_type,
                item.uri,
                item.material_chain,
                item.frame_index,
                item.resolved_frame_index,
                item.start_time_seconds,
            )
            for item in plan.materials
        )
        return (
            plan.task,
            plan.prompt,
            materials,
            self.freeze_for_dedup(plan.shape),
        )

    def run_grouped_requests(
        self,
        batches: list[Req],
        server_args: ServerArgs,
    ) -> list[Req]:
        """Distribute independent H3 presentations over replicated encoders.

        H3 presentations have variable multimodal layouts, so they cannot be
        stacked into the generic text batch without changing padding/kernels.
        Assigning one complete request to a rank preserves the exact
        single-request encoder path, then broadcasts that request's native
        payload to the other ranks.
        """
        grouped = self._group_requests_by_fingerprint(
            batches,
            lambda batch: self.build_dedup_fingerprint(batch, server_args),
        )
        if not grouped:
            return []

        encoder_config = server_args.pipeline_config.text_encoder_configs[0]
        dp_group = self._text_encode_dp_group(
            server_args,
            encoder_config,
            len(grouped),
            self.text_encoder,
        )
        if dp_group is None:
            return super().run_grouped_requests(batches, server_args)

        results: list[Req | None] = [None] * len(batches)
        for group_index, (_, equivalent) in enumerate(grouped):
            owner = group_index % dp_group.world_size
            first_index, first_batch = equivalent[0]
            owner_exception = None
            owner_error = None
            payload = None
            if dp_group.rank_in_group == owner:
                try:
                    first_result = self(first_batch, server_args)
                    payload = first_result.extra.get(
                        MINIMAX_H3_TEXT_EMBEDDINGS_EXTRA_KEY
                    )
                    if not isinstance(payload, dict):
                        raise ValueError(
                            "MiniMax H3 text encode produced no native payload"
                        )
                except Exception as exc:
                    owner_exception = exc
                    owner_error = f"{type(exc).__name__}: {exc}"

            owner_error = dp_group.broadcast_object(owner_error, src=owner)
            if owner_error is not None:
                if owner_exception is not None:
                    raise owner_exception
                raise RuntimeError(
                    f"MiniMax H3 text encode failed on rank {owner}: {owner_error}"
                )
            payload = dp_group.broadcast_tensor_dict(payload, src=owner)
            if not isinstance(payload, dict):
                raise RuntimeError("MiniMax H3 text payload broadcast failed")

            if dp_group.rank_in_group != owner:
                first_batch.extra[MINIMAX_H3_TEXT_EMBEDDINGS_EXTRA_KEY] = payload
                self._publish_native_text_conditioning(first_batch)
                first_result = first_batch
            results[first_index] = first_result

            for index, batch in equivalent[1:]:
                self.copy_deduplicated_outputs(first_result, batch)
                results[index] = batch

        return [result for result in results if result is not None]

    def _log_dp_choice(self, batch_size: int, world_size: int) -> None:
        if self._dp_choice_logged:
            return
        self._dp_choice_logged = True
        logger.info(
            "encoder_parallel: distributing %d independent MiniMax H3 "
            "presentations over %d replicated encoder ranks",
            batch_size,
            world_size,
        )

    @staticmethod
    def _publish_native_text_conditioning(batch: Req) -> None:
        """Mirror H3's rich payload onto the native text-stage fields.

        H3 keeps token tags and presentation metadata in ``Req.extra``, but
        the shared TextEncodingStage contract still owns ``prompt_embeds``.
        Publishing the same tensor there preserves native verification,
        grouped-request deduplication, and downstream memory accounting
        without duplicating the embedding storage.
        """
        payload = batch.extra.get(MINIMAX_H3_TEXT_EMBEDDINGS_EXTRA_KEY)
        positive = payload.get("positive") if isinstance(payload, dict) else None
        hidden_states = (
            positive.get("hidden_states") if isinstance(positive, dict) else None
        )
        text_len = positive.get("text_len") if isinstance(positive, dict) else None
        if not isinstance(hidden_states, torch.Tensor) or hidden_states.ndim < 2:
            raise ValueError(
                "MiniMax H3 text payload must contain positive.hidden_states "
                "with at least two dimensions"
            )
        if not isinstance(text_len, int) or text_len != int(hidden_states.shape[0]):
            raise ValueError(
                "MiniMax H3 text payload positive.text_len must match the "
                "hidden-state sequence dimension"
            )
        batch.prompt_embeds = [hidden_states]
        batch.prompt_seq_lens = [[text_len]]

    def _encode_from_plan(self, batch: Req, plan) -> None:
        """Encode the positive Qwen3VL presentation into layer-50 states.

        MiniMax H3 only supports the CFG-distilled model path, so every task
        emits exactly one positive embedding payload. ComponentManager owns
        residency/offload, while every folded-TP rank enters the encoder
        collectives and receives the same replicated hidden states.
        """
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.presentation import (
            minimax_h3_text_only_ids,
        )

        prompt = plan.prompt
        keyframes = [
            m for m in plan.materials if m.material_chain == "image.target_canvas"
        ]
        if plan.task == "fl2va":
            frame_indices = tuple(material.frame_index for material in keyframes)
            if frame_indices not in MINIMAX_H3_FL2VA_KEYFRAME_SIGNATURES:
                raise ValueError(
                    "fl2va text encoding requires an ordered keyframe signature "
                    f"in {MINIMAX_H3_FL2VA_KEYFRAME_SIGNATURES!r}, got "
                    f"{frame_indices!r}"
                )
        elif keyframes:
            raise ValueError(
                f"task {plan.task!r} cannot carry image.target_canvas materials"
            )
        if MINIMAX_H3_TEXT_EMBEDDINGS_EXTRA_KEY in batch.extra:
            return
        if self.text_encoder is None:
            raise ValueError(
                "MiniMaxH3TextEncodingStage direct encode requires a text_encoder "
                "component"
            )
        encode_ids = getattr(self.text_encoder, "encode_ids", None)
        if not callable(encode_ids):
            raise TypeError(
                "MiniMax H3 text_encoder component must expose callable "
                "encode_ids(...) for direct encode (MiniMaxH3Qwen3VLEncoder)"
            )
        if self.tokenizer is None:
            raise ValueError(
                "MiniMaxH3TextEncodingStage direct encode requires a tokenizer component"
            )
        self._manage_text_encoder_use(0)
        with set_forward_context(current_timestep=0, attn_metadata=None):
            if plan.task == "ref2va":
                embeddings = self._encode_ref2va(batch, plan, encode_ids)
            elif keyframes:
                embeddings = self._encode_fl2va_keyframes(
                    batch,
                    plan,
                    encode_ids,
                    prompt=prompt,
                )
            else:
                positive_ids = minimax_h3_text_only_ids(self.tokenizer, prompt)
                embeddings = {
                    "positive": {
                        "hidden_states": encode_ids(positive_ids),
                        "text_len": int(positive_ids.shape[0]),
                        "text_token_tags": torch.ones(
                            int(positive_ids.shape[0]), dtype=torch.long
                        ),
                    }
                }
        batch.extra[MINIMAX_H3_TEXT_EMBEDDINGS_EXTRA_KEY] = embeddings

    def _encode_fl2va_keyframes(
        self,
        batch: Req,
        plan,
        encode_ids,
        *,
        prompt: str,
    ) -> dict:
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.canvas import (
            minimax_h3_prepared_keyframes,
        )
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.presentation import (
            minimax_h3_multi_image_presentation,
        )

        # The SAME prepared target-canvas images feed
        # Qwen and the visual-condition tokenizer; preparation is cached per request.
        prepared = minimax_h3_prepared_keyframes(batch, plan)
        images = [item["image"] for item in prepared["images"]]
        frame_indices = tuple(prepared.get("semantic_frame_indices") or ())
        if frame_indices not in MINIMAX_H3_FL2VA_KEYFRAME_SIGNATURES or len(
            images
        ) != len(frame_indices):
            raise ValueError(
                "fl2va Qwen preparation requires one or two ordered images with "
                "a supported semantic_frame_indices signature, got "
                f"{frame_indices!r}"
            )
        processor = self.processor
        vision = processor.image_processor(images=images, return_tensors="pt")
        pixel_values = vision["pixel_values"]
        image_grid_thw = vision["image_grid_thw"]
        if int(image_grid_thw.shape[0]) != len(images):
            raise ValueError(
                f"expected {len(images)} image grids, got {list(image_grid_thw.shape)}"
            )
        merge = int(processor.image_processor.merge_size) ** 2
        image_token_counts = [
            int(image_grid_thw[i].prod().item()) // merge for i in range(len(images))
        ]
        pos_ids, pos_tags = minimax_h3_multi_image_presentation(
            self.tokenizer,
            prompt=prompt,
            image_token_counts=image_token_counts,
        )
        pos_hidden = encode_ids(
            pos_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )
        return {
            "positive": {
                "hidden_states": pos_hidden,
                "text_len": int(pos_ids.shape[0]),
                "text_token_tags": pos_tags,
            },
        }

    def _encode_ref2va(self, batch: Req, plan, encode_ids) -> dict:
        """Encode the positive ref2va presentation.

        Per condition in order — image i: '<Picture i>: ' label +
        vision block (prepared reference image); audio j: '<Audio j>: ' label
        only — then the verbatim prompt.
        """
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.presentation import (
            minimax_h3_ref2va_presentation,
            minimax_h3_ref2va_video_presentation,
        )
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.reference_encoding import (
            minimax_h3_prepared_reference_image,
            minimax_h3_prepared_reference_videos,
            minimax_h3_sample_reference_video_frames,
        )

        prepared_videos = None
        if any(
            material.material_chain
            in ("video.reference_preserve", "video_audio.reference_preserve")
            for material in plan.materials
        ):
            prepared_videos = minimax_h3_prepared_reference_videos(batch, plan)
        video_has_audio: dict[int, bool] = {}
        for video_index, item in enumerate((prepared_videos or {}).get("videos") or []):
            if item.get("condition_index") is None:
                continue
            if "input_has_audio" not in item:
                raise ValueError(
                    f"prepared reference video {video_index} is missing "
                    "'input_has_audio'; the canonical minimax_h3 producer must "
                    "supply the audio probe for every video condition"
                )
            video_has_audio[int(item["condition_index"])] = bool(
                item["input_has_audio"]
            )
        condition_labels: list[tuple[str, int]] = []
        counters = {"image": 0, "audio": 0, "video": 0}
        has_image = False
        has_video = False
        for material in plan.materials:
            if material.material_chain == "image.reference_preserve":
                counters["image"] += 1
                condition_labels.append(("image", counters["image"]))
                has_image = True
            elif material.material_chain == "audio":
                counters["audio"] += 1
                condition_labels.append(("audio", counters["audio"]))
            elif material.material_chain in (
                "video.reference_preserve",
                "video_audio.reference_preserve",
            ):
                # A plain video contributes an Audio label only when its
                # probed source actually has a soundtrack. ``video_audio`` is
                # an explicit caller promise and remains fail-closed in the
                # audio stage if its stream is missing.
                if material.material_chain == "video_audio.reference_preserve":
                    contributes_audio = True
                else:
                    condition_index = int(material.condition_index)
                    if condition_index not in video_has_audio:
                        raise KeyError(
                            "prepared reference videos carry no "
                            f"'input_has_audio' probe for condition "
                            f"{condition_index}; the canonical minimax_h3 "
                            "producer must supply it"
                        )
                    contributes_audio = video_has_audio[condition_index]
                if contributes_audio:
                    counters["audio"] += 1
                    condition_labels.append(("audio", counters["audio"]))
                counters["video"] += 1
                condition_labels.append(("video", counters["video"]))
                has_video = True
            else:
                raise NotImplementedError(
                    f"ref2va does not support chain {material.material_chain!r}"
                )

        pixel_values = None
        image_grid_thw = None
        n_image_tokens = None
        processor = self.processor

        if has_image:
            prepared = minimax_h3_prepared_reference_image(batch, plan)
            images = [item["image"] for item in prepared["images"]]
            proc = processor
            vision = proc.image_processor(images=images, return_tensors="pt")
            pixel_values = vision["pixel_values"]
            image_grid_thw = vision["image_grid_thw"]
            if int(image_grid_thw.shape[0]) != len(images):
                raise ValueError(
                    f"expected {len(images)} image grids, got "
                    f"{list(image_grid_thw.shape)}"
                )
            merge = int(proc.image_processor.merge_size) ** 2
            counts = [
                int(image_grid_thw[i].prod().item()) // merge
                for i in range(len(images))
            ]
            # presentation takes an int for one image, a list for several
            n_image_tokens = counts[0] if len(counts) == 1 else counts

        pixel_values_videos = None
        video_grid_thw = None
        video_block_token_counts = None
        video_block_timestamps = None
        if has_video:
            prepared = prepared_videos or minimax_h3_prepared_reference_videos(
                batch, plan
            )
            videos = []
            sampled_videos = []
            for item in prepared["videos"]:
                sampled = minimax_h3_sample_reference_video_frames(item["frames"])
                videos.append(torch.from_numpy(sampled["frames"]).permute(0, 3, 1, 2))
                sampled_videos.append(sampled)
            proc = processor
            vout = proc.video_processor(
                videos=videos,
                do_sample_frames=False,
                input_data_format="channels_first",
                return_tensors="pt",
            )
            pixel_values_videos = vout["pixel_values_videos"]
            video_grid_thw = vout["video_grid_thw"]
            if int(video_grid_thw.shape[0]) != len(videos):
                raise ValueError(
                    f"expected {len(videos)} video grids, got "
                    f"{list(video_grid_thw.shape)}"
                )
            merge = int(proc.image_processor.merge_size) ** 2
            video_block_token_counts = []
            video_block_timestamps = []
            for index, sampled in enumerate(sampled_videos):
                n_blocks = int(video_grid_thw[index, 0])
                per_block = (
                    int(video_grid_thw[index, 1])
                    * int(video_grid_thw[index, 2])
                    // merge
                )
                timestamps = [float(ts) for ts in sampled["block_timestamps"]]
                if len(timestamps) != n_blocks:
                    raise ValueError(
                        f"video block count mismatch: processor {n_blocks} vs "
                        f"timestamps {len(timestamps)} for video {index}"
                    )
                video_block_token_counts.append([per_block] * n_blocks)
                video_block_timestamps.append(timestamps)

        if has_video:
            pos_ids, pos_tags = minimax_h3_ref2va_video_presentation(
                self.tokenizer,
                prompt=plan.prompt,
                condition_labels=condition_labels,
                image_token_count=n_image_tokens,
                video_block_token_counts=video_block_token_counts,
                video_block_timestamps=video_block_timestamps,
            )
        else:
            pos_ids, pos_tags = minimax_h3_ref2va_presentation(
                self.tokenizer,
                prompt=plan.prompt,
                condition_labels=condition_labels,
                image_token_count=n_image_tokens,
            )
        pos_hidden = encode_ids(
            pos_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
        )
        return {
            "positive": {
                "hidden_states": pos_hidden,
                "text_len": int(pos_ids.shape[0]),
                "text_token_tags": pos_tags,
            },
        }


__all__ = ["MiniMaxH3TextEncodingStage"]
