"""Generic decoded media bridge; model semantics stay in registered processors.

Public descriptors never carry internal objects or model features. Rust supplies
owned, bounded snapshots; this module only constructs ordinary media inputs and
preserves the complete MultimodalProcessorOutput through the owned ring carrier.
"""

import asyncio
import json
import weakref

import numpy as np
from PIL import Image

from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.managers.req_types import MultimodalProcessorOutput
from sglang.srt.multimodal.media_io import DecodedAudio, DecodedVideoFrames


def materialize_media(descriptor, payloads):
    """Snapshot order is images, each video's frames, then audio clips."""
    payloads = iter(payloads)

    def rgb(image):
        data = next(payloads)
        width, height = image["width"], image["height"]
        stride = image["row_stride_bytes"]
        # A view of owned Rust storage; PIL takes an independent RGB image.
        array = np.ndarray(
            (height, width, 3), dtype=np.uint8, buffer=data, strides=(stride, 3, 1)
        )
        return Image.fromarray(array).copy()

    images = [rgb(image) for image in descriptor.get("images", [])]
    videos = []
    for video in descriptor.get("videos", []):
        frames = video["frames"]
        videos.append(
            DecodedVideoFrames(
                frames=[np.asarray(rgb(f["image"])) for f in frames],
                timestamps_us=[f.get("timestamp_us", 0) for f in frames],
                source_fps=video["source_fps"],
                source_frame_count=video["source_frame_count"],
                source_frame_indices=[f.get("source_frame_index", 0) for f in frames],
            )
        )
    audios = []
    for audio in descriptor.get("audios", []):
        data = next(payloads)
        samples = np.frombuffer(data, dtype="<f4").reshape(
            audio["sample_frames"], audio["channels"]
        )
        if not np.isfinite(samples).all() or np.any(np.abs(samples) > 1):
            raise ValueError("decoded PCM samples must be finite and in [-1,1]")
        audios.append(DecodedAudio(samples, audio["sample_rate_hz"]))
    if next(payloads, None) is not None:
        raise ValueError("unexpected decoded media payload")
    return images, videos, audios


class NativeDecodedMediaProcessor:
    """One synchronous callback on one native MM worker, owning its async loop.

    There is no global RID dictionary. The complete processor result is owned
    by the request, including auxiliary tensors and model-specific metadata.
    Initial bridge scope: unified single-node single-rank serving, CPU feature
    transport. Other topologies fail at launch, rather than silently losing
    cross-rank feature ownership or P/D semantics.
    """

    def __init__(self, scheduler):
        from sglang.srt.managers.multimodal_processor import get_mm_processor
        from sglang.srt.runtime_context import get_disagg, get_mm, get_parallel

        parallel = get_parallel()
        if (
            any(
                getattr(parallel, name) != 1
                for name in ("nnodes", "tp_size", "dp_size", "pp_size")
            )
            or parallel.attn_dp_enabled
        ):
            raise ValueError(
                "decoded_media currently requires unified single-node single-rank serving"
            )
        if get_disagg().disaggregation_mode != "null":
            raise ValueError("decoded_media P/D ownership has not been enabled")
        if get_mm().mm_feature_transport not in ("cpu", "auto"):
            raise ValueError(
                "decoded_media bridge currently requires CPU MM feature transport"
            )
        # _resolve_multimodal has initialized the same registration and HF
        # processor used by TokenizerManager. No model-specific registry here.
        self.processor = get_mm_processor(
            scheduler.model_config.hf_config,
            scheduler.server_args,
            scheduler.processor,
            "default",
            model_config=scheduler.model_config,
            skip_mm_pool=True,
            cpu_executor_start_method="spawn",
            cpu_executor_workers=1,
        )
        self.max_input_len = scheduler.model_config.context_len
        self.runner = asyncio.Runner()
        self._finalizer = weakref.finalize(
            self, self._close, self.runner, self.processor
        )

    @staticmethod
    def _close(runner, processor):
        runner.close()
        processor.shutdown()

    def __call__(self, input_ids, descriptor, payloads):
        descriptor = json.loads(descriptor)
        if descriptor.get("videos") and not getattr(
            self.processor, "supports_decoded_video", False
        ):
            raise ValueError(
                "registered processor has not enabled already-sampled video frames"
            )
        tokens = getattr(self.processor, "mm_tokens", None)
        if descriptor.get("audios") and not (tokens and tokens.audio_token):
            raise ValueError("registered processor does not support decoded audio")
        images, videos, audios = materialize_media(descriptor, payloads)
        request = GenerateReqInput(
            input_ids=input_ids,
            image_data=images or None,
            video_data=videos or None,
            audio_data=audios or None,
        )
        return self.runner.run(self._process(input_ids, request))

    async def _process(self, input_ids, request):
        output = await self.processor.process_mm_data_async(
            image_data=request.image_data,
            audio_data=request.audio_data,
            input_text=input_ids,
            request_obj=request,
            max_req_input_len=self.max_input_len,
        )
        if isinstance(output, dict):
            output = MultimodalProcessorOutput.from_dict(output)
        if not isinstance(output, MultimodalProcessorOutput) or not output.mm_items:
            raise RuntimeError("registered processor did not return multimodal output")
        self.processor.finalize_cache_identities(output)
        # Hash on this worker, not the scheduler. Some legacy processors leave
        # this lazy when Python TokenizerManager precomputation is disabled.
        for item in output.mm_items:
            item.set_pad_value()
        ids = output.input_ids if output.input_ids is not None else input_ids
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        ids = [int(token) for token in ids]
        if not ids or any(token < 0 for token in ids):
            raise RuntimeError("registered processor returned invalid token IDs")
        for item in output.mm_items:
            for start, end in item.offsets or []:
                if not 0 <= start <= end < len(ids):
                    raise RuntimeError(
                        "registered processor returned invalid token spans"
                    )
            if item.identity is None or len(item.identity) != 32:
                raise RuntimeError(
                    "registered processor did not finalize content identity"
                )
        if output.padded_input_ids is not None:
            output.padded_input_ids = output.build_padded_input_ids(
                ids, output.mm_items
            )
        if output.token_type_ids is not None and len(output.token_type_ids) != len(ids):
            raise RuntimeError("registered processor returned mismatched token types")
        return ids, output
