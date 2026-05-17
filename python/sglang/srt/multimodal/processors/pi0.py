# Copyright 2025 SGLang Team
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
"""Multimodal processor for the π0 VLA model.

Converts per-request images + language prompt + robot state into the
tensors that ``Pi0ForActionPrediction.forward`` consumes:

  * pixel_values   : (num_cameras, 3, 224, 224)  padded with -1 for missing cams
  * image_masks    : (num_cameras,) bool          True = real camera
  * lang_tokens    : (L,) long                    PaliGemma tokenizer, \\n-terminated
  * lang_attention_mask : (L,) bool
  * state          : (state_dim,) float32         optional, zero-padded
  * num_inference_steps : int, optional           override for the denoise loop

Image/prompt preprocessing matches OpenPI and LeRobot bit-for-bit — see
the parity test (``test_pi0_parity.py``) for verification.

Reference implementations:
  - OpenPI: openpi/src/openpi/shared/image_tools.py        (resize_with_pad)
  - OpenPI: openpi/src/openpi/models_pytorch/preprocessing_pytorch.py
  - LeRobot: lerobot/src/lerobot/policies/pi0/modeling_pi0.py  (Pi0NewLineProcessor)
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalProcessorOutput,
)

from sglang.srt.models.pi0 import Pi0ForActionPrediction
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)
from sglang.srt.utils.common import load_image


logger = logging.getLogger(__name__)

# Defaults straight from the π0 reference configs.
PI0_IMAGE_SIZE = 224                 # openpi/models/model.py IMAGE_RESOLUTION
PI0_NUM_IMAGE_TOKENS = 256           # SigLIP So400m/14 on 224×224 → (224/14)**2
PI0_IMAGE_TOKEN_INDEX = 257152       # openpi/models_pytorch/gemma_pytorch.py
PI0_MAX_CAMERAS = 3                  # openpi/models/model.py (3 camera slots)
PI0_MAX_TOKEN_LEN = 48               # openpi/models/pi0_config.py


def resize_with_pad(
    images: torch.Tensor,
    target_height: int,
    target_width: int,
    mode: str = "bilinear",
) -> torch.Tensor:
    """Resize ``(B, C, H, W)`` images to the target shape, preserving aspect
    ratio with -1 padding on the short side.

    Matches openpi ``image_tools.resize_with_pad_torch`` — the clamp to
    [-1, 1] is what lets the padded region blend with SigLIP-normalized
    pixels without adding signal at the boundary.
    """
    if images.ndim != 4:
        raise ValueError(f"Expected 4-D (B,C,H,W), got {images.ndim}-D")
    _, _, cur_h, cur_w = images.shape
    ratio = max(cur_w / target_width, cur_h / target_height)
    rh, rw = int(cur_h / ratio), int(cur_w / ratio)
    align_corners = False if mode == "bilinear" else None
    resized = F.interpolate(images, size=(rh, rw), mode=mode, align_corners=align_corners)
    resized = resized.clamp(-1.0, 1.0)
    ph, rem_h = divmod(target_height - rh, 2)
    pw, rem_w = divmod(target_width - rw, 2)
    return F.pad(resized, (pw, pw + rem_w, ph, ph + rem_h), value=-1.0)


def pil_image_to_tensor(image: Image.Image) -> torch.Tensor:
    """PIL → ``(1, C, H, W)`` float32 in ``[-1, 1]`` (SigLIP normalization)."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    arr = np.array(image, dtype=np.float32) / 255.0 * 2.0 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


class Pi0ImageProcessor:
    """Minimal image preprocessor: PIL → normalized + padded tensor."""

    def __init__(self, image_size: int = PI0_IMAGE_SIZE):
        self.image_size = image_size

    def preprocess_single(self, image: Image.Image) -> torch.Tensor:
        t = pil_image_to_tensor(image)
        if t.shape[2] != self.image_size or t.shape[3] != self.image_size:
            t = resize_with_pad(t, self.image_size, self.image_size)
        return t

    def make_empty_image(self) -> torch.Tensor:
        """Fill tensor for an unused camera slot — pure -1, matches OpenPI."""
        return torch.full((1, 3, self.image_size, self.image_size), -1.0)


def _as_tokenizer(processor_or_tokenizer):
    """Return the underlying tokenizer regardless of what SGLang handed us.

    ``sglang.srt.utils.hf_transformers_utils.get_processor`` routes π0 through
    the ``_tokenizer_only_model_types`` path — so what we receive is often an
    ``AutoTokenizer`` directly, not an ``AutoProcessor`` wrapper. We accept
    both (plus the unlikely degenerate case where nothing has ``.tokenizer``
    and the object itself is already a tokenizer).
    """
    if isinstance(processor_or_tokenizer, (PreTrainedTokenizer, PreTrainedTokenizerFast)):
        return processor_or_tokenizer
    if hasattr(processor_or_tokenizer, "tokenizer"):
        return processor_or_tokenizer.tokenizer
    return processor_or_tokenizer


class Pi0Processor(BaseMultimodalProcessor):
    """Multimodal processor for π0."""

    models = [Pi0ForActionPrediction]

    def __init__(self, hf_config, server_args, _processor, transport_mode):
        super().__init__(hf_config, server_args, _processor, transport_mode)

        # image_resolution lives on the Pi0Config; fall back to the default if
        # the caller rolled a raw config that doesn't set it.
        res = getattr(hf_config, "image_resolution", None)
        if isinstance(res, (list, tuple)) and len(res) == 2:
            self.image_size = res[0]
        else:
            self.image_size = PI0_IMAGE_SIZE

        self.image_processor = Pi0ImageProcessor(image_size=self.image_size)
        self.image_token_id = PI0_IMAGE_TOKEN_INDEX
        self.num_image_tokens = PI0_NUM_IMAGE_TOKENS
        self.max_cameras = PI0_MAX_CAMERAS
        self.max_token_len = getattr(
            hf_config, "tokenizer_max_length",
            getattr(hf_config, "max_token_len", PI0_MAX_TOKEN_LEN),
        )

        self._tokenizer = _as_tokenizer(_processor)
        self.tokenizer = self._tokenizer

        self.mm_tokens = MultimodalSpecialTokens(
            image_token_id=self.image_token_id,
        ).build(self)

    async def process_mm_data_async(
        self,
        image_data: Optional[List] = None,
        audio_data: Optional[List] = None,   # π0 has no audio; base-class API
        input_text: str = "",
        request_obj: Any = None,
        **kwargs,
    ) -> Optional[MultimodalProcessorOutput]:
        if image_data is None:
            image_data = []

        # Clients may supply already-preprocessed tensors (e.g. from a
        # warm-start cache) via a dict-of-dicts payload — route those
        # straight through without re-running the image pipeline.
        # The detection is deliberately strict (requires a typed ``format``
        # or ``feature`` key) so that generic OpenAI-compatible image
        # payloads (``[{"url": "..."}]``, ``[{"type": "image_url", ...}]``)
        # do NOT accidentally get routed here.
        if self._looks_like_precomputed(image_data):
            return self._process_precomputed(image_data, input_text, request_obj)

        # NOTE: we deliberately do *not* call ``self.load_mm_data`` here.
        # That path expects the user prompt to contain one text-level image
        # placeholder per image (e.g. Gemma-3 uses ``<start_of_image>``,
        # Pixtral uses ``[IMG]``) and warns when the counts don't match. π0
        # from OpenPI / LeRobot has a different contract: the prompt is
        # just the natural-language instruction ("pick up the red block")
        # and the image tokens are synthesized at a fixed position
        # (``[image_token_id × 256] × num_cameras`` prepended to the
        # language tokens) — placeholders are **positional, not textual**.
        # We therefore load the PIL images ourselves with SGLang's shared
        # ``load_image`` helper (base64 / URL / file / PIL all supported)
        # and skip the text-scanning / mismatch-warning machinery entirely.
        # Ref: lerobot/policies/pi0/modeling_pi0.py (no <image> tokens in
        # the prompt) and openpi/models/pi0_config.py.
        # ``gpu_image_decode=False`` guarantees ``load_image`` returns PIL;
        # our preprocessor starts from numpy so we need PIL here.
        loaded_images: List[Image.Image] = []
        for image_file in image_data:
            image, _ = load_image(image_file, gpu_image_decode=False)
            if image.mode != "RGB":
                image = image.convert("RGB")
            loaded_images.append(image)
        if len(loaded_images) > self.max_cameras:
            raise ValueError(
                f"π0 supports at most {self.max_cameras} camera images per "
                f"request (the stock model uses 3 fixed camera slots); got "
                f"{len(loaded_images)}."
            )
        pixel_values_list: List[torch.Tensor] = [
            self.image_processor.preprocess_single(img) for img in loaded_images
        ]
        image_masks_list: List[bool] = [True] * len(pixel_values_list)
        for _ in range(max(0, self.max_cameras - len(pixel_values_list))):
            pixel_values_list.append(self.image_processor.make_empty_image())
            image_masks_list.append(False)

        pixel_values = torch.cat(pixel_values_list, dim=0)
        image_masks = torch.tensor(image_masks_list, dtype=torch.bool)

        # Tokenize the prompt the same way LeRobot's Pi0 pipeline does:
        #   - append "\n" terminator  (Pi0NewLineProcessor)
        #   - add BOS/EOS via tokenizer defaults
        #   - right-pad to ``max_token_len`` so every request has the same length
        lang_tokens, lang_attention_mask = self._tokenize_prompt(input_text)

        model_specific_data: Dict[str, Any] = {
            "image_masks": image_masks,
            "num_real_cameras": sum(image_masks_list),
            "lang_tokens": lang_tokens,
            "lang_attention_mask": lang_attention_mask,
        }
        state = self._extract_state(request_obj)
        if state is not None:
            model_specific_data["state"] = state
        num_steps = self._extract_num_steps(request_obj)
        if num_steps is not None:
            model_specific_data["num_inference_steps"] = num_steps

        # input_ids only exist to carry the image placeholder runs (for
        # RadixAttention hashing) plus the padded language chunk (for length
        # accounting). The model's forward reads the real lang tokens back
        # out of ``model_specific_data`` rather than from ``input_ids``.
        num_cams = len(pixel_values_list)
        total_image_tokens = self.num_image_tokens * num_cams

        # ``pad_input_tokens`` in SGLang's multimodal plumbing walks
        # ``item.offsets`` to stamp per-item ``pad_value``s onto the placeholder
        # tokens. We use one ``MultimodalDataItem`` with a single contiguous
        # offset covering all camera placeholders (the whole batch of images
        # is carried as one feature tensor, so they share one hash / pad_value).
        # The range is [start, end] inclusive.
        mm_items = [MultimodalDataItem(
            feature=pixel_values,
            modality=Modality.IMAGE,
            model_specific_data=model_specific_data,
            offsets=[(0, total_image_tokens - 1)] if total_image_tokens > 0 else [],
        )]

        image_placeholder_ids = [self.image_token_id] * total_image_tokens
        # Return the typed MultimodalProcessorOutput that SGLang's tokenizer
        # manager + MultimodalInputs.from_processor_output expect; plain
        # dicts don't have ``.input_ids`` / ``.mm_items`` attributes.
        return MultimodalProcessorOutput(
            mm_items=mm_items,
            input_ids=image_placeholder_ids + list(lang_tokens),
            im_token_id=self.image_token_id,
        )

    def _tokenize_prompt(self, text: str):
        """Return ``(input_ids, attention_mask)``, both of length ``max_token_len``.

        Matches LeRobot's ``Pi0NewLineProcessor``: appends ``\\n`` if the
        caller didn't, then runs the PaliGemma tokenizer with right-padding.
        """
        prompt = text or ""
        if not prompt.endswith("\n"):
            prompt = prompt + "\n"
        enc = self._tokenizer(
            prompt,
            padding="max_length",
            max_length=self.max_token_len,
            truncation=True,
            add_special_tokens=True,
            return_tensors=None,
        )
        return list(enc["input_ids"]), list(enc["attention_mask"])

    def _extract_state(self, request_obj: Any) -> Optional[List[float]]:
        """Pull ``state`` out of the request's ``extra_body`` (if any).

        We accept both dict and object-with-``.state``-attr because SGLang's
        OpenAI-compatible path and its raw-payload path normalize the extras
        differently.
        """
        if request_obj is None:
            return None
        extra = getattr(request_obj, "extra_body", None)
        if not extra:
            return None
        if isinstance(extra, dict):
            return extra.get("state")
        return getattr(extra, "state", None)

    def _extract_num_steps(self, request_obj: Any) -> Optional[int]:
        """Same contract as ``_extract_state`` but for ``num_inference_steps``."""
        if request_obj is None:
            return None
        extra = getattr(request_obj, "extra_body", None)
        if not extra:
            return None
        if isinstance(extra, dict):
            v = extra.get("num_inference_steps")
        else:
            v = getattr(extra, "num_inference_steps", None)
        return int(v) if v is not None else None

    @staticmethod
    def _looks_like_precomputed(image_data) -> bool:
        """Check whether ``image_data`` is a client-supplied warm-start payload.

        The check is narrow on purpose: must be a non-empty list of dicts,
        and the first element must carry one of our typed keys
        (``format`` / ``feature``). Generic OpenAI image dicts
        (``{"url": ...}``, ``{"type": "image_url", ...}``) don't have these
        keys and therefore still go through the normal image-loading path.
        """
        if not (isinstance(image_data, list) and image_data):
            return False
        first = image_data[0]
        if not isinstance(first, dict):
            return False
        return any(k in first for k in ("format", "feature"))

    def _process_precomputed(
        self,
        image_data: List[Dict],
        input_text: str,
        request_obj: Any,
    ) -> MultimodalProcessorOutput:
        """Fast path for clients that provide pre-normalized pixel values.

        **Contract (strict)**: ``image_data`` must be a single-element list
        containing one dict with:
          - ``format`` == ``"processor_output"`` or ``"precomputed_embedding"``
          - ``feature`` — a stacked camera tensor of shape
            ``(num_cameras, 3, H, W)`` where ``num_cameras <= max_cameras``
          - ``model_specific_data`` (optional but recommended) — a dict with
            the same keys the normal path produces: ``image_masks``,
            ``lang_tokens``, ``lang_attention_mask``, ``state``,
            ``num_inference_steps``. If missing, we synthesize minimal
            defaults, but the caller is responsible for ensuring they match
            the feature tensor's actual camera count and state.

        This strict contract ensures the precomputed path produces output
        structurally identical to the normal path: one ``MultimodalDataItem``
        with a stacked feature tensor + complete metadata. Violating it
        (e.g. passing multiple dicts, one per camera) raises ``ValueError``.

        Why so strict? The model's ``_extract_request_inputs`` only reads
        ``mm_input.mm_items[0]``, so any additional items are silently
        ignored. Accepting "one dict per camera" would either require
        re-stacking them here (duplicating the normal path's padding/masking
        logic) or silently drop cameras. Better to fail fast and let the
        caller use the normal path if they don't have a pre-stacked tensor.
        """
        if not image_data or len(image_data) != 1:
            raise ValueError(
                f"Pi0Processor: precomputed path requires exactly one dict "
                f"(containing a pre-stacked camera tensor); got {len(image_data)} "
                f"items. If you have separate per-camera images, use the normal "
                f"image-loading path (pass image URLs/files without 'format' key)."
            )

        item = image_data[0]
        fmt = item.get("format")
        if fmt is None:
            raise ValueError(
                f"Pi0Processor: precomputed image payload is missing ``format`` "
                f"(expected 'processor_output' or 'precomputed_embedding')."
            )
        if fmt not in ("processor_output", "precomputed_embedding"):
            raise ValueError(
                f"Pi0Processor: unknown precomputed format {fmt!r}; "
                f"expected 'processor_output' or 'precomputed_embedding'."
            )

        feature = item.get("feature")
        if feature is None:
            raise ValueError(
                f"Pi0Processor: precomputed item has no ``feature`` tensor."
            )
        if not isinstance(feature, torch.Tensor):
            raise ValueError(
                f"Pi0Processor: precomputed feature must be a torch.Tensor, "
                f"got {type(feature).__name__}."
            )
        if feature.ndim != 4:
            raise ValueError(
                f"Pi0Processor: precomputed feature must be 4D "
                f"(num_cameras, 3, H, W); got shape {feature.shape}."
            )

        num_cams = feature.shape[0]
        if num_cams > self.max_cameras:
            raise ValueError(
                f"Pi0Processor: precomputed feature has {num_cams} cameras but "
                f"max_cameras={self.max_cameras}. The stock π0 model requires "
                f"exactly 3 camera slots (real or padded); trim or re-configure."
            )

        # Extract or synthesize model_specific_data. If the caller provided it,
        # trust them; otherwise build minimal defaults (all cameras active,
        # re-tokenize the prompt, no state/num_steps).
        model_specific_data = item.get("model_specific_data")
        if model_specific_data is None:
            lang_tokens, lang_attention_mask = self._tokenize_prompt(input_text)
            model_specific_data = {
                "image_masks": torch.ones(num_cams, dtype=torch.bool),
                "num_real_cameras": num_cams,
                "lang_tokens": lang_tokens,
                "lang_attention_mask": lang_attention_mask,
            }
            state = self._extract_state(request_obj)
            if state is not None:
                model_specific_data["state"] = state
            num_steps = self._extract_num_steps(request_obj)
            if num_steps is not None:
                model_specific_data["num_inference_steps"] = num_steps
        else:
            # Caller provided model_specific_data; do minimal validation so
            # downstream doesn't crash with cryptic KeyError / shape mismatch.
            if "lang_tokens" not in model_specific_data:
                raise ValueError(
                    "Pi0Processor: precomputed model_specific_data is missing "
                    "'lang_tokens'. Either provide a complete dict or omit it "
                    "to let the processor synthesize defaults."
                )
            if "image_masks" not in model_specific_data:
                raise ValueError(
                    "Pi0Processor: precomputed model_specific_data is missing "
                    "'image_masks'."
                )

        total_image_tokens = self.num_image_tokens * num_cams
        mm_items = [MultimodalDataItem(
            feature=feature,
            modality=Modality.IMAGE,
            model_specific_data=model_specific_data,
            offsets=[(0, total_image_tokens - 1)] if total_image_tokens > 0 else [],
        )]

        # Use lang_tokens from model_specific_data (either caller-provided or
        # just synthesized above) so input_ids length matches the contract.
        lang_tokens = model_specific_data["lang_tokens"]
        return MultimodalProcessorOutput(
            mm_items=mm_items,
            input_ids=[self.image_token_id] * total_image_tokens + list(lang_tokens),
            im_token_id=self.image_token_id,
        )

    def get_estimated_frames_list(self, image_data):
        """π0 treats every input image as a single frame (no video)."""
        return [1] * len(image_data) if image_data else []
