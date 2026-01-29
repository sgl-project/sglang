import json
import os
import time
from typing import Dict, List, Union

from sglang.srt.managers.multimodal_processor import (
    BaseMultimodalProcessor as SGLangBaseProcessor,
)
from sglang.srt.models.lfm2_vl import Lfm2VlForConditionalGeneration
from sglang.srt.multimodal.processors.base_processor import MultimodalSpecialTokens

# #region agent log
_DEBUG_LOG_PATH = "/sgl-workspace/sglang/.cursor/debug.log"


def _dbg_proc(hypothesis_id, location, message, data=None):
    entry = {
        "sessionId": "lfm2vl-debug",
        "runId": "initial",
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": time.time(),
    }
    try:
        os.makedirs(os.path.dirname(_DEBUG_LOG_PATH), exist_ok=True)
        with open(_DEBUG_LOG_PATH, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")
    except Exception:
        pass


# #endregion


class Lfm2VlImageProcessor(SGLangBaseProcessor):
    models = [Lfm2VlForConditionalGeneration]

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)

        self.IMAGE_TOKEN_ID = hf_config.image_token_id

        self.mm_tokens = MultimodalSpecialTokens(
            image_token="<image>",
            image_token_id=hf_config.image_token_id,
        ).build(_processor)

    async def process_mm_data_async(
        self,
        image_data: List[Union[str, bytes, Dict]],
        input_text,
        request_obj,
        *args,
        **kwargs,
    ):
        base_output = self.load_mm_data(
            prompt=input_text,
            image_data=image_data,
            multimodal_tokens=self.mm_tokens,
        )

        # #region agent log
        try:
            _bo_keys = (
                list(vars(base_output).keys())
                if hasattr(base_output, "__dict__")
                else dir(base_output)
            )
            _dbg_proc(
                "H3",
                "processor:base_output",
                "HF processor output keys",
                {
                    "base_output_type": str(type(base_output)),
                    "base_output_keys": _bo_keys[:20],
                    "has_pixel_attention_mask": hasattr(
                        base_output, "pixel_attention_mask"
                    )
                    and base_output.pixel_attention_mask is not None,
                    "has_spatial_shapes": hasattr(base_output, "spatial_shapes")
                    and base_output.spatial_shapes is not None,
                    "has_pixel_values": hasattr(base_output, "pixel_values")
                    and base_output.pixel_values is not None,
                },
            )
        except Exception as e:
            _dbg_proc(
                "H3", "processor:base_output", f"Error logging base_output: {e}", {}
            )
        # #endregion

        mm_items, input_ids, _ = self.process_and_combine_mm_data(
            base_output, self.mm_tokens
        )

        # #region agent log
        try:
            _item_details = []
            _items_list = (
                mm_items.mm_items
                if hasattr(mm_items, "mm_items")
                else (mm_items if isinstance(mm_items, list) else [])
            )
            for idx, item in enumerate(_items_list):
                _item_details.append(
                    {
                        "idx": idx,
                        "modality": (
                            str(item.modality) if hasattr(item, "modality") else None
                        ),
                        "has_feature": (
                            item.feature is not None
                            if hasattr(item, "feature")
                            else False
                        ),
                        "feature_shape": (
                            list(item.feature.shape)
                            if hasattr(item, "feature")
                            and item.feature is not None
                            and hasattr(item.feature, "shape")
                            else None
                        ),
                        "has_pixel_attention_mask": hasattr(
                            item, "pixel_attention_mask"
                        )
                        and item.pixel_attention_mask is not None,
                        "pixel_attention_mask_shape": (
                            list(item.pixel_attention_mask.shape)
                            if hasattr(item, "pixel_attention_mask")
                            and item.pixel_attention_mask is not None
                            and hasattr(item.pixel_attention_mask, "shape")
                            else None
                        ),
                        "has_spatial_shapes": hasattr(item, "spatial_shapes")
                        and item.spatial_shapes is not None,
                        "spatial_shapes_val": (
                            item.spatial_shapes.tolist()
                            if hasattr(item, "spatial_shapes")
                            and item.spatial_shapes is not None
                            and hasattr(item.spatial_shapes, "tolist")
                            else None
                        ),
                    }
                )
            _dbg_proc(
                "H3",
                "processor:mm_items",
                "Processed mm_items details",
                {
                    "num_items": len(_items_list),
                    "item_details": _item_details,
                    "input_ids_len": (
                        len(input_ids) if hasattr(input_ids, "__len__") else None
                    ),
                    "image_token_count": (
                        int((input_ids == self.mm_tokens.image_token_id).sum())
                        if hasattr(input_ids, "__eq__")
                        else None
                    ),
                },
            )
        except Exception as e:
            _dbg_proc("H3", "processor:mm_items", f"Error logging mm_items: {e}", {})
        # #endregion

        return {
            "input_ids": input_ids.tolist(),
            "mm_items": mm_items,
            "im_token_id": self.mm_tokens.image_token_id,
        }
