# Copyright 2023-2024 SGLang Team
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

from typing import Any, Dict, List, Union

import einops
import numpy as np
import torch

# from sglang.srt.models.qwen3_vl import Qwen3VLForConditionalGeneration
from sglang.srt.models.alpamayo_r1 import Alpamayo1_5, AlpamayoR1
from sglang.srt.multimodal.processors.qwen_vl import QwenVLImageProcessor
from sglang.utils import logger

# Constants from alpamayo_r1/models/base_model.py
TRAJ_TOKEN = {
    "history": "<|traj_history|>",
    "future": "<|traj_future|>",
    "history_start": "<|traj_history_start|>",
    "future_start": "<|traj_future_start|>",
    "history_end": "<|traj_history_end|>",
    "future_end": "<|traj_future_end|>",
}

SPECIAL_TOKENS_KEYS = [
    "prompt_start",
    "prompt_end",
    "image_start",
    "image_pre_tkn",
    "image_end",
    "traj_history_start",
    "traj_history_pre_tkn",
    "traj_history_end",
    "cot_start",
    "cot_end",
    "meta_action_start",
    "meta_action_end",
    "traj_future_start",
    "traj_future_pre_tkn",
    "traj_future_end",
    "traj_history",
    "traj_future",
    "image_pad",
    "vectorized_wm",
    "vectorized_wm_start",
    "vectorized_wm_end",
    "vectorized_wm_pre_tkn",
    "route_start",
    "route_pad",
    "route_end",
    "question_start",
    "question_end",
    "answer_start",
    "answer_end",
]


# Copy from alpamayo_r1/models/delta_tokenizer.py
class DeltaTrajectoryTokenizer:
    """Delta trajectory tokenizers."""

    def __init__(
        self,
        ego_xyz_min: tuple[float, float, float] = (-4, -4, -10),
        ego_xyz_max: tuple[float, float, float] = (4, 4, 10),
        ego_yaw_min: float = -np.pi,
        ego_yaw_max: float = np.pi,
        num_bins: int = 1000,
        predict_yaw: bool = False,
        load_weights: bool = False,
    ):
        """Initializes the tokenizer."""
        self.ego_xyz_min = ego_xyz_min
        self.ego_xyz_max = ego_xyz_max
        self.num_bins = num_bins
        self._predict_yaw = predict_yaw
        self.ego_yaw_min = ego_yaw_min
        self.ego_yaw_max = ego_yaw_max

    @property
    def vocab_size(self) -> int:
        """Tokens are integers from the set {0, 1, ..., vocab_size - 1}"""
        return self.num_bins

    def encode(
        self,
        hist_xyz: torch.Tensor,
        hist_rot: torch.Tensor,
        fut_xyz: torch.Tensor,
        fut_rot: torch.Tensor,
        hist_tstamp: torch.Tensor | None = None,
        fut_tstamp: torch.Tensor | None = None,
    ) -> torch.LongTensor:
        """Encodes the trajectories as discrete tokens."""
        del hist_xyz, hist_rot, hist_tstamp, fut_tstamp
        xyz = torch.nn.functional.pad(fut_xyz, [0, 0, 1, 0, 0, 0])
        xyz = xyz[:, 1:] - xyz[:, :-1]
        ego_xyz_max = torch.tensor(self.ego_xyz_max, dtype=xyz.dtype, device=xyz.device)
        ego_xyz_min = torch.tensor(self.ego_xyz_min, dtype=xyz.dtype, device=xyz.device)
        xyz = (xyz - ego_xyz_min) / (ego_xyz_max - ego_xyz_min)
        xyz = (xyz * (self.num_bins - 1)).round().long()
        xyz = xyz.clamp(0, self.num_bins - 1)
        if not self._predict_yaw:
            return einops.rearrange(xyz, "b n m -> b (n m)")
        # Extract yaw angles from rotation matrices
        yaw = torch.atan2(fut_rot[..., 0, 1], fut_rot[..., 0, 0])

        # Calculate delta yaw
        yaw_padded = torch.nn.functional.pad(yaw, [1, 0, 0, 0])
        delta_yaw = yaw_padded[:, 1:] - yaw_padded[:, :-1]

        # Normalize delta yaw to [-pi, pi]
        delta_yaw = torch.atan2(torch.sin(delta_yaw), torch.cos(delta_yaw))

        # Scale and quantize delta yaw
        delta_yaw = (delta_yaw - self.ego_yaw_min) / (
            self.ego_yaw_max - self.ego_yaw_min
        )
        delta_yaw = (delta_yaw * (self.num_bins - 1)).round().long()
        delta_yaw = delta_yaw.clamp(0, self.num_bins - 1)

        xyzw = torch.cat([xyz, delta_yaw.unsqueeze(-1)], dim=-1)  # Shape: (B, Tf, 4)
        return einops.rearrange(xyzw, "b n m -> b (n m)")


# Copy from alpamayo_r1/models/base_model.py
def replace_pad_token(
    input_ids: torch.Tensor, new_ids: torch.Tensor, pad_idx: int
) -> torch.Tensor:
    """Replace pad tokens in input_ids with new token values."""
    mask = input_ids == pad_idx
    # Make sure new_ids matches the number of masked elements
    # Since masked_scatter expects same number of elements as sum(mask)
    return input_ids.masked_scatter(mask, new_ids)


def tokenize_history_trajectory(
    tokenizer: Any, traj_data: dict[str, Any], start_idx: int = 0
) -> torch.Tensor:
    """Tokenize the history trajectory with prefix shape of (B, n_traj, ...)."""
    assert "ego_history_xyz" in traj_data
    assert (
        traj_data["ego_history_xyz"].ndim == 4
    ), "ego_history_xyz must be 4D of [B, n_traj, T, 3]"

    B = traj_data["ego_history_xyz"].shape[0]
    hist_xyz = traj_data["ego_history_xyz"].flatten(start_dim=0, end_dim=1)
    hist_rot = traj_data["ego_history_rot"].flatten(start_dim=0, end_dim=1)

    hist_idx = (
        tokenizer.encode(
            hist_xyz=hist_xyz[:, :1],
            hist_rot=hist_rot[:, :1],
            fut_xyz=hist_xyz,
            fut_rot=hist_rot,
        )
        + start_idx
    )
    hist_idx = einops.rearrange(hist_idx, "(b n_traj) n -> b (n_traj n)", b=B)

    return hist_idx


class AlpamayoR1Processor(QwenVLImageProcessor):
    models = [AlpamayoR1, Alpamayo1_5]

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        # IMPORTANT: Add trajectory tokens to the tokenizer BEFORE calling super().__init__
        # This ensures tokens are available during processor initialization
        self._add_trajectory_tokens_to_processor(_processor, hf_config)

        super().__init__(hf_config, server_args, _processor, *args, **kwargs)

        # Apply min_pixels / max_pixels from model config to the HF image processor.
        if hasattr(hf_config, "min_pixels"):
            _processor.image_processor.min_pixels = hf_config.min_pixels
        if hasattr(hf_config, "max_pixels"):
            _processor.image_processor.max_pixels = hf_config.max_pixels

        # Initialize trajectory tokenizer configs
        self.hist_traj_tokenizer = DeltaTrajectoryTokenizer()

    def _add_trajectory_tokens_to_processor(self, processor, hf_config):
        """Add trajectory tokens to the tokenizer before it's used.

        This mirrors alpamayo_r1/models/base_model.py::_build_processor logic.
        """
        tokenizer = processor.tokenizer

        # Add discrete trajectory tokens <i0> to <i999>
        traj_vocab_size = getattr(hf_config, "traj_vocab_size", 4000)

        discrete_tokens = [f"<i{v}>" for v in range(traj_vocab_size)]
        num_new_tokens = tokenizer.add_tokens(discrete_tokens)
        logger.info(f"Added {num_new_tokens} discrete trajectory tokens to tokenizer")

        # Store indices for later use
        self.traj_token_start_idx = tokenizer.convert_tokens_to_ids("<i0>")
        self.traj_token_end_idx = tokenizer.convert_tokens_to_ids(
            f"<i{traj_vocab_size - 1}>"
        )

        # Check if we should add all special tokens or just traj tokens
        add_special_tokens = getattr(hf_config, "add_special_tokens", True)

        if add_special_tokens:
            # Add all special tokens defined in alpamayo
            special_tokens = [f"<|{k}|>" for k in SPECIAL_TOKENS_KEYS]
            tokenizer.add_tokens(special_tokens, special_tokens=True)
            logger.info(f"Added {len(special_tokens)} special tokens to tokenizer")
        else:
            # Only add trajectory tokens
            tokenizer.add_tokens(list(TRAJ_TOKEN.values()), special_tokens=True)
            logger.info(
                f"Added {len(TRAJ_TOKEN)} trajectory special tokens to tokenizer"
            )

        # Store traj_token_ids mapping
        self.traj_token_ids = {
            k: tokenizer.convert_tokens_to_ids(v) for k, v in TRAJ_TOKEN.items()
        }

    async def process_mm_data_async(
        self,
        image_data: List[Union[str, bytes]],
        input_text,
        request_obj,
        *args,
        **kwargs,
    ):
        # 1. Reuse Qwen3-VL processor logic
        result = await super().process_mm_data_async(
            image_data, input_text, request_obj, *args, **kwargs
        )

        # 2. Check and fuse history_traj
        # history_traj is passed via GenerateReqInput/request_obj as Dict[str, Any]
        history_traj: Dict[str, Any] = getattr(request_obj, "history_traj", None)

        if history_traj:
            input_ids_list = result.input_ids
            # Convert to tensor for manipulation. Note: SGLang processors return CPU lists usually.
            input_ids_tensor = torch.tensor(input_ids_list, dtype=torch.long).unsqueeze(
                0
            )  # [B=1, L]

            # Prepare traj_data converting lists to tensors if necessary
            traj_data = {}
            for k, v in history_traj.items():
                traj_data[k] = (
                    torch.tensor(v) if isinstance(v, (list, np.ndarray)) else v
                )

            # Ensure traj_data elements have correct dimensions.
            # If the user passes numpy array without batch dim (since it's per request), we might need to unsqueeze.
            # But wait, input_ids_tensor [1, L] implies batch=1.
            # So traj_data should have batch=1.

            if "ego_history_xyz" in traj_data:
                t = traj_data["ego_history_xyz"]
                while t.ndim < 4:  # 需要 [B, n_traj, T, 3]
                    t = t.unsqueeze(0)
                traj_data["ego_history_xyz"] = t

            # Fuse tokens
            fused_ids = self.fuse_traj_tokens(input_ids_tensor, traj_data)

            # Update result
            result.input_ids = fused_ids.squeeze(0).tolist()

        return result

    def fuse_traj_tokens(
        self, input_ids: torch.Tensor, traj_data: dict[str, Any] | None = None
    ) -> torch.Tensor:
        """Fuse the trajectory tokens into the input ids."""
        if (
            traj_data is None
            or traj_data.get("ego_history_xyz") is None
            or traj_data.get("ego_history_rot") is None
        ):
            logger.warning(
                "Trajectory data is missing or incomplete. Skipping trajectory fusion."
            )
            return input_ids

        # Reuse tokenize_history_trajectory helper
        hist_idx = tokenize_history_trajectory(
            self.hist_traj_tokenizer, traj_data, self.traj_token_start_idx
        )

        # We need the history placeholder token ID.
        placeholder_id = self.traj_token_ids.get("history")
        if placeholder_id is None:
            raise ValueError(
                "History placeholder token ID not found in tokenizer. Ensure it was added correctly."
            )

        input_ids = replace_pad_token(input_ids, hist_idx, placeholder_id)

        return input_ids
