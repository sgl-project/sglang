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
# ==============================================================================
"""A post-norm residual: each stage reads the residual as it is, and its
output is normalized before it is added (EXAONE 4.0)."""

from sglang.srt.layers.layer_boundary.adapters.attention import (
    attn_tp_gather,
    attn_tp_slice,
)


class PostNormAdd:
    """``residual + norm(output)``. It is not a plain add, so it runs only on a
    complete output. ``applied_at_exit`` makes the stage write it itself."""

    is_plain_add = False
    outlives_layer = True

    def __init__(self, norm, *, applied_at_exit: bool = False):
        self.norm = norm
        self.applied_at_exit = applied_at_exit

    def update(self, hidden_states, residual):
        return self.norm(hidden_states) + residual

    def slice_residual_attn_tp(self, residual):
        return attn_tp_slice(residual)

    def gather_residual_attn_tp(self, residual):
        return attn_tp_gather(residual)


class PlainReadout:
    """The input is the residual itself; the stage's norm is not applied."""

    is_plain_norm = False
    reads_before_dp_gather = False

    def init_residual(self, hidden_states):
        return hidden_states

    def read(self, residual, norm, quant_format="", post_residual_addition=None):
        if quant_format or post_residual_addition is not None:
            raise NotImplementedError(
                f"a plain read with {quant_format=} or a post-residual addition"
            )
        return residual, residual

    def update_and_read(
        self,
        update,
        hidden_states,
        residual,
        norm,
        quant_format="",
        post_residual_addition=None,
    ):
        if residual is not None:
            hidden_states = update.update(hidden_states, residual)
        return self.read(hidden_states, norm, quant_format, post_residual_addition)


PLAIN_READOUT = PlainReadout()
