# SPDX-License-Identifier: Apache-2.0
# Copyright 2023-2025 SGLang Team
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
"""Inference-only Mamba model (state-spaces/mamba-*) for SGLang.

The canonical Mamba-1 (selective-scan) state-space model. It is Falcon-Mamba
without the weightless B/C/dt RMSNorm, so it reuses FalconMambaForCausalLM and
only flips ``use_bc_dt_rms`` off; the tied LM head follows
``config.tie_word_embeddings``.

Reference: https://huggingface.co/state-spaces/mamba-130m-hf
"""

from sglang.srt.models.falcon_mamba import FalconMambaForCausalLM


class MambaForCausalLM(FalconMambaForCausalLM):
    # Plain Mamba has no B/C/dt RMSNorm (that is Falcon-Mamba's variant).
    use_bc_dt_rms: bool = False


EntryClass = MambaForCausalLM
