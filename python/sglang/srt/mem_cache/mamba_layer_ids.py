# Copyright 2023-2026 SGLang Team
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
"""Layer-id list for the request pool's mamba (conv + SSM) state.

Torch-free on purpose: the decision of WHICH layers own a slice of the mamba
state pool is plain arithmetic over the model config, and keeping it here lets
it be unit-tested without a GPU (``test/registered/unit/mem_cache/
test_mamba_layer_ids.py``). ``KVCacheConfigurator._get_mamba_layer_ids_for_req_pool``
is the only production caller.

Every id in the returned list costs one layer of conv + SSM state in the device
``MambaPool`` and in ``MambaPoolHost`` (the pools size their layer axis from
``len(mamba_layer_ids)``), so an id that never indexes mamba state is dead
weight: on GLM-5.3-Flash (34 KDA layers + 1 DSA NextN draft layer) it is 1/35
of the state pool.
"""

from __future__ import annotations

from typing import Callable, Iterable, List, Optional


def resolve_req_pool_mamba_layer_ids(
    mamba_layers: Iterable[int],
    start_layer: int,
    end_layer: int,
    nextn_layer_ids: Iterable[int],
    is_linear_layer: Optional[Callable[[int], bool]],
    spec_enabled: bool,
) -> List[int]:
    """Return the layer ids whose conv/SSM state the request pool must hold.

    Args:
        mamba_layers: the config's linear-attention (mamba-ish) layer ids,
            ``mambaish_config.mamba2_cache_params.layers``.
        start_layer, end_layer: this pipeline rank's ``[start, end)`` slice;
            only ``mamba_layers`` inside it are kept.
        nextn_layer_ids: the config's NextN (MTP draft) layer ids, or empty when
            the config does not expose them.
        is_linear_layer: the config's layer classifier (``is_kda_layer`` on
            GLM-5.3-Flash / Kimi Linear configs) or ``None`` when the config
            has none.
        spec_enabled: whether speculative decoding is on
            (``max_speculative_num_draft_tokens()`` truthy).

    With speculative decoding on, a NextN layer is appended (regardless of the
    pipeline slice, as before) unless the config can classify it AND classifies
    it as non-linear: such a draft layer (GLM-5.3-Flash's NextN block is a DSA
    attention layer that never touches ``mamba_map``) owns no mamba state.
    Configs without a classifier keep the unconditional append.
    """
    mamba_layer_ids = [i for i in mamba_layers if start_layer <= i < end_layer]
    if not spec_enabled:
        return mamba_layer_ids
    for layer_id in nextn_layer_ids:
        if layer_id in mamba_layer_ids:
            continue
        if is_linear_layer is not None and not is_linear_layer(layer_id):
            continue
        mamba_layer_ids.append(layer_id)
    return mamba_layer_ids
