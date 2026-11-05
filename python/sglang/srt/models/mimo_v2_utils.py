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

import os


def mimo_flydsl_target_verify_enabled() -> bool:
    """Both target attention families must support the Flash native-V cache."""
    return (
        os.environ.get("SGLANG_AITER_PA_DECODE_IMPL", "gluon").strip().lower()
        == "flydsl"
        and os.environ.get("SGLANG_AITER_TARGET_VERIFY_SWA_IMPL", "gluon")
        .strip()
        .lower()
        == "flydsl"
    )


def supports_native_mimo_vectorized_v_cache(
    *,
    attention_backend: str,
    kv_cache_layout: str,
    head_dim: int,
    swa_head_dim: int | None,
    v_head_dim: int | None,
    swa_v_head_dim: int | None,
    full_num_kv_heads: int,
    swa_num_kv_heads: int,
) -> bool:
    """Return whether every MiMo attention pool can store native V128."""
    return (
        attention_backend == "aiter"
        and kv_cache_layout == "vectorized_5d"
        and head_dim == 192
        and swa_head_dim == 192
        and v_head_dim == 128
        and swa_v_head_dim == 128
        and full_num_kv_heads == 1
        and (
            swa_num_kv_heads == 1
            or (swa_num_kv_heads == 2 and mimo_flydsl_target_verify_enabled())
        )
    )


def mimo_effective_prefix_lengths(forward_batch) -> list[int] | None:
    """Recover logical prefix lengths when chunk-cache accounting reports zero."""
    reported = getattr(forward_batch, "extend_prefix_lens_cpu", None)
    if reported is None:
        return None
    reported = [int(length) for length in reported]

    extend = getattr(forward_batch, "extend_seq_lens_cpu", None)
    seq_lens = getattr(forward_batch, "seq_lens_cpu", None)
    if extend is None or seq_lens is None or len(extend) != len(seq_lens):
        return reported

    derived = [
        int(seq_len) - int(extend_len) for seq_len, extend_len in zip(seq_lens, extend)
    ]
    return derived if all(length >= 0 for length in derived) else reported
