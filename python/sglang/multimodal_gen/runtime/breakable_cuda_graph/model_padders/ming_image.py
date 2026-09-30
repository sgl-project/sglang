# SPDX-License-Identifier: Apache-2.0
"""Keep Ming's trained padding registers and RoPE offsets unchanged."""

from sglang.multimodal_gen.runtime.breakable_cuda_graph import prompt_padding


def is_ming_image(current_model, call_kwargs):
    return prompt_padding.transformer_class_name_matches(current_model, "MingImage")


def preserve_ming_prompt(call_kwargs, current_model, buckets):
    # Changing caption length also shifts every image's first RoPE coordinate.
    # Capture exact signatures; uncaptured prompt lengths run eager.
    return call_kwargs


prompt_padding.register_prompt_padder(is_ming_image, preserve_ming_prompt)
