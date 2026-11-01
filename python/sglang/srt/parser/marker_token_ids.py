# Copyright 2026 SGLang Team
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
"""Token ids of framing markers that the model emits, such as reasoning end tags."""

from typing import List, Optional

from transformers import PreTrainedTokenizerBase


def marker_token_ids(*, tokenizer: PreTrainedTokenizerBase, marker: str) -> List[int]:
    # A tokenizer that does not parse special tokens out of text, for example one
    # loaded with split_special_tokens=True, encodes a special marker's printed
    # name as ordinary tokens the model never emits.
    token_id = _single_token_id(tokenizer=tokenizer, marker=marker)
    if token_id is not None:
        return [token_id]
    return tokenizer.encode(marker, add_special_tokens=False)


def _single_token_id(
    *, tokenizer: PreTrainedTokenizerBase, marker: str
) -> Optional[int]:
    try:
        token_id = tokenizer.convert_tokens_to_ids(marker)
    except Exception:
        # A tokenizer that cannot look the marker up keeps the text-encoding path.
        return None
    if token_id in tokenizer.all_special_ids:
        # An unknown marker maps to the unknown token, which is itself special.
        return token_id if tokenizer.convert_ids_to_tokens(token_id) == marker else None
    # Added tokens marked special are split too, but are not all in all_special_ids.
    return token_id if tokenizer.get_added_vocab().get(marker) == token_id else None
