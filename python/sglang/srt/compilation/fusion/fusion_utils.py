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

import hashlib
import inspect
import json
import types
from typing import Any, Union


def hash_source(*srcs: Union[str, Any]):
    """
    Utility method to hash the sources of functions or objects.
    :param srcs: strings or objects to add to the hash.
    Objects and functions have their source inspected.
    :return:
    """
    hasher = hashlib.sha256()
    for src in srcs:
        if isinstance(src, str):
            src_str = src
        elif isinstance(src, (types.FunctionType, type)):
            src_str = inspect.getsource(src)
        else:
            # object instance
            src_str = inspect.getsource(src.__class__)
        hasher.update(src_str.encode("utf-8"))
    return hasher.hexdigest()


def hash_dict(dict_: dict[Any, Any]):
    """
    Utility method to hash a dictionary, can alternatively be used for uuid.
    :return: A sha256 hash of the json rep of the dictionary.
    """
    encoded = json.dumps(dict_, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
