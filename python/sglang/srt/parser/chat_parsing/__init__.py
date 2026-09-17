# Copyright 2026 The HuggingFace Team. All rights reserved.
#
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
"""Streaming `response_template` parser.

Ported from `transformers.utils.chat_parsing` at commit
`9cfd6ab7c95f080dc1f33bcc7f97725ce783876e`. SGLang adapts logging and tool
schemas to avoid Transformers imports, and adds strict XML validation and raw
delimiter metadata for serving adapters.
"""

from .response_parser import ResponseParser, parse_response

__all__ = ["ResponseParser", "parse_response"]
