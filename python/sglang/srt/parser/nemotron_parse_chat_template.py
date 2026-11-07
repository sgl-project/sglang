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


def render_nemotron_parse_segments(messages, tools=None, **kwargs):
    if tools:
        raise ValueError("Nemotron Parse does not support tool calls")
    # Image embeddings are prepended by the processor.
    segments = []
    for message in messages:
        content = message["content"]
        if isinstance(content, str):
            segments.append(content)
            continue
        for part in content:
            if part["type"] == "text":
                segments.append(part["text"])
            elif part["type"] not in ("image", "image_url"):
                raise ValueError(
                    "Nemotron Parse supports text and document images only"
                )
    return segments
