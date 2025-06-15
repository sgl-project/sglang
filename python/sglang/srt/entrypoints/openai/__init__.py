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
"""
OpenAI-compatible API server module for SGLang.

This module provides OpenAI-compatible API endpoints that allow existing OpenAI client
applications to seamlessly work with SGLang models. The implementation includes:

Key Features:
- Full OpenAI API compatibility for chat completions, text completions, and embeddings
- Streaming support for real-time response generation
- Batch processing capabilities for multiple requests
- Function calling and tool use support
- Multimodal input support (text, images, audio)
- Advanced reasoning capabilities with separate reasoning content
- Custom sampling parameters and constraints (regex, JSON schema, EBNF)
- LoRA adapter support for fine-tuned models
- Cache reporting and token usage tracking

Supported Endpoints:
- /v1/chat/completions - Chat-based completions with conversation history
- /v1/completions - Text completions for single prompts
- /v1/embeddings - Text/multimodal embeddings generation
- /v1/models - Model listing and information

The module is structured with separate handlers for each endpoint type, all inheriting
from a common base class that provides shared functionality like request validation,
error handling, and response formatting.

Architecture:
- OpenAIServingBase: Abstract base class for all endpoint handlers
- OpenAIServingChat: Handles chat completion requests
- OpenAIServingCompletion: Handles text completion requests
- OpenAIServingEmbedding: Handles embedding requests
- Protocol classes: Pydantic models for request/response validation
- Utility functions: Shared helpers for formatting and validation
"""
