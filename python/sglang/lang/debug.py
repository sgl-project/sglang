import json
import urllib.request
from datetime import datetime
from typing import Any, List, Optional, Union

import requests
from pydantic import BaseModel, Field

from sglang.utils import http_request


# Base models equivalent to TypeScript types
class PromptType(BaseModel):
    type: str
    created_at: str
    updated_at: str


class Prompt(BaseModel):
    id: str
    type: str
    requests: str  # JSON string of array of PromptRequest
    created_at: str
    updated_at: str


class PromptRequest(BaseModel):
    id: str
    requestPrompt: str
    requestTimestamp: str
    requestMetadata: Any  # JSON object
    responseContent: Optional[str] = None
    responseTimestamp: Optional[str] = None
    responseMetadata: Optional[Any] = None  # JSON object


class PromptRequestUpdate(BaseModel):
    id: str
    responseContent: str
    responseTimestamp: str
    responseMetadata: Any  # JSON object


# Union type for request body
PostBodyRequest = Union[PromptRequestUpdate, PromptRequest]


class PostBody(BaseModel):
    type: str
    id: Optional[str] = None  # ULID field
    requests: List[PostBodyRequest]


class DebugInfo(BaseModel):
    base_url: str
    port: int
    debug_name: Optional[str] = None
    debug_prompt_id: Optional[str] = None


def post_studio_prompt(body: PostBody, debug_info: DebugInfo) -> Any:
    """
    Post a studio prompt to the API endpoint using requests.

    Args:
        body: The PostBody object containing the prompt data
        debug_info: Debug configuration information

    Returns:
        The response data from the API

    Raises:
        requests.RequestException: If the HTTP request fails
    """
    print("DEBUGGGGGGING")
    url = f"{debug_info.base_url}:{debug_info.port}/api/prompt"
    return http_request(url, json=body.model_dump(exclude_none=True))
