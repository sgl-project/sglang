"""
Ollama-compatible API protocol definitions.

These models match the Ollama API format:
https://github.com/ollama/ollama/blob/main/docs/api.md
"""

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field


class OllamaMessage(BaseModel):
    """Ollama message format."""

    role: str
    content: str
    images: Optional[List[str]] = None


class OllamaChatRequest(BaseModel):
    """Ollama /api/chat request format."""

    model: str
    messages: List[OllamaMessage]
    stream: bool = True
    format: Optional[Union[Literal["json"], Dict[str, Any]]] = None
    options: Optional[Dict[str, Any]] = None
    keep_alive: Optional[Union[float, str]] = None
    think: Optional[Union[bool, Literal["low", "medium", "high"]]] = None


class OllamaChatResponse(BaseModel):
    """Ollama /api/chat response format (non-streaming)."""

    model: str
    created_at: str
    message: OllamaMessage
    done: bool = True
    done_reason: Optional[str] = "stop"
    total_duration: Optional[int] = None
    load_duration: Optional[int] = None
    prompt_eval_count: Optional[int] = None
    prompt_eval_duration: Optional[int] = None
    eval_count: Optional[int] = None
    eval_duration: Optional[int] = None


class OllamaChatStreamResponse(BaseModel):
    """Ollama /api/chat streaming response chunk."""

    model: str
    created_at: str
    message: OllamaMessage
    done: bool = False
    done_reason: Optional[str] = None


class OllamaGenerateRequest(BaseModel):
    """Ollama /api/generate request format."""

    model: str
    prompt: str
    suffix: Optional[str] = None
    system: Optional[str] = None
    template: Optional[str] = None
    context: Optional[List[int]] = None
    stream: bool = True
    raw: bool = False
    format: Optional[Union[Literal["json"], Dict[str, Any]]] = None
    options: Optional[Dict[str, Any]] = None
    keep_alive: Optional[Union[float, str]] = None
    images: Optional[List[str]] = None
    think: Optional[bool] = None


class OllamaGenerateResponse(BaseModel):
    """Ollama /api/generate response format (non-streaming)."""

    model: str
    created_at: str
    response: str
    done: bool = True
    done_reason: Optional[str] = "stop"
    context: Optional[List[int]] = None
    total_duration: Optional[int] = None
    load_duration: Optional[int] = None
    prompt_eval_count: Optional[int] = None
    prompt_eval_duration: Optional[int] = None
    eval_count: Optional[int] = None
    eval_duration: Optional[int] = None


class OllamaGenerateStreamResponse(BaseModel):
    """Ollama /api/generate streaming response chunk."""

    model: str
    created_at: str
    response: str
    done: bool = False
    done_reason: Optional[str] = None


class OllamaModelInfo(BaseModel):
    """Model information for /api/tags response."""

    name: str
    model: str
    modified_at: str
    size: int
    digest: str
    details: Optional[Dict[str, Any]] = None


class OllamaTagsResponse(BaseModel):
    """Ollama /api/tags response format."""

    models: List[OllamaModelInfo]


class OllamaShowRequest(BaseModel):
    """Ollama /api/show request format."""

    model: str


class OllamaShowResponse(BaseModel):
    """Ollama /api/show response format."""

    license: str = ""
    modelfile: str = ""
    parameters: str = ""
    template: str = ""
    modified_at: str = ""
    details: Dict[str, Any] = Field(default_factory=dict)
    model_info: Dict[str, Any] = Field(default_factory=dict)
    capabilities: List[str] = Field(default_factory=list)
