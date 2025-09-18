from typing import Optional, Dict

from pydantic import BaseModel, Field, field_validator


class DeploymentConfig(BaseModel):
    # Auth and model selection
    hf_token: Optional[str] = Field(default=None, description="Hugging Face access token")
    model_id: str = Field(description="Hugging Face model ID or local path")
    revision: Optional[str] = Field(default=None, description="Optional model revision")

    # Precision and quantization
    dtype: str = Field(default="auto", description="Data type: auto|fp16|bf16|fp32")
    quantization: Optional[str] = Field(
        default=None, description="Quantization strategy: fp8|int8|int4|awq|gptq|auto"
    )
    kv_cache_dtype: str = Field(default="auto", description="KV cache dtype: auto|fp8_e5m2|fp16|bf16")

    # Parallelism and resources
    tp_size: int = Field(default=1, description="Tensor parallel size (1,2,4,8)")
    max_total_tokens: int = Field(default=4096, description="Global token budget for KV cache")
    mem_fraction_static: Optional[float] = Field(default=None, description="Fraction of GPU mem reserved for weights+KV")
    schedule_conservativeness: float = Field(default=1.0, description="Scheduler aggressiveness 0.1-2.0")
    max_queued_requests: Optional[int] = Field(default=None, description="Max requests queued before 429")

    # Server
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    log_level: str = Field(default="info")
    enable_metrics: bool = Field(default=True)
    log_requests: bool = Field(default=True)
    enable_p2p_check: bool = Field(default=False)
    enable_torch_compile: bool = Field(default=False)
    enable_memory_saver: bool = Field(default=True, description="Enable memory saver to reduce VRAM usage")
    enable_expandable_segments: bool = Field(
        default=True,
        description="If true, set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to reduce fragmentation",
    )

    # Platform constraints and heuristics
    h100_only: bool = Field(default=True, description="Fail fast if GPUs are not H100 class")
    auto_use_fp8_on_h100: bool = Field(default=False, description="If H100, prefer fp8 weights + fp8 kv cache when safe (opt-in)")

    # Speculative decoding (EAGLE / EAGLE3)
    speculative_algorithm: Optional[str] = Field(default=None, description="None|EAGLE|EAGLE3|STANDALONE")
    speculative_draft_model_path: Optional[str] = Field(default=None, description="Draft model for speculative decoding")
    speculative_draft_model_revision: Optional[str] = Field(default=None, description="Revision of draft model")
    speculative_num_steps: Optional[int] = Field(default=None, description="Depth of autoregressive drafting")
    speculative_eagle_topk: Optional[int] = Field(default=None, description="Branching factor per step")
    speculative_num_draft_tokens: Optional[int] = Field(default=None, description="Max parallel verification capacity")
    speculative_token_map: Optional[str] = Field(default=None, description="FR-Spec vocabulary subset path")
    speculative_attention_mode: Optional[str] = Field(default=None, description="prefill|decode; defaults to prefill")

    # Autoscaling hints (consumed by external autoscaler)
    autoscale_target_tokens_per_s: Optional[float] = Field(default=None)
    autoscale_target_queue_depth: Optional[int] = Field(default=None)

    # Tenancy / metadata
    namespace: Optional[str] = Field(default=None)
    deployment_id: Optional[str] = Field(default=None)
    customer_id: Optional[str] = Field(default=None)
    labels: Optional[Dict[str, str]] = Field(default=None)

    # Model artifact location (optional). If set to a local path, overrides model_id
    artifacts_uri: Optional[str] = Field(default=None)

    # Enrollment callback endpoint (Developer Platform)
    enrollment_url: Optional[str] = Field(default=None)

    @field_validator("tp_size")
    @classmethod
    def _validate_tp(cls, v: int) -> int:
        if v not in (1, 2, 4, 8):
            raise ValueError("tp_size must be one of 1,2,4,8")
        return v


