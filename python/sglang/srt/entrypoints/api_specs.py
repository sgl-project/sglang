from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Union,
)

from sglang.srt.entrypoints.openai.protocol import Tool

# All spec classes used for APIs are defined here.
# They are extracted from the io_struct.py.


@dataclass
class AttachHiCacheStorageReqInputSpec:
    """Dynamically attach (enable) HiCache storage backend at runtime.

    Note: `hicache_storage_backend_extra_config_json` is a JSON string. It may contain both:
    - backend-specific configs (e.g., mooncake master address)
    - prefetch-related knobs (prefetch_threshold, prefetch_timeout_*, hicache_storage_pass_prefix_keys)
    """

    hicache_storage_backend: str
    hicache_storage_backend_extra_config_json: Optional[str] = None
    hicache_storage_prefetch_policy: Optional[str] = None
    hicache_write_policy: Optional[str] = None

    def __post_init__(self):
        if self.hicache_storage_prefetch_policy is None:
            pass
        else:
            allowed = ["best_effort", "wait_complete", "timeout"]
            if self.hicache_storage_prefetch_policy not in allowed:
                raise ValueError(
                    f"Invalid hicache_storage_prefetch_policy: {self.hicache_storage_prefetch_policy!r}. "
                    f"Expected one of {allowed}."
                )

        if self.hicache_write_policy is None:
            return
        allowed = ["write_back", "write_through", "write_through_selective"]
        if self.hicache_write_policy not in allowed:
            raise ValueError(
                f"Invalid hicache_write_policy: {self.hicache_write_policy!r}. "
                f"Expected one of {allowed}."
            )


@dataclass
class UpdateWeightFromDiskReqInputSpec:
    # The model path with the new weights
    model_path: str
    # The format to load the weights
    load_format: Optional[str] = None
    # Whether to abort all requests before updating weights
    abort_all_requests: bool = False
    # Optional: Update weight version along with weights
    weight_version: Optional[str] = None
    # Whether to update weights asynchronously
    is_async: bool = False
    # Whether to call torch.cuda.empty_cache() during flush
    torch_empty_cache: bool = False
    # Whether to keep the scheduler paused after weight update
    keep_pause: bool = False
    # Whether to recapture cuda graph after weight update
    recapture_cuda_graph: bool = False
    # The trainer step id. Used to know which step's weights are used for sampling.
    token_step: int = 0
    # Whether to flush the cache after updating weights
    flush_cache: bool = True
    # Tensor metadata
    manifest: Optional[Dict[str, Any]] = None


@dataclass
class InitWeightsSendGroupForRemoteInstanceReqInputSpec:
    # The master address
    master_address: str
    # The ports for each rank's communication group
    ports: str
    # The rank in the communication group
    group_rank: int
    # The world size
    world_size: int
    # The group name
    group_name: str = "weight_send_group"
    # The backend
    backend: str = "nccl"


@dataclass
class InitWeightsUpdateGroupReqInputSpec:
    # The master address
    master_address: str
    # The master port
    master_port: int
    # The rank offset
    rank_offset: int
    # The world size
    world_size: int
    # The group name
    group_name: str = "weight_update_group"
    # The backend
    backend: str = "nccl"


@dataclass
class SendWeightsToRemoteInstanceReqInputSpec:
    # The master address
    master_address: str
    # The ports for each rank's communication group
    ports: str
    # The group name
    group_name: str = "weight_send_group"


@dataclass
class DestroyWeightsUpdateGroupReqInputSpec:
    group_name: str = "weight_update_group"


@dataclass
class UpdateWeightsFromTensorReqInputSpec:
    # List[Union[str, bytes]]
    serialized_named_tensors: List[Union[str, bytes]]
    # Optional format specification for loading
    load_format: Optional[str] = None
    # Whether to flush the cache after updating weights
    flush_cache: bool = True
    # Whether to abort all requests before updating weights
    abort_all_requests: bool = False
    # Optional: Update weight version along with weights
    weight_version: Optional[str] = None
    # Optional: Determine whether to disable updating the draft model
    disable_draft_model: Optional[bool] = None
    # Whether to call torch.cuda.empty_cache() during flush
    torch_empty_cache: bool = False


@dataclass
class UpdateWeightsFromDistributedReqInputSpec:
    names: List[str]
    dtypes: List[str]
    shapes: List[List[int]]
    # The group name
    group_name: str = "weight_update_group"
    # Whether to flush the cache after updating weights
    flush_cache: bool = True
    # Whether to abort all requests before updating weights
    abort_all_requests: bool = False
    # Optional: Update weight version along with weights
    weight_version: Optional[str] = None
    # Optional format specification for loading
    load_format: Optional[str] = None
    # Whether to call torch.cuda.empty_cache() during flush
    torch_empty_cache: bool = False


@dataclass
class UpdateWeightsFromIPCReqInputSpec:
    # ZMQ socket paths for each device UUID
    zmq_handles: Dict[str, str]
    # Whether to flush cache after weight update
    flush_cache: bool = True
    # Optional: Update weight version along with weights
    weight_version: Optional[str] = None
    # Whether to call torch.cuda.empty_cache() during flush
    torch_empty_cache: bool = False


@dataclass
class UpdateWeightVersionReqInputSpec:
    # The new weight version
    new_version: str
    # Whether to abort all running requests before updating
    abort_all_requests: bool = True


@dataclass
class GetWeightsByNameReqInputSpec:
    name: str
    truncate_size: int = 100


@dataclass
class SetInternalStateReqSpec:
    server_args: Dict[str, Any]


@dataclass
class ReleaseMemoryOccupationReqInputSpec:
    tags: Optional[List[str]] = None


@dataclass
class ResumeMemoryOccupationReqInputSpec:
    tags: Optional[List[str]] = None


@dataclass
class CheckWeightsReqInputSpec:
    action: str = "checksum"


@dataclass
class SlowDownReqInputSpec:
    forward_sleep_time: Optional[float]


@dataclass
class LoadLoRAAdapterReqInputSpec:
    # The name of the lora module to newly loaded.
    lora_name: str
    # The path of loading.
    lora_path: str
    # Whether to pin the LoRA adapter in memory.
    pinned: bool = False
    # The unique identifier for the LoRA adapter, which automatically generated in the `TokenizerManager`.
    lora_id: Optional[str] = None


@dataclass
class LoadLoRAAdapterFromTensorsReqInputSpec:
    lora_name: str
    config_dict: Dict[str, Any]
    serialized_tensors: str
    pinned: bool = False
    added_tokens_config: Optional[Dict[str, Any]] = None
    lora_id: Optional[str] = None
    load_format: Optional[str] = None


@dataclass
class UnloadLoRAAdapterReqInputSpec:
    # The name of lora module to unload.
    lora_name: str
    # The unique identifier for the LoRA adapter, which automatically generated in the `TokenizerManager`.
    lora_id: Optional[str] = None


@dataclass
class OpenSessionReqInputSpec:
    capacity_of_str_len: int
    session_id: Optional[str] = None
    streaming: Optional[bool] = None
    timeout: Optional[float] = None


@dataclass
class CloseSessionReqInputSpec:
    session_id: str


@dataclass
class ConfigureLoggingReqSpec:
    log_requests: Optional[bool] = None
    log_requests_level: Optional[int] = None
    log_requests_format: Optional[str] = None
    log_level: Optional[str] = None
    dump_requests_folder: Optional[str] = None
    dump_requests_threshold: Optional[int] = None
    crash_dump_folder: Optional[str] = None
    dump_requests_exclude_meta_keys: Optional[List[str]] = None


@dataclass
class AbortReqSpec:
    rid: Optional[str] = None
    # Whether to abort all requests
    abort_all: bool = False


@dataclass
class PauseGenerationReqInputSpec:
    mode: Literal["abort", "retract", "in_place"] = "abort"

    def __post_init__(self):
        allowed = ["abort", "retract", "in_place"]
        if self.mode not in allowed:
            raise ValueError(
                f"Invalid mode: {self.mode!r}. " f"Expected one of {allowed}."
            )


@dataclass
class ContinueGenerationReqInputSpec:
    torch_empty_cache: bool = True


@dataclass
class SeparateReasoningReqInputSpec:
    text: str  # The text to parse.
    reasoning_parser: str  # Specify the parser type, e.g., "deepseek-r1".
    return_blocks: bool = False  # If True, also return segmented reasoning blocks.


@dataclass
class VertexGenerateReqInputSpec:
    instances: List[dict]
    parameters: Optional[dict] = None


@dataclass
class ParseFunctionCallReqSpec:
    text: str  # The text to parse.
    tools: List[Tool] = field(
        default_factory=list
    )  # A list of available function tools (name, parameters, etc.).
    tool_call_parser: Optional[str] = (
        None  # Specify the parser type, e.g. 'llama3', 'qwen25', or 'mistral'. If not specified, tries all.
    )


@dataclass
class ProfileReqInputSpec:
    # The output directory
    output_dir: Optional[str] = None
    # Specify the steps to start the profiling
    start_step: Optional[int] = None
    # If set, it profile as many as this number of steps.
    # If it is set, profiling is automatically stopped after this step, and
    # the caller doesn't need to run stop_profile.
    num_steps: Optional[int] = None
    # The activities to record. The choices are ["CPU", "GPU", "MEM", "RPD"]
    activities: Optional[List[str]] = None
    # Whether profile by stages (e.g., prefill and decode) separately
    profile_by_stage: bool = False
    # Whether to record source information (file and line number) for the ops.
    with_stack: Optional[bool] = None
    # Whether to save information about operator’s input shapes.
    record_shapes: Optional[bool] = None
    # Merge profiles from all ranks into a single trace
    merge_profiles: bool = False
    # The prefix of the profile filenames
    profile_prefix: Optional[str] = None
    # Only profile these stages and ignore others
    profile_stages: Optional[List[str]] = None
