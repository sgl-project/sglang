"""
Copyright 2023-2024 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""The LLM engine arguments."""

import dataclasses
import logging
import os
import random
from typing import List, Optional, Union

from sglang.srt.constrained import disable_cache
from sglang.srt.utils import (
    allocate_init_ports,
    assert_pkg_version,
    enable_show_time_cost,
    maybe_set_triton_cache_manager,
    set_ulimit,
)

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class EngineArgs:
    # Model and tokenizer
    model_path: str
    tokenizer_path: Optional[str] = None
    tokenizer_mode: str = "auto"
    skip_tokenizer_init: bool = False
    load_format: str = "auto"
    dtype: str = "auto"
    kv_cache_dtype: str = "auto"
    trust_remote_code: bool = True
    context_length: Optional[int] = None
    quantization: Optional[str] = None
    served_model_name: Optional[str] = None
    random_seed: Optional[int] = None
    stream_interval: int = 1
    tokenizer_port: Optional[int] = 0
    detokenizer_port: Optional[int] = 0
    controller_port: Optional[int] = 0
    is_embedding: bool = False
    model_override_args: Optional[dict] = None

    # Scheduling
    mem_fraction_static: Optional[float] = None
    max_running_requests: Optional[int] = None
    max_num_reqs: Optional[int] = None
    max_total_tokens: Optional[int] = None
    chunked_prefill_size: int = 8192
    max_prefill_tokens: int = 16384
    schedule_policy: str = "lpm"
    schedule_conservativeness: float = 1.0

    # Parallelism
    tp_size: int = 1
    dp_size: int = 1
    load_balance_method: str = "round_robin"
    nccl_init_addr: Optional[str] = None
    nccl_ports: Optional[List[int]] = None
    additional_ports: Optional[Union[List[int], int]] = None
    nnodes: int = 1
    node_rank: Optional[int] = None

    # Optimization
    disable_flashinfer: bool = False
    disable_flashinfer_sampling: bool = False
    disable_radix_cache: bool = False
    disable_regex_jump_forward: bool = False
    disable_cuda_graph: bool = False
    disable_cuda_graph_padding: bool = False
    disable_disk_cache: bool = False
    disable_custom_all_reduce: bool = False
    enable_mixed_chunk: bool = False
    enable_torch_compile: bool = False
    enable_p2p_check: bool = False
    enable_mla: bool = False
    triton_attention_reduce_in_fp32: bool = False
    efficient_weight_load: bool = False

    # Observability
    log_level: str = "info"
    log_level_http: Optional[str] = None
    log_requests: bool = False
    show_time_cost: bool = False

    def __post_init__(self):
        if self.tokenizer_path is None:
            self.tokenizer_path = self.model_path

        if self.served_model_name is None:
            self.served_model_name = self.model_path

        if self.chunked_prefill_size <= 0:
            # Disable chunked prefill
            self.chunked_prefill_size = None

        if self.mem_fraction_static is None:
            if self.tp_size >= 16:
                self.mem_fraction_static = 0.79
            elif self.tp_size >= 8:
                self.mem_fraction_static = 0.83
            elif self.tp_size >= 4:
                self.mem_fraction_static = 0.85
            elif self.tp_size >= 2:
                self.mem_fraction_static = 0.87
            else:
                self.mem_fraction_static = 0.88

        if isinstance(self.additional_ports, int):
            self.additional_ports = [self.additional_ports]
        elif self.additional_ports is None:
            self.additional_ports = []

        if self.random_seed is None:
            self.random_seed = random.randint(0, 1 << 30)

        self._check_args()

        self._alloc_port_args()

        self._set_envs_and_config()

    def _alloc_port_args(self):
        if isinstance(self.additional_ports, int):
            self.additional_ports = [self.additional_ports]
        elif self.additional_ports is None:
            self.additional_ports = []

        _, ports = allocate_init_ports(
            30000,
            self.additional_ports,
            self.dp_size,
        )
        self.tokenizer_port = ports[0]
        self.controller_port = ports[1]
        self.detokenizer_port = ports[2]
        self.nccl_ports = ports[3:]
        logger.info(
            f"Allocated port args: tokenizer_port({self.tokenizer_port}), controller_port({self.controller_port}),"
            f"detokenizer_port({self.detokenizer_port}), nccl_ports({self.nccl_ports})"
        )

    def _check_args(self):
        assert (
            self.tp_size % self.nnodes == 0
        ), "tp_size must be divisible by number of nodes"

        assert not (
            self.dp_size > 1 and self.node_rank is not None
        ), "multi-node data parallel is not supported"

        if "Alibaba-NLP/gte-Qwen2-1.5B-instruct" == self.model_path:
            logger.info(
                "Not sure why, the tokenizer will add an additional token at the end of the prompt when trust_remote_mode=True"
            )
            self.trust_remote_code = False

        if "gemma-2" in self.model_path.lower():
            logger.info(
                f"When using sliding window in gemma-2, disable radix_cache, regex_jump_forward, and turn on flashinfer."
            )
            # FIXME: compatibility with radix attention
            self.disable_radix_cache = True
            # FIXME: compatibility with jump forward
            self.disable_regex_jump_forward = True
            self.disable_flashinfer = False
            # FIXME: compatibility with chunked prefill
            self.chunked_prefill_size = None

    def _set_envs_and_config(self):
        # Set global environments
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        os.environ["NCCL_CUMEM_ENABLE"] = "0"
        os.environ["NCCL_NVLS_ENABLE"] = "0"
        os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"

        # Set ulimit
        set_ulimit()

        # Enable show time cost for debugging
        if self.show_time_cost:
            enable_show_time_cost()

        # Disable disk cache
        if self.disable_disk_cache:
            disable_cache()

        # Fix triton bugs
        if self.tp_size * self.dp_size > 1:
            # FIXME: remove this after https://github.com/triton-lang/triton/pull/4295 is used as a dependency.
            maybe_set_triton_cache_manager()

        # Check flashinfer version
        if not self.disable_flashinfer:
            assert_pkg_version(
                "flashinfer",
                "0.1.6",
                "Please uninstall the old version and "
                "reinstall the latest version by following the instructions "
                "at https://docs.flashinfer.ai/installation.html.",
            )
