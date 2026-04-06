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
"""CLI argument definitions for memory and scheduling."""

import argparse


class MemoryArgs:
    """CLI argument definitions for memory and scheduling."""

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        from sglang.srt.server_args import ServerArgs
        from sglang.srt.args.constants import RADIX_EVICTION_POLICY_CHOICES
        from sglang.srt.args.actions import DeprecatedAction
        from sglang.srt.utils.common import human_readable_int

        # Memory and scheduling
        parser.add_argument(
            "--mem-fraction-static",
            type=float,
            default=ServerArgs.mem_fraction_static,
            help="The fraction of the memory used for static allocation (model weights and KV cache memory pool). Use a smaller value if you see out-of-memory errors.",
        )
        parser.add_argument(
            "--max-running-requests",
            type=int,
            default=ServerArgs.max_running_requests,
            help="The maximum number of running requests.",
        )
        parser.add_argument(
            "--max-queued-requests",
            type=int,
            default=ServerArgs.max_queued_requests,
            help="The maximum number of queued requests. This option is ignored when using disaggregation-mode.",
        )
        parser.add_argument(
            "--max-total-tokens",
            type=human_readable_int,
            default=ServerArgs.max_total_tokens,
            help="The maximum number of tokens in the memory pool. If not specified, it will be automatically calculated based on the memory usage fraction. "
            "This option is typically used for development and debugging purposes."
            + f"\n\n{human_readable_int.__doc__}",
        )
        parser.add_argument(
            "--chunked-prefill-size",
            type=int,
            default=ServerArgs.chunked_prefill_size,
            help="The maximum number of tokens in a chunk for the chunked prefill. Setting this to -1 means disabling chunked prefill.",
        )
        parser.add_argument(
            "--prefill-max-requests",
            type=int,
            default=ServerArgs.prefill_max_requests,
            help="The maximum number of requests in a prefill batch. If not specified, there is no limit.",
        )
        parser.add_argument(
            "--enable-dynamic-chunking",
            action="store_true",
            default=ServerArgs.enable_dynamic_chunking,
            help="Enable dynamic chunk size adjustment for pipeline parallelism. When enabled, chunk sizes are dynamically calculated based on fitted function to maintain consistent execution time across chunks.",
        )
        parser.add_argument(
            "--max-prefill-tokens",
            type=human_readable_int,
            default=ServerArgs.max_prefill_tokens,
            help="The maximum number of tokens in a prefill batch. The real bound will be the maximum of this value and the model's maximum context length."
            + f"\n\n{human_readable_int.__doc__}",
        )
        parser.add_argument(
            "--schedule-policy",
            type=str,
            default=ServerArgs.schedule_policy,
            choices=[
                "lpm",
                "random",
                "fcfs",
                "dfs-weight",
                "lof",
                "priority",
                "routing-key",
            ],
            help="The scheduling policy of the requests.",
        )
        parser.add_argument(
            "--enable-priority-scheduling",
            action="store_true",
            default=ServerArgs.enable_priority_scheduling,
            help="Enable priority scheduling. Requests with higher priority integer values will be scheduled first by default.",
        )
        parser.add_argument(
            "--disable-priority-preemption",
            action="store_true",
            default=ServerArgs.disable_priority_preemption,
            help="Disable priority scheduling preemption.",
        )
        parser.add_argument(
            "--default-priority-value",
            type=int,
            default=ServerArgs.default_priority_value,
            help="Default priority for requests without explicit priority.",
        )
        parser.add_argument(
            "--abort-on-priority-when-disabled",
            action="store_true",
            default=ServerArgs.abort_on_priority_when_disabled,
            help="If set, abort requests that specify a priority when priority scheduling is disabled.",
        )
        parser.add_argument(
            "--schedule-low-priority-values-first",
            action="store_true",
            default=ServerArgs.schedule_low_priority_values_first,
            help="If specified with --enable-priority-scheduling, the scheduler will schedule requests with lower priority integer values first.",
        )
        parser.add_argument(
            "--priority-scheduling-preemption-threshold",
            type=int,
            default=ServerArgs.priority_scheduling_preemption_threshold,
            help="Minimum difference in priorities for an incoming request to have to preempt running request(s).",
        )
        parser.add_argument(
            "--schedule-conservativeness",
            type=float,
            default=ServerArgs.schedule_conservativeness,
            help="How conservative the schedule policy is. A larger value means more conservative scheduling. Use a larger value if you see requests being retracted frequently.",
        )
        parser.add_argument(
            "--page-size",
            type=int,
            default=ServerArgs.page_size,
            help="The number of tokens in a page.",
        )
        parser.add_argument(
            "--hybrid-kvcache-ratio",
            action=DeprecatedAction,
            help="Note: --hybrid-kvcache-ratio is deprecated now. Please use --swa-full-tokens-ratio instead.",
        )
        parser.add_argument(
            "--swa-full-tokens-ratio",
            type=float,
            default=ServerArgs.swa_full_tokens_ratio,
            help="The ratio of SWA layer KV tokens / full layer KV tokens, regardless of the number of swa:full layers. It should be between 0 and 1. "
            "E.g. 0.5 means if each swa layer has 50 tokens, then each full layer has 100 tokens.",
        )
        parser.add_argument(
            "--disable-hybrid-swa-memory",
            action="store_true",
            help="Disable the hybrid SWA memory pool.",
        )
        parser.add_argument(
            "--radix-eviction-policy",
            type=str,
            choices=RADIX_EVICTION_POLICY_CHOICES,
            default=ServerArgs.radix_eviction_policy,
            help="The eviction policy of radix trees. 'lru' stands for Least Recently Used, 'lfu' stands for Least Frequently Used, and 'slru' stands for Segmented Least Recently Used.",
        )
        parser.add_argument(
            "--enable-prefill-delayer",
            action="store_true",
            help="Enable prefill delayer for DP attention to reduce idle time.",
        )
        parser.add_argument(
            "--prefill-delayer-max-delay-passes",
            type=int,
            default=ServerArgs.prefill_delayer_max_delay_passes,
            help="Maximum forward passes to delay prefill.",
        )
        parser.add_argument(
            "--prefill-delayer-token-usage-low-watermark",
            type=float,
            default=None,
            help="Token usage low watermark for prefill delayer.",
        )
        parser.add_argument(
            "--prefill-delayer-forward-passes-buckets",
            type=float,
            nargs="+",
            default=None,
            help="Custom buckets for prefill delayer forward passes histogram. 0 and max_delay_passes-1 will be auto-added.",
        )
        parser.add_argument(
            "--prefill-delayer-wait-seconds-buckets",
            type=float,
            nargs="+",
            default=None,
            help="Custom buckets for prefill delayer wait seconds histogram. 0 will be auto-added.",
        )
