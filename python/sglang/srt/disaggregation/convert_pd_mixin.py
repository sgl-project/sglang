from __future__ import annotations
from sglang.srt.disaggregation.utils import (
    DisaggregationMode,
)
import os
from sglang.srt.managers.io_struct import ConvertDisaggregationRoleReqInput, ConvertDisaggregationRoleReqOutput
from typing import TYPE_CHECKING, Any, Dict, Optional
from sglang.srt.disaggregation.utils import (
    DisaggregationMode,
    KVClassType,
    get_kv_class,
)
from sglang.srt.server_args import is_port_available
if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.managers.tokenizer_manager import TokenizerManager

class SchedulerDisaggregationConvertMixin:
    """Mixin class for convert prefill/decode roles."""

    def check_disaggregation_idle(self: Scheduler) -> bool:
        """Check if the disaggregation server is idle."""
        # only for enable_pd_convert mode
        if (
            len(self.waiting_queue) == 0
            and self.running_batch.is_empty()
            and (self.pp_size == 1 or all(x.is_empty() for x in self.running_mbs))
        ):
            pass
        else:
            return False

        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            return (
                len(self.disagg_prefill_bootstrap_queue.queue) == 0
                and len(self.disagg_prefill_inflight_queue) == 0
            )
        elif self.disaggregation_mode == DisaggregationMode.DECODE:
            return (
                len(self.disagg_decode_prealloc_queue.queue) == 0
                and len(self.disagg_decode_transfer_queue.queue) == 0
            )
        return False

    def convert_server_args(self: Scheduler, recv_req: ConvertDisaggregationRoleReqInput):
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            # change server args for prefill mode
            self.convert_prefill_server_args(recv_req)
            # stop prefill event loop
            self.stop_prefill_event.set()
            return ConvertDisaggregationRoleReqOutput(
                success=True,
                message="The role of this server is now DECODE.",
                bootstrap_port=recv_req.bootstrap_port,
            )
        else:
            # change server args for prefill mode
            self.convert_decode_server_args(recv_req)
            # stop decode event loop
            self.stop_decode_event.set()
            return ConvertDisaggregationRoleReqOutput(
                success=True,
                message="The role of this server is now PREFILL.",
                bootstrap_port=recv_req.bootstrap_port,
            )

    def convert_disaggregation_resources(self: Scheduler):
        """convert the p/d resources to d/p resources."""
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            self.convert_prefill_resources()
        else:
            self.convert_decode_resources()

def set_env_vars(env_vars: Optional[Dict[str, Any]]):
    if env_vars:
        for k, v in env_vars.items():
            if v is not None:
                os.environ[k] = v
            else:
                os.environ.pop(k, None)

def set_bootstrap_server(manager: TokenizerManager, obj: ConvertDisaggregationRoleReqInput):
    if obj.check_idle or obj.failed_bootstrap_addr:
        return
    bootstrap_port = None
    if manager.disaggregation_mode == DisaggregationMode.DECODE:
        # start a bootstarp server first
        kv_bootstrap_server_class = get_kv_class(
            manager.disaggregation_transfer_backend, KVClassType.BOOTSTRAP_SERVER
        )
        # find a free port
        bootstrap_port = 8998
        while True:
            if is_port_available(bootstrap_port):
                break
            else:
                bootstrap_port += 1
        manager.bootstrap_server = kv_bootstrap_server_class(
            bootstrap_port,
        )
        manager.server_args.disaggregation_bootstrap_port = bootstrap_port
        obj.bootstrap_port = bootstrap_port
        set_env_vars(obj.disaggregation_prefill_envs)
        manager.disaggregation_mode = DisaggregationMode.PREFILL
    else:
        # stop the bootstrap server
        manager.bootstrap_server.close()
        del manager.bootstrap_server
        set_env_vars(obj.disaggregation_decode_envs)
        manager.disaggregation_mode = DisaggregationMode.DECODE