"""Connex transfer backend for PD disaggregation.

The Mooncake backend's protocol (bootstrap, decode pre-allocation, chunked
KV sends, aux data, aborts) with the transfer engine replaced by connex-rs,
the data plane of Connex (https://github.com/HeterInfra/connex): one-sided
RDMA writes over UCX (GPUDirect when available) posted together, peer-death
events within milliseconds, and endpoints keyed by a logical identity so a
restarted instance rejoins under the same session id. The engine wrapper
comes from the ``connex-ai`` package (``pip install "connex-ai[rs]"``).
"""

from __future__ import annotations

from sglang.srt.disaggregation.mooncake.conn import (
    MooncakeKVBootstrapServer,
    MooncakeKVManager,
    MooncakeKVReceiver,
    MooncakeKVSender,
)
from sglang.srt.disaggregation.utils import DisaggregationMode


class ConnexKVManager(MooncakeKVManager):
    def init_engine(self):
        try:
            from connex.adapters.sglang.pd_backend import get_engine
        except ImportError as e:
            raise ImportError(
                "The connex transfer backend requires connex-ai: "
                'pip install "connex-ai[rs]"'
            ) from e
        role = (
            "prefill"
            if self.disaggregation_mode == DisaggregationMode.PREFILL
            else "decode"
        )
        self.engine = get_engine(self.local_ip, role, int(self.kv_args.gpu_id))


class ConnexKVSender(MooncakeKVSender):
    pass


class ConnexKVReceiver(MooncakeKVReceiver):
    pass


class ConnexKVBootstrapServer(MooncakeKVBootstrapServer):
    pass
