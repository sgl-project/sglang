import multiprocessing as mp

from sglang.srt.managers.scheduler import run_scheduler_process
from sglang.srt.server_args import EngineFragmentArgs


class EngineFragment:
    def __init__(
        self,
        tp_rank: int,
        gpu_id: int,
        fragment_args: "EngineFragmentArgs",
    ):
        self._proc = mp.Process(
            target=run_scheduler_process,
            kwargs=dict(
                server_args=fragment_args.server_args,
                port_args=fragment_args.port_args,
                gpu_id=gpu_id,
                tp_rank=tp_rank,
                dp_rank=None,
                ready_ipc_name=fragment_args.scheduler_ready_ipc_names[tp_rank],
            ),
        )
        self._proc.start()
