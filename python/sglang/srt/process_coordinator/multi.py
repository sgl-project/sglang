import multiprocessing as mp

from sglang.srt.managers.detokenizer_manager import run_detokenizer_process
from sglang.srt.process_coordinator.base import BaseProcessCoordinator


class MultiProcessCoordinator(BaseProcessCoordinator):
    def _launch_detoken_proc(self):
        detoken_proc = mp.Process(
            target=run_detokenizer_process,
            args=(server_args, port_args),
        )
        detoken_proc.start()
