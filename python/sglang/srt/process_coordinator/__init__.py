from sglang.srt.process_coordinator.base import BaseProcessCoordinator
from sglang.srt.process_coordinator.multi import MultiProcessCoordinator
from sglang.srt.process_coordinator.single import SingleProcessCoordinator


def create_process_coordinator(mode: str) -> BaseProcessCoordinator:
    return {
        'multi': MultiProcessCoordinator,
        'single': SingleProcessCoordinator,
    }[mode]()
