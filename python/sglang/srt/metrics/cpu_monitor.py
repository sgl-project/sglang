import threading
import time

import psutil


def start_cpu_monitor_thread(component: str, interval: float = 1.0) -> threading.Thread:
    from prometheus_client import Gauge

    cpu_busy_ratio = Gauge(
        name="sglang:process_cpu_busy_ratio",
        documentation="CPU busy time ratio of this process (user+system time / wall time)",
        labelnames=["component"],
        multiprocess_mode="mostrecent",
    )

    def monitor():
        process = psutil.Process()
        last_times = process.cpu_times()
        last_check = time.monotonic()

        while True:
            time.sleep(interval)
            now = time.monotonic()
            curr_times = process.cpu_times()
            elapsed = now - last_check
            if elapsed > 0:
                cpu_busy = (
                    (curr_times.user - last_times.user)
                    + (curr_times.system - last_times.system)
                ) / elapsed
                cpu_busy_ratio.labels(component=component).set(cpu_busy)
            last_times = curr_times
            last_check = now

    t = threading.Thread(target=monitor, daemon=True)
    t.start()
    return t
