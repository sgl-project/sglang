import threading
import time

import psutil


def start_cpu_monitor_thread(component: str, interval: float = 5.0) -> threading.Thread:
    from prometheus_client import Counter

    cpu_seconds_total = Counter(
        name="sglang:process_cpu_seconds_total",
        documentation="Total CPU time consumed by this process (user + system)",
        labelnames=["component"],
    )

    def monitor():
        process = psutil.Process()
        last_times = process.cpu_times()

        while True:
            time.sleep(interval)
            curr_times = process.cpu_times()
            delta = (curr_times.user - last_times.user) + (
                curr_times.system - last_times.system
            )
            cpu_seconds_total.labels(component=component).inc(delta)
            last_times = curr_times

    t = threading.Thread(target=monitor, daemon=True)
    t.start()
    return t
