import time

def now_wall_s() -> float:
    return time.time()

def now_mono_s() -> float:
    return time.perf_counter()

def now_wall_ns() -> int:
    return int(time.time() * 1_000_000_000)
