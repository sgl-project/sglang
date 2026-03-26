"""
VLM Pipeline Stage Timer for SGLang.

Records wall-clock and CUDA-synchronized times for the 5 common VLM stages:
  Stage 1: Processor (HF multimodal preprocessing)
  Stage 2a: Transfer Tokenizer→Scheduler (ZMQ send/recv)
  Stage 2b: Transfer Scheduler→TP Workers (broadcast)
  Stage 3: ViT Encoding (vision encoder forward)
  Stage 4: Embedding Merge (scatter vision embeds into text embeds)
  Stage 5: LLM Prefill (language model forward)
"""

import time
from contextlib import contextmanager
from dataclasses import dataclass

import torch


@dataclass
class StageRecord:
    stage: str
    req_id: str
    wall_ms: float


_records: list[StageRecord] = []


def is_enabled() -> bool:
    return True


def _get_tp_rank() -> int:
    try:
        from sglang.srt.distributed.parallel_state import get_tensor_model_parallel_rank
        return get_tensor_model_parallel_rank()
    except Exception:
        return -1


def _log(msg: str):
    tp = _get_tp_rank()
    prefix = f"[TP{tp}] " if tp >= 0 else ""
    print(f"{prefix}{msg}", flush=True)


def _cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@contextmanager
def record_stage(stage: str, req_id: str = "", cuda_sync: bool = False):
    """Context manager to time a VLM pipeline stage."""
    if cuda_sync:
        _cuda_sync()

    t0 = time.perf_counter()
    yield
    if cuda_sync:
        _cuda_sync()
    t1 = time.perf_counter()

    wall_ms = (t1 - t0) * 1000.0
    _records.append(StageRecord(stage=stage, req_id=req_id, wall_ms=wall_ms))
    _log(f"[VLM_STAGE_TIMER] {stage} | req={req_id} | wall={wall_ms:.3f} ms")


def record_stage_timestamp(stage: str, req_id: str = ""):
    """Record a point-in-time timestamp (for send/recv correlation)."""
    ts = time.perf_counter()
    _log(f"[VLM_STAGE_TIMER] {stage} | req={req_id} | timestamp={ts:.6f}")


def get_records() -> list[StageRecord]:
    return list(_records)


def clear_records():
    _records.clear()


def dump_records():
    for r in _records:
        _log(f"[VLM_STAGE_TIMER_DUMP] {r.stage} | req={r.req_id} | wall={r.wall_ms:.3f} ms")
