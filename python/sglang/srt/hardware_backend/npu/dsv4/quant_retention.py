"""Retention ring for every npu_dynamic_quant payload on this rank.

The req_to_token row clobber's payload is fp16[T] (per-token, +/-2,
byte-identical across CP ranks) whose source tensor is already freed by
the time the send-path fingerprint runs. Wrapping npu_dynamic_quant here
keeps the last N (quant, scale) pairs alive so the fingerprint can
sequence-match them; the ring is unbounded-cost-bounded (fixed N) and
only holds references (no copies).
"""

import torch
import torch_npu

_recent: list = []
_KEEP = 16


_NOP = False


def _wrap(fn):
    def wrapped(*args, **kwargs):
        out = fn(*args, **kwargs)
        if _NOP:
            # Same per-call Python overhead as retention but holds nothing:
            # if the clobber disappears here too, the masking is pure timing
            # (M2), not memory pinning (M1).
            return out
        try:
            q, s = out
            _recent.append(
                (q.detach(), s.detach() if torch.is_tensor(s) else s)
            )
            del _recent[:-_KEEP]
        except Exception:
            pass
        return out

    return wrapped


def install() -> None:
    global _NOP
    import os

    _NOP = os.getenv("SGLANG_MF_RETAIN_NOP", "0") == "1"
    torch_npu.npu_dynamic_quant = _wrap(torch_npu.npu_dynamic_quant)


def payloads():
    return list(_recent)
