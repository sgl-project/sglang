# TODO shall we put this file elsewhere?
import json
import logging

from sglang.srt.utils import get_bool_env_var

logger = logging.getLogger(__name__)

_ENABLE_CAPTURE = get_bool_env_var("SGLANG_ENABLE_WARMUP_DEEPGEMM_CAPTURE")


def capturer_on_execution(lhs, rhs, out):
    if not _ENABLE_CAPTURE:
        return
    _capturer.on_execution(lhs=lhs, rhs=rhs)


class _Capturer:
    def __init__(self):
        self._seen_infos = set()

    def on_execution(self, lhs, rhs):
        info = _compute_shape_from_args(lhs=lhs, rhs=rhs)

        if info in self._seen_infos:
            return

        self._seen_infos.add(info)
        logger.info(f"CAPTURER_NEW_INFO={json.dumps(info)}")


_capturer = _Capturer() if _ENABLE_CAPTURE else None


def _compute_shape_from_args(lhs, rhs):
    m, k = lhs[0].shape
    n, k_ = rhs[0].shape
    assert k == k_
    return dict(m=m, k=k, n=n)
