# TODO shall we put this file elsewhere?

from sglang.srt.utils import get_bool_env_var

_ENABLE_CAPTURE = get_bool_env_var("SGLANG_ENABLE_WARMUP_DEEPGEMM_CAPTURE")


def on_execution(lhs, rhs, out):
    if not _ENABLE_CAPTURE:
        return

    info = _compute_shape_from_args(lhs, rhs, out)


def _compute_shape_from_args(lhs, rhs, out):
    m, k = lhs[0].shape
    n, k_ = rhs[0].shape
    assert k == k_
    return m, k, n


class _Capturer:
    def __init__(self):
        self._seen_mkn_arr = {}

    def on_execution(self, shape):
        TODO
