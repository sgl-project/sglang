# TODO shall we put this file elsewhere?

from sglang.srt.utils import get_bool_env_var

_ENABLE_CAPTURE = get_bool_env_var("SGLANG_ENABLE_WARMUP_DEEPGEMM_CAPTURE")


def on_execution(lhs, rhs, out):
    if not _ENABLE_CAPTURE:
        return

    m, k = lhs[0].shape
    n, k_ = rhs[0].shape
    assert k == k_
    mkn = m, k, n
    TODO
