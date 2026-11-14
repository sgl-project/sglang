"""The DASH dispatch wrappers keep the signatures of the Triton entry points they replace.

Off MPS each wrapper forwards its arguments to a Triton entry point; a parameter or
default that differs would change CUDA calls, and the CUDA path is not run on a Mac.
"""

import inspect
import unittest

from sglang.kernels.ops.attention.unlimited_ocr_dash import (
    attend,
    decode_meta,
    dispatch,
    page2d_rope,
    route,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_TRITON_ENTRY_POINTS = {
    "dash_sparse_attn": attend.dash_sparse_attn,
    "dash_sparse_attn_decode": attend.dash_sparse_attn_decode,
    "win_bounds_decode": route.win_bounds_decode,
    "route_prefill": route.route_prefill_triton,
    "kbar_refresh_tail": route.kbar_refresh_tail_triton,
    "route_decode": route.route_decode_triton,
    "dash_decode_meta": decode_meta.dash_decode_meta,
    "dash_decode_allow": decode_meta.dash_decode_allow,
    "apply_page2d_rope_pair": page2d_rope.apply_page2d_rope_pair,
}


def _parameters(fn):
    return [
        (p.name, p.kind, p.default) for p in inspect.signature(fn).parameters.values()
    ]


class TestDashDispatchSignatures(CustomTestCase):
    def test_every_wrapper_has_a_triton_entry_point(self):
        wrappers = {
            name
            for name, fn in vars(dispatch).items()
            if inspect.isfunction(fn)
            and fn.__module__ == dispatch.__name__
            and not name.startswith("_")
        }
        self.assertEqual(wrappers, set(_TRITON_ENTRY_POINTS))

    def test_wrapper_parameters_and_defaults_match_triton(self):
        for name, triton_fn in _TRITON_ENTRY_POINTS.items():
            with self.subTest(wrapper=name):
                self.assertEqual(
                    _parameters(getattr(dispatch, name)), _parameters(triton_fn)
                )


if __name__ == "__main__":
    unittest.main()
