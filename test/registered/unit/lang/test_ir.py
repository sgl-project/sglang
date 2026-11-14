import importlib.util
import sys
import types
import unittest


def _ensure_stub_modules():
    for mod_name in ("pybase64", "IPython"):
        if mod_name not in sys.modules and importlib.util.find_spec(mod_name) is None:
            sys.modules[mod_name] = types.ModuleType(mod_name)


_ensure_stub_modules()

from sglang.lang.ir import (  # noqa: E402
    SglExpr,
    SglFork,
    SglFunction,
    SglGetForkItem,
    SglImage,
    SglVideo,
)
from sglang.test.ci.ci_register import register_cuda_ci  # noqa: E402
from sglang.test.test_utils import CustomTestCase  # noqa: E402

register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-small")


class TestSglLangIR(CustomTestCase):
    def test_bind_preserves_num_api_spec_tokens(self):
        def prog(s, x, y=2):
            s += f"{x}-{y}"

        fn = SglFunction(prog, num_api_spec_tokens=64)
        bound = fn.bind(x="hello")
        self.assertEqual(bound.num_api_spec_tokens, 64)
        self.assertEqual(bound.bind_arguments, {"x": "hello"})

        rebound = bound.bind(y=10)
        self.assertEqual(rebound.num_api_spec_tokens, 64)
        self.assertEqual(rebound.bind_arguments, {"x": "hello", "y": 10})

    def test_sgl_image_and_video_initialize_base_expr_fields(self):
        img = SglImage("example.png")
        vid = SglVideo("example.mp4", 8)

        self.assertIsInstance(img, SglExpr)
        self.assertIsInstance(vid, SglExpr)
        self.assertIsInstance(img.node_id, int)
        self.assertIsInstance(vid.node_id, int)
        self.assertIsNone(img.prev_node)
        self.assertIsNone(vid.prev_node)
        self.assertIsNone(img.pid)
        self.assertIsNone(vid.pid)

        fork_expr = SglFork(2)
        fork_expr.prev_node = img
        self.assertIn(f"%{img.node_id}", repr(fork_expr))

        get_item_expr = SglGetForkItem(0)
        get_item_expr.prev_node = vid
        self.assertIn(f"%{vid.node_id}", repr(get_item_expr))

    def test_trace_and_print_graph_dfs_with_image_and_video(self):
        def multimodal_prog(s):
            s += " prefix "
            s += SglImage("photo.png")
            s += SglVideo("clip.mp4", 4)
            s += " suffix "

        fn = SglFunction(multimodal_prog)
        trace_info = fn.trace()
        printed = trace_info.last_node.print_graph_dfs()
        self.assertIn("SglImage(photo.png)", printed)
        self.assertIn("SglVideo(clip.mp4, 4)", printed)


if __name__ == "__main__":
    unittest.main()
