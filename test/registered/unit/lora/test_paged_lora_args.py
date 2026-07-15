"""Unit tests for paged LoRA server arguments.

Usage:
    python -m pytest test/registered/unit/lora/test_paged_lora_args.py -v
"""


def _patch_kernels_revision():
    try:
        from kernels.layer.func import FuncRepository as _FR
        from kernels.layer.layer import LayerRepository as _LR

        _lr_orig = _LR.__init__

        def _lr_patched(
            self, repo_id, *, layer_name, revision=None, version=None, **kw
        ):
            if revision is None and version is None:
                revision = "main"
            _lr_orig(
                self,
                repo_id,
                layer_name=layer_name,
                revision=revision,
                version=version,
                **kw,
            )

        _LR.__init__ = _lr_patched

        _fr_orig = _FR.__init__

        def _fr_patched(self, repo_id, *, func_name, revision=None, version=None, **kw):
            if revision is None and version is None:
                revision = "main"
            _fr_orig(
                self,
                repo_id,
                func_name=func_name,
                revision=revision,
                version=version,
                **kw,
            )

        _FR.__init__ = _fr_patched
    except ImportError:
        pass
    except Exception as e:
        import logging

        logging.getLogger(__name__).warning(f"patch_kernels failed: {e}")


_patch_kernels_revision()

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest

from sglang.srt.server_args import ServerArgs, prepare_server_args


class TestPagedLoRAArgs(unittest.TestCase):
    def test_lora_page_rank_size_default_disabled(self):
        server_args = ServerArgs(model_path="dummy")
        self.assertEqual(server_args.lora_page_rank_size, 0)

    def test_lora_pages_default_auto(self):
        server_args = ServerArgs(model_path="dummy")
        self.assertEqual(server_args.lora_pages, 0)

    def test_lora_page_rank_size_from_cli(self):
        server_args = prepare_server_args(
            ["--model-path", "dummy", "--lora-page-rank-size", "8"]
        )
        self.assertEqual(server_args.lora_page_rank_size, 8)

    def test_lora_pages_from_cli(self):
        server_args = prepare_server_args(
            ["--model-path", "dummy", "--lora-pages", "64"]
        )
        self.assertEqual(server_args.lora_pages, 64)


if __name__ == "__main__":
    unittest.main()
