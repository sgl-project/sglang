import os
import subprocess
import sys
import textwrap


def test_eagle_worker_v2_import_does_not_require_flashinfer_mla_api():
    code = textwrap.dedent(
        """
        import importlib.machinery
        import sys
        import types

        import sglang.srt.utils

        # Reproduce the partial FlashInfer namespace exposed while the SM70
        # runtime initializes. Qwen QSA EAGLE must not import MLA backends.
        sglang.srt.utils.is_flashinfer_available = lambda: True
        flashinfer = types.ModuleType("flashinfer")
        flashinfer.__spec__ = importlib.machinery.ModuleSpec(
            "flashinfer", loader=None, is_package=True
        )
        flashinfer.__path__ = []
        sys.modules["flashinfer"] = flashinfer

        import sglang.srt.speculative.eagle_worker_v2
        """
    )
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    subprocess.run([sys.executable, "-c", code], check=True, env=env)
