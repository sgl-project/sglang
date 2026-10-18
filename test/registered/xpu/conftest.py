"""Make the diffusion perf fixture reachable from this directory.

``DiffusionServerBase`` (subclassed by the diffusion tests here) has an autouse
fixture that requests ``perf_results``, which is declared in
``sglang/multimodal_gen/test/server/conftest.py``. pytest resolves conftest.py
by the collected file's directory, so that declaration never reaches
``test/registered/``; re-exporting it here is what makes it resolvable.
"""

from sglang.multimodal_gen.test.server.conftest import perf_results  # noqa: F401
