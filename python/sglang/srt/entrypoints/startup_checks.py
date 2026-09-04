"""Advisory startup checks that do not import GPU kernel libraries."""

import importlib.metadata
import importlib.util
import logging

logger = logging.getLogger(__name__)


def warn_missing_mxfp4_jit_cache(moe_runner_backend: str, cuda_version: str | None):
    """Explain potentially expensive cold JIT builds for the MXFP4 backend."""
    if moe_runner_backend != "flashinfer_mxfp4" or cuda_version is None:
        return
    if importlib.util.find_spec("flashinfer_jit_cache") is not None:
        # FlashInfer validates compatibility when importing the cache. Package
        # presence alone does not guarantee coverage of every requested kernel.
        return
    try:
        flashinfer_version = importlib.metadata.version("flashinfer-python")
    except importlib.metadata.PackageNotFoundError:
        # Leave missing FlashInfer itself to the existing dependency checks.
        return

    logger.warning(
        "flashinfer_mxfp4 is selected but the optional flashinfer-jit-cache "
        "package is not installed. On a cold cache, MoE kernels may require "
        "lengthy JIT compilation before the server is ready. Existing local "
        "or bundled kernel caches may avoid compilation. To reduce cold-start "
        "time, install a flashinfer-jit-cache build matching flashinfer-python "
        "%s and CUDA %s, if available for your platform; see "
        "https://docs.flashinfer.ai/installation.html.",
        flashinfer_version,
        cuda_version,
    )
