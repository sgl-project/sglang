# SPDX-License-Identifier: Apache-2.0
"""One runtime construction context for diffusion workers and weight owners."""

import os
import tempfile

import torch

from sglang.multimodal_gen import envs
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    maybe_init_distributed_environment_and_model_parallel,
)
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.server_args import ServerArgs, set_global_server_args
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.srt.environ import third_party_cache_defaults
from sglang.srt.utils.network import NetworkAddress

logger = init_logger(__name__)


def worker_cpu_intra_op_threads(num_gpus: int) -> int | None:
    """Divide host cores across colocated workers; respect explicit OMP policy."""
    if "OMP_NUM_THREADS" in os.environ:
        return None
    return max(1, min(16, (os.cpu_count() or 1) // max(1, num_gpus)))


def configure_persistent_torch_compile_cache() -> None:
    """Persist Inductor/Triton cache, retaining non-ephemeral user overrides."""
    compile_cache_root = os.path.join(
        envs.SGLANG_DIFFUSION_CACHE_ROOT, "torch_compile_cache"
    )
    tmp_root = tempfile.gettempdir()
    sglang_defaults = third_party_cache_defaults()
    for env_name, sub in (
        ("TORCHINDUCTOR_CACHE_DIR", "inductor"),
        ("TRITON_CACHE_DIR", "triton"),
    ):
        current = os.environ.get(env_name)
        if (
            current
            and current != sglang_defaults.get(env_name)
            and not current.startswith(tmp_root)
        ):
            continue
        cache_path = os.path.join(compile_cache_root, sub)
        try:
            os.makedirs(cache_path, exist_ok=True)
        except OSError as error:
            logger.warning(
                "Could not create torch.compile cache dir %s: %s", cache_path, error
            )
            continue
        os.environ[env_name] = cache_path
    logger.info(
        "torch.compile cache: TORCHINDUCTOR_CACHE_DIR=%s TRITON_CACHE_DIR=%s",
        os.environ.get("TORCHINDUCTOR_CACHE_DIR"),
        os.environ.get("TRITON_CACHE_DIR"),
    )


def bootstrap_diffusion_runtime(
    server_args: ServerArgs,
    *,
    local_rank: int,
    rank: int,
    rendezvous: NetworkAddress,
    role: str = "diffusion_gpu_worker",
) -> None:
    """Establish model-construction globals without loading any component.

    local_rank is the launcher's resolved process-local device index. A cache
    owner passes its own rendezvous, never the client's live process group.
    Launcher preparation must not call this function.
    """
    if not current_platform.is_mps():
        current_platform.set_device(current_platform.get_device(local_rank))
    set_global_server_args(server_args)
    intra_op_threads = worker_cpu_intra_op_threads(
        server_args.num_gpus // server_args.nnodes
    )
    if intra_op_threads is not None:
        torch.set_num_threads(intra_op_threads)
    os.environ.update(
        MASTER_ADDR=rendezvous.host,
        MASTER_PORT=str(rendezvous.port),
        LOCAL_RANK=str(local_rank),
        RANK=str(rank),
        WORLD_SIZE=str(server_args.num_gpus),
    )
    configure_persistent_torch_compile_cache()
    maybe_init_distributed_environment_and_model_parallel(
        tp_size=server_args.tp_size,
        cfg_degree=server_args.cfg_parallel_degree or 1,
        ulysses_degree=server_args.ulysses_degree,
        ring_degree=server_args.ring_degree,
        sp_size=server_args.sp_degree,
        dp_size=server_args.dp_size,
        distributed_init_method=rendezvous.to_tcp(),
        dist_timeout=server_args.dist_timeout,
    )
    from sglang.srt.runtime_context import get_context, publish
    from sglang.srt.server_args import ServerArgs as SrtServerArgs

    if get_context()._server_args is None:
        publish(
            SrtServerArgs(model_path="dummy", tp_size=server_args.tp_size), role=role
        )

    from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions

    if current_platform.is_cuda():
        monkey_patch_torch_reductions()
