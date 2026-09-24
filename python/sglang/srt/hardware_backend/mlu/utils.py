from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from sglang.srt.arg_groups.overrides import (
    declare_resolution,
    resolving_view,
)

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


def set_default_server_args(args: "ServerArgs"):
    """
    Set default server arguments for MLU backend.
    """

    cfg = resolving_view(args)

    for name, value in (
        ("attention_backend", cfg.attention_backend),
        ("prefill_attention_backend", cfg.prefill_attention_backend),
        ("decode_attention_backend", cfg.decode_attention_backend),
    ):
        # The MLU kernels implement no other backend; None is filled by the
        # declaration below.
        if value is not None and value != "mlu":
            raise ValueError(
                f"MLU currently supports only the 'mlu' attention backend; "
                f"got {name}={value!r}."
            )
    declare_resolution(
        args,
        "set_default_server_args",
        attention_backend="mlu",
        prefill_attention_backend="mlu",
        decode_attention_backend="mlu",
    )
    if cfg.sampling_backend is None:
        declare_resolution(
            args,
            "set_default_server_args",
            sampling_backend="pytorch",
        )
    elif cfg.sampling_backend != "pytorch":
        raise ValueError(
            "MLU currently supports only the 'pytorch' sampling backend; "
            f"got sampling_backend={cfg.sampling_backend!r}."
        )
    if cfg.page_size is None:
        # reshape_paged_cache scatters whole pages; _page_size_default would
        # otherwise resolve 1.
        declare_resolution(
            args,
            "set_default_server_args",
            page_size=16,
        )
    # torch_mlu_ops ships no custom-allreduce kernel; CNCL covers collectives.
    declare_resolution(
        args,
        "set_default_server_args",
        disable_custom_all_reduce=True,
    )
    if cfg.enable_hierarchical_cache:
        logger.warning("MLU does not support hierarchical cache; disabling it.")
        declare_resolution(
            args,
            "set_default_server_args",
            enable_hierarchical_cache=False,
        )
