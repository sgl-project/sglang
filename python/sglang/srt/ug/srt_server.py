# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class UGSRTSchedulerHandle:
    """Owns the SRT scheduler and optional temporary model view."""

    scheduler: Any
    model_view_dir: tempfile.TemporaryDirectory | None = None

    def close(self) -> None:
        for name in (
            "recv_from_tokenizer",
            "recv_from_rpc",
            "send_metrics_from_scheduler",
        ):
            socket = getattr(self.scheduler, name, None)
            close = getattr(socket, "close", None)
            if callable(close):
                close(linger=0)
        if self.model_view_dir is not None:
            self.model_view_dir.cleanup()


UGBAGELSRTSchedulerHandle = UGSRTSchedulerHandle


class _NoopSender:
    def send_output(self, *args, **kwargs):
        del args, kwargs


def is_real_bagel_ug_model(model_path: str | None, model_id: str | None = None) -> bool:
    identifier = " ".join(str(value or "") for value in (model_path, model_id)).lower()
    return "bagel" in identifier


def is_real_u1_ug_model(model_path: str | None, model_id: str | None = None) -> bool:
    identifier = " ".join(str(value or "") for value in (model_path, model_id)).lower()
    return "sensenova-u1" in identifier or "sensenova_u1" in identifier


def build_bagel_language_model_view(
    checkpoint_dir: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
) -> Path:
    checkpoint_path = Path(checkpoint_dir).expanduser()
    output_path = Path(output_dir)
    config_path = checkpoint_path / "llm_config.json"
    weight_path = checkpoint_path / "ema.safetensors"
    if not config_path.exists() or not weight_path.exists():
        raise FileNotFoundError(
            "BAGEL SRT scheduler requires llm_config.json and "
            f"ema.safetensors under {checkpoint_path}"
        )

    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    config.update(
        {
            "architectures": ["BAGELQwen2MoTForCausalLM"],
            "bagel_checkpoint_dir": str(checkpoint_path),
            "bagel_enable_visual_feature_extractors": True,
            "bagel_connector_act": "gelu_pytorch_tanh",
            "bagel_latent_patch_size": 2,
            "bagel_max_latent_size": 64,
            "bagel_max_latent_tokens": 64 * 64,
            "bagel_vit_max_num_patch_per_side": 70,
            "layer_module": "Qwen2MoTDecoderLayer",
            "qk_norm": True,
            "tie_word_embeddings": False,
        }
    )
    (output_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    os.symlink(weight_path, output_path / "model.safetensors")
    return output_path


def create_bagel_srt_scheduler(
    *,
    checkpoint_dir: str,
    gpu_id: int = 0,
    dtype: str = "bfloat16",
    mem_fraction_static: float = 0.35,
    chunked_prefill_size: int = 256,
    attention_backend: str | None = None,
    log_level: str = "error",
) -> UGSRTSchedulerHandle:
    """Create the SRT Scheduler used as the BAGEL UG session owner."""

    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.server_args import (
        PortArgs,
        ServerArgs,
        set_global_server_args_for_scheduler,
    )

    model_view_dir = tempfile.TemporaryDirectory(prefix="sglang-ug-bagel-srt-")
    model_path = build_bagel_language_model_view(checkpoint_dir, model_view_dir.name)
    server_args = ServerArgs(
        model_path=str(model_path),
        tokenizer_path=str(Path(checkpoint_dir).expanduser()),
        trust_remote_code=True,
        dtype=dtype,
        tp_size=1,
        pp_size=1,
        dp_size=1,
        disable_cuda_graph=True,
        disable_piecewise_cuda_graph=True,
        disable_overlap_schedule=True,
        skip_server_warmup=True,
        attention_backend=attention_backend,
        mem_fraction_static=float(mem_fraction_static),
        chunked_prefill_size=int(chunked_prefill_size),
        log_level=log_level,
    )
    server_args.check_server_args()
    set_global_server_args_for_scheduler(server_args)

    scheduler = Scheduler(
        server_args,
        PortArgs.init_new(server_args),
        gpu_id=int(gpu_id),
        tp_rank=0,
        moe_ep_rank=0,
        pp_rank=0,
        attn_cp_rank=0,
        moe_dp_rank=0,
        dp_rank=None,
    )
    _replace_sender_with_noop(scheduler, "send_to_tokenizer")
    _replace_sender_with_noop(scheduler, "send_to_detokenizer")
    return UGSRTSchedulerHandle(
        scheduler=scheduler,
        model_view_dir=model_view_dir,
    )


def create_u1_srt_scheduler(
    *,
    checkpoint_dir: str,
    gpu_id: int = 0,
    dtype: str = "bfloat16",
    mem_fraction_static: float = 0.35,
    chunked_prefill_size: int = 256,
    attention_backend: str | None = None,
    log_level: str = "error",
) -> UGSRTSchedulerHandle:
    """Create the SRT Scheduler used as the SenseNova U1 UG session owner."""

    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.server_args import (
        PortArgs,
        ServerArgs,
        set_global_server_args_for_scheduler,
    )

    server_args = ServerArgs(
        model_path=str(Path(checkpoint_dir).expanduser()),
        tokenizer_path=str(Path(checkpoint_dir).expanduser()),
        trust_remote_code=False,
        dtype=dtype,
        tp_size=1,
        pp_size=1,
        dp_size=1,
        disable_cuda_graph=True,
        disable_piecewise_cuda_graph=True,
        disable_overlap_schedule=True,
        skip_server_warmup=True,
        attention_backend=attention_backend,
        mem_fraction_static=float(mem_fraction_static),
        chunked_prefill_size=int(chunked_prefill_size),
        log_level=log_level,
    )
    server_args.check_server_args()
    set_global_server_args_for_scheduler(server_args)

    scheduler = Scheduler(
        server_args,
        PortArgs.init_new(server_args),
        gpu_id=int(gpu_id),
        tp_rank=0,
        moe_ep_rank=0,
        pp_rank=0,
        attn_cp_rank=0,
        moe_dp_rank=0,
        dp_rank=None,
    )
    _replace_sender_with_noop(scheduler, "send_to_tokenizer")
    _replace_sender_with_noop(scheduler, "send_to_detokenizer")
    return UGSRTSchedulerHandle(scheduler=scheduler)


def _replace_sender_with_noop(scheduler: Any, name: str) -> None:
    sender = getattr(scheduler, name, None)
    socket = getattr(sender, "socket", None)
    if socket is not None:
        socket.close(linger=0)
    setattr(scheduler, name, _NoopSender())
