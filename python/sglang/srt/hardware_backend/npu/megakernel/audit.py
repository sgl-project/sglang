"""Opt-in synchronous numerical diagnostics; never enable during timing."""

import os
from pathlib import Path

import torch


def directory(binding):
    from sglang.srt.runtime_context import get_parallel

    dest = Path(binding.audit) / f"tp{get_parallel().tp_rank}"
    dest.mkdir(exist_ok=True, parents=True)
    return dest


def write_metadata(binding, route, infos, select):
    number = binding.counters.get(route, 0)
    if number >= int(os.getenv("SGLANG_ASCEND_AUDIT_ROUNDS", "4")):
        return
    binding.counters[route] = number + 1
    fields = (
        "index_rope_positions",
        "slot_mapping",
        "block_tables",
        "q_seq_len",
        "kv_seq_len",
    )
    rows = [
        {name: getattr(info, name).detach().cpu() for name in fields} for info in infos
    ]
    torch.save(
        (rows, None if select is None else select.detach().cpu()),
        directory(binding) / f"{route}-{number:05}.pt",
    )


def write_binding(binding, model):
    import json

    def layout(tensor):
        return [
            tensor.data_ptr(),
            list(tensor.shape),
            list(tensor.stride()),
            str(tensor.dtype),
        ]

    cache = binding.bridge._shared_cache
    report = {
        field: [layout(tensor) for tensor in getattr(cache, field)]
        for field in ("k_cache", "index_k_buffer", "index_k_scale_buffer")
    }
    for key, obj in (("host", model), ("blockrt", binding.runtime.model)):
        report[key] = {
            name: layout(tensor)
            for name, tensor in obj.named_parameters()
            if not name.startswith("_megakernel")
            and (
                "q_b_proj" in name
                or name.endswith(("embed_tokens.weight", "lm_head.weight"))
            )
        }
    (directory(binding) / f"bind-{binding.role}.json").write_text(
        json.dumps(report, indent=2)
    )
