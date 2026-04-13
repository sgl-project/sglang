import os
from pathlib import Path

import torch


def _get_ltx23_ti2v_dump_dir() -> Path | None:
    out_dir = os.environ.get("LTX23_SGLANG_TI2V_DUMP_DIR")
    if not out_dir:
        return None
    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def maybe_save_ltx23_ti2v_tensor(name: str, value: torch.Tensor | None) -> None:
    out_dir = _get_ltx23_ti2v_dump_dir()
    if out_dir is None or value is None:
        return
    torch.save(value.detach().cpu(), out_dir / f"{name}.pt")
