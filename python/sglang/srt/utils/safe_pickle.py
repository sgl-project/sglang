# SPDX-License-Identifier: Apache-2.0
"""Restricted pickle loaders shared by torch-free protocol modules."""

import io
import pickle


class SafeUnpickler(pickle.Unpickler):
    ALLOWED_MODULE_PREFIXES = {
        # --- Python types ---
        "builtins.",
        "collections.",
        "copyreg.",
        "functools.",
        "itertools.",
        "operator.",
        "types.",
        "weakref.",
        # --- PyTorch types ---
        "torch.",
        "torch._tensor.",
        "torch.storage.",
        "torch.nn.parameter.",
        "torch.autograd.function.",
        # --- torch distributed ---
        "torch.distributed.",
        "torch.distributed._shard.",
        "torch.distributed._composable.",
        "torch._C._distributed_c10d.",
        "torch._C._distributed_fsdp.",
        "torch.distributed.optim.",
        # --- multiprocessing ---
        "multiprocessing.resource_sharer.",
        "multiprocessing.reduction.",
        "pickletools.",
        # --- PEFT / LoRA ---
        "peft.",
        "transformers.",
        "huggingface_hub.",
        # --- SGLang & Unitest ---
        "sglang.srt.weight_sync.tensor_bucket.",
        "sglang.srt.model_executor.model_runner.",
        "sglang.srt.model_executor.model_runner_components.weight_updater.",
        "sglang.srt.layers.",
        "sglang.srt.utils.",
        "sglang.srt.disaggregation.",
        "sglang.srt.managers.",
        "torch_npu.",
    }

    DENY_CLASSES = {
        ("builtins", "eval"),
        ("builtins", "exec"),
        ("builtins", "compile"),
        ("os", "system"),
        ("subprocess", "Popen"),
        ("subprocess", "run"),
        ("codecs", "decode"),
        ("types", "CodeType"),
        ("types", "FunctionType"),
    }

    def find_class(self, module, name):
        # Block deterministic attacks
        if (module, name) in self.DENY_CLASSES:
            raise RuntimeError(
                f"Blocked unsafe class loading ({module}.{name}), "
                f"to prevent exploitation of CVE-2025-10164"
            )
        # Allowlist of safe-to-load modules.
        if any(
            (module + ".").startswith(prefix) for prefix in self.ALLOWED_MODULE_PREFIXES
        ):
            return super().find_class(module, name)
        # Block everything else. (Potential attack surface)
        raise RuntimeError(
            f"Blocked unsafe class loading ({module}.{name}), "
            f"to prevent exploitation of CVE-2025-10164"
        )


def safe_pickle_load(fp):
    """Drop-in replacement for pickle.load() that blocks unsafe class loading."""
    return SafeUnpickler(fp).load()


def safe_pickle_loads(data):
    """Drop-in replacement for pickle.loads() that blocks unsafe class loading."""
    if isinstance(data, (bytes, bytearray, memoryview)):
        buf = bytes(data)
    else:
        # zmq.Frame and other buffer-protocol objects
        buf = bytes(memoryview(data))
    return SafeUnpickler(io.BytesIO(buf)).load()
