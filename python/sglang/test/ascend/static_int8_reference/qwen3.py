"""Reference model plugin; nonquantized Qwen3 operations use the NPU backend."""

import inspect
import json
import os
from pathlib import Path
from types import MethodType

from sglang.srt.hardware_backend.npu.quantization.linear_method_npu import (
    NPUW8A8Int8StaticLinearMethod,
)
from sglang.srt.models.qwen3 import Qwen3ForCausalLM as OriginalQwen3ForCausalLM
from sglang.test.ascend.static_int8_reference.checkpoint import CheckpointIntegerOracle


class Qwen3ForCausalLM(OriginalQwen3ForCausalLM):
    def _install_integer_reference(self):
        self._reference_config = json.loads(
            Path(os.environ["SGLANG_STATIC_INT8_REFERENCE_CONFIG"]).read_text()
        )
        self._integer_oracle = CheckpointIntegerOracle(
            self._reference_config["checkpoint"]
        )
        self._integer_oracle.verify_model(self)
        if (
            len(self._integer_oracle.logical_names) != 196
            or len(self._integer_oracle.groups) != 112
            or len(self._integer_oracle.checks) != 675
        ):
            raise ValueError("Expected the cached Qwen3-0.6B checkpoint layout")
        self._reference_active = False
        found = set()
        for name, module in self.named_modules():
            kernel = getattr(getattr(module, "scheme", None), "kernel", None)
            if not isinstance(kernel, NPUW8A8Int8StaticLinearMethod):
                continue
            original = kernel.apply
            found.add(name)

            def apply(kernel, layer, x, bias=None, name=name, original=original):
                if not self._reference_active:
                    return original(layer, x, bias)
                return self._integer_oracle.linear(name, x, bias)

            kernel.apply = MethodType(apply, kernel)
        if found != set(self._integer_oracle.groups):
            raise ValueError("Reference model missed a serialized fused linear")

    def forward(self, *args, **kwargs):
        if not hasattr(self, "_integer_oracle"):
            self._install_integer_reference()
        input_ids = kwargs.get("input_ids", args[0] if args else None)
        target = self._reference_config["first_input_ids"]
        if (
            not self._reference_active
            and input_ids is not None
            and input_ids.numel() == len(target)
            and input_ids.detach().cpu().flatten().tolist() == target
        ):
            self._reference_active = True
        try:
            return super().forward(*args, **kwargs)
        finally:
            if self._reference_active:
                path = Path(self._reference_config["receipt"])
                temporary = path.with_suffix(".tmp")
                receipt = self._integer_oracle.receipt()
                receipt["reference_model_source"] = __file__
                receipt["original_nonquantized_model_source"] = inspect.getsourcefile(
                    OriginalQwen3ForCausalLM
                )
                temporary.write_text(json.dumps(receipt, indent=2) + "\n")
                os.replace(temporary, path)


EntryClass = Qwen3ForCausalLM
