import threading
from dataclasses import dataclass
from typing import Callable, Optional

import torch
from torch.nn.parameter import Parameter

from sglang.srt.hardware_backend.npu.utils import npu_format_cast
from sglang.srt.layers.quantization.dequantization import copy_missing_attrs
from sglang.srt.layers.quantization.online_quantization import CopyNumelCounter
from sglang.srt.runtime_context import get_server_args


@dataclass(frozen=True)
class NPUOnlineIntegerQuantSpec:
    mode: str
    weight_dtype: torch.dtype
    activation_dtype: torch.dtype
    dispatcher_output_dtype: str


_ONLINE_INTEGER_QUANT_SPECS = {
    "w8a8_int8": NPUOnlineIntegerQuantSpec(
        mode="w8a8_int8",
        weight_dtype=torch.int8,
        activation_dtype=torch.int8,
        dispatcher_output_dtype="int8",
    ),
    "w4a8_int8": NPUOnlineIntegerQuantSpec(
        mode="w4a8_int8",
        weight_dtype=torch.quint4x2,
        activation_dtype=torch.int8,
        dispatcher_output_dtype="int8",
    ),
    "w4a4_int4": NPUOnlineIntegerQuantSpec(
        mode="w4a4_int4",
        weight_dtype=torch.quint4x2,
        activation_dtype=torch.quint4x2,
        dispatcher_output_dtype="bf16",
    ),
}


def get_npu_online_integer_quant_spec(
    mode: Optional[str] = None,
) -> Optional[NPUOnlineIntegerQuantSpec]:
    if mode is None:
        mode = get_server_args().online_quantization
    return _ONLINE_INTEGER_QUANT_SPECS.get(mode)


def validate_npu_online_source_dtype(params_dtype: torch.dtype) -> None:
    if params_dtype != torch.float16:
        raise ValueError(
            "Ascend online integer quantization currently requires --dtype "
            "float16; the dynamic-scale matmul contract does not support "
            "non-FP16 output."
        )


def npu_dynamic_quantize_weight(
    weight: torch.Tensor, spec: NPUOnlineIntegerQuantSpec
) -> tuple[torch.Tensor, torch.Tensor]:
    kwargs = {}
    if spec.weight_dtype != torch.int8:
        kwargs["dst_type"] = spec.weight_dtype
    return torch.ops.npu.npu_dynamic_quant(weight, **kwargs)


def npu_format_online_weight(weight: torch.Tensor) -> torch.Tensor:
    return npu_format_cast(weight.transpose(-2, -1).contiguous())


class NPUOnlineDenseWeightLoader:
    def __init__(
        self,
        layer: torch.nn.Module,
        params_dtype: torch.dtype,
        original_weight_loader: Callable,
        spec: NPUOnlineIntegerQuantSpec,
        on_complete: Callable[[], None],
    ) -> None:
        validate_npu_online_source_dtype(params_dtype)
        self.layer = layer
        self.params_dtype = params_dtype
        self.original_weight_loader = original_weight_loader
        self.spec = spec
        self.on_complete = on_complete
        self.load_device = torch.get_default_device()
        self.lock = threading.Lock()
        self.loaded_numel = 0
        self.state = "loading"
        self.source_shape = None
        self.target_numel = 0

    def register_source(self, weight: Parameter) -> None:
        self.source_shape = tuple(weight.shape)
        self.target_numel = weight.numel()
        self.layer._npu_online_dense_loader = self

    def _materialize_source(self) -> Parameter:
        current = self.layer.weight
        source = current.__class__(
            data=torch.empty(
                self.source_shape,
                dtype=self.params_dtype,
                device=self.load_device,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=self.weight_loader,
        )
        copy_missing_attrs(current, source)
        self.layer.weight = source
        return source

    def weight_loader(
        self,
        param: Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id=None,
    ) -> None:
        with self.lock:
            if self.state == "converted":
                raise RuntimeError(
                    "Duplicate dense online-quantized weight load before "
                    "post-load processing completed."
                )
            was_ready_reload = self.state == "ready_reload"
            if was_ready_reload:
                self.loaded_numel = 0
                self.state = "loading"
            current = self.layer.weight
            if current.device.type == "meta" or was_ready_reload or (
                tuple(current.shape) != self.source_shape
                or current.dtype != self.params_dtype
            ):
                current = self._materialize_source()
            param = current

        loaded_weight = loaded_weight.to(self.load_device)
        kwargs = {}
        if loaded_shard_id is not None:
            kwargs["loaded_shard_id"] = loaded_shard_id
        copy_counter = CopyNumelCounter()
        with copy_counter:
            self.original_weight_loader(param, loaded_weight, **kwargs)

        should_quantize = False
        with self.lock:
            copied = self.loaded_numel + copy_counter.copied_numel
            if copied > self.target_numel:
                raise RuntimeError(
                    "Dense online-quantized weight load overflow: copied "
                    f"{copied} elements, expected {self.target_numel}."
                )
            self.loaded_numel = copied
            if copied == self.target_numel:
                self.state = "quantizing"
                should_quantize = True

        if should_quantize:
            self.on_complete()
            with self.lock:
                self.state = "converted"

    def finish_post_load(self) -> None:
        if self.state == "ready_reload":
            return
        if self.state != "converted":
            raise RuntimeError(
                "Ascend online integer dense weight was not completely loaded "
                "through its completion-tracked loader."
            )
        self.state = "ready_reload"


class NPUOnlineMoEWeightLoader:
    def __init__(
        self,
        layer: torch.nn.Module,
        params_dtype: torch.dtype,
        original_weight_loader: Callable,
        spec: NPUOnlineIntegerQuantSpec,
    ) -> None:
        validate_npu_online_source_dtype(params_dtype)
        self.layer = layer
        self.params_dtype = params_dtype
        self.original_weight_loader = original_weight_loader
        self.spec = spec
        self.load_device = torch.get_default_device()
        self.lock = threading.Lock()
        self.loaded_numel = {"w13": 0, "w2": 0}
        self.state = {"w13": "loading", "w2": "loading"}
        self.source_shapes = {}
        self.target_numel = {}

    def register_sources(
        self, w13_weight: Parameter, w2_weight: Parameter
    ) -> None:
        self.source_shapes = {
            "w13": tuple(w13_weight.shape),
            "w2": tuple(w2_weight.shape),
        }
        self.target_numel = {
            "w13": w13_weight.numel(),
            "w2": w2_weight.numel(),
        }
        self.layer._npu_online_moe_loader = self

    def _materialize_source(self, weight_prefix: str) -> Parameter:
        weight_name = f"{weight_prefix}_weight"
        current = getattr(self.layer, weight_name)
        source = Parameter(
            torch.empty(
                self.source_shapes[weight_prefix],
                dtype=self.params_dtype,
                device=self.load_device,
            ),
            requires_grad=False,
        )
        copy_missing_attrs(current, source)
        setattr(self.layer, weight_name, source)
        return source

    @staticmethod
    def _weight_prefix(weight_name: str) -> str:
        if "w13" in weight_name:
            return "w13"
        if "w2" in weight_name:
            return "w2"
        raise ValueError(f"Expected an online MoE w13/w2 weight, got {weight_name!r}.")

    def weight_loader(
        self,
        param: Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str,
        expert_id: int,
    ) -> None:
        weight_prefix = self._weight_prefix(weight_name)
        weight_attr = f"{weight_prefix}_weight"

        with self.lock:
            state = self.state[weight_prefix]
            if state == "converted":
                raise RuntimeError(
                    f"Duplicate load for online-quantized {weight_attr} before "
                    "post-load processing completed."
                )
            if state == "ready_reload":
                self.loaded_numel[weight_prefix] = 0
                self.state[weight_prefix] = "loading"

            current = getattr(self.layer, weight_attr)
            if current.device.type == "meta" or state == "ready_reload":
                current = self._materialize_source(weight_prefix)
            param = current

        loaded_weight = loaded_weight.to(self.load_device)
        copy_counter = CopyNumelCounter()
        with copy_counter:
            self.original_weight_loader(
                param, loaded_weight, weight_name, shard_id, expert_id
            )

        should_quantize = False
        with self.lock:
            copied = self.loaded_numel[weight_prefix] + copy_counter.copied_numel
            target = self.target_numel[weight_prefix]
            if copied > target:
                raise RuntimeError(
                    f"Online-quantized {weight_attr} load overflow: copied "
                    f"{copied} elements, expected {target}."
                )
            self.loaded_numel[weight_prefix] = copied
            if copied == target:
                self.state[weight_prefix] = "quantizing"
                should_quantize = True

        if should_quantize:
            kernel = getattr(self.layer, f"{weight_prefix}_kernel")
            kernel.process_weights_after_loading(self.layer, weight_prefix)
            with self.lock:
                self.state[weight_prefix] = "converted"


def create_npu_online_moe_weight_loader(
    layer: torch.nn.Module,
    params_dtype: torch.dtype,
    original_weight_loader: Callable,
) -> Optional[NPUOnlineMoEWeightLoader]:
    spec = get_npu_online_integer_quant_spec()
    if spec is None:
        return None
    return NPUOnlineMoEWeightLoader(
        layer=layer,
        params_dtype=params_dtype,
        original_weight_loader=original_weight_loader,
        spec=spec,
    )
