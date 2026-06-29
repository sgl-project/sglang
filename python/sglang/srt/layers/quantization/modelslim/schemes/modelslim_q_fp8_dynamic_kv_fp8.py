import torch
from torch import nn

from sglang.srt.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from sglang.srt.layers.quantization.modelslim.schemes.modelslim_scheme import (
    ModelSlimKVSchemeBase,
)


def _modelslim_kv_weight_loader(
    param: torch.Tensor, loaded_weight: torch.Tensor
) -> None:
    if param.numel() == 1 and loaded_weight.numel() == 1:
        param.data.fill_(loaded_weight.item())
        return

    loaded_weight = loaded_weight.to(param.dtype)
    if loaded_weight.shape != param.shape:
        tp_rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_world_size()
        if loaded_weight.shape[0] % tp_size != 0:
            raise ValueError(
                f"Cannot shard ModelSlim KV weight {tuple(loaded_weight.shape)} "
                f"across tensor parallel size {tp_size}."
            )
        shard_size = loaded_weight.shape[0] // tp_size
        loaded_weight = loaded_weight.narrow(0, shard_size * tp_rank, shard_size)

    if loaded_weight.shape != param.shape:
        raise ValueError(
            f"Attempted to load ModelSlim KV weight {tuple(loaded_weight.shape)} "
            f"into parameter {tuple(param.shape)}."
        )

    param.data.copy_(loaded_weight)


class ModelSlimQFP8DynamicKVFP8Scheme(ModelSlimKVSchemeBase):
    def __init__(self, quant_config, prefix: str):
        self.quant_config = quant_config
        self.prefix = prefix

    def create_weights(
        self,
        layer: nn.Module,
        num_heads: int,
        num_kv_heads: int,
    ) -> None:

        for name in ("fa_q", "fa_k", "fa_v"):
            if not hasattr(layer, name):
                setattr(layer, name, nn.Module())

        params = {
            "fa_q.scale": torch.ones((num_heads, 1), dtype=torch.float32),
            "fa_k.scale": torch.ones((num_kv_heads, 1), dtype=torch.float32),
            "fa_v.scale": torch.ones((num_kv_heads, 1), dtype=torch.float32),
            "fa_q.offset": torch.zeros((num_heads, 1), dtype=torch.float32),
            "fa_k.offset": torch.zeros((num_kv_heads, 1), dtype=torch.float32),
            "fa_v.offset": torch.zeros((num_kv_heads, 1), dtype=torch.float32),
        }

        for name, value in params.items():
            module_name, param_name = name.rsplit(".", 1)
            module = getattr(layer, module_name)
            if hasattr(module, param_name):
                continue
            param = nn.Parameter(value, requires_grad=False)
            param.weight_loader = _modelslim_kv_weight_loader
            module.register_parameter(param_name, param)

        runtime_params = {
            "fak_descale_float": torch.ones((1, num_kv_heads), dtype=torch.float32),
            "fak_descale_reciprocal": torch.ones(
                (1, num_kv_heads), dtype=torch.float32
            ),
        }
        for name, value in runtime_params.items():
            if hasattr(layer, name):
                continue
            layer.register_parameter(name, nn.Parameter(value, requires_grad=False))

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        if not hasattr(layer, "fak_descale_float") or not hasattr(
            layer, "fak_descale_reciprocal"
        ):
            raise RuntimeError(
                f"ModelSlimQFP8DynamicKVFP8Scheme for {self.prefix} has not created weights yet."
            )

        fa_k_scale = torch.squeeze(layer.fa_k.scale).reshape(1, -1).to(torch.float32)
        fa_k_descale_reciprocal = torch.reciprocal(fa_k_scale)

        layer.fak_descale_float.data.copy_(fa_k_scale)
        layer.fak_descale_reciprocal.data.copy_(fa_k_descale_reciprocal)
