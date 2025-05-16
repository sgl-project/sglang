"""
Common utilities for quark.
"""

import logging
from tqdm.auto import tqdm
import torch

logger = logging.getLogger(__name__)


def apply_quark_quant_config_to_model(model_config, quark_config):
    if quark_config == "" or quark_config is None or hasattr(model_config.hf_config, 'quantization_config'):
        return
    elif "int4fp8_moe" in quark_config:
        quantization_config = {
            "quant_method": "quark_int4fp8_moe",  # int4-fp8
            "activation_scheme": "dynamic",
            "weight_scheme": {
                "bits": 4,
                "group": "column",
                "sym": True
            },
        }
        model_config.hf_config.update({"quantization_config": quantization_config})
        model_config._verify_quantization()
    else:
        raise ValueError(f"Unexpected config: {quark_config}")
    

def online_quant(model_config, weights_dict):
    if model_config.quantization == 'quark_int4fp8_moe':

        def quantize_fp8_scale_tensorwise(w):
            FP8_MAX = 448.0
            scale = w.abs().amax().float() / FP8_MAX
            scaled = (w / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
            return scaled, scale


        def quantize_int4_scale_columnwise(w):
            S4_MAX = 7
            w_flat = w.reshape(-1, w.shape[-1]).float()
            scale = w_flat.abs().amax(axis=-1) / S4_MAX
            scaled = torch.round(w_flat / scale[:, None]).to(torch.int8).clamp(-S4_MAX, S4_MAX)
            return scaled.reshape(w.shape), scale.reshape(w.shape[:-1])


        def pack(to_pack: torch.Tensor, reorder: bool = True) -> torch.Tensor:
            if to_pack.ndim > 2:
                raise ValueError("Pack: Only supports tensors with dimensions not greater than 2.")

            if reorder:
                order_map = [0, 2, 4, 6, 1, 3, 5, 7]
            else:
                order_map = [0, 1, 2, 3, 4, 5, 6, 7]
            pack_num = 8
            if to_pack.ndim == 2:
                packed = torch.zeros(to_pack.shape[0], to_pack.shape[1] // pack_num, dtype=torch.int32, device=to_pack.device)
                new_c = to_pack.shape[1] // pack_num
                for c in range(new_c):
                    for i in range(pack_num):
                        # Use -3 as an example, high_position is 11111111,cause bit_or generate errors, so we can't use int4 directly
                        packed_col = to_pack[:, c * pack_num + order_map[i]].to(torch.int32)
                        packed_col = packed_col & 0x0F
                        packed[:, c] = torch.bitwise_or(packed[:, c], torch.bitwise_left_shift(packed_col, i * 4))
            elif to_pack.ndim == 0:
                packed = to_pack.to(torch.int32)
            else:
                packed = torch.zeros(to_pack.shape[0] // pack_num, dtype=torch.int32, device=to_pack.device)
                new_c = to_pack.shape[0] // pack_num
                for c in range(new_c):
                    for i in range(pack_num):
                        # Use -3 as an example, high_position is 11111111,cause bit_or generate errors, so we can't use int4 directly
                        packed_col = to_pack[c * pack_num + order_map[i]]
                        packed_col = packed_col & 0x0F
                        packed[c] = torch.bitwise_or(packed[c], torch.bitwise_left_shift(packed_col, i * 4))

            return packed

        def quark_quant_weights(weights_dict):
            for name, loaded_weight in tqdm(weights_dict, desc="Quark Online Quantizating "):
                if "w1.weight" in name or "w2.weight" in name or "w3.weight" in name: 
                    fp8_w, fp8_scale = quantize_fp8_scale_tensorwise(loaded_weight)
                    int4_w, int4_scale = quantize_int4_scale_columnwise(loaded_weight)

                    int4_w = pack(int4_w)
                    int4_scale /= fp8_scale

                    yield name, int4_w  
                    yield name + "_scale", fp8_scale
                    yield name + "_scale1", int4_scale
                elif "proj.weight" in name:
                    fp8_w, fp8_scale = quantize_fp8_scale_tensorwise(loaded_weight)

                    yield name, fp8_w  
                    yield name + "_scale", fp8_scale
                else:
                    yield name, loaded_weight  

        weights_dict = quark_quant_weights(weights_dict)
    return weights_dict
        

