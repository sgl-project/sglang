"""A temporary, calibrated static-INT8 checkpoint from the cached Qwen3 fixture."""

from pathlib import Path

import torch
from compressed_tensors.quantization import QuantizationConfig
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

CALIBRATION_TEXTS = (
    "The capital of France is Paris. The capital of Japan is Tokyo.",
    "Two plus three equals five. Twelve divided by four equals three.",
    "Write a short explanation of how rain forms from water vapor.",
    "A binary search repeatedly divides a sorted list into smaller halves.",
    "A triangle has three sides. A square has four equal sides.",
    "Translate the word hello into French and explain its meaning.",
    "The temperature changed from minus five degrees to ten degrees.",
    "Summarize the difference between a noun and a verb in one sentence.",
)
REFERENCE_PROMPTS = (
    "The capital of France is",
    "A triangle has three sides. A square has",
)
NEW_TOKENS = 8
MAX_LOGPROB_ERROR = 0.05


def prepare_static_int8_qwen(base_model, output_dir):
    """Calibrate cached real weights; return tokenized test prompts."""
    output_dir = Path(output_dir)
    output_dir.mkdir()
    tokenizer = AutoTokenizer.from_pretrained(base_model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float32,
        local_files_only=True,
        attn_implementation="eager",
    ).eval()
    linears = {
        name: layer
        for name, layer in model.named_modules()
        if isinstance(layer, torch.nn.Linear) and name != "lm_head"
    }
    if not linears:
        raise ValueError("The cached fixture has no dense linear layers")
    maxima = dict.fromkeys(linears, 0.0)
    observations = dict.fromkeys(linears, 0)
    hooks = []
    for name, layer in linears.items():

        def observe(module, inputs, name=name):
            value = inputs[0].detach().float()
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"Nonfinite calibration input for {name}")
            maxima[name] = max(maxima[name], float(value.abs().max()))
            observations[name] += value.numel()

        hooks.append(layer.register_forward_pre_hook(observe))
    try:
        with torch.inference_mode():
            for text in CALIBRATION_TEXTS:
                tokens = tokenizer(
                    text, return_tensors="pt", truncation=True, max_length=64
                )
                model(**tokens, use_cache=False)
    finally:
        for hook in hooks:
            hook.remove()
    if not all(observations.values()):
        raise ValueError("Every quantized layer must execute during calibration")
    input_scales = {name: max(value / 127.0, 1e-8) for name, value in maxima.items()}
    for name in linears:
        if name.endswith(".q_proj"):
            fused = [name[:-6] + suffix for suffix in ("q_proj", "k_proj", "v_proj")]
        elif name.endswith(".gate_proj"):
            fused = [name, name[:-9] + "up_proj"]
        else:
            continue
        if len({input_scales[part] for part in fused}) != 1:
            raise ValueError("Fused projections need identical calibration scales")
    state = {
        name: value.detach().cpu().to(torch.bfloat16).contiguous().clone()
        if value.is_floating_point()
        else value.detach().cpu().contiguous().clone()
        for name, value in model.state_dict().items()
    }
    for name, layer in linears.items():
        weight = layer.weight.detach().float()
        if not bool(torch.isfinite(weight).all()):
            raise ValueError(f"Nonfinite model weight for {name}")
        scale = (weight.abs().amax(dim=1, keepdim=True) / 127.0).clamp_min(1e-8)
        state[name + ".weight"] = (
            torch.round(weight / scale).clamp(-128, 127).to(torch.int8).contiguous()
        )
        state[name + ".weight_scale"] = scale.contiguous()
        state[name + ".input_scale"] = torch.tensor(
            [input_scales[name]], dtype=torch.float32
        )
    model.config.quantization_config = QuantizationConfig.model_validate(
        {
            "quant_method": "compressed-tensors",
            "format": "int-quantized",
            "quantization_status": "compressed",
            "ignore": ["lm_head"],
            "config_groups": {
                "group_0": {
                    "targets": ["Linear"],
                    "weights": {
                        "num_bits": 8,
                        "type": "int",
                        "strategy": "channel",
                        "symmetric": True,
                        "dynamic": False,
                    },
                    "input_activations": {
                        "num_bits": 8,
                        "type": "int",
                        "strategy": "tensor",
                        "symmetric": True,
                        "dynamic": False,
                    },
                }
            },
        }
    ).model_dump(mode="json", exclude_none=True)
    model.config.torch_dtype = torch.bfloat16
    model.config.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    save_file(state, output_dir / "model.safetensors", metadata={"format": "pt"})
    return [
        {"input_ids": tokenizer.encode(prompt, add_special_tokens=True)}
        for prompt in REFERENCE_PROMPTS
    ]
