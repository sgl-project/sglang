"""Calibrate a temporary static-INT8 checkpoint from real cached Qwen3 weights."""

from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

REFERENCE_PROMPTS = (
    "The capital of France is",
    "A triangle has three sides. A square has",
)
NEW_TOKENS = 8
MAX_LOGPROB_ERROR = 0.05


def prepare_static_int8_qwen(base_model, output_dir, strategy="channel"):
    if not Path(base_model).is_dir():
        base_model = snapshot_download(
            base_model, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"]
        )
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
        n: m
        for n, m in model.named_modules()
        if isinstance(m, torch.nn.Linear) and n != "lm_head"
    }
    maxima = dict.fromkeys(linears, 0.0)
    hooks = []
    for name, module in linears.items():

        def observe(module, inputs, name=name):
            maxima[name] = max(maxima[name], float(inputs[0].detach().abs().max()))

        hooks.append(module.register_forward_pre_hook(observe))
    try:
        with torch.inference_mode():
            for text in REFERENCE_PROMPTS:
                model(**tokenizer(text, return_tensors="pt"), use_cache=False)
    finally:
        for hook in hooks:
            hook.remove()
    if not maxima or not all(
        v > 0 and torch.isfinite(torch.tensor(v)) for v in maxima.values()
    ):
        raise ValueError("Every linear must receive finite, nonzero calibration inputs")
    state = {
        n: t.detach().cpu().to(torch.bfloat16).contiguous().clone()
        if t.is_floating_point()
        else t.detach().cpu().contiguous().clone()
        for n, t in model.state_dict().items()
    }
    for name, layer in linears.items():
        weight = layer.weight.detach().float()
        maximum = (
            weight.abs().amax(dim=1, keepdim=True)
            if strategy == "channel"
            else weight.abs().amax().reshape(1)
        )
        scale = (maximum / 127).clamp_min(1e-8)
        state[name + ".weight"] = (
            torch.round(weight / scale).clamp(-128, 127).to(torch.int8).contiguous()
        )
        state[name + ".weight_scale"] = scale.contiguous()
        state[name + ".input_scale"] = torch.tensor([max(maxima[name] / 127, 1e-8)])
    args = dict(num_bits=8, type="int", symmetric=True, dynamic=False)
    model.config.quantization_config = {
        "quant_method": "compressed-tensors",
        "format": "int-quantized",
        "quantization_status": "compressed",
        "ignore": ["lm_head"],
        "config_groups": {
            "group_0": {
                "targets": ["Linear"],
                "weights": dict(args, strategy=strategy),
                "input_activations": dict(args, strategy="tensor"),
            }
        },
    }
    model.config.torch_dtype = torch.bfloat16
    model.config.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    save_file(state, output_dir / "model.safetensors", metadata={"format": "pt"})
    return [
        tokenizer.encode(prompt, add_special_tokens=True)
        for prompt in REFERENCE_PROMPTS
    ]
