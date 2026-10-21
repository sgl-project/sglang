"""Read serialized INT8 codes independently of the runtime's fused loader."""

import hashlib
from pathlib import Path

import torch
from safetensors import safe_open


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class CheckpointIntegerOracle:
    """Exact CPU integer GEMM; floating nonlinears stay on the tested backend."""

    def __init__(self, checkpoint):
        self.path = Path(checkpoint) / "model.safetensors"
        self.weight_sha256 = file_sha256(self.path)
        self.reader = safe_open(self.path, framework="pt", device="cpu")
        self.keys = set(self.reader.keys())
        self.logical_names = sorted(
            name[: -len(".input_scale")]
            for name in self.keys
            if name.endswith(".input_scale")
        )
        self.groups = {}
        self.cache = {}
        self.checks = []
        self.calls = []
        for name in self.logical_names:
            if name.endswith((".q_proj", ".k_proj", ".v_proj")):
                projection = name.rsplit(".", 1)[1]
                target = name.rsplit(".", 1)[0] + ".qkv_proj"
                position = ("q_proj", "k_proj", "v_proj").index(projection)
            elif name.endswith((".gate_proj", ".up_proj")):
                projection = name.rsplit(".", 1)[1]
                target = name.rsplit(".", 1)[0] + ".gate_up_proj"
                position = ("gate_proj", "up_proj").index(projection)
            else:
                target, position = name, 0
            self.groups.setdefault(target, []).append((position, name))
        for entries in self.groups.values():
            entries.sort()
        self.logical_map = [
            {
                "fused": name,
                "shards": [
                    {
                        "position": position,
                        "logical": logical,
                        "weight_shape": list(
                            self.reader.get_slice(logical + ".weight").get_shape()
                        ),
                        "scale_shape": list(
                            self.reader.get_slice(logical + ".weight_scale").get_shape()
                        ),
                    }
                    for position, logical in entries
                ],
            }
            for name, entries in self.groups.items()
        ]

    def _serialized_group(self, name):
        weights, scales, steps = [], [], []
        for _, logical in self.groups[name]:
            weight = self.reader.get_tensor(logical + ".weight")
            scale = self.reader.get_tensor(logical + ".weight_scale").float()
            step = self.reader.get_tensor(logical + ".input_scale").float()
            if weight.dtype != torch.int8 or weight.ndim != 2:
                raise ValueError("Reference needs serialized INT8 [out, in] codes")
            if scale.numel() not in (1, weight.shape[0]):
                raise ValueError("Invalid tensor/channel weight scale")
            if step.numel() != 1:
                raise ValueError("Reference needs one static activation scale")
            if not bool(torch.isfinite(scale).all() & (scale > 0).all()):
                raise ValueError("Invalid serialized weight scale")
            if not bool(torch.isfinite(step).all() & (step > 0).all()):
                raise ValueError("Invalid serialized activation scale")
            weights.append(weight)
            scales.append(scale.reshape(-1).expand(weight.shape[0]))
            steps.append(step.reshape(()))
        if not all(torch.equal(step, steps[0]) for step in steps):
            raise ValueError("Serialized fused input scales differ")
        return torch.cat(weights).T.contiguous(), torch.cat(scales), steps[0]

    def verify_model(self, model):
        """Check every checkpoint tensor before computing expected outputs."""
        actual = model.state_dict()
        checks = []

        def check(name, expected):
            alias = (
                "model.embed_tokens.weight"
                if name == "lm_head.weight"
                else "lm_head.weight"
                if name == "model.embed_tokens.weight"
                else None
            )
            key = name if name in actual else alias
            if key is None or key not in actual:
                raise ValueError("Missing loaded checkpoint tensor: " + name)
            value = actual[key].detach().cpu()
            if (
                value.dtype != expected.dtype
                or value.shape != expected.shape
                or not torch.equal(value, expected)
            ):
                raise ValueError("Loaded checkpoint tensor differs: " + name)
            checks.append({"name": name, "actual_name": key, "passed": True})

        for name in self.groups:
            weight, scale, step = self._serialized_group(name)
            check(name + ".weight", weight)
            check(name + ".weight_scale", scale)
            check(name + ".input_scale", step.reshape(1))
            check(name + ".npu_static_quant_scale", step.expand(weight.shape[0]))
            check(
                name + ".npu_static_quant_offset",
                torch.zeros(weight.shape[0], dtype=torch.int32),
            )
        excluded = {
            name + suffix
            for name in self.logical_names
            for suffix in (".weight", ".weight_scale", ".input_scale")
        }
        for name in sorted(self.keys - excluded):
            check(name, self.reader.get_tensor(name))
        self.checks = checks
        return checks

    def linear(self, name, x, bias=None):
        if name not in self.cache:
            weight, scale, step = self._serialized_group(name)
            self.cache[name] = weight.to(torch.int64), scale, step
        weight, scale, step = self.cache[name]
        codes = torch.round(x.detach().cpu().float() / step).clamp(-128, 127)
        accumulator = (
            codes.to(torch.int8).reshape(-1, weight.shape[0]).to(torch.int64) @ weight
        )
        if accumulator.numel() and (
            int(accumulator.min()) < -(2**31) or int(accumulator.max()) >= 2**31
        ):
            raise OverflowError("Reference integer accumulator exceeds INT32")
        value = accumulator.float() * (scale * step)
        if bias is not None:
            value += bias.detach().cpu().float()
        self.calls.append({"module": name, "input_shape": list(x.shape)})
        return value.reshape(*x.shape[:-1], weight.shape[-1]).to(x.dtype).to(x.device)

    def receipt(self):
        return {
            "checkpoint_weight_sha256": self.weight_sha256,
            "checkpoint_logical_linears": len(self.logical_names),
            "fused_modules": len(self.groups),
            "logical_to_fused": self.logical_map,
            "loader_checks": self.checks,
            "cpu_integer_linear_calls": self.calls,
            "reference_math": "CPU static divide-round-clamp, INT64 GEMM, FP32 combined scales, floating bias, output cast",
        }
