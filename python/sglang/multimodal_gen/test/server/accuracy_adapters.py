from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from sglang.multimodal_gen.test.server.accuracy_utils import seed_and_broadcast


class ComponentAdapter(ABC):
    @abstractmethod
    def generate_inputs(
        self,
        case: Any,
        model: nn.Module,
        device: str,
        ref_model: Optional[nn.Module] = None,
    ) -> Dict[str, Any]:
        pass

    @abstractmethod
    def run_sglang(self, model: nn.Module, inputs: Dict[str, Any]) -> torch.Tensor:
        pass

    @abstractmethod
    def run_reference(self, model: nn.Module, inputs: Dict[str, Any]) -> torch.Tensor:
        pass


class _DeterministicRNG:
    """Generate deterministic tensors across ranks (CPU -> device + broadcast)."""

    def __init__(self, seed: int = 42) -> None:
        self._seed = seed

    def randn(
        self, shape: tuple[int, ...], device: str, dtype: torch.dtype
    ) -> torch.Tensor:
        torch.manual_seed(self._seed)
        tensor = torch.randn(shape, device="cpu", dtype=dtype).to(device)
        seed_and_broadcast(self._seed, tensor)
        self._seed += 1
        return tensor


class FluxTransformerAdapter(ComponentAdapter):
    def generate_inputs(
        self,
        case: Any,
        model: nn.Module,
        device: str,
        ref_model: Optional[nn.Module] = None,
    ) -> Dict[str, Any]:
        rng = _DeterministicRNG()
        arch = getattr(model.config, "arch_config", model.config)

        cid = case.id.lower()
        is_flux2 = (
            "flux_2" in cid or "flux2" in cid or "flux2" in str(type(model)).lower()
        )

        in_c = 128 if is_flux2 else 64
        text_c = getattr(arch, "joint_attention_dim", 4096)

        h, w = 16, 16
        img_len = h * w
        txt_len = 64

        if is_flux2:
            img_ids = torch.zeros(img_len, 4, device=device, dtype=torch.bfloat16)
            img_ids[:, 1] = torch.arange(h).repeat_interleave(w)
            img_ids[:, 2] = torch.arange(w).repeat(h)
            txt_ids = torch.zeros(txt_len, 4, device=device, dtype=torch.bfloat16)
        else:
            img_ids = torch.zeros(img_len, 3, device=device, dtype=torch.bfloat16)
            img_ids[:, 0] = torch.arange(h).repeat_interleave(w)
            img_ids[:, 1] = torch.arange(w).repeat(h)
            txt_ids = torch.zeros(txt_len, 3, device=device, dtype=torch.bfloat16)

        return {
            "hidden_states": rng.randn((1, img_len, in_c), device, torch.bfloat16),
            "encoder_hidden_states": rng.randn(
                (1, txt_len, text_c), device, torch.bfloat16
            ),
            "pooled_projections": rng.randn(
                (1, getattr(arch, "pooled_projection_dim", 768)), device, torch.bfloat16
            ),
            "timestep": torch.tensor([500.0], device=device, dtype=torch.bfloat16),
            "img_ids": img_ids,
            "txt_ids": txt_ids,
            "guidance": torch.tensor([1.0], device=device, dtype=torch.bfloat16),
        }

    def run_sglang(self, model: nn.Module, inputs: Dict[str, Any]) -> torch.Tensor:
        kwargs = {
            "hidden_states": inputs["hidden_states"],
            "encoder_hidden_states": inputs["encoder_hidden_states"],
            "timestep": inputs["timestep"],
        }
        if "flux2" in str(type(model)).lower():
            ids = torch.cat([inputs["txt_ids"], inputs["img_ids"]], dim=0)
            kwargs["freqs_cis"] = model.rotary_emb(ids)
        sig = inspect.signature(model.forward)
        if "guidance" in sig.parameters and "guidance" in inputs:
            kwargs["guidance"] = inputs["guidance"] * 1000.0
        for k in ["pooled_projections", "img_ids", "txt_ids"]:
            if k in sig.parameters and k in inputs:
                kwargs[k] = inputs[k]
        return model(**kwargs)

    def run_reference(self, model: nn.Module, inputs: Dict[str, Any]) -> torch.Tensor:
        kwargs = {
            "hidden_states": inputs["hidden_states"],
            "encoder_hidden_states": inputs["encoder_hidden_states"],
            "timestep": inputs["timestep"] / 1000.0,
            "img_ids": inputs["img_ids"],
            "txt_ids": inputs["txt_ids"],
            "return_dict": True,
        }
        sig = inspect.signature(model.forward)
        if "guidance" in sig.parameters and "guidance" in inputs:
            kwargs["guidance"] = inputs["guidance"]
        if "pooled_projections" in sig.parameters:
            kwargs["pooled_projections"] = inputs["pooled_projections"]
        out = model(**kwargs)
        return out.sample if hasattr(out, "sample") else out


class WanTransformerAdapter(ComponentAdapter):
    def generate_inputs(
        self,
        case: Any,
        model: nn.Module,
        device: str,
        ref_model: Optional[nn.Module] = None,
    ) -> Dict[str, Any]:
        rng = _DeterministicRNG()
        cid = case.id.lower()
        is_i2v = "i2v" in cid
        is_wan22 = "wan2.2" in cid or "wan2_2" in cid or "wan2-2" in cid

        # Wan 2.2 TI2V 5B uses 48 channels (16 video + 32 image conditioning)
        if is_wan22 and "ti2v" in cid:
            in_c = 48
        else:
            in_c = 36 if is_i2v else 16

        num_frames = 4

        inner_model = model.module if hasattr(model, "module") else model
        if (
            hasattr(inner_model, "config")
            and getattr(inner_model.config, "rope_max_seq_len", None) != 1024
        ):
            inner_model.config.rope_max_seq_len = 1024

        # Generate on CPU and broadcast to keep ranks aligned under SP.
        inputs = {
            "hidden_states": rng.randn(
                (1, in_c, num_frames, 32, 32), device, torch.bfloat16
            ),
            "encoder_hidden_states": rng.randn((1, 64, 4096), device, torch.bfloat16),
            "timestep": torch.tensor([500.0], device=device, dtype=torch.bfloat16),
            "guidance": torch.tensor([1.0], device=device, dtype=torch.bfloat16),
        }
        if is_i2v:
            inputs["encoder_hidden_states_image"] = rng.randn(
                (1, 257, 1280), device, torch.bfloat16
            )
        return inputs

    def run_sglang(self, model: nn.Module, inputs: Dict[str, Any]) -> torch.Tensor:
        x = inputs["hidden_states"]
        context = inputs["encoder_hidden_states"]
        context_img = inputs.get("encoder_hidden_states_image")
        kwargs = {
            "hidden_states": x,
            "encoder_hidden_states": context,
            "timestep": inputs["timestep"],
        }
        if context_img is not None:
            kwargs["encoder_hidden_states_image"] = [context_img]

        sig = inspect.signature(model.forward)
        # Diffusers WanTransformer3DModel does not take guidance; keep SGLang aligned.
        if "guidance" in sig.parameters and "guidance" in inputs:
            if "wan" not in str(type(model)).lower():
                kwargs["guidance"] = inputs["guidance"] * 1000.0

        out_shard = model(**kwargs)
        return out_shard

    def run_reference(self, model: nn.Module, inputs: Dict[str, Any]) -> torch.Tensor:
        kwargs = {
            "hidden_states": inputs["hidden_states"],
            "encoder_hidden_states": inputs["encoder_hidden_states"],
            "timestep": inputs["timestep"],
            "return_dict": True,
        }
        if "encoder_hidden_states_image" in inputs:
            kwargs["encoder_hidden_states_image"] = inputs[
                "encoder_hidden_states_image"
            ]

        if "wan" in str(type(model)).lower():
            kwargs.pop("guidance", None)
        else:
            sig = inspect.signature(model.forward)
            if "guidance" in sig.parameters and "guidance" in inputs:
                kwargs["guidance"] = inputs["guidance"]

        out = model(**kwargs)
        return out.sample if hasattr(out, "sample") else out


class HunyuanTransformerAdapter(ComponentAdapter):
    def generate_inputs(
        self,
        case: Any,
        model: nn.Module,
        device: str,
        ref_model: Optional[nn.Module] = None,
    ) -> Dict[str, Any]:
        rng = _DeterministicRNG()
        txt_seq = rng.randn((1, 64, 4096), device, torch.bfloat16)
        pooled = rng.randn((1, 768), device, torch.bfloat16)
        return {
            "hidden_states": rng.randn((1, 16, 4, 16, 16), device, torch.bfloat16),
            "encoder_hidden_states_ref": txt_seq,
            "pooled_projections": pooled,
            "timestep": torch.tensor([500.0], device=device, dtype=torch.bfloat16),
            "encoder_attention_mask": torch.ones(
                1, 64, device=device, dtype=torch.bool
            ),
            "guidance": torch.tensor([1.0], device=device, dtype=torch.bfloat16),
        }

    def run_sglang(self, model: nn.Module, inputs: Dict[str, Any]) -> torch.Tensor:
        kwargs = {
            "hidden_states": inputs["hidden_states"],
            "encoder_hidden_states": [
                inputs["encoder_hidden_states_ref"],
                inputs["pooled_projections"],
            ],
            "timestep": inputs["timestep"],
        }
        sig = inspect.signature(model.forward)
        if "guidance" in sig.parameters and "guidance" in inputs:
            kwargs["guidance"] = inputs["guidance"] * 1000.0
        return model(**kwargs)

    def run_reference(self, model: nn.Module, inputs: Dict[str, Any]) -> torch.Tensor:
        kwargs = {
            "hidden_states": inputs["hidden_states"],
            "timestep": inputs["timestep"],
            "encoder_hidden_states": inputs["encoder_hidden_states_ref"],
            "encoder_attention_mask": inputs["encoder_attention_mask"],
            "pooled_projections": inputs["pooled_projections"],
            "return_dict": True,
        }
        sig = inspect.signature(model.forward)
        if "guidance" in sig.parameters and "guidance" in inputs:
            kwargs["guidance"] = inputs["guidance"]
        out = model(**kwargs)
        return out.sample if hasattr(out, "sample") else out


class QwenTransformerAdapter(ComponentAdapter):
    def generate_inputs(
        self,
        case: Any,
        model: nn.Module,
        device: str,
        ref_model: Optional[nn.Module] = None,
    ) -> Dict[str, Any]:
        rng = _DeterministicRNG()
        arch = getattr(model.config, "arch_config", model.config)
        in_c = getattr(arch, "in_channels", 64)
        joint_c = getattr(arch, "joint_attention_dim", 4096)
        h, w = 32, 32
        seq_len = (h // 2) * (w // 2)
        inputs = {
            "hidden_states": rng.randn((1, seq_len, in_c), device, torch.bfloat16),
            "encoder_hidden_states": rng.randn(
                (1, 64, joint_c), device, torch.bfloat16
            ),
            "timestep": torch.tensor([500.0], device=device, dtype=torch.bfloat16),
            "img_shapes": [[(1, h // 2, w // 2)]],
            "txt_seq_lens": [64],
        }
        if "layered" in case.id.lower() or getattr(
            model.config, "additional_t_cond", False
        ):
            # Qwen expects integer indices (0/1) for the additional timestep embedding.
            inputs["additional_t_cond"] = torch.zeros(
                (1,), device=device, dtype=torch.long
            )
        return inputs

    def run_sglang(self, model: nn.Module, inputs: Dict[str, Any]) -> torch.Tensor:
        device = inputs["hidden_states"].device
        img_freqs, txt_freqs = model.rotary_emb(
            inputs["img_shapes"], inputs["txt_seq_lens"], device=device
        )
        freqs_cis = (
            torch.cat([img_freqs.real.float(), img_freqs.imag.float()], dim=-1),
            torch.cat([txt_freqs.real.float(), txt_freqs.imag.float()], dim=-1),
        )
        kwargs = {
            "hidden_states": inputs["hidden_states"],
            "encoder_hidden_states": inputs["encoder_hidden_states"],
            "timestep": inputs["timestep"],
            "img_shapes": inputs["img_shapes"],
            "txt_seq_lens": inputs["txt_seq_lens"],
            "freqs_cis": freqs_cis,
        }
        if (
            "additional_t_cond" in inputs
            and "additional_t_cond" in inspect.signature(model.forward).parameters
        ):
            kwargs["additional_t_cond"] = inputs["additional_t_cond"]
        return model(**kwargs)

    def run_reference(self, model: nn.Module, inputs: Dict[str, Any]) -> torch.Tensor:
        kwargs = {
            "hidden_states": inputs["hidden_states"],
            "encoder_hidden_states": inputs["encoder_hidden_states"],
            "timestep": inputs["timestep"] / 1000.0,
            "img_shapes": inputs["img_shapes"],
            "txt_seq_lens": inputs["txt_seq_lens"],
            "return_dict": True,
        }
        if (
            "additional_t_cond" in inputs
            and "additional_t_cond" in inspect.signature(model.forward).parameters
        ):
            kwargs["additional_t_cond"] = inputs["additional_t_cond"]
        out = model(**kwargs)
        return out.sample if hasattr(out, "sample") else out


class ZImageTransformerAdapter(ComponentAdapter):
    def generate_inputs(
        self,
        case: Any,
        model: nn.Module,
        device: str,
        ref_model: Optional[nn.Module] = None,
    ) -> Dict[str, Any]:
        rng = _DeterministicRNG()
        # Generate on CPU and broadcast to keep ranks aligned under SP/TP.
        in_c = getattr(model, "in_channels", 16)
        h, w = 32, 32
        x = rng.randn((1, in_c, 1, h, w), device, torch.bfloat16)
        context = rng.randn((1, 64, 2560), device, torch.bfloat16)
        cap_len = 64
        cap_pad_len = (-cap_len) % 32
        freqs_cis = None
        if ref_model is not None and hasattr(ref_model, "patchify_and_embed"):
            try:
                patch_out = ref_model.patchify_and_embed(
                    [x.squeeze(0).clone()],
                    [context.squeeze(0).clone()],
                    patch_size=2,
                    f_patch_size=1,
                )
                if len(patch_out) >= 5:
                    x_pos_ids = patch_out[3]
                    cap_pos_ids = patch_out[4]
                    if isinstance(x_pos_ids, (list, tuple)):
                        x_pos_ids = torch.cat(x_pos_ids, dim=0)
                    if isinstance(cap_pos_ids, (list, tuple)):
                        cap_pos_ids = torch.cat(cap_pos_ids, dim=0)
                    x_pos_ids = x_pos_ids.to(device=device, dtype=torch.long)
                    cap_pos_ids = cap_pos_ids.to(device=device, dtype=torch.long)
                    cos_img, sin_img = model.rotary_emb(x_pos_ids)
                    cos_cap, sin_cap = model.rotary_emb(cap_pos_ids)
                    freqs_cis = ((cos_cap, sin_cap), (cos_img, sin_img))
            except Exception:
                freqs_cis = None
        if freqs_cis is None:
            cap_ids = (
                torch.stack(
                    torch.meshgrid(
                        torch.arange(cap_len + cap_pad_len),
                        torch.arange(1),
                        torch.arange(1),
                        indexing="ij",
                    ),
                    dim=-1,
                )
                .flatten(0, 2)
                .to(device)
            )
            img_ids = (
                torch.stack(
                    torch.meshgrid(
                        torch.arange(1),
                        torch.arange(h // 2),
                        torch.arange(w // 2),
                        indexing="ij",
                    ),
                    dim=-1,
                )
                .flatten(0, 2)
                .to(device)
            )
            cos_cap, sin_cap = model.rotary_emb(cap_ids)
            cos_img, sin_img = model.rotary_emb(img_ids)
            freqs_cis = ((cos_cap, sin_cap), (cos_img, sin_img))
        return {
            "hidden_states": x,
            "encoder_hidden_states": context,
            "timestep": torch.tensor([500.0], device=device, dtype=torch.bfloat16),
            "freqs_cis": freqs_cis,
        }

    def run_sglang(self, model: nn.Module, inputs: Dict[str, Any]) -> torch.Tensor:
        return model(
            hidden_states=inputs["hidden_states"],
            encoder_hidden_states=inputs["encoder_hidden_states"],
            timestep=inputs["timestep"],
            freqs_cis=inputs["freqs_cis"],
        )

    def _run_reference_impl(
        self, model: nn.Module, inputs: Dict[str, Any], *, invert_t: bool, negate: bool
    ) -> torch.Tensor:
        t_val = inputs["timestep"].item()
        if invert_t:
            t_norm = (1000.0 - t_val) / 1000.0
        else:
            t_norm = t_val / 1000.0
        kwargs = {
            "x": [inputs["hidden_states"].squeeze(0)],
            "t": torch.tensor(
                [t_norm], device=inputs["timestep"].device, dtype=torch.bfloat16
            ),
            "cap_feats": [inputs["encoder_hidden_states"].squeeze(0)],
            "return_dict": True,
        }
        out = model(**kwargs)
        res = (
            out.sample
            if hasattr(out, "sample")
            else (out[0] if isinstance(out, (list, tuple)) else out)
        )
        if isinstance(res, list):
            res = res[0]
        return -res if negate else res

    def run_reference(self, model: nn.Module, inputs: Dict[str, Any]) -> torch.Tensor:
        return self._run_reference_impl(model, inputs, invert_t=True, negate=True)


def _decode_vae(model: nn.Module, z: torch.Tensor) -> torch.Tensor:
    if (
        any(isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)) for m in model.modules())
        and z.ndim == 4
    ):
        z = z.unsqueeze(2)
    out = model.decode(z)
    res = out.sample if hasattr(out, "sample") else out
    return res.squeeze(2) if res.ndim == 5 else res


class BaseVAEAdapter(ComponentAdapter):
    def run_sglang(self, model: nn.Module, inputs: Dict[str, Any]) -> torch.Tensor:
        return _decode_vae(model, inputs["z"])

    def run_reference(self, model: nn.Module, inputs: Dict[str, Any]) -> torch.Tensor:
        return _decode_vae(model, inputs["z"])


class FluxVAEAdapter(BaseVAEAdapter):
    def generate_inputs(
        self,
        case: Any,
        model: nn.Module,
        device: str,
        ref_model: Optional[nn.Module] = None,
    ) -> Dict[str, Any]:
        rng = _DeterministicRNG()
        cid = case.id.lower()
        in_c = 32 if ("flux_2" in cid or "flux2" in cid) else 16
        z = rng.randn((1, in_c, 32, 32), device, torch.bfloat16)
        return {"z": z}


class WanVAEAdapter(BaseVAEAdapter):
    def generate_inputs(
        self,
        case: Any,
        model: nn.Module,
        device: str,
        ref_model: Optional[nn.Module] = None,
    ) -> Dict[str, Any]:
        rng = _DeterministicRNG()
        cid = case.id.lower()
        is_wan22 = "wan2.2" in cid or "wan2_2" in cid or "wan2-2" in cid
        in_c = 48 if (is_wan22 and "ti2v" in cid) else 16
        z = rng.randn((1, in_c, 32, 32), device, torch.bfloat16)
        return {"z": z}


class QwenVAEAdapter(BaseVAEAdapter):
    def generate_inputs(
        self,
        case: Any,
        model: nn.Module,
        device: str,
        ref_model: Optional[nn.Module] = None,
    ) -> Dict[str, Any]:
        rng = _DeterministicRNG()
        z = rng.randn((1, 16, 32, 32), device, torch.bfloat16)
        return {"z": z}


class ZImageVAEAdapter(BaseVAEAdapter):
    def generate_inputs(
        self,
        case: Any,
        model: nn.Module,
        device: str,
        ref_model: Optional[nn.Module] = None,
    ) -> Dict[str, Any]:
        rng = _DeterministicRNG()
        z = rng.randn((1, 16, 32, 32), device, torch.bfloat16)
        return {"z": z}


class HunyuanVAEAdapter(BaseVAEAdapter):
    def generate_inputs(
        self,
        case: Any,
        model: nn.Module,
        device: str,
        ref_model: Optional[nn.Module] = None,
    ) -> Dict[str, Any]:
        rng = _DeterministicRNG()
        z = rng.randn((1, 16, 32, 32), device, torch.bfloat16)
        return {"z": z}


def get_adapter(case_id: str, module_name: str) -> Optional[ComponentAdapter]:
    key = next(
        (
            k
            for k in ["flux", "wan", "qwen", "zimage", "hunyuan"]
            if k in case_id.lower()
        ),
        None,
    )
    if module_name == "transformer":
        mapping = {
            "flux": FluxTransformerAdapter,
            "wan": WanTransformerAdapter,
            "qwen": QwenTransformerAdapter,
            "zimage": ZImageTransformerAdapter,
            "hunyuan": HunyuanTransformerAdapter,
        }
        adapter_cls = mapping.get(key)
        return adapter_cls() if adapter_cls else None
    if module_name == "vae":
        mapping = {
            "flux": FluxVAEAdapter,
            "wan": WanVAEAdapter,
            "qwen": QwenVAEAdapter,
            "zimage": ZImageVAEAdapter,
            "hunyuan": HunyuanVAEAdapter,
        }
        adapter_cls = mapping.get(key)
        return adapter_cls() if adapter_cls else None
    return None
