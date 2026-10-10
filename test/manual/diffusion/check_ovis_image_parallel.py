# SPDX-License-Identifier: Apache-2.0
"""Tiny DiT oracle under real torchrun collectives, including both SP padding tails."""

import argparse
import json

import torch
from diffusers import OvisImageTransformer2DModel as ReferenceDiT

from sglang.multimodal_gen.configs.models.dits.ovis_image import OvisImageConfig
from sglang.multimodal_gen.configs.pipeline_configs.ovis_image import (
    OvisImagePipelineConfig,
)
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    cleanup_dist_env_and_memory,
    maybe_init_distributed_environment_and_model_parallel,
)
from sglang.multimodal_gen.runtime.loader.utils import set_default_torch_dtype
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.models.dits.ovis_image import (
    OvisImageTransformer2DModel,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs, set_global_server_args


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--ulysses", type=int, default=1)
    parser.add_argument("--ring", type=int, default=1)
    parser.add_argument(
        "--edge-cases",
        action="store_true",
        help="Also check one text token with 35 or 1 image tokens, for B1 and B4",
    )
    args = parser.parse_args()
    head_dim = 32 if args.ring > 1 else 8
    kwargs = dict(
        in_channels=64,
        out_channels=64,
        num_attention_heads=4,
        attention_head_dim=head_dim,
        joint_attention_dim=16,
        num_layers=1,
        num_single_layers=1,
        axes_dims_rope=(8, 12, 12) if args.ring > 1 else (2, 2, 4),
    )
    config = OvisImageConfig()
    config.update_model_arch(kwargs)
    server = ServerArgs(
        model_path="ATH-MaaS/Ovis-Image-7B",
        num_gpus=args.tp * args.ulysses * args.ring,
        tp_size=args.tp,
        ulysses_degree=args.ulysses,
        ring_degree=args.ring,
        pipeline_config=OvisImagePipelineConfig(dit_config=config),
        attention_backend="fa" if args.ring > 1 else "torch_sdpa",
        performance_mode="manual",
    )
    set_global_server_args(server)
    maybe_init_distributed_environment_and_model_parallel(
        tp_size=args.tp,
        sp_size=args.ulysses * args.ring,
        ulysses_degree=args.ulysses,
        ring_degree=args.ring,
    )
    from sglang.srt.runtime_context import publish
    from sglang.srt.server_args import ServerArgs as SrtServerArgs

    publish(
        SrtServerArgs(model_path="dummy", tp_size=args.tp), role="diffusion_gpu_worker"
    )
    report = []
    dtypes = (torch.bfloat16,) if args.ring > 1 else (torch.float32, torch.bfloat16)
    cases = [(1, 5, 15), (4, 7, 25)]
    if args.edge_cases:
        cases.extend([(1, 1, 35), (4, 1, 35), (1, 1, 1), (4, 1, 1)])
    for dtype in dtypes:
        for batch_size, text_len, image_len in cases:
            torch.manual_seed(42)
            reference = ReferenceDiT(**kwargs).cuda().to(dtype).eval()
            # Attention selects its backend during construction, as in the
            # production loader. Casting a FP32 module afterward would already
            # have resolved FA to SDPA, which cannot supply Ring's softmax LSE.
            with set_default_torch_dtype(dtype):
                model = OvisImageTransformer2DModel(config, kwargs).cuda().eval()
            model.load_weights(reference.state_dict().items())
            image = torch.randn(batch_size, image_len, 64, device="cuda", dtype=dtype)
            text = torch.randn(batch_size, text_len, 16, device="cuda", dtype=dtype)
            text[:, -2:] = 0
            t = torch.full((batch_size,), 1000.0, device="cuda")
            txt_ids = torch.zeros(text_len, 3, device="cuda")
            txt_ids[:, 1:] = torch.arange(text_len, device="cuda")[:, None]
            img_ids = torch.zeros(image_len, 3, device="cuda")
            img_ids[:, 1] = torch.arange(image_len, device="cuda") // 5
            img_ids[:, 2] = torch.arange(image_len, device="cuda") % 5
            rope = model.rotary_emb(torch.cat([txt_ids, img_ids]))
            with set_forward_context(current_timestep=0, attn_metadata=None):
                actual = model(image, text, t, rope)
            expected = reference(image, text, t / 1000, img_ids, txt_ids).sample
            atol, rtol = (1e-4, 1e-4) if dtype == torch.float32 else (0.05, 0.02)
            torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
            report.append(
                {
                    "dtype": str(dtype),
                    "batch": batch_size,
                    "text_length": text_len,
                    "image_length": image_len,
                    "attention_backend": model.transformer_blocks[
                        0
                    ].attn.attn.backend.name,
                    "max_abs": (actual.float() - expected.float()).abs().max().item(),
                }
            )
    if torch.distributed.get_rank() == 0:
        print(json.dumps(report, indent=2))
    cleanup_dist_env_and_memory()


if __name__ == "__main__":
    main()
