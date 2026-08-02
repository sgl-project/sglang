# SPDX-License-Identifier: Apache-2.0
# VAE distribution and aggregation helpers for the MiniMax H3 visual VAE.
import torch


class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, upcast_fp32=True):
        if upcast_fp32:
            parameters = parameters.to(dtype=torch.float32)

        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.std = self.logvar.mul(0.5).exp_()

    @torch.compiler.disable
    def sample(self, generator=None):
        noise = torch.randn(self.mean.shape, generator=generator)
        noise = noise.to(device=self.parameters.device)
        return noise.mul_(self.std).add_(self.mean)


class ClsTokenAggregator:
    def __init__(self, vae_model):
        self.vae = vae_model
        self.cls_tokens = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.cls_tokens and hasattr(self.vae.encoder, "loss_info"):
            self.vae.encoder.loss_info["cls_token"] = torch.stack(
                self.cls_tokens, dim=0
            ).mean(dim=0)
        return False

    def collect(self):
        if (
            hasattr(self.vae.encoder, "loss_info")
            and "cls_token" in self.vae.encoder.loss_info
        ):
            self.cls_tokens.append(self.vae.encoder.loss_info["cls_token"].clone())

    def collect_stacked(self, num_tiles, sample_batch_size):
        if (
            hasattr(self.vae.encoder, "loss_info")
            and "cls_token" in self.vae.encoder.loss_info
        ):
            cls_token = self.vae.encoder.loss_info["cls_token"]
            cls_token = cls_token.unflatten(0, (num_tiles, sample_batch_size))
            self.cls_tokens.extend(token.clone() for token in cls_token)
