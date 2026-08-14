# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""EarTTS stage for NVIDIA NemotronLabs VoiceChat online inference.

Adapted from NVIDIA NeMo's vLLM-Omni EarTTS implementation. The transformer
backbone runs eagerly because per-turn codec inputs change independently of the
placeholder token used for KV state. The fixed-shape MaskGIT sampler is compiled
separately on CUDA and covered by the server's startup warm-up.
"""

from collections.abc import Iterable

import torch
from torch import nn
from transformers.generation.logits_process import TopKLogitsWarper, TopPLogitsWarper

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.gemma3_causal import Gemma3ForCausalLM

TEXT_PAD_TOKEN_ID = 12
EOS_TOKEN_ID = 2


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        out = x.float() * torch.rsqrt(
            x.float().pow(2).mean(-1, keepdim=True) + self.eps
        )
        return (out * (1.0 + self.weight.float())).type_as(x)


class MLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(
            nn.functional.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x)
        )


class MLPLayer(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, eps: float):
        super().__init__()
        self.pre_norm = RMSNorm(hidden_size, eps)
        self.mlp = MLP(hidden_size, intermediate_size)
        self.post_norm = RMSNorm(hidden_size, eps)

    def forward(self, x):
        return x + self.post_norm(self.mlp(self.pre_norm(x)))


class GatedProjectedSumRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, num_codebooks: int):
        super().__init__()
        self.num_codebooks = num_codebooks
        self.audio_proj = nn.Linear(hidden_size, hidden_size)
        self.text_proj = nn.Linear(hidden_size, hidden_size)
        self.gate = nn.Parameter(torch.zeros(hidden_size), requires_grad=False)
        self.residual_scale = nn.Parameter(torch.tensor(0.5), requires_grad=False)
        self.final_norm = RMSNorm(hidden_size)

    def forward(self, audio_emb, text_emb):
        audio_h = self.audio_proj(audio_emb / self.num_codebooks)
        text_h = self.text_proj(text_emb)
        dtype = audio_h.dtype
        mixed = torch.sigmoid(self.gate).to(dtype) * audio_h
        mixed += (1 - torch.sigmoid(self.gate)).to(dtype) * text_h
        mixed *= torch.sigmoid(self.residual_scale).to(dtype)
        return self.final_norm(mixed.float()).to(dtype)


class EarTTSInputEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.rvq_embs = nn.ModuleList(
            [
                nn.Embedding(config.codebook_size + 1, config.latent_size)
                for _ in range(config.num_quantizers)
            ]
        )
        self.embed_code = nn.Linear(config.latent_size, config.hidden_size, bias=False)
        self.embed_subword = nn.Embedding(config.emb_vocab_size, config.hidden_size)
        self.bos_emb = nn.Parameter(torch.empty(config.hidden_size))
        self.use_gated_fusion_for_text_audio = config.use_gated_fusion_for_text_audio
        self.use_audio_prompt_frozen_projection = (
            config.use_audio_prompt_frozen_projection
        )
        if self.use_gated_fusion_for_text_audio:
            self.gated_fusion_audio_text = GatedProjectedSumRMSNorm(
                config.hidden_size, config.num_quantizers
            )
        if self.use_audio_prompt_frozen_projection:
            self.audio_prompt_projection_W = nn.Parameter(
                torch.empty(config.hidden_size, config.hidden_size), requires_grad=False
            )

    def forward(self, acoustic, text, text_mask, bos_mask, speaker_latent):
        audio = sum(emb(acoustic[:, i]) for i, emb in enumerate(self.rvq_embs))
        audio = self.embed_code(audio)
        if self.use_audio_prompt_frozen_projection:
            provided = speaker_latent.abs().sum(-1, keepdim=True) > 0
            replace = (bos_mask.unsqueeze(-1) == 0) & provided
            audio = torch.where(replace, speaker_latent, audio)
        audio = audio + bos_mask.unsqueeze(-1) * self.bos_emb
        text_emb = self.embed_subword(text) * text_mask.unsqueeze(-1)
        if self.use_gated_fusion_for_text_audio:
            return self.gated_fusion_audio_text(audio, text_emb)
        return audio + text_emb


def _gumbel_like(x, eps=1e-8):
    u = torch.rand_like(x)
    return -torch.log(-torch.log(u + eps) + eps)


def _batch_matmul(x, w, indices):
    return torch.bmm(w[indices], x.unsqueeze(2)).squeeze(2)


class MoGHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        h, n, rank = config.hidden_size, config.mog_num_predictions, config.mog_low_rank
        self.out_size, self.num_predictions, self.low_rank = config.latent_size, n, rank
        self.min_log_std = config.mog_min_log_std
        self.logits_warper = (
            TopPLogitsWarper(config.top_p_or_k)
            if isinstance(config.top_p_or_k, float)
            else TopKLogitsWarper(config.top_p_or_k)
        )
        self.mlp_stack = nn.Sequential(
            *[
                MLPLayer(h, config.intermediate_size, config.mog_eps)
                for _ in range(config.mog_num_layers)
            ],
            RMSNorm(h, config.mog_eps),
        )
        self.proj_logits = nn.Linear(h, n, bias=False)
        self.proj_mus = nn.Linear(h, n * rank, bias=False)
        self.proj_logs = nn.Linear(h, 1, bias=False)
        self.proj_else = nn.Linear(h, config.latent_size, bias=False)
        self.low_mat = nn.Parameter(torch.empty(n, config.latent_size, rank))

    def forward(self, x):
        x = self.mlp_stack(x)
        logits = self.logits_warper(None, self.proj_logits(x))
        indices = (nn.functional.log_softmax(logits, -1) + _gumbel_like(logits)).argmax(
            -1
        )
        mus = _batch_matmul(
            x,
            self.proj_mus.weight.view(self.num_predictions, self.low_rank, -1),
            indices,
        )
        mus = _batch_matmul(mus, self.low_mat, indices)
        logs = self.proj_logs(x).clamp_min(self.min_log_std)
        return mus * torch.exp(logs) + self.proj_else(x), logs


class MaskGITSampler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_quantizers = config.num_quantizers
        self.codebook_size = config.codebook_size
        self.noise_scale = config.noise_scale
        rates = torch.linspace(0.0, 1.0, config.num_iter + 1)[:-1]
        masks = torch.ceil(
            (1 - rates.pow(config.exponent)).pow(1 / config.exponent)
            * config.num_quantizers
        ).to(torch.int64)
        shifted = torch.cat((masks[1:], torch.zeros(1, dtype=torch.int64)))
        self.num_to_sample = [int(v) for v in (masks - shifted) if v]
        self.rvq_embs = nn.Parameter(
            torch.empty(config.num_quantizers, config.codebook_size, config.latent_size)
        )
        self.embed_code = nn.Linear(config.latent_size, config.hidden_size, bias=False)
        self.mog_head = MoGHead(config)
        self._compiled_forward = torch.compile(
            self._forward_impl, fullgraph=True, mode="reduce-overhead"
        )

    def _depthsum(self, codes):
        tables = nn.functional.pad(self.rvq_embs, [0, 0, 0, 1])
        return sum(
            nn.functional.embedding(codes[i], tables[i]) for i in range(len(tables))
        )

    def _forward_impl(self, hidden):
        codes = torch.full(
            (self.num_quantizers, hidden.shape[0]),
            self.codebook_size,
            dtype=torch.long,
            device=hidden.device,
        )
        depth = 0
        for count in self.num_to_sample:
            mu, logs = self.mog_head(self.embed_code(self._depthsum(codes)) + hidden)
            residual = mu + torch.exp(logs) * torch.randn_like(mu) * self.noise_scale
            for i in range(depth, depth + count):
                selected = (
                    self.rvq_embs[i].pow(2).sum(-1) - 2 * residual @ self.rvq_embs[i].T
                ).argmin(-1)
                residual = residual - nn.functional.embedding(
                    selected, self.rvq_embs[i]
                )
                codes[i] = selected
            depth += count
        return codes.T

    def forward(self, hidden):
        # The production stream always samples one fixed-shape CUDA frame. Keep
        # CPU tests eager, while allowing the startup warm-up to compile and
        # capture this launch-heavy loop before the first client arrives.
        if hidden.is_cuda:
            return self._compiled_forward(hidden)
        return self._forward_impl(hidden)


class EarTTSForCausalLM(nn.Module):
    """Streaming EarTTS stage. Requires eager execution and `custom_inputs`."""

    def __init__(self, *, config, quant_config=None, prefix=""):
        super().__init__()
        self.config = config
        self.total_emb = EarTTSInputEmbedding(config)
        self.backbone = Gemma3ForCausalLM(config, quant_config, prefix="backbone")
        self.sampler = MaskGITSampler(config)
        self.register_buffer(
            "sil_tokens", torch.zeros(config.num_quantizers, dtype=torch.int32)
        )

    def get_attention_sliding_window_size(self):
        return self.backbone.get_attention_sliding_window_size()

    def _prepare_inputs(self, input_ids, forward_batch):
        items = forward_batch.custom_inputs
        if items is None or any(item is None for item in items):
            raise ValueError("EarTTS requires custom_inputs for every request.")
        lengths = forward_batch.extend_seq_lens_cpu or [1] * forward_batch.batch_size
        acoustic, text, text_mask, bos_mask, latent = [], [], [], [], []
        decode_rows, offset = [], 0
        device = input_ids.device
        dtype = self.total_emb.bos_emb.dtype
        for request_index, (item, length) in enumerate(
            zip(items, lengths, strict=True)
        ):
            if item.get("is_speaker_prefill", False):
                speaker = torch.as_tensor(
                    item["speaker_latent"], device=device, dtype=dtype
                )
                if speaker.shape != (length, self.config.hidden_size):
                    raise ValueError(
                        "speaker_latent must match the prefill token span."
                    )
                ac = self.sil_tokens.to(device=device, dtype=torch.long).expand(
                    length, -1
                )
                tx = torch.full(
                    (length,), TEXT_PAD_TOKEN_ID, device=device, dtype=torch.long
                )
                tx[-1] = EOS_TOKEN_ID
                tm = torch.zeros(length, device=device, dtype=torch.long)
                tm[max(0, length - 2) :] = 1
                bm = torch.zeros(length, device=device, dtype=torch.long)
                bm[-1] = 1
            else:
                token = int(item["text_token"])
                previous = item.get("previous_audio_codes")
                if token == EOS_TOKEN_ID:
                    ac = self.sil_tokens.to(device=device, dtype=torch.long).reshape(
                        1, -1
                    )
                elif previous is None:
                    ac = torch.full(
                        (1, self.config.num_quantizers),
                        self.config.codebook_size,
                        device=device,
                        dtype=torch.long,
                    )
                else:
                    ac = torch.as_tensor(
                        previous, device=device, dtype=torch.long
                    ).reshape(1, -1)
                tx = torch.tensor([token], device=device)
                tm = torch.ones(1, device=device, dtype=torch.long)
                bm = torch.zeros(1, device=device, dtype=torch.long)
                speaker = torch.zeros(
                    1, self.config.hidden_size, device=device, dtype=dtype
                )
                if length != 1:
                    raise ValueError("EarTTS decode turns must contain one token.")
                decode_rows.append((request_index, offset))
            acoustic.append(ac)
            text.append(tx)
            text_mask.append(tm)
            bos_mask.append(bm)
            latent.append(speaker)
            offset += length
        return (
            tuple(map(torch.cat, (acoustic, text, text_mask, bos_mask, latent))),
            decode_rows,
        )

    @torch.no_grad()
    def forward(self, input_ids, positions, forward_batch: ForwardBatch, **_):
        prepared, decode_rows = self._prepare_inputs(input_ids, forward_batch)
        embeddings = self.total_emb(*prepared)
        hidden = self.backbone.model(input_ids, positions, forward_batch, embeddings)
        batch = forward_batch.batch_size
        logits = torch.full((batch, 2), -torch.inf, device=input_ids.device)
        logits[:, 0] = 0
        output = LogitsProcessorOutput(next_token_logits=logits)
        frames = [[0] * self.config.num_quantizers for _ in range(batch)]
        if decode_rows:
            sampled = self.sampler(hidden[[row for _, row in decode_rows]]).tolist()
            for (request_index, _), codes in zip(decode_rows, sampled, strict=True):
                frames[request_index] = codes
        output.customized_info = {"audio_codes": frames}
        return output

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        params = dict(self.named_parameters())
        buffers = dict(self.named_buffers())
        loaded_params = set()
        for name, weight in weights:
            if name.startswith("model.backbone."):
                suffix = name[len("model.backbone.") :]
                if suffix == "embed_tokens.weight":
                    continue
                backbone_params = self.backbone.load_weights(
                    [(f"model.{suffix}", weight)]
                )
                loaded_params.update(
                    f"backbone.{param_name}" for param_name in backbone_params
                )
                continue
            name = name.replace("model.total_emb.", "total_emb.", 1)
            name = name.replace("model.sampler.", "sampler.", 1)
            name = name.replace("sampler_module.sampler.", "sampler.", 1)
            name = name.replace("model.sil_tokens", "sil_tokens", 1)
            name = name.replace(
                "total_emb.embed_subword.embed_subwords.",
                "total_emb.embed_subword.",
                1,
            )
            target = params.get(name, buffers.get(name))
            if target is not None:
                loader = getattr(target, "weight_loader", default_weight_loader)
                loader(target, weight)
                loaded_params.add(name)

        # EarTTS never reads the backbone token embedding because every call
        # supplies fused audio/text embeddings. All other parameters, plus the
        # codec silence-token buffer, must come from the converted checkpoint.
        required = set(params)
        required.discard("backbone.model.embed_tokens.weight")
        required.add("sil_tokens")
        missing = required - loaded_params
        if missing:
            raise RuntimeError(
                "Some EarTTS weights are not initialized from the checkpoint: "
                f"{sorted(missing)}"
            )
        return loaded_params


EntryClass = EarTTSForCausalLM
