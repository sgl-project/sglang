# srt/models/gigarembed.py

import os, json
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
import logging

from sglang.srt.layers.pooler import EmbeddingPoolerOutput
from sglang.srt.layers.quantization.base_config import QuantizationConfig

logger = logging.getLogger(__name__)

def _first_not_none(*vals):
    for v in vals:
        if v is not None:
            return v
    return None


class GigarEmbedModel(nn.Module):
    def __init__(self, config: AutoConfig, quant_config: Optional[QuantizationConfig]=None, prefix: str=""):
        super().__init__()
        self.config = config
        self.quant_config = quant_config

        # Всегда локальный путь из --model-path
        self.model_path = prefix or getattr(config, "name_or_path", "")
        if (not self.model_path) or str(self.model_path).startswith(("ai-sage/","https://","hf://","hub://")):
            raise RuntimeError(f"Ожидался локальный путь (--model-path), получено: {self.model_path!r}")

        # Надёжно определяем pad_id: text_config.pad_token_id -> text_config.eos_token_id -> top -> 0
        text_cfg = getattr(config, "text_config", None)
        pad_from_text = getattr(text_cfg, "pad_token_id", None) if text_cfg is not None else None
        eos_from_text = getattr(text_cfg, "eos_token_id", None) if text_cfg is not None else None
        pad_top = getattr(config, "pad_token_id", None)

        pad_id = _first_not_none(pad_from_text, eos_from_text, pad_top)
        if pad_id is None:
            # оффлайн-фолбэк: попробуем прочитать tokenizer_config/special_tokens_map
            for fn in ("tokenizer_config.json", "special_tokens_map.json"):
                p = os.path.join(self.model_path, fn)
                if os.path.exists(p):
                    try:
                        with open(p, "r", encoding="utf-8") as f:
                            tcfg = json.load(f)
                        pad_id = pad_id or tcfg.get("pad_token_id")
                        if pad_id is None and tcfg.get("eos_token_id") is not None:
                            pad_id = tcfg["eos_token_id"]
                    except Exception:
                        pass
        if pad_id is None:
            pad_id = 0
        self.pad_id = int(pad_id)

        self.hf_model: Optional[nn.Module] = None
        self._weights_loaded = False

    # Лоадер SGLang зовёт это
    def load_weights(self, weights):
        if self._weights_loaded:
            return self

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        state_dict = None

        if isinstance(weights, dict):
            state_dict = weights
        else:
            get_sd = getattr(weights, "get_state_dict", None)
            if callable(get_sd):
                try:
                    state_dict = get_sd()
                except Exception:
                    state_dict = None

        if state_dict is not None:
            self.hf_model = AutoModel.from_config(self.config, trust_remote_code=True)
            self.hf_model.load_state_dict(state_dict, strict=False)
            self.hf_model.to(dtype=dtype).eval()
        else:
            # ВАЖНО: не просим flash_attention_2 — иначе Transformers будет искать пакет flash_attn
            self.hf_model = AutoModel.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                torch_dtype=dtype,
                local_files_only=True,
                # НЕ указываем attn_implementation, оставляем SDPA по умолчанию
                # attn_implementation="sdpa",  # можно явно так, если хотите
            ).eval()
            self.hf_model.cuda()

        # (опционально) насильно для пула: пусть возвращает 2D маску-путь, если нужно
        try:
            if hasattr(self.hf_model, "pooler") and hasattr(self.hf_model.pooler, "config"):
                # Можно оставить как есть; при SDPA тут не требуется FA2
                pass
        except Exception:
            pass

        self._weights_loaded = True
        return self

    @torch.no_grad()
    def forward(self, input_ids, positions, forward_batch,
                pp_proxy_tensors=None, input_embeds=None, get_embedding=False):

        if self.hf_model is None:
            raise RuntimeError("forward called before load_weights()")

        if get_embedding:
            # Check if we have sequence length information
            if hasattr(forward_batch, 'extend_seq_lens') and forward_batch.extend_seq_lens is not None:
                # Get sequence lengths
                seq_lens = forward_batch.extend_seq_lens.tolist()

                # Convert the combined tensor into a batch of tensors
                # Split input_ids into individual sequences
                text_tensors = []
                start_idx = 0

                for seq_len in seq_lens:
                    if seq_len > 0:
                        # Extract tokens for the current sequence
                        text_tokens = input_ids[start_idx:start_idx + seq_len]
                        text_tensors.append(text_tokens)
                    start_idx += seq_len

                # Add padding so all sequences have the same length
                # and create a batch tensor
                if text_tensors:
                    # Find the maximum length
                    max_len = max(len(t) for t in text_tensors)

                    # Add padding
                    padded_tensors = []
                    attention_masks = []

                    for text_tensor in text_tensors:
                        # Padding
                        padded = torch.cat([
                            text_tensor,
                            torch.full((max_len - len(text_tensor),), self.pad_id,
                                     dtype=text_tensor.dtype, device=text_tensor.device)
                        ])
                        padded_tensors.append(padded)

                        # Attention mask
                        mask = torch.cat([
                            torch.ones(len(text_tensor), dtype=torch.bool, device=text_tensor.device),
                            torch.zeros(max_len - len(text_tensor), dtype=torch.bool, device=text_tensor.device)
                        ])
                        attention_masks.append(mask)

                    # Create batch
                    batch_input_ids = torch.stack(padded_tensors, dim=0)
                    batch_attention_mask = torch.stack(attention_masks, dim=0)

                    # Pass batch to model
                    out = self.hf_model(
                        input_ids=batch_input_ids,
                        attention_mask=batch_attention_mask,
                        pool_mask=batch_attention_mask,
                        return_embeddings=True,
                        return_dict=True,
                    )

                    # Extract embeddings
                    emb = out.get("sentence_embeddings") if isinstance(out, dict) else out
                    if emb is None:
                        emb = out.get("embeddings") if isinstance(out, dict) else None
                    if emb is None:
                        raise RuntimeError("HF model did not return embeddings")

                    final_embeddings = emb
            else:
                # Backward compatibility: process as a single request
                if input_ids.dim() == 1:
                    input_ids = input_ids.unsqueeze(0)

                device = input_ids.device
                attn_mask = (input_ids != self.pad_id).to(device=device)

                if attn_mask.dim() == 1:
                    attn_mask = attn_mask.unsqueeze(0)
                attn_mask = attn_mask.to(device=device)

                out = self.hf_model(
                    input_ids=input_ids,
                    attention_mask=attn_mask,
                    pool_mask=attn_mask,
                    return_embeddings=True,
                    return_dict=True,
                )
                emb = out.get("sentence_embeddings") if isinstance(out, dict) else out
                if emb is None:
                    emb = out.get("embeddings") if isinstance(out, dict) else None
                if emb is None:
                    raise RuntimeError("HF model did not return embeddings")

                final_embeddings = emb.float()

            # Ensure we return a tensor with the correct shape [batch_size, embedding_dim]
            if final_embeddings.dim() == 1:
                final_embeddings = final_embeddings.unsqueeze(0)

            logger.info(f"{final_embeddings.shape=}")
            return EmbeddingPoolerOutput(embeddings=final_embeddings)

        # For non-embedding mode, return a dummy tensor
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        
        device = input_ids.device
        vocab_size = int(getattr(getattr(self.config, "text_config", self.config), "vocab_size", 128256))
        dummy = torch.zeros((input_ids.size(0), vocab_size), dtype=torch.float32, device=device)
        return EmbeddingPoolerOutput(embeddings=dummy)

    def forward_batch(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


EntryClass = [GigarEmbedModel]
