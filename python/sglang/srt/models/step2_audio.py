import logging
from typing import Iterable, List, Literal, Optional, Tuple, Union, overload

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, nn

from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
from sglang.srt.managers.mm_utils import (
    MultiModalityDataPaddingPatternMultimodalTokens,
    general_mm_embed_routine,
)
from sglang.srt.managers.schedule_batch import MultimodalDataItem, MultimodalInputs
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.qwen2 import Qwen2Model
from sglang.srt.utils import add_prefix

logger = logging.getLogger(__name__)

step2audioConfig = None
StepAudio2EncoderConfig = None


@overload
def flatten_bn(x: list[torch.Tensor]) -> list[torch.Tensor]: ...


@overload
def flatten_bn(
    x: Union[list[torch.Tensor], torch.Tensor],
    *,
    concat: Literal[True],
) -> torch.Tensor: ...


@overload
def flatten_bn(
    x: Union[list[torch.Tensor], torch.Tensor],
    *,
    concat: bool = False,
) -> Union[list[torch.Tensor], torch.Tensor]: ...


def flatten_bn(
    x,
    *,
    concat: bool = False,
):
    """
    Flatten the ``B`` and ``N`` dimensions of batched multimodal inputs.

    The input tensor should have shape ``(B, N, ...)```.
    """
    if isinstance(x, torch.Tensor):
        return x.flatten(0, 1)

    if concat:
        return torch.cat(x)

    return [x_n for x_b in x for x_n in x_b]


def make_non_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    """Make mask tensor containing indices of non-padded part.
    The sequences in a batch may have different lengths. To enable
    batch computing, padding is need to make all sequence in same
    size. To avoid the padding part pass value to context dependent
    block such as attention or convolution , this padding part is
    masked.
    1 for non-padded part and 0 for padded part.
    Parameters
    ----------
        lengths (torch.Tensor): Batch of lengths (B,).
    Returns:
    -------
        torch.Tensor: Mask tensor containing indices of padded part (B, max_T).
    Examples:
        >>> import torch
        >>> import s3tokenizer
        >>> lengths = torch.tensor([5, 3, 2])
        >>> masks = s3tokenizer.make_non_pad_mask(lengths)
        masks = [[1, 1, 1, 1, 1],
                 [1, 1, 1, 0, 0],
                 [1, 1, 0, 0, 0]]
    """
    batch_size = lengths.size(0)
    max_len = max_len if max_len > 0 else lengths.max().item()
    seq_range = torch.arange(0, max_len, dtype=torch.int64, device=lengths.device)
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_length_expand = lengths.unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand
    return ~mask


def mask_to_bias(mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Convert bool-tensor to float-tensor for flash attention.
    Parameters
    ----------
        lengths (torch.Tensor): Batch of lengths (B, ?).
    Returns:
    -------
        torch.Tensor: Mask tensor containing indices of padded part (B, ?).
    Examples:
        >>> import torch
        >>> import s3tokenizer
        >>> lengths = torch.tensor([5, 3, 2])
        >>> masks = s3tokenizer.make_non_pad_mask(lengths)
        masks = [[1, 1, 1, 1, 1],
                 [1, 1, 1, 0, 0],
                 [1, 1, 0, 0, 0]]
        >>> new_masks = s3tokenizer.mask_to_bias(masks, torch.float32)
        new_masks = [[-0.0000e+00, -0.0000e+00, -0.0000e+00, -0.0000e+00, -0.0000e+00],
                    [-0.0000e+00, -0.0000e+00, -0.0000e+00, -1.0000e+10, -1.0000e+10],
                    [-0.0000e+00, -0.0000e+00, -1.0000e+10, -1.0000e+10, -1.0000e+10]]
    """
    assert mask.dtype == torch.bool
    assert dtype in [torch.float32, torch.bfloat16, torch.float16]
    mask = mask.to(dtype)
    # attention mask bias
    # NOTE(Mddct): torch.finfo jit issues
    #     chunk_masks = (1.0 - chunk_masks) * torch.finfo(dtype).min
    mask = (1.0 - mask) * -1.0e10
    return mask


class LayerNorm(nn.LayerNorm):
    def forward(self, input: Tensor) -> Tensor:
        return super().forward(input).type(input.dtype)


class Linear(nn.Linear):
    def forward(self, input: Tensor) -> Tensor:
        return F.linear(
            input,
            self.weight.to(input.dtype),
            None if self.bias is None else self.bias.to(input.dtype),
        )


class Conv1d(nn.Conv1d):
    def _conv_forward(
        self, input: Tensor, weight: Tensor, bias: Optional[Tensor]
    ) -> Tensor:
        return super()._conv_forward(
            input,
            weight.to(input.dtype),
            None if bias is None else bias.to(input.dtype),
        )


class MultiHeadAttention(nn.Module):
    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.query = Linear(n_state, n_state)
        self.key = Linear(n_state, n_state, bias=False)
        self.value = Linear(n_state, n_state)
        self.out = Linear(n_state, n_state)

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
    ):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        wv, qk = self.qkv_attention(q, k, v, mask)
        return self.out(wv), qk

    def qkv_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        _, T, D = q.shape
        scale = (D // self.n_head) ** -0.25
        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3) * scale
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 3, 1) * scale
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

        qk = q @ k  # (B, n_head, T, T)
        if mask is not None:
            qk = qk + mask
        qk = qk.float()

        w = F.softmax(qk, dim=-1).to(q.dtype)
        return (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2), qk.detach()


class ResidualAttentionBlock(nn.Module):
    def __init__(self, n_state: int, n_head: int):
        super().__init__()

        self.attn = MultiHeadAttention(n_state, n_head)
        self.attn_ln = LayerNorm(n_state)

        n_mlp = n_state * 4
        self.mlp = nn.Sequential(
            Linear(n_state, n_mlp), nn.GELU(), Linear(n_mlp, n_state)
        )
        self.mlp_ln = LayerNorm(n_state)

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
    ):
        x = x + self.attn(self.attn_ln(x.contiguous()), mask=mask)[0]
        x = x + self.mlp(self.mlp_ln(x.contiguous()))
        return x


class StepAudio2Encoder(nn.Module):
    def __init__(
        self,
        config: StepAudio2EncoderConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        # TODO: use the attention implementation of sglang
        super().__init__()
        self.conv1 = Conv1d(
            config.n_mels, config.n_audio_state, kernel_size=3, padding=1
        )
        self.conv2 = Conv1d(
            config.n_audio_state,
            config.n_audio_state,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.positional_embedding = nn.Embedding(
            config.n_audio_ctx, config.n_audio_state
        )
        self.positional_embedding.requires_grad_(False)
        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [
                ResidualAttentionBlock(config.n_audio_state, config.n_audio_head)
                for _ in range(config.n_audio_layer)
            ]
        )
        self.avg_pooler = nn.AvgPool1d(2, stride=2)
        self.after_norm = LayerNorm(config.n_audio_state)

    def forward(self, x: Tensor, x_len: Tensor) -> Tuple[Tensor, Tensor]:
        T = x.size(-1)
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)  # (B, T // 2, n_state)
        mask = make_non_pad_mask(x_len, T).unsqueeze(1)  # (B, 1, T)
        mask = mask_to_bias(mask[:, :, (T + 1) % 2 :: 2], x.dtype)  # (B, 1, T // 2)
        x = (x + self.positional_embedding.weight[: x.shape[1], :]).to(x.dtype)
        for block in self.blocks:
            x = block(x, mask.unsqueeze(1))
        x = x.permute(0, 2, 1)
        x = self.avg_pooler(x)
        x = x.permute(0, 2, 1)
        x_len = (x_len + 1) // 2 // 2
        x = self.after_norm(x.contiguous())
        return x, x_len


class Step2Adaptor(nn.Module):
    def __init__(
        self,
        config: StepAudio2EncoderConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.stride = config.adapter_stride
        if self.stride != -1:
            self.conv = Conv1d(
                config.n_audio_state,
                config.n_audio_state,
                config.kernel_size,
                config.adapter_stride,
                padding=1,
            )
        self.linear1 = nn.Linear(config.n_audio_state, 2048)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(2048, config.llm_dim)
        self.gradient_checkpointing = False

    def forward(self, x: Tensor) -> Tuple[Tensor]:
        T = x.size(-1)
        if self.stride != -1:
            if self.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(self.conv, x.permute(0, 2, 1))
                x = x.permute(0, 2, 1)
            else:
                x = x.permute(0, 2, 1)
                x = F.gelu(self.conv(x))
                x = x.permute(0, 2, 1)

        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


class StepAudio2ForCausalLM(nn.Module):
    def __init__(
        self,
        config: step2audioConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.model = Qwen2Model(config.text_config)
        self.quant_config = quant_config
        if isinstance(config.torch_dtype, str):
            dtype = getattr(torch, config.torch_dtype)
        else:
            dtype = config.torch_dtype
        # TODO: modify this
        self.bf16 = True  # dtype==torch.bfloat16
        self.encoder = StepAudio2Encoder(
            config.audio_encoder_config,
            quant_config,
            prefix=add_prefix("encoder", prefix),
        )
        self.adapter = Step2Adaptor(
            config.audio_encoder_config,
            quant_config,
            prefix=add_prefix("adapter", prefix),
        )
        if self.bf16:
            self.encoder = self.encoder.bfloat16()
            self.adapter = self.adapter.bfloat16()

        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("lm_head", prefix),
        )

        self.logits_processor = LogitsProcessor(config)

        self.pattern = MultiModalityDataPaddingPatternMultimodalTokens()

    def pad_input_ids(self, input_ids: List[int], mm_inputs: MultimodalInputs):
        return self.pattern.pad_input_tokens(input_ids, mm_inputs)

    def _parse_and_validate_audio_input(self, audio_mels=None, audio_lens=None):

        if audio_mels is None:
            return None

        audio_mels_lst = []
        cur_idx = 0
        for audio_len in audio_lens:
            audio_mels_lst.append(audio_mels[cur_idx : cur_idx + audio_len])
            cur_idx += audio_len

        max_len = max(x.size(0) for x in audio_mels_lst)
        audio_mels = torch.stack(
            [F.pad(x, (0, 0, 0, max_len - x.size(0))) for x in audio_mels_lst], dim=0
        )

        return {
            "audio_mels": audio_mels,
            "audio_lens": audio_lens,
        }

    def get_audio_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        # TODO: convert items to wavs(the input encoder needs)
        wavs = items[0].feature
        wav_lens = torch.tensor(items[0].audio_feature_lens, device=wavs.device)
        ret = self._parse_and_validate_audio_input(audio_mels=wavs, audio_lens=wav_lens)
        wavs = ret["audio_mels"]
        wav_lens = ret["audio_lens"]
        wavs = wavs.permute(0, 2, 1)
        wavs = wavs.to(torch.bfloat16)
        out, feat_lens = self.encoder(wavs, wav_lens)
        out = self.adapter(out)
        audio_feature_lens = (feat_lens - 1) // 2 + 1

        audio_feature_list = [
            out[i, : audio_feature_lens[i]] for i in range(out.size(0))
        ]
        return torch.cat(audio_feature_list, dim=0)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def forward(
        self,
        input_ids,
        positions,
        forward_batch: ForwardBatch,
    ):
        hidden_states = general_mm_embed_routine(
            input_ids=input_ids,
            forward_batch=forward_batch,
            language_model=self.model,
            multimodal_model=self,
            positions=positions,
        )

        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]

        params_dict = dict(self.named_parameters())
        loaded_params = set()
        for name, loaded_weight in weights:
            # name = self.maybe_remap_params(name)
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(name)
                break
            else:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(name)


EntryClass = StepAudio2ForCausalLM
