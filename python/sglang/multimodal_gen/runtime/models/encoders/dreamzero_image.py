# SPDX-License-Identifier: Apache-2.0

"""
Concise re-implementation of
``https://github.com/openai/CLIP'' and
``https://github.com/mlfoundations/open_clip''.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T


def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_heads: int,
    compatibility_mode: bool = False,
) -> torch.Tensor:
    del compatibility_mode
    batch, seq_len, channels = q.shape
    head_dim = channels // num_heads
    q = q.reshape(batch, seq_len, num_heads, head_dim).transpose(1, 2)
    k = k.reshape(batch, -1, num_heads, head_dim).transpose(1, 2)
    v = v.reshape(batch, -1, num_heads, head_dim).transpose(1, 2)
    out = F.scaled_dot_product_attention(q, k, v)
    return out.transpose(1, 2).reshape(batch, seq_len, channels)


class SelfAttention(nn.Module):

    def __init__(self, dim, num_heads, dropout=0.1, eps=1e-5):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.eps = eps

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        """
        x:   [B, L, C].
        """
        b, s, c, n, d = *x.size(), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.q(x).reshape(b, s, n, d).permute(0, 2, 1, 3)
        k = self.k(x).reshape(b, s, n, d).permute(0, 2, 1, 3)
        v = self.v(x).reshape(b, s, n, d).permute(0, 2, 1, 3)

        # compute attention
        p = self.dropout.p if self.training else 0.0
        x = F.scaled_dot_product_attention(q, k, v, mask, p)
        x = x.permute(0, 2, 1, 3).reshape(b, s, c)

        # output
        x = self.o(x)
        x = self.dropout(x)
        return x


class AttentionBlock(nn.Module):

    def __init__(self, dim, num_heads, post_norm, dropout=0.1, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.post_norm = post_norm
        self.eps = eps

        # layers
        self.attn = SelfAttention(dim, num_heads, dropout, eps)
        self.norm1 = nn.LayerNorm(dim, eps=eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim),
            nn.Dropout(dropout))
        self.norm2 = nn.LayerNorm(dim, eps=eps)

    def forward(self, x, mask):
        if self.post_norm:
            x = self.norm1(x + self.attn(x, mask))
            x = self.norm2(x + self.ffn(x))
        else:
            x = x + self.attn(self.norm1(x), mask)
            x = x + self.ffn(self.norm2(x))
        return x


class XLMRoberta(nn.Module):
    """
    XLMRobertaModel with no pooler and no LM head.
    """

    def __init__(self,
                 vocab_size=250002,
                 max_seq_len=514,
                 type_size=1,
                 pad_id=1,
                 dim=1024,
                 num_heads=16,
                 num_layers=24,
                 post_norm=True,
                 dropout=0.1,
                 eps=1e-5):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.type_size = type_size
        self.pad_id = pad_id
        self.dim = dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.post_norm = post_norm
        self.eps = eps

        # embeddings
        self.token_embedding = nn.Embedding(vocab_size, dim, padding_idx=pad_id)
        self.type_embedding = nn.Embedding(type_size, dim)
        self.pos_embedding = nn.Embedding(max_seq_len, dim, padding_idx=pad_id)
        self.dropout = nn.Dropout(dropout)

        # blocks
        self.blocks = nn.ModuleList([
            AttentionBlock(dim, num_heads, post_norm, dropout, eps)
            for _ in range(num_layers)
        ])

        # norm layer
        self.norm = nn.LayerNorm(dim, eps=eps)

    def forward(self, ids):
        """
        ids: [B, L] of torch.LongTensor.
        """
        b, s = ids.shape
        mask = ids.ne(self.pad_id).long()

        # embeddings
        x = self.token_embedding(ids) + \
            self.type_embedding(torch.zeros_like(ids)) + \
            self.pos_embedding(self.pad_id + torch.cumsum(mask, dim=1) * mask)
        if self.post_norm:
            x = self.norm(x)
        x = self.dropout(x)

        # blocks
        mask = torch.where(
            mask.view(b, 1, 1, s).gt(0), 0.0,
            torch.finfo(x.dtype).min)
        for block in self.blocks:
            x = block(x, mask)

        # output
        if not self.post_norm:
            x = self.norm(x)
        return x


def xlm_roberta_large(pretrained=False,
                      return_tokenizer=False,
                      device='cpu',
                      **kwargs):
    """
    XLMRobertaLarge adapted from Huggingface.
    """
    # params
    cfg = dict(
        vocab_size=250002,
        max_seq_len=514,
        type_size=1,
        pad_id=1,
        dim=1024,
        num_heads=16,
        num_layers=24,
        post_norm=True,
        dropout=0.1,
        eps=1e-5)
    cfg.update(**kwargs)

    # init model
    if pretrained:
        from sora import DOWNLOAD_TO_CACHE

        # init a meta model
        with torch.device('meta'):
            model = XLMRoberta(**cfg)

        # load checkpoint
        model.load_state_dict(
            torch.load(
                DOWNLOAD_TO_CACHE('models/xlm_roberta/xlm_roberta_large.pth'),
                map_location=device),
            assign=True)
    else:
        # init a model on device
        with torch.device(device):
            model = XLMRoberta(**cfg)

    # init tokenizer
    if return_tokenizer:
        from sora.data import HuggingfaceTokenizer
        tokenizer = HuggingfaceTokenizer(
            name='xlm-roberta-large',
            seq_len=model.text_len,
            clean='whitespace')
        return model, tokenizer
    else:
        return model



def pos_interpolate(pos, seq_len):
    if pos.size(1) == seq_len:
        return pos
    else:
        src_grid = int(math.sqrt(pos.size(1)))
        tar_grid = int(math.sqrt(seq_len))
        n = pos.size(1) - src_grid * src_grid
        return torch.cat([
            pos[:, :n],
            F.interpolate(
                pos[:, n:].float().reshape(1, src_grid, src_grid, -1).permute(
                    0, 3, 1, 2),
                size=(tar_grid, tar_grid),
                mode='bicubic',
                align_corners=False).flatten(2).transpose(1, 2)
        ],
                         dim=1)


class QuickGELU(nn.Module):

    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)


class LayerNorm(nn.LayerNorm):

    def forward(self, x):
        return super().forward(x).type_as(x)


class SelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 causal=False,
                 attn_dropout=0.0,
                 proj_dropout=0.0):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.causal = causal
        self.attn_dropout = attn_dropout
        self.proj_dropout = proj_dropout

        # layers
        self.to_qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        """
        x:   [B, L, C].
        """
        # compute query, key, value
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)

        # compute attention
        x = flash_attention(q, k, v, num_heads=self.num_heads, compatibility_mode=True)

        # output
        x = self.proj(x)
        x = F.dropout(x, self.proj_dropout, self.training)
        return x


class SwiGLU(nn.Module):

    def __init__(self, dim, mid_dim):
        super().__init__()
        self.dim = dim
        self.mid_dim = mid_dim

        # layers
        self.fc1 = nn.Linear(dim, mid_dim)
        self.fc2 = nn.Linear(dim, mid_dim)
        self.fc3 = nn.Linear(mid_dim, dim)

    def forward(self, x):
        x = F.silu(self.fc1(x)) * self.fc2(x)
        x = self.fc3(x)
        return x


class AttentionBlock(nn.Module):

    def __init__(self,
                 dim,
                 mlp_ratio,
                 num_heads,
                 post_norm=False,
                 causal=False,
                 activation='quick_gelu',
                 attn_dropout=0.0,
                 proj_dropout=0.0,
                 norm_eps=1e-5):
        assert activation in ['quick_gelu', 'gelu', 'swi_glu']
        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.num_heads = num_heads
        self.post_norm = post_norm
        self.causal = causal
        self.norm_eps = norm_eps

        # layers
        self.norm1 = LayerNorm(dim, eps=norm_eps)
        self.attn = SelfAttention(dim, num_heads, causal, attn_dropout,
                                  proj_dropout)
        self.norm2 = LayerNorm(dim, eps=norm_eps)
        if activation == 'swi_glu':
            self.mlp = SwiGLU(dim, int(dim * mlp_ratio))
        else:
            self.mlp = nn.Sequential(
                nn.Linear(dim, int(dim * mlp_ratio)),
                QuickGELU() if activation == 'quick_gelu' else nn.GELU(),
                nn.Linear(int(dim * mlp_ratio), dim), nn.Dropout(proj_dropout))

    def forward(self, x):
        if self.post_norm:
            x = x + self.norm1(self.attn(x))
            x = x + self.norm2(self.mlp(x))
        else:
            x = x + self.attn(self.norm1(x))
            x = x + self.mlp(self.norm2(x))
        return x


class AttentionPool(nn.Module):

    def __init__(self,
                 dim,
                 mlp_ratio,
                 num_heads,
                 activation='gelu',
                 proj_dropout=0.0,
                 norm_eps=1e-5):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.proj_dropout = proj_dropout
        self.norm_eps = norm_eps

        # layers
        gain = 1.0 / math.sqrt(dim)
        self.cls_embedding = nn.Parameter(gain * torch.randn(1, 1, dim))
        self.to_q = nn.Linear(dim, dim)
        self.to_kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)
        self.norm = LayerNorm(dim, eps=norm_eps)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            QuickGELU() if activation == 'quick_gelu' else nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim), nn.Dropout(proj_dropout))

    def forward(self, x):
        """
        x:  [B, L, C].
        """
        b, s, c, n, d = *x.size(), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.to_q(self.cls_embedding).view(1, 1, n*d).expand(b, -1, -1)
        k, v = self.to_kv(x).chunk(2, dim=-1)

        # compute attention
        x = flash_attention(q, k, v, num_heads=self.num_heads, compatibility_mode=True)
        x = x.reshape(b, 1, c)

        # output
        x = self.proj(x)
        x = F.dropout(x, self.proj_dropout, self.training)

        # mlp
        x = x + self.mlp(self.norm(x))
        return x[:, 0]


class VisionTransformer(nn.Module):

    def __init__(self,
                 image_size=224,
                 patch_size=16,
                 dim=768,
                 mlp_ratio=4,
                 out_dim=512,
                 num_heads=12,
                 num_layers=12,
                 pool_type='token',
                 pre_norm=True,
                 post_norm=False,
                 activation='quick_gelu',
                 attn_dropout=0.0,
                 proj_dropout=0.0,
                 embedding_dropout=0.0,
                 norm_eps=1e-5):
        if image_size % patch_size != 0:
            print(
                '[WARNING] image_size is not divisible by patch_size',
                flush=True)
        assert pool_type in ('token', 'token_fc', 'attn_pool')
        out_dim = out_dim or dim
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size)**2
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.pool_type = pool_type
        self.post_norm = post_norm
        self.norm_eps = norm_eps

        # embeddings
        gain = 1.0 / math.sqrt(dim)
        self.patch_embedding = nn.Conv2d(
            3,
            dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=not pre_norm)
        if pool_type in ('token', 'token_fc'):
            self.cls_embedding = nn.Parameter(gain * torch.randn(1, 1, dim))
        self.pos_embedding = nn.Parameter(gain * torch.randn(
            1, self.num_patches +
            (1 if pool_type in ('token', 'token_fc') else 0), dim))
        self.dropout = nn.Dropout(embedding_dropout)

        # transformer
        self.pre_norm = LayerNorm(dim, eps=norm_eps) if pre_norm else None
        self.transformer = nn.Sequential(*[
            AttentionBlock(dim, mlp_ratio, num_heads, post_norm, False,
                           activation, attn_dropout, proj_dropout, norm_eps)
            for _ in range(num_layers)
        ])
        self.post_norm = LayerNorm(dim, eps=norm_eps)

        # head
        if pool_type == 'token':
            self.head = nn.Parameter(gain * torch.randn(dim, out_dim))
        elif pool_type == 'token_fc':
            self.head = nn.Linear(dim, out_dim)
        elif pool_type == 'attn_pool':
            self.head = AttentionPool(dim, mlp_ratio, num_heads, activation,
                                      proj_dropout, norm_eps)

    def forward(self, x, interpolation=False, use_31_block=False):
        b = x.size(0)

        # embeddings
        x = self.patch_embedding(x).flatten(2).permute(0, 2, 1)
        # print("x in forward: ", x[0,0,100:105], x.shape)
        # print("patch_embedding: ", self.patch_embedding.module.weight[0:10, 0, 0, 3], self.patch_embedding.module.weight.shape)     
        if self.pool_type in ('token', 'token_fc'):
            x = torch.cat([self.cls_embedding.expand(b, -1, -1).to(dtype=x.dtype, device=x.device), x], dim=1)
        if interpolation:
            e = pos_interpolate(self.pos_embedding, x.size(1))
        else:
            e = self.pos_embedding
        e = e.to(dtype=x.dtype, device=x.device)
        x = self.dropout(x + e)
        if self.pre_norm is not None:
            x = self.pre_norm(x)

        # transformer
        if use_31_block:
            # print("x before transformer: ", x[0,0,100:105], x.shape)
            x = self.transformer[:-1](x)
            return x
        else:
            # print("x before transformer: ", x[0,0,100:105], x.shape)
            x = self.transformer(x)
            return x


class CLIP(nn.Module):

    def __init__(self,
                 embed_dim=512,
                 image_size=224,
                 patch_size=16,
                 vision_dim=768,
                 vision_mlp_ratio=4,
                 vision_heads=12,
                 vision_layers=12,
                 vision_pool='token',
                 vision_pre_norm=True,
                 vision_post_norm=False,
                 vocab_size=49408,
                 text_len=77,
                 text_dim=512,
                 text_mlp_ratio=4,
                 text_heads=8,
                 text_layers=12,
                 text_causal=True,
                 text_pool='argmax',
                 text_head_bias=False,
                 logit_bias=None,
                 activation='quick_gelu',
                 attn_dropout=0.0,
                 proj_dropout=0.0,
                 embedding_dropout=0.0,
                 norm_eps=1e-5):
        super().__init__()
        self.embed_dim = embed_dim
        self.image_size = image_size
        self.patch_size = patch_size
        self.vision_dim = vision_dim
        self.vision_mlp_ratio = vision_mlp_ratio
        self.vision_heads = vision_heads
        self.vision_layers = vision_layers
        self.vision_pool = vision_pool
        self.vision_pre_norm = vision_pre_norm
        self.vision_post_norm = vision_post_norm
        self.vocab_size = vocab_size
        self.text_len = text_len
        self.text_dim = text_dim
        self.text_mlp_ratio = text_mlp_ratio
        self.text_heads = text_heads
        self.text_layers = text_layers
        self.text_causal = text_causal
        self.text_pool = text_pool
        self.text_head_bias = text_head_bias
        self.norm_eps = norm_eps

        # models
        self.visual = VisionTransformer(
            image_size=image_size,
            patch_size=patch_size,
            dim=vision_dim,
            mlp_ratio=vision_mlp_ratio,
            out_dim=embed_dim,
            num_heads=vision_heads,
            num_layers=vision_layers,
            pool_type=vision_pool,
            pre_norm=vision_pre_norm,
            post_norm=vision_post_norm,
            activation=activation,
            attn_dropout=attn_dropout,
            proj_dropout=proj_dropout,
            embedding_dropout=embedding_dropout,
            norm_eps=norm_eps)
        self.textual = TextTransformer(
            vocab_size=vocab_size,
            text_len=text_len,
            dim=text_dim,
            mlp_ratio=text_mlp_ratio,
            out_dim=embed_dim,
            num_heads=text_heads,
            num_layers=text_layers,
            causal=text_causal,
            pool_type=text_pool,
            head_bias=text_head_bias,
            activation=activation,
            attn_dropout=attn_dropout,
            proj_dropout=proj_dropout,
            embedding_dropout=embedding_dropout,
            norm_eps=norm_eps)
        self.log_scale = nn.Parameter(math.log(1 / 0.07) * torch.ones([]))
        if logit_bias is not None:
            self.logit_bias = nn.Parameter(logit_bias * torch.ones([]))

        # initialize weights
        self.init_weights()

    def forward(self, imgs, txt_ids):
        """
        imgs:       [B, 3, H, W] of torch.float32.
        - mean:     [0.48145466, 0.4578275, 0.40821073]
        - std:      [0.26862954, 0.26130258, 0.27577711]
        txt_ids:    [B, L] of torch.long. Encoded by data.CLIPTokenizer.
        """
        xi = self.visual(imgs)
        xt = self.textual(txt_ids)
        return xi, xt

    def init_weights(self):
        # embeddings
        nn.init.normal_(self.textual.token_embedding.weight, std=0.02)
        nn.init.normal_(self.visual.patch_embedding.weight, std=0.1)

        # attentions
        for modality in ['visual', 'textual']:
            dim = self.vision_dim if modality == 'visual' else self.text_dim
            transformer = getattr(self, modality).transformer
            proj_gain = (1.0 / math.sqrt(dim)) * (
                1.0 / math.sqrt(2 * len(transformer)))
            attn_gain = 1.0 / math.sqrt(dim)
            mlp_gain = 1.0 / math.sqrt(2.0 * dim)
            for block in transformer:
                nn.init.normal_(block.attn.to_qkv.weight, std=attn_gain)
                nn.init.normal_(block.attn.proj.weight, std=proj_gain)
                nn.init.normal_(block.mlp[0].weight, std=mlp_gain)
                nn.init.normal_(block.mlp[2].weight, std=proj_gain)

    def param_groups(self):
        groups = [{
            'params': [
                p for n, p in self.named_parameters()
                if 'norm' in n or n.endswith('bias')
            ],
            'weight_decay': 0.0
        }, {
            'params': [
                p for n, p in self.named_parameters()
                if not ('norm' in n or n.endswith('bias'))
            ]
        }]
        return groups


class XLMRobertaWithHead(XLMRoberta):

    def __init__(self, **kwargs):
        self.out_dim = kwargs.pop('out_dim')
        super().__init__(**kwargs)

        # head
        mid_dim = (self.dim + self.out_dim) // 2
        self.head = nn.Sequential(
            nn.Linear(self.dim, mid_dim, bias=False), nn.GELU(),
            nn.Linear(mid_dim, self.out_dim, bias=False))

    def forward(self, ids):
        # xlm-roberta
        x = super().forward(ids)

        # average pooling
        mask = ids.ne(self.pad_id).unsqueeze(-1).to(x)
        x = (x * mask).sum(dim=1) / mask.sum(dim=1)

        # head
        x = self.head(x)
        return x


class XLMRobertaCLIP(nn.Module):

    def __init__(self,
                 embed_dim=1024,
                 image_size=224,
                 patch_size=14,
                 vision_dim=1280,
                 vision_mlp_ratio=4,
                 vision_heads=16,
                 vision_layers=32,
                 vision_pool='token',
                 vision_pre_norm=True,
                 vision_post_norm=False,
                 activation='gelu',
                 vocab_size=250002,
                 max_text_len=514,
                 type_size=1,
                 pad_id=1,
                 text_dim=1024,
                 text_heads=16,
                 text_layers=24,
                 text_post_norm=True,
                 text_dropout=0.1,
                 attn_dropout=0.0,
                 proj_dropout=0.0,
                 embedding_dropout=0.0,
                 norm_eps=1e-5):
        super().__init__()
        self.embed_dim = embed_dim
        self.image_size = image_size
        self.patch_size = patch_size
        self.vision_dim = vision_dim
        self.vision_mlp_ratio = vision_mlp_ratio
        self.vision_heads = vision_heads
        self.vision_layers = vision_layers
        self.vision_pre_norm = vision_pre_norm
        self.vision_post_norm = vision_post_norm
        self.activation = activation
        self.vocab_size = vocab_size
        self.max_text_len = max_text_len
        self.type_size = type_size
        self.pad_id = pad_id
        self.text_dim = text_dim
        self.text_heads = text_heads
        self.text_layers = text_layers
        self.text_post_norm = text_post_norm
        self.norm_eps = norm_eps

        # models
        self.visual = VisionTransformer(
            image_size=image_size,
            patch_size=patch_size,
            dim=vision_dim,
            mlp_ratio=vision_mlp_ratio,
            out_dim=embed_dim,
            num_heads=vision_heads,
            num_layers=vision_layers,
            pool_type=vision_pool,
            pre_norm=vision_pre_norm,
            post_norm=vision_post_norm,
            activation=activation,
            attn_dropout=attn_dropout,
            proj_dropout=proj_dropout,
            embedding_dropout=embedding_dropout,
            norm_eps=norm_eps)
        self.textual = None
        self.log_scale = nn.Parameter(math.log(1 / 0.07) * torch.ones([]))

    def forward(self, imgs, txt_ids):
        """
        imgs:       [B, 3, H, W] of torch.float32.
        - mean:     [0.48145466, 0.4578275, 0.40821073]
        - std:      [0.26862954, 0.26130258, 0.27577711]
        txt_ids:    [B, L] of torch.long.
                    Encoded by data.CLIPTokenizer.
        """
        xi = self.visual(imgs)
        xt = self.textual(txt_ids)
        return xi, xt

    def param_groups(self):
        groups = [{
            'params': [
                p for n, p in self.named_parameters()
                if 'norm' in n or n.endswith('bias')
            ],
            'weight_decay': 0.0
        }, {
            'params': [
                p for n, p in self.named_parameters()
                if not ('norm' in n or n.endswith('bias'))
            ]
        }]
        return groups


def _clip(pretrained=False,
          pretrained_name=None,
          model_cls=CLIP,
          return_transforms=False,
          return_tokenizer=False,
          tokenizer_padding='eos',
          dtype=torch.float32,
          device='cpu',
          **kwargs):
    # init model
    if pretrained and pretrained_name:
        from sora import BUCKET, DOWNLOAD_TO_CACHE

        # init a meta model
        with torch.device('meta'):
            model = model_cls(**kwargs)

        # checkpoint path
        checkpoint = f'models/clip/{pretrained_name}'
        if dtype in (torch.float16, torch.bfloat16):
            suffix = '-' + {
                torch.float16: 'fp16',
                torch.bfloat16: 'bf16'
            }[dtype]
            if object_exists(BUCKET, f'{checkpoint}{suffix}.pth'):
                checkpoint = f'{checkpoint}{suffix}'
        checkpoint += '.pth'

        # load
        model.load_state_dict(
            torch.load(DOWNLOAD_TO_CACHE(checkpoint), map_location=device),
            assign=True,
            strict=False)
    else:
        # init a model on device
        with torch.device(device):
            model = model_cls(**kwargs)

    # set device
    output = (model,)

    # init transforms
    if return_transforms:
        # mean and std
        if 'siglip' in pretrained_name.lower():
            mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        else:
            mean = [0.48145466, 0.4578275, 0.40821073]
            std = [0.26862954, 0.26130258, 0.27577711]

        # transforms
        transforms = T.Compose([
            T.Resize((model.image_size, model.image_size),
                     interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std)
        ])
        output += (transforms,)

    # init tokenizer
    if return_tokenizer:
        from sora import data
        if 'siglip' in pretrained_name.lower():
            tokenizer = data.HuggingfaceTokenizer(
                name=f'timm/{pretrained_name}',
                seq_len=model.text_len,
                clean='canonicalize')
        elif 'xlm' in pretrained_name.lower():
            tokenizer = data.HuggingfaceTokenizer(
                name='xlm-roberta-large',
                seq_len=model.max_text_len - 2,
                clean='whitespace')
        elif 'mba' in pretrained_name.lower():
            tokenizer = data.HuggingfaceTokenizer(
                name='facebook/xlm-roberta-xl',
                seq_len=model.max_text_len - 2,
                clean='whitespace')
        else:
            tokenizer = data.CLIPTokenizer(
                seq_len=model.text_len, padding=tokenizer_padding)
        output += (tokenizer,)
    return output[0] if len(output) == 1 else output


def clip_xlm_roberta_vit_h_14(
        pretrained=False,
        pretrained_name='open-clip-xlm-roberta-large-vit-huge-14',
        **kwargs):
    cfg = dict(
        embed_dim=1024,
        image_size=224,
        patch_size=14,
        vision_dim=1280,
        vision_mlp_ratio=4,
        vision_heads=16,
        vision_layers=32,
        vision_pool='token',
        activation='gelu',
        vocab_size=250002,
        max_text_len=514,
        type_size=1,
        pad_id=1,
        text_dim=1024,
        text_heads=16,
        text_layers=24,
        text_post_norm=True,
        text_dropout=0.1,
        attn_dropout=0.0,
        proj_dropout=0.0,
        embedding_dropout=0.0)
    cfg.update(**kwargs)
    return _clip(pretrained, pretrained_name, XLMRobertaCLIP, **cfg)


class WanImageEncoder(torch.nn.Module):

    def __init__(self, image_encoder_pretrained_path: str=None):
        super().__init__()
        # init model
        self.model, self.transforms = clip_xlm_roberta_vit_h_14(
            pretrained=False,
            return_transforms=True,
            return_tokenizer=False,
            dtype=torch.float32,
            device="cpu")
        self.image_encoder_pretrained_path = image_encoder_pretrained_path

    def encode_image(self, videos):
        # preprocess
        size = (self.model.image_size,) * 2
        videos = torch.cat([
            F.interpolate(
                u,
                size=size,
                mode='bicubic',
                align_corners=False) for u in videos
        ])
        videos = self.transforms.transforms[-1](videos.mul_(0.5).add_(0.5))

        # forward
        dtype = next(iter(self.model.visual.parameters())).dtype
        videos = videos.to(dtype)
        out = self.model.visual(videos, use_31_block=True)
        # The outputs of torch compile always need to be cloned before being used.
        out = out.clone()
        return out
        
    @staticmethod
    def state_dict_converter():
        return WanImageEncoderStateDictConverter()
    
    
class WanImageEncoderStateDictConverter:
    def __init__(self):
        pass

    def from_diffusers(self, state_dict):
        return state_dict
    
    def from_civitai(self, state_dict):
        state_dict_ = {}
        for name, param in state_dict.items():
            if name.startswith("textual."):
                continue
            name = "model." + name
            state_dict_[name] = param
        return state_dict_
