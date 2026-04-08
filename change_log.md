# Change Log

## Session: 2026-04-03 — ErnieImage sglang 启动问题修复

---

### 1. 新增 `python/sglang/multimodal_gen/configs/models/vaes/ernie_image.py`

**问题**：`sglang serve --model-path baidu/ERNIE-Image-Turbo` 启动时报错：
```
ModuleNotFoundError: No module named 'sglang.multimodal_gen.configs.models.vaes.ernie_image'
```

**原因**：`configs/pipeline_configs/ernie_image.py` 中 import 了 `ErnieImageVAEConfig`，但对应的文件从未创建。

**修复**：新建文件，定义 `ErnieImageVAEArchConfig` 和 `ErnieImageVAEConfig`，参考 `flux.py` 和 `glmimage.py` 的实现模式。关键参数：
- `z_dim = 32`（对应 pipeline 中 `_patchify_latents` 注释的 `[B, 32, H, W]` 潜空间）
- `scale_factor_spatial = 8`（VAE 本身空间压缩比，总倍率 16 = 8 spatial × 2 patch）
- `use_tiling = False`（与图像类 VAE 保持一致）

**状态**：✅ 保留，已由用户 commit

---

### 2. 修改 `python/sglang/multimodal_gen/runtime/pipelines/ernie_image.py` — `pipeline_name`

**问题**：registry 读取 `model_index.json` 中的 `_class_name`，与 pipeline 文件中的 `pipeline_name` 不一致，导致 native pipeline 无法匹配，回退到 diffusers backend 后再次失败。

**过程**：
- 第一次错误日志：`model_index.json._class_name = "Text2ImgDiTPipeline"`，而代码中 `pipeline_name = "ErnieImagePipeline"` → 不匹配
- 修改为 `pipeline_name = "Text2ImgDiTPipeline"`
- 第二次错误日志：模型更新到新 snapshot（`1b7e912...`），`_class_name` 已变为 `"ErnieImagePipeline"`，与修改后的 `"Text2ImgDiTPipeline"` 又不一致
- 再次修改回 `pipeline_name = "ErnieImagePipeline"`

**状态**：✅ 当前值为 `"ErnieImagePipeline"`，与最新 `model_index.json` 一致

---

### 3. 修改 `python/sglang/multimodal_gen/runtime/loader/component_loaders/component_loader.py` — 后回退

**问题**：`model_index.json` 中 transformer 组件声明为：
```json
"transformer": ["transformers", "ErnieImageTransformer2DModel"]
```
而 `TransformerLoader.expected_library = "diffusers"`，导致断言失败：
```
AssertionError: transformer must be loaded from diffusers, got transformers
```

**修复（已回退）**：曾将 `"transformer"` 加入 `for_component_type()` 中的特殊处理列表，强制将库名覆盖为 `"diffusers"`。

**回退原因**：用户直接修改了 `model_index.json`，将 transformer 的库改为 `"diffusers"`，从根源修复，代码层面无需改动。

**状态**：⏪ 已回退，`component_loader.py` 恢复原样

---

### 4. 复制 `model_index.json` 到仓库根目录

**操作**：将 HuggingFace 本地缓存中的 `model_index.json` 复制到 `/root/paddlejob/workspace/env_run/output/sglang/model_index.json`，供用户查阅和修改（将 transformer 的库声明从 `"transformers"` 改为 `"diffusers"`）。

**文件路径（缓存）**：
```
/root/.cache/huggingface/hub/models--baidu--ERNIE-Image-Turbo/snapshots/1b7e91251206129fc8d4084024088fb3d37854dc/model_index.json
```

**状态**：已由用户 commit 并随后在 `507c9b279` 中从仓库删除

---

## Session: 2026-04-03 — Attention 权重名适配新 diffusers 命名规范

---

### 5. 修改 `python/sglang/multimodal_gen/runtime/models/dits/ernie_image.py` — attention 模块重命名

**背景**：模型 checkpoint 的 attention 权重名从旧格式（checkpoint 自定义命名）更新为 diffusers Attention 标准命名：

| 旧名 | 新名 |
|------|------|
| `*.self_attention.q_proj.weight` | `*.self_attention.to_q.weight` |
| `*.self_attention.k_proj.weight` | `*.self_attention.to_k.weight` |
| `*.self_attention.v_proj.weight` | `*.self_attention.to_v.weight` |
| `*.self_attention.linear_proj.weight` | `*.self_attention.to_out.0.weight` |
| `*.self_attention.q_layernorm.weight` | `*.self_attention.norm_q.weight` |
| `*.self_attention.k_layernorm.weight` | `*.self_attention.norm_k.weight` |

**修改内容**（`ErnieImageSelfAttention`）：
- `self.q_proj` → `self.to_q`（`ColumnParallelLinear`，prefix 同步更新）
- `self.k_proj` → `self.to_k`（`ColumnParallelLinear`）
- `self.v_proj` → `self.to_v`（`ColumnParallelLinear`）
- `self.linear_proj` → `self.to_out = nn.ModuleList([RowParallelLinear(...)])` — 用 `ModuleList` 使权重路径变为 `to_out.0.*`，与 diffusers 一致
- `self.q_layernorm` → `self.norm_q`（`RMSNorm`）
- `self.k_layernorm` → `self.norm_k`（`RMSNorm`）
- `forward()` 中所有引用同步更新

**状态**：✅ 已完成

---

### 6. 修改 `python/sglang/multimodal_gen/configs/models/dits/ernie_image.py` — 更新注释

**修改内容**：更新 `param_names_mapping` 注释，说明 attention 权重名已与模型模块名直接对齐，无需 remapping。`param_names_mapping` 内容本身不变（仅保留 MLP gate/up 融合规则）。

**状态**：✅ 已完成

---

## 当前有效变更汇总

| 文件 | 变更类型 | 说明 |
|------|---------|------|
| `python/sglang/multimodal_gen/configs/models/vaes/ernie_image.py` | 新增 | ErnieImage VAE 配置，修复 ModuleNotFoundError |
| `python/sglang/multimodal_gen/runtime/pipelines/ernie_image.py` | 修改 | `pipeline_name` 保持为 `"ErnieImagePipeline"`，与 model_index.json 对齐 |
| `python/sglang/multimodal_gen/runtime/models/dits/ernie_image.py` | 修改 | attention 模块重命名，适配新 diffusers 权重名 |
| `python/sglang/multimodal_gen/configs/models/dits/ernie_image.py` | 修改 | 更新 `param_names_mapping` 注释 |
