# 修改记录 (Issue #18077 相关)

本文档记录为修复单卡 GLM-Image 启动问题及支持 SP 所做的**核心修改**，并给出干净的 PR 提交清单。

---

## 测试与复现概要（写入本节的说明）

1. **只靠文档里的代码就能复现**
   - 在「三、PR 提交清单」里「PR 前自检」后面有 **「代码是否足够复现？」**：明确写 **是，三处改动就够**。
   - `base.py` 里已有 `get_sp_world_size` / `get_sp_parallel_rank` / `sequence_model_parallel_all_gather` 的 import，新增的 `SpatialImagePipelineConfig` 不用再补 import。
   - 在新 branch 上只要按文档改这 **3 个文件**，就能百分百复现行为。

2. **在新 branch 上如何验证（不依赖 bug_18077_analysis 里的脚本）**
   - 与 upstream 完全一致的干净 branch + 只应用文档中的 3 处代码后，**不用拷任何 test 脚本**也可验证：
   - **单卡**：一条 `sglang serve --model-path zai-org/GLM-Image --backend sglang --resolution 512x512`，能起来且没有 `parallel_folding` 报错即通过。
   - **多卡 SP**：`sglang serve ... --num-gpus 2 --sp-degree 2`，再用文档 **「四、在新 branch 上如何验证」** 里给的 curl 或 Python 一行发一次生成请求；没有 einops/4D 形状相关报错即通过。
   - 若仍想用原有 test 脚本：把 `bug_18077_analysis/code/` 里用到的脚本拷到新 branch 任意目录再跑即可，验证逻辑与上面一致。

3. **总结**
   - 只靠文档里的代码就能百分百复现；在新 branch 上不拷 test 脚本时，用文档 **「四」** 的几条命令即可完成单卡和多卡 SP 的验证。

---

## 一、修改了哪些文件

**SGLang 源码（3 个已存在文件）：**

| # | 文件路径 |
|---|----------|
| 1 | `python/sglang/multimodal_gen/configs/pipeline_configs/glm_image.py` |
| 2 | `python/sglang/multimodal_gen/configs/pipeline_configs/base.py` |
| 3 | `python/sglang/multimodal_gen/runtime/models/dits/glm_image.py` |

---

## 二、每个文件的 Change

### 1. `python/sglang/multimodal_gen/configs/pipeline_configs/glm_image.py`

**目的**：修复单卡启动时 `EncoderConfig` 缺少 `parallel_folding` 导致的 AttributeError 及 T5 fallback/权重 MISSING；并支持 4D latents 的 SP。

**Change：**

- **Import**
  - 从 `sglang.multimodal_gen.configs.models` 增加 `EncoderConfig`。
  - 从 `sglang.multimodal_gen.configs.models.encoders.t5` 引入 `T5Config`。
  - 从 `sglang.multimodal_gen.configs.pipeline_configs.base` 改为引入 `SpatialImagePipelineConfig`（不再用 `ImagePipelineConfig`）。

- **基类**
  - `GlmImagePipelineConfig` 改为继承 `SpatialImagePipelineConfig`，以便走 4D latents 的 SP shard/gather 路径。

- **配置**
  - 在 `GlmImagePipelineConfig` 中显式定义 `text_encoder_configs`，使用 `T5Config()`，不再沿用基类默认的 `EncoderConfig()`。

**效果**：单卡 `sglang serve` 启动 GLM-Image 不再报 `'EncoderConfig' object has no attribute 'parallel_folding'`，T5 正常加载；多卡 SP 时使用 4D 切分，形状正确。

---

### 2. `python/sglang/multimodal_gen/configs/pipeline_configs/base.py`

**目的**：为 GLM-Image 等使用 4D latents `[B, C, H', W']` 的 pipeline 提供 SP，避免走基类 3D token 路径导致形状错误。

**Change：**

- 在 `ImagePipelineConfig` 之后新增类 **`SpatialImagePipelineConfig(ImagePipelineConfig)`**：
  - **`shard_latents_for_sp(batch, latents)`**：若 `latents.dim() == 4`，沿 H'（dim=2）切分，必要时 padding 使可被 `sp_world_size` 整除；否则回退 `super().shard_latents_for_sp(...)`。
  - **`gather_latents_for_sp(latents)`**：若 4D，沿 dim=2 `sequence_model_parallel_all_gather`；否则回退基类。

---

### 3. `python/sglang/multimodal_gen/runtime/models/dits/glm_image.py`

**目的**：SP 时 latents 在 H 维切分，当前 rank 的 `hidden_states` patch 数少于全局 `prior_hidden_states`，需对 prior 按 seq 维切分再相加。

**Change：**

- 在 **`GlmImageTransformer2DModel`** 中，prior 与 hidden_states 相加前：
  - 若 `get_sp_world_size() > 1` 且 `prior_hidden_states.shape[1] != hidden_states.shape[1]`，则按 `get_sp_parallel_rank()` 对 `prior_hidden_states` 在 seq 维做等分切块，再与 `hidden_states` 相加。
- 增加 import：`get_sp_parallel_rank`, `get_sp_world_size`（来自 `runtime.distributed.parallel_state`）。

**Prior Sharding 方案选择（最终确认）**  
当时有两种可选做法：  
- **方案 A**：在 **Model 文件** `runtime/models/dits/glm_image.py` 的 `forward` 里，在 prior 与 hidden_states 相加前按 SP rank 切分 prior（即上面写的这段）。  
- **方案 B**：在 **Config 文件** `configs/pipeline_configs/glm_image.py` 里加 helper（如 `shard_prior_for_sp`），通过 `prepare_pos_cond_kwargs` 传进去，Model 不动。  

**当前实现是方案 A**：只改了 Model 文件，Config 里没有 prior 相关 helper，也没有动 `prepare_pos_cond_kwargs` 的传参。这在 SGLang 里是常见做法。提交时按现有清单（含 `dits/glm_image.py`）即可，无需再改 Config。

---

## 三、PR 提交清单（干净 PR）

为保持 PR 干净，**只提交与单卡修复 + SP 支持直接相关的代码**，建议如下。

### 建议纳入本 PR 的文件（3 个）

| # | 文件 | 说明 |
|---|------|------|
| 1 | `python/sglang/multimodal_gen/configs/pipeline_configs/glm_image.py` | 单卡 T5 配置 + 继承 SpatialImagePipelineConfig |
| 2 | `python/sglang/multimodal_gen/configs/pipeline_configs/base.py` | 新增 SpatialImagePipelineConfig（4D SP shard/gather） |
| 3 | `python/sglang/multimodal_gen/runtime/models/dits/glm_image.py` | SP 时 prior_hidden_states 按 seq 切分 |

### 代码变更（可直接复现）

**1. `configs/pipeline_configs/glm_image.py`**

```diff
@@ -3,17 +3,18 @@ from dataclasses import dataclass, field
 import torch
 from diffusers.image_processor import VaeImageProcessor
 
-from sglang.multimodal_gen.configs.models import DiTConfig, VAEConfig
+from sglang.multimodal_gen.configs.models import DiTConfig, EncoderConfig, VAEConfig
 from sglang.multimodal_gen.configs.models.dits.glmimage import GlmImageDitConfig
+from sglang.multimodal_gen.configs.models.encoders.t5 import T5Config
 from sglang.multimodal_gen.configs.models.vaes.glmimage import GlmImageVAEConfig
 from sglang.multimodal_gen.configs.pipeline_configs.base import (
-    ImagePipelineConfig,
     ModelTaskType,
+    SpatialImagePipelineConfig,
 )
 
 
 @dataclass
-class GlmImagePipelineConfig(ImagePipelineConfig):
+class GlmImagePipelineConfig(SpatialImagePipelineConfig):
     """Configuration for the GlmImage pipeline."""
 
     vae_precision: str = "bf16"
@@ -29,6 +30,12 @@ class GlmImagePipelineConfig(ImagePipelineConfig):
     # VAE
     vae_config: VAEConfig = field(default_factory=GlmImageVAEConfig)
 
+    # GLM-Image uses T5 text encoder; base default is EncoderConfig() which lacks
+    # parallel_folding and causes AttributeError + fallback to native T5 with missing weights.
+    text_encoder_configs: tuple[EncoderConfig, ...] = field(
+        default_factory=lambda: (T5Config(),)
+    )
+
     enable_autocast: bool = False
```

**2. `configs/pipeline_configs/base.py`**（在 `ImagePipelineConfig` 类定义结束、`SlidingTileAttnConfig` 之前插入）

```diff
@@ -762,6 +762,49 @@ class ImagePipelineConfig(PipelineConfig):
         return latents, batch_size, channels, height, width
 
 
+@dataclass
+class SpatialImagePipelineConfig(ImagePipelineConfig):
+    """Base config for spatial image pipelines (e.g. GLM-Image) with 4D latents (B, C, H', W').
+
+    Overrides shard_latents_for_sp / gather_latents_for_sp to shard along the height dimension
+    so that each SP rank gets (B, C, H'_local, W') instead of using the token-style (B, S, C) path.
+    """
+
+    def shard_latents_for_sp(self, batch, latents):
+        # 4D latents (B, C, H', W') -> shard along H' (dim=2); otherwise fall back to base (B, S, C)
+        sp_world_size = get_sp_world_size()
+        if sp_world_size <= 1:
+            return latents, False
+        if latents.dim() != 4:
+            return super().shard_latents_for_sp(batch, latents)
+
+        # (B, C, H', W')
+        _, _, h_lat, w_lat = latents.shape
+        if h_lat % sp_world_size != 0:
+            pad_len = sp_world_size - (h_lat % sp_world_size)
+            pad = torch.zeros(
+                (latents.shape[0], latents.shape[1], pad_len, latents.shape[3]),
+                dtype=latents.dtype,
+                device=latents.device,
+            )
+            latents = torch.cat([latents, pad], dim=2)
+            h_lat = latents.shape[2]
+        rank_in_sp_group = get_sp_parallel_rank()
+        chunk_size = h_lat // sp_world_size
+        h0 = rank_in_sp_group * chunk_size
+        h1 = h0 + chunk_size
+        sharded = latents[:, :, h0:h1, :].contiguous()
+        return sharded, True
+
+    def gather_latents_for_sp(self, latents):
+        if get_sp_world_size() <= 1:
+            return latents
+        if latents.dim() != 4:
+            return super().gather_latents_for_sp(latents)
+        # Gather along dim=2 (H') to match shard_latents_for_sp
+        return sequence_model_parallel_all_gather(latents, dim=2)
+
+
 @dataclass
 class SlidingTileAttnConfig(PipelineConfig):
```

**3. `runtime/models/dits/glm_image.py`**

- 在 import 区、`CachableDiT` 之后增加：

```python
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_sp_parallel_rank,
    get_sp_world_size,
)
```

- 在 `prior_hidden_states = self.prior_projector(prior_embedding)` 与 `hidden_states = hidden_states + prior_hidden_states` 之间增加：

```python
        # SP: when latents are H-sharded, hidden_states has fewer patches than prior_hidden_states.
        # Shard prior_hidden_states along seq dim to match (prior is row-major, same as latent patches).
        if (
            get_sp_world_size() > 1
            and prior_hidden_states.shape[1] != hidden_states.shape[1]
        ):
            rank = get_sp_parallel_rank()
            sp_world_size = get_sp_world_size()
            chunk = prior_hidden_states.shape[1] // sp_world_size
            prior_hidden_states = prior_hidden_states[
                :, rank * chunk : (rank + 1) * chunk, :
            ]
```

### PR 前自检

- [ ] 仅包含上述 3 个核心文件
- [ ] `git diff upstream/main` 仅涉及上述文件，无多余改动
- [ ] 单卡启动 GLM-Image 无 `parallel_folding` 报错
- [ ] 多卡 `sp_degree > 1` 时 GLM-Image 推理形状正确、无 einops 错误

### 代码是否足够复现？

**是。** 上面三处代码变更即全部改动：`base.py` 文件顶部已有 `get_sp_world_size`、`get_sp_parallel_rank`、`sequence_model_parallel_all_gather` 的 import，新增的 `SpatialImagePipelineConfig` 无需再加 import。按文档改完 3 个文件即可百分百复现。

---

## 四、在新 branch 上如何验证（不依赖 bug_18077_analysis 里的脚本）

新开一个与 upstream 完全一致的 branch、只应用本文档中的 3 处代码后，**不需要**把 `bug_18077_analysis/code/` 下的脚本拷过去，用下面命令即可自验。

**1. 单卡验证（修单卡启动 + T5 加载）**

在 repo 根目录（或 `python` 下）执行：

```bash
# 启动服务（能正常起来、无 parallel_folding 报错即通过）
sglang serve --model-path zai-org/GLM-Image --backend sglang --resolution 512x512
```

若出现 `'EncoderConfig' object has no attribute 'parallel_folding'` 或大量 decoder 权重 MISSING，说明 config 未按文档改对。

**2. 多卡 SP 验证（修 SP 形状错误）**

至少 2 张 GPU 时：

```bash
# 2 卡、sp_degree=2 启动
sglang serve --model-path zai-org/GLM-Image --backend sglang --num-gpus 2 --sp-degree 2 --resolution 512x512
```

再另开终端发一次生成请求（任选其一）：

```bash
# 用 curl 调 OpenAI 兼容接口
curl -X POST http://localhost:30000/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a cat", "size": "512x512", "n": 1}'
```

或 Python 一行：

```bash
python -c "
from openai import OpenAI
c = OpenAI(base_url='http://localhost:30000/v1')
r = c.images.generate(model='', prompt='a cat', size='512x512', n=1)
print('ok' if r else 'fail')
"
```

若出现 `einops.EinopsError` 或 `Wrong shape: expected 3 dims. Received 4-dim tensor`，说明 SP 相关两处（base.py 或 dits/glm_image.py）未改对或未生效。

**3. 若仍想用原有 test 脚本**

把 `bug_18077_analysis/code/` 里用到的脚本（如 `B01_single_gpu.sh`、`B02_multiple_gpu.sh`）**单独拷到新 branch 的任意目录**（不必在 bug_18077_analysis 下），在脚本里保证 `sglang serve` 指向当前环境、再跑即可。验证逻辑与上面 1、2 一致。

---

## 五、参考（未改动的相关文件）

- `python/sglang/multimodal_gen/runtime/distributed/__init__.py` — `_get_folding_tp_group(config)` 要求 `TextEncoderConfig`
- `python/sglang/multimodal_gen/configs/models/encoders/base.py` — `EncoderConfig` / `TextEncoderConfig` 定义
- `python/sglang/multimodal_gen/configs/pipeline_configs/base.py` — 基类 `text_encoder_configs` 默认 `(EncoderConfig(),)`

---

## 六、已 Revert 的调试日志（A03_GLM）

曾为多卡 SP 排查在以下位置加过 `[A03_GLM]` 的 `logger.info`，已全部删除，PR 中不包含：

- `python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/glm_image.py`（3 处）
- `python/sglang/multimodal_gen/runtime/models/dits/glm_image.py`（1 处）

---

*最后更新：加入「代码是否足够复现」说明；新增「四、在新 branch 上如何验证」，不依赖 bug_18077_analysis 脚本即可单卡/多卡 SP 自验。*
