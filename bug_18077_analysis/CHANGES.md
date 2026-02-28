# 修改记录 (Issue #18077 相关)

本文档记录在 bug_18077_analysis 与 sglang 代码库中为修复单卡 GLM-Image 启动问题所做的文件和代码修改。与 `NEXT_STEPS.md` 配套使用。

---

## 1. SGLang 源码修改

### 1.1 `python/sglang/multimodal_gen/configs/pipeline_configs/glm_image.py`

**目的**：修复单卡启动时 `EncoderConfig` 缺少 `parallel_folding` 导致的 AttributeError，以及由此引发的 T5 fallback 与大量 decoder 权重 MISSING。

**修改内容**：

- **新增 import**  
  - 从 `sglang.multimodal_gen.configs.models` 增加 `EncoderConfig`。  
  - 从 `sglang.multimodal_gen.configs.models.encoders.t5` 引入 `T5Config`。

- **新增配置覆盖**  
  在 `GlmImagePipelineConfig` 中显式定义 `text_encoder_configs`，使用 `T5Config()` 而非基类默认的 `EncoderConfig()`。

**具体代码**：

```python
# 在文件顶部 import 区域增加/调整为：
from sglang.multimodal_gen.configs.models import DiTConfig, EncoderConfig, VAEConfig
from sglang.multimodal_gen.configs.models.encoders.t5 import T5Config

# 在 GlmImagePipelineConfig 中，在 vae_config 之后增加：
    # GLM-Image uses T5 text encoder; base default is EncoderConfig() which lacks
    # parallel_folding and causes AttributeError + fallback to native T5 with missing weights.
    text_encoder_configs: tuple[EncoderConfig, ...] = field(
        default_factory=lambda: (T5Config(),)
    )
```

**效果**：单卡 `sglang serve` 启动 GLM-Image 时不再出现 `'EncoderConfig' object has no attribute 'parallel_folding'`，T5 走定制加载路径，权重正常加载，无 fallback 与 MISSING 告警。

---

## 2. 文档修改

### 2.1 `bug_18077_analysis/NEXT_STEPS.md`

**目的**：在下一步计划中记录单卡 text encoder 问题的现象、原因与修复说明。

**修改内容**：在 “1. ✅ Completed” 与 “2. 🚧 Next: Fix Sequence Parallelism” 之间新增小节 **“1.5 Single-GPU 启动时的 text encoder 问题（A01 报错）— 已修”**，包括：

- 现象：`AttributeError: 'EncoderConfig' object has no attribute 'parallel_folding'`，以及 fallback 后大量 decoder 权重 MISSING。
- 原因：`GlmImagePipelineConfig` 未覆盖 `text_encoder_configs`，沿用基类 `(EncoderConfig(),)`，T5 加载需要 `TextEncoderConfig`（含 `parallel_folding`）。
- 为何 01/03 可能未遇到：同一套加载逻辑，可能因版本或是否真正启动 serve 而表现不同。
- 修复：在 `GlmImagePipelineConfig` 中设置 `text_encoder_configs = (T5Config(),)`，见 `glm_image.py`。

---

## 3. 新增文件

### 3.1 `bug_18077_analysis/CHANGES.md`（本文件）

**目的**：集中记录为 issue #18077 及单卡修复所改动的文件与具体代码，便于复查和提交 PR 时说明。

---

## 4. 未改动的相关文件（参考）

以下文件与问题或后续 Step 2 相关，本次**未做修改**，仅作定位参考：

- `python/sglang/multimodal_gen/runtime/distributed/__init__.py` — `_get_folding_tp_group(config)` 要求 `TextEncoderConfig`。
- `python/sglang/multimodal_gen/configs/models/encoders/base.py` — `EncoderConfig` 与 `TextEncoderConfig` 定义；`TextEncoderConfig` 含 `parallel_folding`。
- `python/sglang/multimodal_gen/configs/pipeline_configs/base.py` — 基类 `text_encoder_configs` 默认 `(EncoderConfig(),)`。

---

*最后更新：与 NEXT_STEPS.md 中 1.5 节及 glm_image.py 修改同步。*
