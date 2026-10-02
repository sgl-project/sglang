# PPU 代码模式速查表

## 1. PPU 平台识别模式

### 设备检测（runtime 层）

```python
# python/sglang/kernels/jit/utils/common.py
@cache_once
def is_ppu_runtime() -> bool:
    return bool(torch.cuda.is_available() and "ZW" in torch.cuda.get_device_name())

# python/sglang/srt/utils/common.py
@lru_cache(maxsize=1)
def is_ppu() -> bool:
    return torch.cuda.is_available() and "ZW" in torch.cuda.get_device_name()

@lru_cache(maxsize=1)
def is_cuda_alike():
    return is_cuda() or is_hip() or is_ppu()
```

### 平台发现（platforms 层）

```python
# python/sglang/srt/platforms/ppu.py
@lru_cache(maxsize=1)
def is_ppu_available() -> bool:
    return bool(torch.cuda.is_available() and _ZW_TAG in torch.cuda.get_device_name())

# 在 python/sglang/srt/platforms/__init__.py 的 _resolve_platform() 中:
# 当无 OOT plugin 激活时，PPU 检测优先于 CUDA fallback
if is_ppu_available():
    return PPUSRTPlatform()
```

### PPU 平台类

```python
# python/sglang/srt/platforms/ppu.py
class PPUDeviceMixin(CudaDeviceMixin):
    _enum = PlatformEnum.PPU
    device_name = "ppu"
    # device_type 保持 "cuda"

    def is_cuda(self) -> bool:
        return True  # CUDA-compatible API

class PPUSRTPlatform(PPUDeviceMixin, SRTPlatform):
    def supports_fp8(self) -> bool:
        return self.get_device_capability().to_int() >= 89

    def support_cuda_graph(self) -> bool:
        return True

    def support_piecewise_cuda_graph(self) -> bool:
        return True

    def get_jit_cuda_arch_suffix(self) -> str:
        return "a" if "810" in self.get_device_name() else ""
```

## 2. PPU 代码分支模式

### 条件分支模式（在社区代码中添加 PPU 分支）

```python
# 模式 A: 扩展 is_cuda_alike 检查
if is_cuda_alike():  # 自动包含 PPU
    ...

# 模式 B: 通过 current_platform 检查
from sglang.srt.platforms import current_platform
if current_platform.is_ppu():
    # PPU 专属逻辑
    ...
elif current_platform.is_cuda():
    # CUDA 逻辑
    ...
```

### Import 兼容模式

```python
# PPU 代码中引用社区函数时，确认 import 路径在目标分支仍有效
# 常见迁移:
#   sglang.srt.utils.common → sglang.srt.utils.device (示例)
# 检查方法:
grep -rn "from sglang.srt.utils.common import" python/sglang/srt/
```

## 3. 数据类型适配模式

### FP8 与其他数据类型

```python
# 社区典型 FP8 路径:
if dtype == torch.float8_e4m3fn:
    weight, scale = per_token_cast_to_fp8(...)
    output = fp8_gemm(weight, scale)

# PPU 扩展为多数据类型:
if dtype == torch.float8_e4m3fn:
    weight, scale = per_token_cast_to_fp8(...)
    output = fp8_gemm(weight, scale)
elif dtype == torch.bfloat16:
    output = bf16_gemm(weight)  # PPU 扩展
elif dtype == torch.float16:
    output = fp16_gemm(weight)  # PPU 扩展
```

### supports_fp8 能力检查

```python
# PPU 平台:
def supports_fp8(self) -> bool:
    return self.get_device_capability().to_int() >= 89  # sm89+

# Cherry-pick 时检查社区是否新增了 FP8 能力判断分支
# 搜索:
grep -n "supports_fp8\|support.*fp8" python/sglang/srt/
```

### KV Cache 数据类型

```python
# 社区 kv-cache-dtype auto 决策:
# 未指定时自动推导为 fp8（如有能力）或 None
# PPU 需确认 auto 路径是否正确命中 PPU

# 搜索相关逻辑:
grep -n "kv_cache_dtype.*auto\|kv.cache.*dtype" python/sglang/srt/
```

## 4. Hook 机制模式

### Hook 类型

| 类型 | 行为 | 使用场景 |
|------|------|---------|
| REPLACE | 完全替换社区函数 | PPU 有独立的等价实现 |
| AROUND | 在社区函数前后插入逻辑 | 需在社区逻辑外包裹额外处理 |

### Hook 注册模式

```python
# PPU Hook 文件典型结构 (如 ppu_flashmla_hooks.py)

# --- Hook 1: get_mla_metadata (REPLACE) ---
@hook_register(target="get_mla_metadata", type="REPLACE")
def ppu_get_mla_metadata(*args, **kwargs):
    # PPU 专属实现
    ...

# --- Hook 2: init_cuda_graph_state (AROUND) ---
@hook_register(target="init_cuda_graph_state", type="AROUND")
def ppu_init_cuda_graph_state(original_fn, *args, **kwargs):
    # 前置处理
    result = original_fn(*args, **kwargs)
    # 后置处理 (如 cap buffer)
    return result
```

### Hook 维护检查清单

```
- [ ] 确认被 Hook 替换的社区函数在目标分支中仍存在
- [ ] 确认社区函数签名未变更（参数个数、类型、关键字参数）
- [ ] 如社区函数已自行处理 PPU 场景 → 移除对应 Hook
- [ ] 如社区函数新增了参数/分支 → PPU Hook 需同步处理
- [ ] 更新 Hook 文件的模块文档字符串和 section 注释
- [ ] 更新后续 Hook 编号
```

## 5. JIT 编译适配模式

```python
# python/sglang/kernels/jit/utils/arch.py
@cache_once
def _init_jit_cuda_arch_once():
    ...
    if is_hip_runtime() or is_musa_runtime():
        suffix = ""
    elif is_ppu_runtime():
        from sglang.srt.platforms import current_platform
        suffix = current_platform.get_jit_cuda_arch_suffix()  # PPU: "a" 或 ""
    else:
        suffix = _cuda_arch_suffix(major, minor)
```

## 6. 构建配置模式

### PPU pyproject 变体

```
python/
├── pyproject.toml          # 社区主配置
├── pyproject_ppu.toml      # PPU 变体（PPU 专属依赖）
├── sglang/kernels/aot/
│   ├── pyproject.toml      # AOT kernels 主配置
│   └── pyproject_ppu.toml  # PPU AOT kernels 变体
```

### PPU 专属依赖（典型）

```toml
# pyproject_ppu.toml 中的 PPU 专属依赖
dependencies = [
    ...,
    # PPU 专属包
    "sglang-kernel-ppu",  # PPU kernel 替换包
    ...
]
```

### Cherry-pick 构建配置时的检查

```
- [ ] pyproject_ppu.toml 中的社区依赖版本与目标分支一致
- [ ] PPU 专属依赖版本正确
- [ ] kernels/aot/pyproject_ppu.toml 同步更新
- [ ] 构建: pip install -e python --config-settings pyproject.toml=python/pyproject_ppu.toml
```

## 7. 测试模式

### PPU 平台单元测试

```python
# test/registered/unit/platforms/test_platform_interface.py
class TestPpuDeviceMixin(CustomTestCase):
    def test_ppu_identity_overrides_cuda_identity(self):
        base = PPUSRTPlatform()
        self.assertEqual(base._enum, PlatformEnum.PPU)
        self.assertEqual(base.device_name, "ppu")
        self.assertEqual(base.device_type, "cuda")
        self.assertTrue(base.is_cuda())      # PPU is CUDA-compatible
        self.assertTrue(base.is_ppu())
        self.assertTrue(base.is_cuda_alike())

    def test_ppu_srt_platform_capabilities(self):
        base = PPUSRTPlatform()
        self.assertTrue(base.support_cuda_graph())
        self.assertTrue(base.support_piecewise_cuda_graph())

    @patch("torch.cuda.get_device_capability", return_value=(8, 9))
    def test_ppu_supports_fp8_at_sm89(self, mock):
        base = PPUSRTPlatform()
        self.assertTrue(base.supports_fp8())
```

### 测试目录约定

```
test/registered/unit/platforms/    # PPU 平台单元测试（社区共享目录）
                                    # 不要放在 PPU 专属目录中
```

## 8. 常见陷阱速查

| 陷阱 | 说明 | 解决 |
|------|------|------|
| `lru_cache` vs `torch.compile` | `lru_cache` 在 `torch.compile` 路径上会导致 graph break | 改用 `@cache_once`（`common.py` 已提供） |
| PPU `is_cuda()` = True | 社区 `is_cuda()` 分支会命中 PPU | 确认这些分支对 PPU 行为正确 |
| `device_type = "cuda"` | PPU 设备类型字符串不变 | 不要尝试用 `"ppu"` 作为 device_type |
| tc_piecewise 不兼容 | PPU 环境 `tc_piecewise` 可能与 CUDA graph 冲突 | cherry-pick 时检查 piecewise 相关代码路径 |
| FlashMLA Hook 已清理 | 部分 Hook 已因社区自行处理而删除 | 不要盲目恢复已删除的 Hook |
| Custom Op 注册 | `torch.compile` 追踪时需正确注册 Custom Op | 确认 `torch.library.custom_op` 注册方式 |

## 类成员归属对账（形态 5：方法落错类）

触发场景：源补丁在类 A 体内新增方法，社区基线重构过类 A（方法抽为混入类、
类间插入新类），cherry-pick 三方合并以类体边界作锚点，锚点漂移导致 hunk 被零
冲突塞进相邻类 B。符号存在性/语法/AST 名称解析全部通过，运行时 `instance.m()`
才 AttributeError。

```bash
python .claude/skills/cherry-pick-ppu-patches/assets/audit_class_placement.py <base_ref>
```

输出判读：`MISPLACED/SUSPECT` 行给出「提交 ← 源拷贝 / 文件 :: def 名 / 源所属类
→ 落点类 / 最匹配目标类」。修复方式：rebase 定停在引入提交，把方法移入最匹配
类（以源补丁中的兄弟方法风格为准，如混入类用 `self.value == X.value` 比较），
amend 折入，不留 fix-later 痕迹。

**pick 执行时快速判别**（每个「向类体内新增 def」的 hunk，pick 完立即做，见 SKILL.md 4.4）：

```bash
# 源补丁中新增块上方上下文（应为同类兄弟成员，如 def is_aiter）
git show <源commit> -- <file>
# 落地结果中同一新增块上方上下文（若变成数据字段/无关成员 → 落错类，需移位）
git show HEAD -- <file>
```

决策原则：零冲突只代表文本可融合，不代表语义落点正确；落点错误的处置不是 abandon
整个补丁，而是把新增块移位到语义正确的类体（参照源补丁兄弟成员 + 社区重构模式
定位归属），`git add` 后 `--continue`/`--amend` 完成。


v0.5.19 实例：`is_acext` 定义落在 `DeepEPv2Fp8ScaleFormat`（NamedTuple）体内，
而 17 个兄弟谓词都在 `_MoeRunnerBackendPredicates` 混入类；`w8a8_int8.py:409` 的
`moe_runner_backend.is_acext()` 在 PPU 实机批量触发 AttributeError。全量 65 提交
审计确认此类错位仅此一处（上游基线自身的 `is_intel_xpu` 悬空死代码为同构先例，
不属迁移引入，未改动）。

## server_args 读法对账（形态 6：访问惯用法漂移）

触发场景：源基线在解析结束把声明「物化」回 ServerArgs 字段（materialize_
declarations），目标基线已删除物化——声明只进 stash、投影进 config bag
（get_schedule()/get_disagg()/...）。源上合法的 `server_args.<field>` 原始
直读被 cherry-pick 逐字搬运后，在目标上读到 None/旧值。迁移代码与源逐字节
一致：冲突检查、源文本对齐、AST 名称解析、符号对账全部通过——它们校验的
都是文本/结构保真，缺陷在「读取契约」维度。

判别判据（两级）：
1. 同文件分歧：本提交新增 `server_args.<field>` 直读，而同文件已有
   `get_<bag>().<field>` 访问器读法 → 读法分歧（最强信号是**同函数
   适配不一致**：兄弟字段已改访问器、它没改）
2. 字段属性：server_args.py 中标注 `resolvable=True`（值可能来自模型覆盖/
   后处理 pass 声明）→ 运行期必须走访问器；CLI 直填字段（resolvable=False）
   直读合法

```bash
python .claude/skills/cherry-pick-ppu-patches/assets/audit_serverargs_reads.py <base_ref>
```

输出判读：`DIVERGENT`（字段 resolvable=True + 同文件有访问器惯例）必须人工
甄别为 0；`DIVERGENT-NO-RESOLVE`（CLI 直填字段）为良性，按家规可顺手统一
为访问器但不强制。修复方式：改为字段 NS 归属 bag 的 getter（page_size →
`get_schedule().page_size`、disaggregation_* → `get_disagg().xxx`），
同文件既有访问器读法为参照；解析期内读法（@register_post_process /
resolving_view）是另一套合法惯例，不要改。

v0.5.19 实例：`_estimate_req_to_token_pool_bytes`（kv_cache_configurator.py）
读 `self.server_args.page_size`（None）传入 `page_size % 128` 断言，DSV4+
EAGLE+dp-attention 启动必炸 `TypeError: unsupported operand type(s) for %:
'NoneType' and 'int'`——同函数的 max_mamba_cache_size 已适配 get_schedule()、
唯独 page_size 漏改；one_batch.py 的 disable_overlap_schedule/page_size
同病（bench 判定与真实 worker 分支静默分叉，同文件 508/570 行已有正确惯例）。
源/目标基线的 materialize_declarations 存废差异是元凶。
