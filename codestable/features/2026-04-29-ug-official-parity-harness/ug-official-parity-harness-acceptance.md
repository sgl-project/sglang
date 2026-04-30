# ug-official-parity-harness 验收报告

> 阶段：阶段 3（验收闭环）
> 验收日期：2026-04-30
> 关联方案 doc：`codestable/features/2026-04-29-ug-official-parity-harness/ug-official-parity-harness-design.md`

## 1. 接口契约核对

- [x] `UGParityCase`：已实现 case id、task、prompt/messages、image path、seed、sampling params、dump points、metadata，并限制 task 为 `vlm`、`text_to_image`、`image_edit`、`interleaved`。落点：`python/sglang/srt/ug/parity.py`。
- [x] `UGParityArtifact`：已实现 text、image summary、tensor summary、debug counters、metadata、error 和 JSON roundtrip。落点：`python/sglang/srt/ug/parity.py`。
- [x] `UGParityReport`/comparator：已实现 pass/fail、reference/candidate runner、字段级 diff，覆盖 text/image/tensor/error 的比较。落点：`python/sglang/srt/ug/parity.py`。
- [x] Opt-in live entry：已实现 env 缺失 skip，env 齐全时写出 `case.json`、`reference.official.json`、`candidate.sglang.json`、`report.json`。落点：`test/registered/scheduler/test_bagel_official_parity_harness.py`。

## 2. 行为与决策核对

- [x] Official reference runner 和 SGLang candidate runner 通过 JSON artifact 解耦，live harness 用 subprocess 隔离 official repo 的 `sys.path`。
- [x] `python/sglang/srt/ug/parity.py` 只包含 schema、summary、comparator、runner protocol，不加载官方 BAGEL 仓库函数。
- [x] 第一版 harness 没有承诺 VLM/T2I/Edit 的最终阈值；真实结果对齐留给 Phase 2。
- [x] 图像和 tensor 默认写摘要：图像记录 sha256/尺寸/mode，tensor 记录 shape/dtype/numel/stats/sha256，不默认 dump 巨量张量。
- [x] 挂载点均在方案清单内：`python/sglang/srt/ug/parity.py`、CPU unit test、opt-in registered test、roadmap item。没有额外 runtime 挂载点。

## 3. 测试约束核对

- [x] `python3 -m py_compile python/sglang/srt/ug/parity.py python/sglang/multimodal_gen/test/unit/test_ug_official_parity.py test/registered/scheduler/test_bagel_official_parity_harness.py`：通过。
- [x] `python3 -m ruff check python/sglang/srt/ug/parity.py python/sglang/multimodal_gen/test/unit/test_ug_official_parity.py test/registered/scheduler/test_bagel_official_parity_harness.py`：通过。
- [x] 远端容器 CPU 单测：`python3 python/sglang/multimodal_gen/test/unit/test_ug_official_parity.py`，7 tests OK。
- [x] 远端容器 opt-in live harness 缺 env：skip OK，不触发 official import。
- [x] 远端容器 opt-in live harness 带 env：`SGLANG_TEST_BAGEL_OFFICIAL_REPO=/data/BAGEL`、`SGLANG_TEST_BAGEL_QWEN2_MOT_MODEL=/data/models/BAGEL-7B-MoT`、`SGLANG_TEST_BAGEL_PARITY_OUTPUT=/tmp/ug-parity-phase1-codex-20260430`，1 test OK，`report.json` passed=true。
- [x] bfloat16 tensor summary smoke：远端容器可稳定记录 `torch.bfloat16` dtype。
- [x] `git diff --check`：通过。

Phase 1 live 验证产物：

- `/tmp/ug-parity-phase1-codex-20260430/case.json`
- `/tmp/ug-parity-phase1-codex-20260430/reference.official.json`
- `/tmp/ug-parity-phase1-codex-20260430/candidate.sglang.json`
- `/tmp/ug-parity-phase1-codex-20260430/report.json`

`report.json` 核心结果：

```json
{
  "candidate_runner": "sglang",
  "case_id": "bagel-parity-harness-probe",
  "differences": [],
  "passed": true,
  "reference_runner": "official"
}
```

## 4. 术语一致性

- [x] `UGParityCase`、`UGParityArtifact`、`UGParityReport`、`UGTensorSummary`、`UGImageSummary` 命名均集中在 `python/sglang/srt/ug/parity.py` 和对应测试中。
- [x] `Official reference runner`/`SGLang candidate runner` 在设计文档、架构文档、测试入口里的含义一致。
- [x] Import firewall 已覆盖官方 BAGEL import hook：`SGLANG_TEST_BAGEL_OFFICIAL_REPO`、`modeling.bagel`、`modeling.autoencoder`、`modeling.qwen2`、`data.data_utils`、`data.transforms`、`InterleaveInferencer`、`_build_official_bagel_inferencer` 不允许进入 `python/sglang/**` runtime。

## 5. 架构归并

- [x] 新增 `codestable/architecture/ug-runtime.md`，归并 Phase 1 稳定架构事实。
- [x] 名词归并：`UGParityCase`、`UGParityArtifact`、`UGParityReport`、official reference runner、SGLang candidate runner。
- [x] 动词骨架归并：case JSON -> official subprocess / SGLang subprocess -> artifacts -> comparator -> report。
- [x] 跨层纪律归并：官方 BAGEL/seed 代码只能在 opt-in harness 或外部对照脚本里出现，不能进入 `python/sglang/**` runtime import 链。

## 6. requirement 回写

- [x] 本 feature 不新增用户可感产品能力，frontmatter `requirement` 为空，无 requirement 回写。

## 7. roadmap 回写

- [x] `codestable/roadmap/ug-official-alignment/ug-official-alignment-items.yaml` 中 `ug-official-parity-harness` 已从 `in-progress` 改为 `done`。
- [x] `codestable/roadmap/ug-official-alignment/ug-official-alignment-roadmap.md` 子 feature 清单已同步为 `done`。
- [x] Phase 1 结束条件满足：`ug-official-parity-harness` 通过 acceptance 并在 items.yaml 标为 `done`。

## 8. 遗留与下一阶段

Phase 1 到这里结束，但它只证明 harness 本身可靠。它不证明 VLM、生图、编辑结果已经对齐官方 BAGEL。

下一阶段进入 Phase 2。第一条建议启动 `ug-vlm-official-parity`：用同 checkpoint、同 seed、同输入图片/文本，通过 Phase 1 harness 对齐 official BAGEL 与 SGLang UG 的 VLM 文本或 logits 结果。这是后续文生图、编辑和 interleaved 的最小结果闭环。
