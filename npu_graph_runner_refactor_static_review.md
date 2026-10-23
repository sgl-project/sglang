# NPUGraphRunner 第一阶段重构与静态完整性检查

范围：只移除普通 Decode / Target Verify 的 NPUGraphRunner；保留 EAGLE draft、draft extend、多层 MTP 的 NPU 专用 Runner。未修改模型、Attention 算子和 MTP 验收算法。

## 实现结果

```text
普通 Decode / Target Verify：
    DecodeCudaGraphRunner
        → NPUCudaGraphBackend / FullCudaGraphBackend

MTP draft / extend：
    仍使用现有 NPU 专用 Runner
        → NPUCudaGraphBackend 的兼容更新入口
```

常规图工厂和 adaptive target 创建均改为公共 Runner。按用户要求，原始 npu_graph_runner.py 暂时原样保留供对照，不接回生产创建入口。公共 Runner 负责 model_context、计时、共享读事件、完整 ShapeKey、输出处理。NPU Backend 保留原来的 update worker、fence 和 update/replay提交顺序。

## 旧方法与新归属

| 原定制 | 新归属 / 检查结论 |
|---|---|
| NPU __init__ / 属性映射 | Backend保存普通decode资源绑定；长度类型保持当前实际list协议，属性选择在_decode_update_attr_name |
| patch_model_npu | Backend.prepare_capture_forward；删除全局monkey-patch，npugraph_ex/fullgraph/dynamic=False语义保留 |
| _create_device_graph / _capture_graph | Backend.capture_one已经负责，删除旧重复入口 |
| _uses_v2_seq_len_update / 两个getter | Backend._decode_update_attr_name，按模型及捕获模式选择actual_seq_kvlen/actual_seq_lengths_kv |
| _update_inputs | 删除无生产调用的旧入口，保留Backend的唯一实际更新实现 |
| _cache_loc_dtype | 公共Runner委托Backend.cache_loc_dtype；在buffer分配前选择Backend，NPU=int32，默认GPU=int64 |
| profiling与收尾 | Backend.capture_profile_context / finish_capture_profile；公共捕获循环及shape tag作用域保留 |
| execute普通/预规划输入准备 | 使用公共load_batch，保留静态输入刷新，不为DSA/V4添加Host长度更新 |
| execute DFlash预规划特例 | Backend.prepare_replay，保留B/DP token、长度、req indices、cache loc、embedding、mRoPE和metadata刷新 |
| Host长度构造 | Backend._decode_update_lengths：CPU mirror优先、DFlash有效长度优先、verify宽度只增加一次 |
| Host更新选择 | Backend.replay：DSA/V4直接replay；普通相关路径内部调用replay_with_input_update |
| graph key与输出裁剪 | 公共Runner，保留变体key、输出扩展字段及PP token宽度裁剪；特殊准备路径额外刷新PP输入 |

## 阶段性接口说明

本阶段采用小范围设备hook：cache_loc_dtype、bind_decode_runner、prepare_replay、prepare_capture_forward、capture_profile_context、finish_capture_profile。公共Backend提供保持原GPU行为的默认实现。

Backend在公共buffer分配前创建，在分配后bind_decode_runner。NPU特殊准备暂时通过绑定的Runner访问既有静态资源；这是第一阶段的窄范围迁移，没有同时实现设计稿中的完整ReplayRequest/Preparation/Context数据结构，也没有将所有Backend改成纯ABC的新接口。后续MTP收敛时应继续收紧资源上下文，避免扩大Backend对Runner内部字段的依赖。

只有公共Decode Runner调用bind_decode_runner。现有专用MTP Runner不绑定普通decode策略，因此其显式Host更新不会被Backend.replay再次执行；保留replay_with_input_update兼容接口，等待后续阶段迁移。

## 已完成检查

- 变更Python文件全部通过AST语法解析。
- git diff --check通过。
- 生产创建入口与迁移后的测试不再导入或实例化普通NPUGraphRunner；旧文件仍保留类定义，仅供对照。
- 审计Backend早期构造读取的资源：当前内置Full/Breakable/NPU构造函数不读取尚未分配的Decode buffers；外部插件仍需遵守这一构造约束。
- MTP专用Runner文件未修改，target adaptive创建改为公共Runner。
- 五项隔离行为检查通过：CPU长度与padding、verify/DFlash防止重复加宽度、V2属性选择、DSA跳过Host更新、DFlash刷新设备长度与KV写入位置。检查执行真实Backend方法，但tensor操作使用轻量替代对象，不是PyTorch/NPU测试。

## 新增/迁移的CPU回归用例

`test/registered/unit/npu/graph_runner/test_npu_graph_runner.py` 已改为测试Backend策略，不再依赖删除的类。包含六个用例；除上述五类外，还覆盖公共replay自动构造更新输入及旧MTP未绑定策略时直接replay的兼容性。

## 尚未验证的边界

当前环境缺少torch、pytest和NPU运行时，未运行真实CPU回归套件、GPU/NPU数值测试或性能测试。不能用AST或轻量行为检查证明异步update/replay顺序、图内存生命周期、通信及多卡同步正确。

设备回归至少包括：普通decode连续长度增长；DSA/V4；V2/IDLE verify；DFlash预规划与非预规划；DP/PP/mRoPE；完整输出字段；现有MTP draft/extend兼容；NPU profiling与torch.compile；GPU Full默认路径。外部Backend插件若依赖构造时已分配的buffer，需要改为在bind阶段使用资源。

## 后续更新：CUDA profiling实现已迁移

CUDA profiler创建、按shape的trace导出、memory history/snapshot及汇总已迁入`runner_backend/capture_profile.py`，由Backend接口调用。Runner只提供CaptureProfileOptions（阶段名、是否按shape、标签读取函数）并管理捕获循环。NPU实现接受同一options协议；XPU原有设备profiler迁入FullXPUGraphBackend以保持其设备活动与导出行为。GPU Backend不再反向调用Runner的profiling私有方法。
