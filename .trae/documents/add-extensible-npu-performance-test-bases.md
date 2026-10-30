# Context

[test_npu_performance_utils.py](file:///d:/mysglang/python/sglang/test/ascend/e2e/test_npu_performance_utils.py) 已有单机、混部和 PD 分离基类，但三者的压测参数组装彼此重复，且 `accept_rate` 依赖单机风格的 `other_args`，无法直接用于多节点配置。需要新增一套可扩展基类，在不改变旧类及既有调用方行为的前提下，让后续加入新单机基类的公共能力自动复用于新混部和新 PD 分离基类。

# 实施方案

仅修改 `python/sglang/test/ascend/e2e/test_npu_performance_utils.py`，在现有类之后追加以下三个类，不修改旧类、全局函数或既有测试的导入：

1. 新增 `TestNpuPerformanceTestCaseBaseV2(TestNpuPerformanceTestCaseBase)`。
   - 作为新的单机扩展点，继承原有配置、输出、服务生命周期、`run_throughput` 和指标断言。
   - 后续通用压测/指标能力添加到此类，避免再修改旧类。

2. 新增 `TestNpuPerfMultiNodePdMixTestCaseBaseV2(TestNpuPerformanceTestCaseBaseV2)`。
   - 参考旧混部类实现 master/worker 的 `setUpClass`、节点启动和原有空 `tearDownClass`，不调用旧单机的生命周期方法，避免误启动单机服务。
   - 启动前将 `model_config["model_path"]` 和 `model_config["other_args"]` 分别适配为 `model`、`other_args`；参数列表做拷贝，避免节点启动过程原地修改配置后影响指标读取。
   - `run_throughput` 只保留 `check_role(master/worker)` 门控并调用 `super()`，完整 AISBench/bench-serving 组装统一复用新单机继承链；不重复添加 `@retry()`，防止嵌套重试。

3. 新增 `TestNpuPerfMultiNodePdSepTestCaseBaseV2(TestNpuPerformanceTestCaseBaseV2)`。
   - 参考旧 PD 类实现 router/prefill/decode 的启动、ConfigMap 等待、router 健康检查和进程清理，不调用旧单机生命周期。
   - 启动前适配 `model_config["model_path"]` 为 `model`，并复制 `model_config["decode_args"]` 为 `other_args`。speculative decoding 的 draft token 参数必须来自 decode 服务，不能错误回退到 prefill/router。
   - `run_throughput` 只保留 `check_role(router)` 门控并调用 `super()`，同样不重复 `@retry()`。

该单线继承结构使 `accept_rate` 直接复用现有 `run_bench_serving` 的 `accept_length` 解析和 `assert_metrics` 的断言逻辑：单机读取 `other_args`，混部读取映射后的 `model_config["other_args"]`，PD 读取映射后的 `decode_args`。不扩展 AISBench 的接受率支持，不迁移任何旧用例。

# 验证

- 对目标文件执行 Python 语法编译检查。
- 通过只读 diff 确认三个旧类及其方法未被改动，新代码仅为追加。
- 用无 NPU 的轻量 mock/内省检查验证：新混部和 PD 的直接父类均为新单机类；二者自身覆盖生命周期；多节点 setup 不触发 `popen_launch_server`；角色门控正确；压测仅有一层重试。
- mock `run_bench_serving` 与指标输出，验证三种部署均能按各自参数来源计算并断言 `accept_rate`，且 PD 只使用 `decode_args` 中的 `--speculative-num-draft-tokens`。
