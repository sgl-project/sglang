
用表格的方式，帮我总结下目录下有哪些文件，以及每个文件的作用和功能

我来帮您分析 mooncake-store 目录下的文件结构和功能。让我先探索各个子目录的内容。
现在让我查看一些关键文件的内容来了解具体功能：
基于我对 mooncake-store 目录结构的详细分析，现在为您提供完整的文件总结：
Mooncake Store 完整文件结构总结

| 文件路径 | 类型 | 大小 | 功能描述 |
|----------|------|------|----------|
| 根目录 |
| CMakeLists.txt | 构建配置 | 533B | 主构建文件，配置项目依赖和子目录 |
| src/ 核心源代码 |
| master_service.cpp | 主服务实现 | 20KB | 主服务核心实现，包含对象存储、副本管理、垃圾回收等功能 |
| master_metric_manager.cpp | 指标管理 | 14KB | 性能指标收集和管理，包括吞吐量、延迟等监控数据 |
| allocator.cpp | 分配器实现 | 7.7KB | 内存分配器实现，管理内存段的分配和释放 |
| client.cpp | 客户端实现 | 18KB | 存储客户端实现，提供与主服务的交互接口 |
| master.cpp | 主程序入口 | 3.2KB | 主服务程序入口点，启动和管理服务 |
| master_client.cpp | 主客户端 | 10KB | 主服务客户端实现，用于服务间通信 |
| types.cpp | 类型实现 | 1.6KB | 基础数据类型的实现，包括错误码转换等 |
| utils.cpp | 工具函数 | 1.4KB | 通用工具函数实现 |
| CMakeLists.txt | 构建配置 | 669B | src目录的构建配置 |
| src/vllm/ VLLM集成 |
| distributed_object_store.h | VLLM接口头文件 | 1.7KB | 为VLLM提供的分布式对象存储接口定义 |
| distributed_object_store.cpp | VLLM接口实现 | 6.1KB | VLLM框架的分布式对象存储实现 |
| src/cachelib_memory_allocator/ 内存分配器 |
| MemoryPoolManager.cpp | 内存池管理器 | 4.5KB | 内存池管理器实现，管理多个内存池 |
| MemoryPool.cpp | 内存池实现 | 13KB | 内存池核心实现，提供高效的内存分配 |
| MemoryAllocator.cpp | 内存分配器 | 7.7KB | 内存分配器实现，基于slab分配算法 |
| AllocationClass.cpp | 分配类管理 | 22KB | 分配类管理，支持不同大小的内存块分配 |
| SlabAllocator.cpp | Slab分配器 | 4.9KB | Slab分配器实现，高效的内存块管理 |
| Slab.cpp | Slab实现 | 907B | Slab基础数据结构实现 |
| CMakeLists.txt | 构建配置 | 460B | 内存分配器模块构建配置 |
| include/ 头文件 |
| types.h | 类型定义 | 10KB | 核心数据类型定义，包括错误码、状态枚举、缓冲区描述符等 |
| master_service.h | 主服务接口 | 10KB | 主服务接口定义，包含对象操作、副本管理、垃圾回收等API |
| rpc_service.h | RPC服务接口 | 11KB | RPC服务接口定义，提供网络通信功能 |
| master_metric_manager.h | 指标管理接口 | 5.0KB | 性能指标管理接口定义 |
| client.h | 客户端接口 | 6.3KB | 存储客户端接口定义 |
| master_client.h | 主客户端接口 | 3.4KB | 主服务客户端接口定义 |
| allocator.h | 分配器接口 | 2.6KB | 内存分配器接口定义 |
| allocation_strategy.h | 分配策略接口 | 3.3KB | 内存分配策略接口定义 |
| eviction_strategy.h | 驱逐策略接口 | 3.0KB | 内存驱逐策略接口定义 |
| utils.h | 工具接口 | 391B | 通用工具函数接口定义 |
| include/utils/ |
| scoped_vlog_timer.h | 计时器工具 | 3.0KB | 作用域计时器，用于性能分析和调试 |
| include/cachelib_memory_allocator/ |
| MemoryPool.h | 内存池头文件 | 14KB | 内存池接口定义，包含池管理、分配策略等 |
| MemoryAllocator.h | 内存分配器头文件 | 19KB | 内存分配器接口定义，支持多种分配策略 |
| AllocationClass.h | 分配类头文件 | 19KB | 分配类接口定义，管理不同大小的内存块 |
| MemoryPoolManager.h | 内存池管理器头文件 | 5.3KB | 内存池管理器接口定义 |
| SlabAllocator.h | Slab分配器头文件 | 7.8KB | Slab分配器接口定义 |
| Slab.h | Slab头文件 | 10KB | Slab数据结构接口定义 |
| proto/ 协议定义 |
| master.proto | 协议定义 | 4.0KB | Protocol Buffers协议定义，包含RPC服务、消息格式等 |
| tests/ 测试代码 |
| master_service_test.cpp | 主服务测试 | 42KB | 主服务的综合测试，包括功能测试和压力测试 |
| client_integration_test.cpp | 客户端集成测试 | 12KB | 客户端集成测试，验证客户端与服务端交互 |
| master_metrics_test.cpp | 指标测试 | 6.3KB | 性能指标收集和管理的测试 |
| eviction_strategy_test.cpp | 驱逐策略测试 | 4.0KB | 内存驱逐策略的测试 |
| stress_workload_test.cpp | 压力测试 | 7.2KB | 高负载压力测试 |
| buffer_allocator_test.cpp | 缓冲区分配器测试 | 5.7KB | 内存分配器的测试 |
| test_distributed_object_store.py | Python分布式存储测试 | 14KB | Python实现的分布式对象存储测试 |
| stress_cluster_benchmark.py | 集群压力测试 | 2.7KB | 集群级别的压力测试脚本 |
| distributed_object_store_provider.py | 分布式存储提供者 | 1.0KB | Python分布式存储提供者实现 |
| CMakeLists.txt | 测试构建配置 | 1.5KB | 测试模块的构建配置 |
| tests/scripts/ |
| setup_dev.sh | 开发环境设置脚本 | 542B | 开发环境初始化脚本 |
| tests/conf/ |
| allocator.conf | 分配器配置 | 405B | 内存分配器的配置文件 |