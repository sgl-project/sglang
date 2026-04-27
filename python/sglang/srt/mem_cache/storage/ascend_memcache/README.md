# Ascend MemCache 作为 L3 KV Cache

本文档说明如何为 SGLang HiCache 的 `ascend_memcache` 后端安装 Ascend MemCache 依赖。

## 安装 Ascend MemCache

**方式 1：使用 pip 安装**

```bash
pip install memcache_hybrid
```

**方式 2：从源码安装**

1. 从官方或内部代码仓库拉取 Ascend MemCache 源码。
2. 按仓库提供的构建文档完成编译与安装。
3. 确保在运行 SGLang 的同一 Python 环境中可以导入 `memcache_hybrid`。

安装校验示例：

```bash
git clone https://gitcode.com/Ascend/memcache.git --recursive
```
```bash
mkdir build
cd build
cmake ..
make -j
```
```bash
sudo make install
```
