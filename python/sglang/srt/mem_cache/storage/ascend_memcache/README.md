# MemCache as L3 KV Cache

本文档说明如何将 **Ascend MemCache** 作为 **SGLang HiCache** 的 L3 KV Cache 后端使用。

相关文档：

- [Ascend MemCache Build Guide](https://gitcode.com/Ascend/memcache/blob/master/doc/build.md)
- [Ascend MemCache Config Guide](https://gitcode.com/Ascend/memcache/blob/master/doc/memcache_config.md)
- [Ascend MemCache Python API](https://gitcode.com/Ascend/memcache/blob/master/doc/memcache_python_api.md)
- [SGLang HiCache Design](https://docs.sglang.io/advanced_features/hicache_design.html)

## 关于 MemCache

MemCache 是 Ascend 提供的分布式缓存系统，底层基于 MemFabric，可以提供高性能的分布式内存池。  
在 SGLang HiCache 中，MemCache 可以作为 L3 KV Cache backend，用于存储和复用 KV cache。
`



## 安装 MemCache

#### 直接 clone MemCache

```bash
git clone https://gitcode.com/Ascend/memcache
cd memcache
git clean -xdf
git reset --hard
```
```
git checkout release/1.1
```

#### 拉取第三方库

```bash
git submodule update --recursive --init
git submodule update --remote 3rdparty/memfabric_hybrid
```

#### 安装memfabric
```bash
cd 3rdparty/memfabric_hybrid
git clean -xdf
git reset --hard
```
##### 切换develop分支
```
git checkout develop
```
##### 拉取第三方库
```
git submodule update --recursive --init
```
##### 编译，执行如下命令进行编译，编译成功后，会生成run包在output目录下
```
bash script/build_and_pack_run.sh --build_mode RELEASE --build_python ON --xpu_type NPU --build_test OFF --build_hcom OFF
```
##### run包默认安装根路径为 /usr/local/

安装完成后需要source安装路径下的memfabric_hybrid/set_env.sh

参考安装命令如下（此处以1.0.0版本为例）

```
cd output
bash memfabric_hybrid-1.1.0_linux_aarch64.run
source /usr/local/memfabric_hybrid/set_env.sh
```
查看版本
```
cat /usr/local/memfabric_hybrid/latest/version.info
```
##### 安装whl包
检查是否自动安装
```
pip show memfabric_hybrid
```
若没有，用run包安装
```
pip install /usr/local/memfabric_hybrid/latest/aarch64-linux/wheel/memfabric_hybrid-1.0.0-cp311-cp311-linux_aarch64.whl
```
or 在线安装
```
pip install memfabric_hybrid==1.0.0
```

设置环境变量
```
export LD_LIBRARY_PATH=/usr/local/lib/python3.11/site-packages/memfabric_hybrid/lib/:$LD_LIBRARY_PATH
```




## 编译 MemCache

cd 至memcache文件夹


##### 编译并打包，编译成功后，会在 `output` 目录下生成 run 包

```bash
bash script/build_and_pack_run.sh --build_mode RELEASE
```


##### 编译并运行ut

```bash
bash script/run_ut.sh
```

##### 安装 run 包：

```bash
cd output
bash memcache_hybrid-1.1.0_linux_aarch64.run
source /usr/local/memcache_hybrid/set_env.sh
```

##### 查看版本
```bash
 cat /usr/local/memcache_hybrid/latest/version.info
```

## 部署 MemCache
### 运行Metaservice

##### 配置文件

1. 方式一：环境变量 + 配置文件（兼容原有方式）
```
export MMC_META_CONFIG_PATH=/usr/local/memcache_hybrid/latest/config/mmc-meta.conf

进入python控制台或者编写python脚本如下即可拉起进程：
from memcache_hybrid import MetaService
MetaService.main()
```

2. 方式二（推荐）：Python 直接设置配置
```
from memcache_hybrid import MetaService, MetaConfig

config = MetaConfig()
config.meta_service_url = "tcp://192.168.1.1:5001"
config.config_store_url = "tcp://192.168.1.2:6000"
config.metrics_url = "http://192.168.1.1:8000"
config.ha_enable = False
config.log_level = "info"

MetaService.setup(config)
MetaService.main()
```
3. 方法三：在ascend_memcache添加metaservice_config.json
```
{
    "meta_service_url": "tcp://141.61.39.233:5001",
    "config_store_url": "tcp://141.61.39.233:6000",
    "metrics_url": "http://141.61.39.233:8000",
    "ha_enable": false,
    "log_level": "info"
}
```
然后运行
```
python python/sglang/srt/mem_cache/storage/ascend_memcache/start_meta_service.py --config_path /home/sxy/sglang/python/sglang/srt/mem_cache/storage/ascend_memcache/metaservice_config.json
```

bin形式
```
安装run包即完成了相应二进制的部署
1、设置环境变量
source /usr/local/memcache_hybrid/set_env.sh
source /usr/local/memfabric_hybrid/set_env.sh

2、环境变量 + 配置文件
export MMC_META_CONFIG_PATH=/usr/local/memcache_hybrid/latest/config/mmc-meta.conf
/usr/local/memcache_hybrid/latest/aarch64-linux/bin/mmc_meta_service
```

### 运行Localservice

##### 配置文件

1. 方式一：环境变量 + 配置文件（兼容原有方式）
```
export MMC_LOCAL_CONFIG_PATH=/usr/local/memcache_hybrid/latest/config/mmc-local.conf

通过MemCache提供的接口初始化客户端并拉起localservice，执行数据写入、查询、获取、删除等，下面的脚本是一个示例：
python3 test_mmc_demo.py
```
2. 方式二（推荐）：Python 入口直接指定 local 配置文件路径
```
from memcache_hybrid import DistributedObjectStore, LocalConfig

config = LocalConfig()
config.protocol = "device_rdma"
config.dram_size = "10GB"
config.max_dram_size = "64GB"
print(config)

store = DistributedObjectStore()
assert store.setup(config) == 0, "setup local config failed"
ret = store.setup("/usr/local/memcache_hybrid/latest/config/mmc-local.conf", device_id=0, init_bm=True)
print(ret)
```
3. 方法三：在ascend_memcache添加localservice_config.json
```
{
    "protocol": "host_tcp",
    "dram_size": "10GB",
    "max_dram_size": "64GB",
    "device_id": 0,
    "init_bm": true,
    "conf_file_path": "/usr/local/memcache_hybrid/latest/config/mmc-local.conf"
}
```
运行
```
python python/sglang/srt/mem_cache/storage/ascend_memcache/start_local_store.py \
    --config_path python/sglang/srt/mem_cache/storage/ascend_memcache/localservice_config.json
```


## 运行Ascend_memcache backend



#### Shell 1: 启动 master

```
docker start -ai sglang_a3_service
```
```
source /usr/local/memfabric_hybrid/set_env.sh
source /usr/local/memcache_hybrid/set_env.sh
```
```
cd /home/sxy/sglang
```
```
python python/sglang/srt/mem_cache/storage/ascend_memcache/start_meta_service.py \
  --config_path python/sglang/srt/mem_cache/storage/ascend_memcache/metaservice_config.json
  ```
 
  
#### Shell2: sglang
```
docker exec -it sglang_a3_service bash
```
```
cd /home/sxy/sglang
```
```
source /usr/local/memfabric_hybrid/set_env.sh
source /usr/local/memcache_hybrid/set_env.sh

export SGLANG_HICACHE_MEMCACHE_CONFIG_PATH=python/sglang/srt/mem_cache/storage/ascend_memcache/localservice_config.json
```
添加local store config
```
{
  "meta_service_url": "tcp://127.0.0.1:5001",
  "config_store_url": "tcp://127.0.0.1:6000",
  "log_level": "info",
  "protocol": "host_tcp",
  "dram_size": "10GB",
  "max_dram_size": "64GB",
  "hbm_size": "0",
  "device_id": 0,
  "init_bm": true,
  "check_server": true,
  "metrics_url": "http://127.0.0.1:8000"
}
```
```
export PYTHONPATH=/home/sxy/sglang/python:$PYTHONPATH
python -m sglang.launch_server \
  --model-path /data/atb_testdata/weights/DeepSeek-V2-Lite-Chat \
  --attention-backend ascend \
  --enable-hierarchical-cache \
  --hicache-storage-backend ascend_memcache
```
