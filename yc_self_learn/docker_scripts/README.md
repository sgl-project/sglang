# Docker 脚本使用说明

这个目录包含用于管理 SGLang Docker 容器的实用脚本。

## 文件说明

- `run_sglang.ps1` - 启动/停止/管理 SGLang 容器
- `docker-compose.yml` - Docker Compose 配置文件
- `test_sglang.ps1` - 测试 SGLang 服务是否正常运行

## 快速开始

### 1. 设置环境变量

在 PowerShell 中设置 HuggingFace token：

```powershell
$env:HF_TOKEN = "your-huggingface-token-here"
```

### 2. 启动容器

使用 PowerShell 脚本：

```powershell
# 基本启动
.\run_sglang.ps1

# 指定模型和端口
.\run_sglang.ps1 -Model "meta-llama/Llama-3.1-8B-Instruct" -Port 30001

# 使用开发镜像（支持代码修改）
.\run_sglang.ps1 -Dev
```

或使用 Docker Compose：

```powershell
# 设置环境变量
$env:HF_TOKEN = "your-token"

# 启动
docker-compose up -d

# 查看日志
docker-compose logs -f

# 停止
docker-compose down
```

### 3. 测试服务

```powershell
.\test_sglang.ps1
```

## 常用命令

### 查看日志

```powershell
# 使用脚本
.\run_sglang.ps1 -Logs

# 或直接使用 docker
docker logs -f sglang_server
```

### 进入容器

```powershell
# 使用脚本
.\run_sglang.ps1 -Shell

# 或直接使用 docker
docker exec -it sglang_server /bin/bash
```

### 停止容器

```powershell
# 使用脚本
.\run_sglang.ps1 -Stop

# 或直接使用 docker
docker stop sglang_server
docker rm sglang_server
```

## 路径配置

默认路径配置（在 `run_sglang.ps1` 中）：

- HuggingFace 缓存: `I:\yc_research\docker_data\huggingface`
- 模型目录: `I:\yc_research\docker_data\models`
- 数据目录: `I:\yc_research\docker_data\data`
- 输出目录: `I:\yc_research\docker_data\outputs`
- SGLang 源码: `I:\yc_research\sglang`

如需修改，请编辑脚本中的 `$BaseDir` 变量。

## 故障排除

### 端口被占用

```powershell
# 检查端口占用
netstat -ano | findstr :30000

# 使用其他端口
.\run_sglang.ps1 -Port 30001
```

### GPU 不可用

```powershell
# 检查 NVIDIA Docker 支持
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi
```

### 容器启动失败

```powershell
# 查看详细日志
docker logs sglang_server

# 检查容器状态
docker ps -a
```

## 更多信息

参考主文档：`Docker_SGLang_本地开发环境设置.md`

