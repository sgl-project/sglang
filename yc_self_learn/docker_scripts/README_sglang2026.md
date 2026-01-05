# SGLang 2026 Docker 容器使用指南

这是一个独立的 SGLang 2026 Docker 容器配置，不会影响你现有的其他容器。

## 📋 快速开始

### 1. 启动容器

```powershell
# 进入脚本目录
cd I:\yc_research\sglang\yc_self_learn\docker_scripts

# 启动容器
.\run_sglang2026.ps1 start
```

### 2. 查看状态

```powershell
.\run_sglang2026.ps1 status
```

### 3. 查看日志

```powershell
.\run_sglang2026.ps1 logs
```

### 4. 停止容器

```powershell
.\run_sglang2026.ps1 stop
```

## 🔧 可用命令

| 命令 | 说明 |
|------|------|
| `start` | 启动容器 |
| `stop` | 停止容器 |
| `restart` | 重启容器 |
| `status` | 查看容器状态 |
| `logs` | 查看容器日志（实时） |
| `shell` | 进入容器 shell |

## ⚙️ 配置说明

### 端口配置

- **容器端口**: 30000
- **主机端口**: 30001（避免与现有容器冲突）

访问地址: `http://localhost:30001`

### 数据卷映射

- `I:/yc_research/docker_data/huggingface` → `/root/.cache/huggingface` (HuggingFace 缓存)
- `I:/yc_research/docker_data/models` → `/models` (模型存储)
- `I:/yc_research/docker_data/data` → `/data` (数据文件)
- `I:/yc_research/docker_data/outputs` → `/outputs` (输出文件)

### 修改配置

编辑 `docker-compose-sglang2026.yml` 文件来修改：

1. **更改端口**: 修改 `ports` 部分的 `30001:30000`
2. **更改模型**: 修改 `command` 部分的 `--model-path`
3. **指定 GPU**: 修改 `device_ids` 部分，例如 `["0"]` 只使用第一个 GPU
4. **添加参数**: 在 `command` 部分添加更多启动参数

## 🔍 常见问题

### 1. 端口被占用

如果 30001 端口被占用，修改 `docker-compose-sglang2026.yml` 中的端口映射：

```yaml
ports:
  - "30002:30000"  # 改为其他端口
```

### 2. GPU 不可用

检查 NVIDIA Docker 支持：

```powershell
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi
```

### 3. 容器无法启动

查看详细日志：

```powershell
docker-compose -f docker-compose-sglang2026.yml logs
```

### 4. 需要进入容器调试

```powershell
.\run_sglang2026.ps1 shell
```

或者直接使用：

```powershell
docker exec -it sglang2026 /bin/bash
```

## 📝 测试容器

容器启动后，测试服务是否正常：

```powershell
# 健康检查
curl http://localhost:30001/health

# 或者使用 PowerShell
Invoke-WebRequest -Uri http://localhost:30001/health
```

## 🔄 更新镜像

如果需要更新到最新版本：

```powershell
# 拉取最新镜像
docker pull lmsysorg/sglang:latest

# 重启容器
.\run_sglang2026.ps1 restart
```

## 💡 提示

- 容器名称是 `sglang2026`，不会与现有容器冲突
- 所有数据都保存在 `I:/yc_research/docker_data/` 目录下
- 使用 `docker ps` 可以查看所有运行中的容器
- 使用 `docker stats sglang2026` 可以查看容器资源使用情况

