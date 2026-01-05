# Docker + SGLang 本地开发环境设置指南（Windows）

本指南将帮助你在 Windows 电脑上设置 Docker + SGLang 的开发环境，用于本地训练和推理。

## 📋 前置要求

- Windows 10/11（64位）
- 至少 16GB RAM（推荐 32GB+）
- NVIDIA GPU（可选，但推荐用于加速）
- 至少 50GB 可用磁盘空间

## 第一步：安装 Docker Desktop

### 1.1 下载 Docker Desktop

1. 访问 [Docker Desktop for Windows](https://www.docker.com/products/docker-desktop/)
2. 下载 Docker Desktop Installer
3. 运行安装程序并按照提示完成安装

### 1.2 配置 Docker Desktop

安装完成后：

1. **启动 Docker Desktop**
2. **启用 WSL 2 后端**（推荐）：
   - 打开 Docker Desktop 设置
   - 进入 `General` → 勾选 `Use the WSL 2 based engine`
   - 如果提示安装 WSL 2，按照提示安装

3. **配置资源**（可选但推荐）：
   - 进入 `Resources` → `Advanced`
   - 设置：
     - CPUs: 至少 4 核（根据你的 CPU 调整）
     - Memory: 至少 8GB（推荐 16GB+）
     - Swap: 2GB
     - Disk image size: 至少 64GB

4. **配置 NVIDIA GPU 支持**（如果有 NVIDIA GPU）：
   - 确保已安装 [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker)
   - 在 Docker Desktop 设置中启用 GPU 支持
   - 或者安装 [NVIDIA Docker Support](https://github.com/NVIDIA/nvidia-docker)

### 1.3 验证 Docker 安装

打开 PowerShell 或命令提示符，运行：

```powershell
docker --version
docker run hello-world
```

如果看到 "Hello from Docker!" 消息，说明安装成功。

## 第二步：准备 SGLang Docker 镜像

### 2.1 拉取预构建的镜像（推荐）

```powershell
# 拉取最新的 SGLang 镜像
docker pull lmsysorg/sglang:latest

# 或者拉取开发版本（用于开发）
docker pull lmsysorg/sglang:dev
```

### 2.2 或者从源码构建镜像（可选）

如果你想使用本地代码构建：

```powershell
# 进入 sglang 项目目录
cd I:\yc_research\sglang

# 构建镜像（这会花费较长时间）
docker build -t sglang:local -f docker/Dockerfile .
```

## 第三步：配置本地目录结构

### 3.1 创建必要的目录

在 Windows 上创建以下目录结构：

```
I:\yc_research\
├── sglang\              # SGLang 源码（已存在）
├── docker_data\         # Docker 数据目录
│   ├── huggingface\     # HuggingFace 模型缓存
│   ├── models\          # 本地模型存储
│   ├── data\            # 训练数据
│   └── outputs\         # 训练输出
└── docker_scripts\       # Docker 相关脚本
```

创建这些目录：

```powershell
# 在 PowerShell 中运行
New-Item -ItemType Directory -Force -Path "I:\yc_research\docker_data\huggingface"
New-Item -ItemType Directory -Force -Path "I:\yc_research\docker_data\models"
New-Item -ItemType Directory -Force -Path "I:\yc_research\docker_data\data"
New-Item -ItemType Directory -Force -Path "I:\yc_research\docker_data\outputs"
New-Item -ItemType Directory -Force -Path "I:\yc_research\docker_scripts"
```

### 3.2 获取 HuggingFace Token（如果需要）

如果你要使用私有模型或需要下载模型：

1. 访问 [HuggingFace](https://huggingface.co/settings/tokens)
2. 创建一个 Access Token
3. 保存 token 供后续使用

## 第四步：运行 SGLang 容器

### 4.1 基本运行命令

创建一个 PowerShell 脚本 `I:\yc_research\docker_scripts\run_sglang.ps1`：

```powershell
# run_sglang.ps1
# 设置变量
$HF_TOKEN = "your-huggingface-token-here"  # 替换为你的 token
$MODEL_PATH = "qwen/qwen2.5-0.5b-instruct"  # 或你想要的模型
$PORT = 30000

# 停止并删除旧容器（如果存在）
docker stop sglang_server 2>$null
docker rm sglang_server 2>$null

# 运行容器
docker run -d `
    --name sglang_server `
    --gpus all `
    --shm-size 32g `
    -p ${PORT}:30000 `
    -v I:\yc_research\docker_data\huggingface:/root/.cache/huggingface `
    -v I:\yc_research\docker_data\models:/models `
    -v I:\yc_research\docker_data\data:/data `
    -v I:\yc_research\docker_data\outputs:/outputs `
    -v I:\yc_research\sglang:/sgl-workspace/sglang `
    --env "HF_TOKEN=$HF_TOKEN" `
    --ipc=host `
    lmsysorg/sglang:latest `
    python3 -m sglang.launch_server `
        --model-path $MODEL_PATH `
        --host 0.0.0.0 `
        --port 30000

# 查看日志
docker logs -f sglang_server
```

### 4.2 运行脚本

```powershell
# 编辑脚本，填入你的 HuggingFace token
notepad I:\yc_research\docker_scripts\run_sglang.ps1

# 运行脚本
.\I:\yc_research\docker_scripts\run_sglang.ps1
```

### 4.3 使用 Docker Compose（推荐）

创建 `I:\yc_research\docker_scripts\docker-compose.yml`：

```yaml
version: '3.8'

services:
  sglang:
    image: lmsysorg/sglang:latest
    container_name: sglang_server
    volumes:
      # Windows 路径需要转换为 Docker 路径格式
      - I:\yc_research\docker_data\huggingface:/root/.cache/huggingface
      - I:\yc_research\docker_data\models:/models
      - I:\yc_research\docker_data\data:/data
      - I:\yc_research\docker_data\outputs:/outputs
      - I:\yc_research\sglang:/sgl-workspace/sglang
    restart: unless-stopped
    network_mode: host  # 或使用 ports: ["30000:30000"]
    privileged: true
    environment:
      HF_TOKEN: ${HF_TOKEN}  # 从环境变量读取
      # 或直接写: HF_TOKEN: "your-token-here"
    entrypoint: python3 -m sglang.launch_server
    command: 
      - --model-path
      - qwen/qwen2.5-0.5b-instruct
      - --host
      - 0.0.0.0
      - --port
      - "30000"
    ulimits:
      memlock: -1
      stack: 67108864
    ipc: host
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ["0"]  # 使用第一个 GPU，可以改为 ["all"] 使用所有 GPU
              capabilities: [gpu]
    healthcheck:
      test: ["CMD-SHELL", "curl -f http://localhost:30000/health || exit 1"]
      interval: 30s
      timeout: 10s
      retries: 3
```

使用 Docker Compose 运行：

```powershell
# 设置环境变量（在 PowerShell 中）
$env:HF_TOKEN = "your-huggingface-token-here"

# 进入脚本目录
cd I:\yc_research\docker_scripts

# 启动服务
docker-compose up -d

# 查看日志
docker-compose logs -f

# 停止服务
docker-compose down
```

## 第五步：验证 SGLang 服务

### 5.1 检查容器状态

```powershell
# 查看运行中的容器
docker ps

# 查看容器日志
docker logs sglang_server

# 进入容器（用于调试）
docker exec -it sglang_server /bin/bash
```

### 5.2 测试 API

在 PowerShell 中测试：

```powershell
# 测试健康检查
curl http://localhost:30000/health

# 测试 API（使用 PowerShell 的 Invoke-RestMethod）
$body = @{
    model = "qwen/qwen2.5-0.5b-instruct"
    messages = @(
        @{
            role = "user"
            content = "你好，请介绍一下你自己"
        }
    )
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:30000/v1/chat/completions" `
    -Method Post `
    -ContentType "application/json" `
    -Body $body
```

或使用 Python 测试（在本地，不在容器内）：

```python
# test_sglang.py
import requests

url = "http://localhost:30000/v1/chat/completions"
data = {
    "model": "qwen/qwen2.5-0.5b-instruct",
    "messages": [{"role": "user", "content": "你好"}],
}

response = requests.post(url, json=data)
print(response.json())
```

## 第六步：用于训练场景的配置

### 6.1 开发模式容器（用于训练和开发）

创建一个开发容器，可以修改代码并运行训练：

```powershell
# run_sglang_dev.ps1
docker run -itd `
    --name sglang_dev `
    --gpus all `
    --shm-size 32g `
    -p 30000:30000 `
    -v I:\yc_research\docker_data\huggingface:/root/.cache/huggingface `
    -v I:\yc_research\docker_data\models:/models `
    -v I:\yc_research\docker_data\data:/data `
    -v I:\yc_research\docker_data\outputs:/outputs `
    -v I:\yc_research\sglang:/sgl-workspace/sglang `
    --env "HF_TOKEN=$env:HF_TOKEN" `
    --ipc=host `
    --network=host `
    --privileged `
    lmsysorg/sglang:dev `
    /bin/zsh

# 进入容器
docker exec -it sglang_dev /bin/zsh
```

在容器内：

```bash
# 进入工作目录
cd /sgl-workspace/sglang

# 安装依赖（如果需要）
pip install -e "python[all]"

# 运行训练脚本
python your_training_script.py

# 或启动服务器
python3 -m sglang.launch_server \
    --model-path /models/your-model \
    --host 0.0.0.0 \
    --port 30000
```

### 6.2 从本地调用容器内的 SGLang

在 Windows 本地运行训练脚本，调用容器内的 SGLang：

```python
# train_with_sglang.py
import openai
import requests

# 连接到容器内的 SGLang 服务
SGLANG_URL = "http://localhost:30000/v1"

# 方法 1: 使用 OpenAI 客户端
client = openai.Client(
    base_url=SGLANG_URL,
    api_key="None"
)

# 方法 2: 使用 requests
def call_sglang(prompt):
    response = requests.post(
        f"{SGLANG_URL}/chat/completions",
        json={
            "model": "qwen/qwen2.5-0.5b-instruct",
            "messages": [{"role": "user", "content": prompt}],
        }
    )
    return response.json()

# 你的训练逻辑
def train():
    # 使用 SGLang 生成数据
    result = call_sglang("生成一些训练数据")
    print(result)
    
    # 继续你的训练流程
    # ...

if __name__ == "__main__":
    train()
```

### 6.3 挂载训练数据

确保训练数据在挂载的目录中：

```powershell
# 将训练数据复制到挂载目录
Copy-Item -Path "I:\your\training\data\*" -Destination "I:\yc_research\docker_data\data\" -Recurse
```

在容器内访问：

```bash
# 在容器内
ls /data  # 查看训练数据
```

## 第七步：常用操作

### 7.1 容器管理

```powershell
# 启动容器
docker start sglang_server

# 停止容器
docker stop sglang_server

# 重启容器
docker restart sglang_server

# 删除容器
docker rm sglang_server

# 查看容器资源使用
docker stats sglang_server
```

### 7.2 查看和导出日志

```powershell
# 实时查看日志
docker logs -f sglang_server

# 导出日志到文件
docker logs sglang_server > sglang_logs.txt

# 查看最近 100 行日志
docker logs --tail 100 sglang_server
```

### 7.3 更新镜像

```powershell
# 拉取最新镜像
docker pull lmsysorg/sglang:latest

# 停止并删除旧容器
docker stop sglang_server
docker rm sglang_server

# 使用新镜像重新运行（使用之前的脚本）
.\run_sglang.ps1
```

## 第八步：故障排除

### 8.1 GPU 不可用

如果遇到 GPU 相关错误：

```powershell
# 检查 NVIDIA Docker 支持
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi

# 如果失败，安装 NVIDIA Container Toolkit
# 参考: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
```

### 8.2 端口被占用

```powershell
# 检查端口占用
netstat -ano | findstr :30000

# 更改端口（在运行脚本中修改）
-p 30001:30000  # 使用 30001 端口
```

### 8.3 内存不足

```powershell
# 增加共享内存大小
--shm-size 64g  # 从 32g 增加到 64g

# 在 Docker Desktop 中增加内存分配
# Settings → Resources → Advanced → Memory
```

### 8.4 路径问题（Windows）

Windows 路径在 Docker 中可能需要特殊处理：

```powershell
# 使用正斜杠或双反斜杠
-v I:/yc_research/docker_data/huggingface:/root/.cache/huggingface
# 或
-v I:\\yc_research\\docker_data\\huggingface:/root/.cache/huggingface
```

### 8.5 模型下载失败

```powershell
# 检查 HuggingFace token
docker exec sglang_server env | grep HF_TOKEN

# 手动设置 token
docker exec -it sglang_server bash
export HF_TOKEN="your-token"
```

## 第九步：优化建议

### 9.1 性能优化

1. **使用 SSD**：将 Docker 数据目录放在 SSD 上
2. **增加共享内存**：`--shm-size 64g` 或更大
3. **使用 GPU**：确保正确配置 NVIDIA Docker
4. **网络模式**：使用 `--network host` 减少网络开销

### 9.2 资源管理

```powershell
# 限制容器资源使用
docker update --memory="16g" --cpus="4" sglang_server

# 查看资源使用
docker stats sglang_server
```

## 快速参考脚本

创建一个快速启动脚本 `quick_start.ps1`：

```powershell
# quick_start.ps1
param(
    [string]$Model = "qwen/qwen2.5-0.5b-instruct",
    [int]$Port = 30000
)

$HF_TOKEN = $env:HF_TOKEN
if (-not $HF_TOKEN) {
    Write-Host "请设置环境变量 HF_TOKEN" -ForegroundColor Red
    exit 1
}

Write-Host "启动 SGLang 容器..." -ForegroundColor Green
Write-Host "模型: $Model" -ForegroundColor Yellow
Write-Host "端口: $Port" -ForegroundColor Yellow

docker run -d `
    --name sglang_server `
    --gpus all `
    --shm-size 32g `
    -p ${Port}:30000 `
    -v I:\yc_research\docker_data\huggingface:/root/.cache/huggingface `
    -v I:\yc_research\docker_data\models:/models `
    -v I:\yc_research\docker_data\data:/data `
    -v I:\yc_research\docker_data\outputs:/outputs `
    --env "HF_TOKEN=$HF_TOKEN" `
    --ipc=host `
    lmsysorg/sglang:latest `
    python3 -m sglang.launch_server `
        --model-path $Model `
        --host 0.0.0.0 `
        --port 30000

Write-Host "等待服务启动..." -ForegroundColor Green
Start-Sleep -Seconds 10

Write-Host "查看日志..." -ForegroundColor Green
docker logs -f sglang_server
```

使用：

```powershell
# 设置 token
$env:HF_TOKEN = "your-token"

# 运行
.\quick_start.ps1 -Model "qwen/qwen2.5-0.5b-instruct" -Port 30000
```

## 下一步

1. ✅ 完成 Docker 安装和配置
2. ✅ 拉取/构建 SGLang 镜像
3. ✅ 运行容器并验证服务
4. ✅ 配置训练数据目录
5. ✅ 开始你的训练项目！

祝你使用愉快！🚀

