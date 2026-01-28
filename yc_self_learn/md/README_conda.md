# SGLang 2026 Conda 环境使用指南

## 快速开始

### 1. 创建环境

运行环境设置脚本：

```powershell
cd I:\yc_research\sglang\yc_self_learn
.\setup_conda_env.ps1
```

这个脚本会：
- ✅ 检查 Conda 是否安装
- ✅ 创建名为 `sglang2026` 的新环境
- ✅ 安装 Python 3.10
- ✅ 安装基础依赖（pip, requests, openai 等）
- ✅ 可选安装 SGLang

### 2. 激活环境

**方法 1：使用 Conda 命令**
```powershell
conda activate sglang2026
```

**方法 2：使用便捷脚本**
```powershell
. .\activate_sglang2026.ps1
```

### 3. 验证环境

```powershell
# 检查 Python 版本
python --version

# 查看已安装的包
pip list

# 测试导入
python -c "import requests; print('requests 可用')"
```

## 环境说明

### 环境名称
- **环境名**: `sglang2026`
- **Python 版本**: 3.10

### 已安装的基础包
- `pip` - Python 包管理器
- `setuptools` - 构建工具
- `wheel` - 打包工具
- `requests` - HTTP 库
- `openai` - OpenAI API 客户端（用于调用 SGLang API）

### 可选安装

#### 如果使用 Docker 运行 SGLang（推荐）
只需要安装客户端库：
```powershell
pip install sglang[openai]
```

#### 如果本地运行 SGLang（需要 GPU）
安装完整版本：
```powershell
pip install "sglang[all]>=0.5.3rc0"
```

## 常用命令

### 环境管理
```powershell
# 激活环境
conda activate sglang2026

# 退出环境
conda deactivate

# 查看所有环境
conda env list

# 删除环境（如果需要）
conda env remove -n sglang2026
```

### 包管理
```powershell
# 查看已安装的包
pip list

# 安装新包
pip install <package-name>

# 安装特定版本
pip install <package-name>==<version>

# 卸载包
pip uninstall <package-name>

# 导出环境配置
pip freeze > requirements.txt

# 从配置文件安装
pip install -r requirements.txt
```

### 更新包
```powershell
# 更新 pip
pip install --upgrade pip

# 更新特定包
pip install --upgrade <package-name>

# 更新所有包（谨慎使用）
pip list --outdated | ForEach-Object { pip install --upgrade $_.Split()[0] }
```

## 使用场景

### 场景 1: 使用 Docker + 本地调用

如果你使用 Docker 运行 SGLang，本地环境只需要客户端库：

```powershell
# 激活环境
conda activate sglang2026

# 安装客户端库
pip install sglang[openai] requests

# 运行你的脚本
python your_script.py
```

### 场景 2: 本地开发（需要 GPU）

如果你要在本地运行 SGLang：

```powershell
# 激活环境
conda activate sglang2026

# 安装完整版本
pip install "sglang[all]>=0.5.3rc0"

# 运行 SGLang
python -m sglang.launch_server --model-path <model-path>
```

### 场景 3: 训练脚本

用于运行训练脚本：

```powershell
# 激活环境
conda activate sglang2026

# 安装训练相关包
pip install torch transformers datasets

# 运行训练
python train.py
```

## 项目结构建议

```
yc_self_learn/
├── setup_conda_env.ps1          # 环境创建脚本
├── activate_sglang2026.ps1      # 环境激活脚本
├── requirements.txt             # 项目依赖（可选）
├── train/                       # 训练脚本目录
│   └── train.py
├── scripts/                     # 其他脚本
│   └── call_sglang.py
└── data/                        # 数据目录
```

## 创建 requirements.txt

导出当前环境的依赖：

```powershell
conda activate sglang2026
pip freeze > requirements.txt
```

安装依赖：

```powershell
conda activate sglang2026
pip install -r requirements.txt
```

## 常见问题

### Q1: 环境激活失败
**A:** 
- 确保已运行 `setup_conda_env.ps1` 创建环境
- 检查环境是否存在：`conda env list`
- 尝试重新创建环境

### Q2: 包安装失败
**A:**
- 更新 pip: `pip install --upgrade pip`
- 检查网络连接
- 使用国内镜像（如果需要）：
  ```powershell
  pip install -i https://pypi.tuna.tsinghua.edu.cn/simple <package>
  ```

### Q3: Python 版本不兼容
**A:**
- 检查 Python 版本：`python --version`
- 如果需要，重新创建环境并指定版本：
  ```powershell
  conda create -n sglang2026 python=3.11 -y
  ```

### Q4: 环境占用空间太大
**A:**
- 清理 pip 缓存：`pip cache purge`
- 删除不需要的包
- 使用 `conda clean` 清理

## 下一步

环境配置完成后：

1. **测试环境**
   ```powershell
   conda activate sglang2026
   python -c "import sys; print(sys.version)"
   ```

2. **安装需要的包**
   ```powershell
   pip install <your-packages>
   ```

3. **开始开发**
   - 编写训练脚本
   - 调用 SGLang API
   - 运行实验

祝你使用愉快！🎉

