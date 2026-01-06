# 数据处理

这是用于3D形状和纹理生成的数据处理流程。

**注意事项**：
1. 该实现是我们工业流程的简化版本。
2. 渲染脚本基于[TRELLIS](https://github.com/microsoft/TRELLIS/blob/main/dataset_toolkits/blender_script/render.py)。

## 渲染

### 动机
渲染脚本`render/render.py`主要有三个目的：
1. 使用Blender将复杂的3D格式转换为PLY文件，以便进行进一步处理。
2. 为DiT训练渲染条件图像。
3. 渲染正交图像、PBR材质以及用于纹理生成的条件信号（世界空间法线和位置）。

### 需求
渲染脚本使用Blender 4.1执行。你需要使用Blender的Python安装`opencv`、`OpenEXR`和`Imath`。以下是Macbook上的示例：
```bash
/Applications/Blender.app/Contents/Resources/4.1/python/bin/python3.11 -m pip install OpenEXR Imath opencv-python
```

### 执行
前两个目的可以通过以下单一命令执行：
```bash
$BLENDER_PATH -b -P render/render.py -- \
    --object ${INPUT_FILE} --geo_mode --resolution 512 \
    --output_folder $OUTPUT_FOLDER
```
对于第三个目的，只需移除`--geo_mode`标志。

## 水密网格处理和采样

### 动机
为了学习3DShape2VecSets的SDF表示，我们需要一个水密输入网格。该流程处理原始三角网格，生成三种必要的数据类型：
1. **表面采样** - 编码器的输入点。
2. **体积采样** - 解码器中SDF评估的查询点。
3. **体积SDFs** - VAE训练的地面真实有符号距离值。

### 执行
处理三角网格（OBJ/OFF格式），生成以下内容：
1. 水密网格（`${OUTPUT_NAME}_watertight.obj`）。
2. 表面点采样（`${OUTPUT_NAME}_surface.npz`）。
3. 带有SDF的体积采样（`${OUTPUT_NAME}_sdf.npz`）。

**命令：**
```bash
python3 watertight/watertight_and_sample.py \
    --input_obj ${INPUT_MESH} \
    --output_prefix ${OUTPUT_NAME}
```

### 输出数据格式

#### 1. 表面采样（`${OUTPUT_NAME}_surface.npz`）
包含两个点云数组，以numpy NPZ格式存储：

| 键             | 形状    | 格式   | 描述                     |
|-----------------|----------|----------|---------------------------------|
| `random_surface` | `(N, 6)` | `float16`| 表面上的均匀点采样 |
| `sharp_surface`  | `(M, 6)` | `float16`| 靠近网格锐边的采样   |

#### 2. 体积SDF采样（`${OUTPUT_NAME}_sdf.npz`）
包含三种采样类型，以数组对的形式存储。对于每种类型`${type}`：

| 采样类型     | 点数组         | SDF标签数组     | 形状    | 格式   | 描述             |
|-----------------|----------------------|----------------------|----------|----------|-------------------------|
| `vol`          | `vol_points`        | `vol_label`         | `(P, 3)/(P,)` | `float16`| 随机空间采样 |
| `random_near`   | `random_near_points` | `random_near_label`  | `(Q, 3)/(Q,)` | `float16`| 靠近表面的采样   |
| `sharp_near`    | `sharp_near_points`  | `sharp_near_label`   | `(R, 3)/(R,)` | `float16`| 靠近锐边的采样 |

**数据规格**：
- 所有点坐标（`*_points`数组）包含以`float16`值存储的3D位置。
- 所有SDF值（`*_label`数组）是表示以下内容的`float16`标量：
  - **正值**：在表面外。
  - **负值**：在表面内。
  - **零值**：在表面上。
- 数组维度：
  - `N`、`M`、`P`、`Q`、`R`表示采样数量（因形状而异）。
  - `3`表示XYZ坐标。
  - `6`表示XYZ/法线坐标。
- 所有数组均以未压缩形式存储在numpy的NPZ格式中。

## 整体脚本
修改pipeline.sh里面这4个变量，
1. **INPUT_FILE** 每个3D数据的路径。
2. **OUTPUT_FOLDER** 输出数据集的总路径。
3. **NAME** 每个数据的输出路径命名。
4. **BLENDER_PATH** Blender可执行路径。

然后运行以下脚本：
```bash
bash pipeline.sh
```