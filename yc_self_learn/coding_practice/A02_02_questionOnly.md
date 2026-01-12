# A02: OA 题目 - BNSF Data Engineer

## 任务目标

给你一份芝加哥郊区房价数据（CSV）。你要：
1. 先做一些数据分析统计（summary）
2. 再拟合一个线性回归模型（linear regression），并用模型做一次预测

---

## 数据字段

数据表包含这些变量（可能有缺失值 NA）：

- **Price**：房价（回归的因变量 / label）
- **Bedroom**：卧室数
- **Space**：房屋面积（平方英尺）
- **Room**：房间数
- **Lot**：地块宽度
- **Tax**：年税
- **Bathroom**：卫生间数
- **Garage**：车库停车位数
- **Condition**：房屋状况（1=好，0=其他/差）

⚠️ **注意**：有些列会有缺失值，你必须正确处理（比如计算统计量前先过滤 NA）。

---

## 你需要写的函数

写一个函数：

```r
analyse_and_fit_lrm <- function(file_path) {
    ...
}
```

**输入**：`file_path`（CSV 路径）

**输出**：一个命名 list，结构必须完全符合下面要求（名字和顺序都要一致）

---

## 输出结构（必须严格匹配）

### A) summary_list（命名 list，长度=3）

包含 3 个元素：

#### 1. statistics
- 一个长度为 5 的 numeric vector，顺序是：**mean, sd, median, min, max**
- 统计对象：满足 **Bathroom=2 且 Bedroom=4** 的房子，它们的 **Tax**
- 计算时要先去掉 Tax 的 NA
- 不需要给 vector 的元素命名（names 可为空）

#### 2. data_frame
- 一个 data frame
- 筛选条件：**Space > 800**
- 排序：按 **Price 降序**（decreasing）

#### 3. number_of_observations
- 一个 numeric（计数）
- 统计：**Lot >= Lot 的第 4 个 5-quantile** 的样本数量
- "第 4 个 5-quantile"意思是：把分位点按 0%,20%,40%,60%,80%,100% 分成 5 段，第4个阈值=**80%分位数（q80）**
- 所以这里就是数 **Lot >= q80** 的数量（Lot 的 NA 要处理好）

---

### B) regression_list（命名 list，长度=2）

包含 2 个元素：

#### 1. model_parameters
- 一个长度为 9 的 numeric vector（线性回归参数）
- 模型：用线性回归描述 **Price ~ 所有其他变量**（predictors）
- 参数向量第一个元素必须命名为 **Intercept**
- 其余参数名要和对应变量名一致（Bedroom/Space/...）

#### 2. price_prediction
- 一个 numeric（预测值）
- 用上面拟合的模型，对以下这套"新房子特征"做预测：

| 变量 | 值 |
|------|-----|
| Bedroom | 3 |
| Space | 1500 |
| Room | 8 |
| Lot | 40 |
| Tax | 1000 |
| Bathroom | 2 |
| Garage | 1 |
| Condition | 0（差/非好） |

---

## 允许使用的工具

你可以用 base R 也可以用 tidyverse 里的任何包（比如 dplyr）

---

## 提示（题目要求）

- 不要在文件里手动调用 `analyse_and_fit_lrm()`（单元测试会自动调用）
- 最终一定要 return 题目要求的那个 list（结构/名字/顺序都要对）
