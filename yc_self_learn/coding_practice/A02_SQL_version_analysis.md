# A02 题目 SQL 版本可行性分析

## 题目要求总结

这道 BNSF Data Engineer OA 题目要求：
1. **数据分析统计**（summary_list）
   - (a) 统计 Tax 的 mean/sd/median/min/max（特定筛选条件）
   - (b) 筛选 Space > 800，按 Price 降序排序
   - (c) 计算 Lot >= 80% 分位数的观测数

2. **线性回归**（regression_list）
   - (a) 拟合线性回归模型：Price ~ 所有其他变量
   - (b) 提取模型参数（9个参数：1个截距 + 8个变量系数）
   - (c) 用模型预测新数据

---

## SQL 可行性分析

### ✅ **完全可以用 SQL 做的部分**（summary_list）

#### 1(a) 统计 Tax 的 mean/sd/median/min/max

```sql
-- 筛选条件：Bathroom=2 & Bedroom=4
-- 去掉 Tax 的 NULL（对应 R 的 NA）
-- 计算统计量
SELECT 
    AVG(Tax) as mean_tax,
    STDDEV(Tax) as sd_tax,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY Tax) as median_tax,
    MIN(Tax) as min_tax,
    MAX(Tax) as max_tax
FROM house_prices
WHERE Bathroom = 2 
  AND Bedroom = 4 
  AND Tax IS NOT NULL;
```

**说明**：
- ✅ `AVG()`, `MIN()`, `MAX()`：标准 SQL，所有数据库都支持
- ✅ `STDDEV()` 或 `STDDEV_SAMP()`：大多数数据库支持（MySQL/PostgreSQL/SQL Server）
- ⚠️ `PERCENTILE_CONT()`：需要数据库支持（PostgreSQL 9.4+, SQL Server 2012+, Oracle）
- ⚠️ MySQL 8.0+ 支持窗口函数，但可能需要用 `PERCENT_RANK()` 或其他方式

**MySQL 兼容版本**：
```sql
-- 使用子查询计算中位数
SELECT 
    AVG(Tax) as mean_tax,
    STDDEV(Tax) as sd_tax,
    (SELECT Tax 
     FROM (SELECT Tax, ROW_NUMBER() OVER (ORDER BY Tax) as rn,
                  COUNT(*) OVER() as cnt
           FROM house_prices
           WHERE Bathroom = 2 AND Bedroom = 4 AND Tax IS NOT NULL) t
     WHERE rn IN ((cnt+1)/2, (cnt+2)/2)) as median_tax,
    MIN(Tax) as min_tax,
    MAX(Tax) as max_tax
FROM house_prices
WHERE Bathroom = 2 
  AND Bedroom = 4 
  AND Tax IS NOT NULL;
```

---

#### 1(b) 筛选 Space > 800，按 Price 降序排序

```sql
-- 简单直接的 SQL
SELECT *
FROM house_prices
WHERE Space > 800
ORDER BY Price DESC;
```

**说明**：
- ✅ 标准 SQL，所有数据库都支持
- ✅ 完全可以用 SQL 做

---

#### 1(c) 计算 Lot >= 80% 分位数的观测数

```sql
-- 方法1：使用窗口函数（PostgreSQL, SQL Server, MySQL 8.0+）
WITH lot_stats AS (
    SELECT 
        Lot,
        PERCENTILE_CONT(0.8) WITHIN GROUP (ORDER BY Lot) OVER() as q80
    FROM house_prices
    WHERE Lot IS NOT NULL
)
SELECT COUNT(*) as number_of_observations
FROM lot_stats
WHERE Lot >= q80;
```

**MySQL 兼容版本**：
```sql
-- 使用子查询
WITH q80_value AS (
    SELECT Lot, 
           ROW_NUMBER() OVER (ORDER BY Lot) as rn,
           COUNT(*) OVER() as total_cnt
    FROM house_prices
    WHERE Lot IS NOT NULL
),
q80 AS (
    SELECT Lot as q80
    FROM q80_value
    WHERE rn = CEIL(total_cnt * 0.8)
)
SELECT COUNT(*) as number_of_observations
FROM house_prices h, q80
WHERE h.Lot IS NOT NULL 
  AND h.Lot >= q80.q80;
```

**更简单的版本**（使用 NTILE 或 PERCENT_RANK）：
```sql
WITH ranked AS (
    SELECT 
        Lot,
        PERCENT_RANK() OVER (ORDER BY Lot) as pct_rank
    FROM house_prices
    WHERE Lot IS NOT NULL
)
SELECT COUNT(*) as number_of_observations
FROM ranked
WHERE pct_rank >= 0.8;
```

**说明**：
- ✅ 可以用 SQL 做，但需要数据库支持窗口函数
- ⚠️ MySQL 5.7 及以下版本不支持窗口函数，需要子查询

---

### ⚠️ **部分可以用 SQL 做，但很复杂**（regression_list）

#### 2(a) 拟合线性回归模型

**问题**：SQL 本身不支持机器学习算法，线性回归需要用数学公式手动实现。

**线性回归的数学公式**：
```
Y = β₀ + β₁X₁ + β₂X₂ + ... + βₙXₙ + ε

其中：
- Y 是因变量（Price）
- X₁, X₂, ..., Xₙ 是自变量（Bedroom, Space, Room, Lot, Tax, Bathroom, Garage, Condition）
- β₀ 是截距（Intercept）
- β₁, β₂, ..., βₙ 是系数
- ε 是误差项
```

**多元线性回归的参数估计（最小二乘法）**：
```
β = (X'X)⁻¹X'Y

其中：
- X 是设计矩阵（包含截距列和所有自变量）
- Y 是因变量向量
- β 是参数向量
```

**SQL 实现（非常复杂）**：

```sql
-- 这个示例只是概念性的，实际实现会非常复杂
-- 需要：
-- 1. 构造设计矩阵 X（包含截距列和所有自变量）
-- 2. 计算 X'X（矩阵乘法）
-- 3. 计算 (X'X)⁻¹（矩阵求逆）- SQL 本身不支持
-- 4. 计算 X'Y
-- 5. 计算 β = (X'X)⁻¹X'Y

-- 实际上，这需要：
-- - 使用递归 CTE 或存储过程
-- - 或者使用数据库的扩展功能（如 PostgreSQL 的 madlib）
-- - 或者使用 Python/R 作为存储过程（如 PostgreSQL 的 plpython）
```

**使用数据库扩展**：

1. **PostgreSQL + MADlib**：
```sql
-- 需要安装 MADlib 扩展
SELECT madlib.linregr_train(
    'house_prices',
    'price_regression_model',
    'Price',
    'ARRAY[Bedroom, Space, Room, Lot, Tax, Bathroom, Garage, Condition]'
);
```

2. **SQL Server + R Services**：
```sql
-- 需要安装 SQL Server Machine Learning Services
EXEC sp_execute_external_script
  @language = N'R',
  @script = N'
    model <- lm(Price ~ ., data = InputDataSet)
    OutputDataSet <- as.data.frame(coef(model))
  ',
  @input_data_1 = N'SELECT * FROM house_prices WHERE ...'
```

3. **Oracle + Oracle Machine Learning**：
```sql
-- 需要 Oracle Machine Learning 功能
BEGIN
    DBMS_DATA_MINING.CREATE_MODEL(
        model_name => 'price_model',
        mining_function => DBMS_DATA_MINING.REGRESSION,
        data_table_name => 'house_prices',
        case_id_column_name => 'id',
        target_column_name => 'Price'
    );
END;
```

**说明**：
- ❌ 标准 SQL 不支持矩阵运算（矩阵求逆等）
- ⚠️ 需要数据库扩展（MADlib, R Services, Oracle ML 等）
- ⚠️ 实现非常复杂，不适合 OA 题目

---

#### 2(b) 提取模型参数

**如果已经拟合了模型**（通过扩展功能），可以用 SQL 查询参数：

```sql
-- PostgreSQL + MADlib 示例
SELECT * FROM price_regression_model;
-- 或
SELECT unnest(array['Intercept', 'Bedroom', 'Space', 'Room', 'Lot', 'Tax', 'Bathroom', 'Garage', 'Condition']) as parameter_name,
       unnest(coef) as coefficient
FROM price_regression_model;
```

**说明**：
- ✅ 如果模型已拟合，可以用 SQL 查询
- ❌ 但拟合模型本身在标准 SQL 中很难实现

---

#### 2(c) 用模型预测新数据

**如果模型参数已知**，可以用 SQL 计算预测值：

```sql
-- 假设我们已经有了模型参数（从之前的查询得到）
-- 参数存储在参数表中：regression_params(parameter_name, coefficient)

WITH params AS (
    SELECT parameter_name, coefficient
    FROM regression_params
),
prediction AS (
    SELECT 
        (SELECT coefficient FROM params WHERE parameter_name = 'Intercept') +
        (SELECT coefficient FROM params WHERE parameter_name = 'Bedroom') * 3 +
        (SELECT coefficient FROM params WHERE parameter_name = 'Space') * 1500 +
        (SELECT coefficient FROM params WHERE parameter_name = 'Room') * 8 +
        (SELECT coefficient FROM params WHERE parameter_name = 'Lot') * 40 +
        (SELECT coefficient FROM params WHERE parameter_name = 'Tax') * 1000 +
        (SELECT coefficient FROM params WHERE parameter_name = 'Bathroom') * 2 +
        (SELECT coefficient FROM params WHERE parameter_name = 'Garage') * 1 +
        (SELECT coefficient FROM params WHERE parameter_name = 'Condition') * 0
        as price_prediction
)
SELECT price_prediction FROM prediction;
```

**更简洁的版本**（使用聚合函数）：
```sql
WITH params AS (
    SELECT parameter_name, coefficient
    FROM regression_params
),
new_house AS (
    SELECT 
        3 as Bedroom,
        1500 as Space,
        8 as Room,
        40 as Lot,
        1000 as Tax,
        2 as Bathroom,
        1 as Garage,
        0 as Condition
)
SELECT 
    (SELECT coefficient FROM params WHERE parameter_name = 'Intercept') +
    SUM(CASE 
        WHEN p.parameter_name = 'Bedroom' THEN p.coefficient * n.Bedroom
        WHEN p.parameter_name = 'Space' THEN p.coefficient * n.Space
        WHEN p.parameter_name = 'Room' THEN p.coefficient * n.Room
        WHEN p.parameter_name = 'Lot' THEN p.coefficient * n.Lot
        WHEN p.parameter_name = 'Tax' THEN p.coefficient * n.Tax
        WHEN p.parameter_name = 'Bathroom' THEN p.coefficient * n.Bathroom
        WHEN p.parameter_name = 'Garage' THEN p.coefficient * n.Garage
        WHEN p.parameter_name = 'Condition' THEN p.coefficient * n.Condition
        ELSE 0
    END) as price_prediction
FROM params p
CROSS JOIN new_house n
WHERE p.parameter_name != 'Intercept';
```

**说明**：
- ✅ 如果模型参数已知，可以用 SQL 计算预测值
- ⚠️ 需要先将模型参数存储到数据库中
- ⚠️ 如果模型是用扩展功能拟合的，可以直接调用预测函数

---

## 总结对比

| 任务 | R 语言 | 标准 SQL | SQL + 扩展 | 难度 |
|------|--------|----------|------------|------|
| **1(a) Tax 统计量** | ✅ `mean()`, `sd()`, `median()`, `min()`, `max()` | ✅ 标准聚合函数（中位数需要窗口函数） | ✅ | 简单 |
| **1(b) 筛选排序** | ✅ `filter()`, `arrange()` | ✅ `WHERE`, `ORDER BY` | ✅ | 简单 |
| **1(c) 分位数统计** | ✅ `quantile()` | ⚠️ 需要窗口函数（MySQL 8.0+） | ✅ | 中等 |
| **2(a) 拟合回归** | ✅ `lm(Price ~ ., data)` | ❌ 不支持矩阵运算 | ⚠️ 需要 MADlib/R Services | 困难 |
| **2(b) 提取参数** | ✅ `coef(model)` | ⚠️ 需要先拟合模型 | ✅ 查询模型表 | 中等 |
| **2(c) 预测** | ✅ `predict(model, newdata)` | ⚠️ 需要手动计算公式 | ✅ 如果参数已知 | 中等 |

---

## 实际建议

### ❌ **不推荐用 SQL 做这道题**

**原因**：
1. **题目明确要求 R 语言**：这是 R 语言的 OA 题目
2. **线性回归在标准 SQL 中几乎不可能**：需要矩阵运算，SQL 不支持
3. **即使使用扩展功能，也很复杂**：
   - 需要特定的数据库（PostgreSQL + MADlib）
   - 需要安装额外的扩展
   - 代码会变得非常复杂
4. **时间成本**：用 SQL 做会花费大量时间，而 OA 题目通常有时间限制

### ✅ **但可以部分用 SQL 做数据分析**

如果你对 SQL 很熟悉，可以先在数据库中：
1. 把 CSV 数据导入数据库
2. 用 SQL 完成 **summary_list** 的所有任务
3. 然后导出结果，用 R 完成线性回归部分

**混合方案示例**：
```sql
-- Step 1: 导入数据（假设已导入到 house_prices 表）

-- Step 2: 用 SQL 完成 summary_list
-- (a) Tax 统计量
SELECT AVG(Tax), STDDEV(Tax), PERCENTILE_CONT(0.5)..., MIN(Tax), MAX(Tax)
FROM house_prices
WHERE Bathroom = 2 AND Bedroom = 4 AND Tax IS NOT NULL;

-- (b) 筛选排序
SELECT * FROM house_prices WHERE Space > 800 ORDER BY Price DESC;

-- (c) 分位数统计
SELECT COUNT(*) FROM house_prices 
WHERE Lot >= (SELECT PERCENTILE_CONT(0.8) WITHIN GROUP (ORDER BY Lot) FROM house_prices)
  AND Lot IS NOT NULL;

-- Step 3: 导出数据，用 R 完成线性回归
```

### ✅ **如果题目允许，可以用 SQL 做数据分析部分**

**适用于**：
- 如果题目允许混合语言
- 如果你想展示 SQL 技能
- 如果线性回归部分可以用其他方式（比如用数据库扩展，或导出数据用 Python/R）

---

## 结论

**这道题能不能用 SQL 做？**

- **数据分析部分（summary_list）**：✅ **可以**
  - 标准 SQL 就能完成大部分
  - 需要数据库支持窗口函数（MySQL 8.0+ 或 PostgreSQL）

- **线性回归部分（regression_list）**：❌ **很难**
  - 标准 SQL 不支持矩阵运算
  - 需要数据库扩展功能（MADlib, R Services 等）
  - 实现非常复杂

**建议**：
- 如果题目要求 R 语言，就用 R 做
- 如果题目允许，可以考虑用 SQL 做数据分析部分，用 R/Python 做线性回归部分
- 如果你对 SQL 很熟悉，可以先在数据库中完成数据分析，然后导出结果

**对于 OA 题目**：
- 通常要求使用指定语言（这道题是 R）
- 用 SQL 做可能不符合题目要求
- 建议按照题目要求，用 R 完成所有任务
