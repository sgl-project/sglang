# SQL 刷题模板对比分析：四大支柱 vs 六模板

## 📊 模板对比表

| 模板名称 | 六模板版本 | 四大支柱版本 | 覆盖情况 |
|---------|----------|------------|---------|
| **排名/TopN** | ✅ 3. topN / 排名（row_number / dense_rank） | ✅ 组件 A：排名与 Top N | ✅ 完全一致 |
| **连续性问题** | ✅ 4. 连续段（gaps & islands） | ✅ 组件 B：连续性问题 | ⚠️ 部分重叠，六模板更具体 |
| **累计/滚动** | ✅ 5. rolling / 累计（sum over order by） | ⚠️ 组件 C 的一部分 | ⚠️ 六模板更具体 |
| **条件聚合** | ✅ 2. group by + having（口径题） | ✅ 组件 D：条件聚合 | ⚠️ 四大支柱更强调 CASE WHEN |
| **找缺失数据** | ✅ 1. left join + is null | ❌ **缺失** | ❌ 四大支柱未覆盖 |
| **多表业务口径** | ✅ 6. 多表业务口径（先 dedup/过滤，再 join，再聚合） | ❌ **缺失** | ❌ 四大支柱未覆盖 |
| **时间转换** | ❌ 未明确列出 | ✅ 组件 C：时间维度转换 | ⚠️ 六模板未单独列出 |

---

## 🎯 结论：六模板更全面

### ✅ **六模板覆盖了四大支柱的所有内容，并且补充了两个重要场景**

六模板 **更全面**，因为：

1. **补充了"找缺失数据"场景**（left join + is null）
   - 这是 Data Engineer 面试中非常常见的题目
   - 例如：找出从未下单的用户、找出缺失的日期等

2. **补充了"多表业务口径"场景**（先 dedup/过滤，再 join，再聚合）
   - 这是实际业务中最常见的复杂 SQL 模式
   - 四大支柱过于强调单表操作，忽略了多表组合

3. **更贴近实际面试场景**
   - 六模板是从实际面试题目中抽象出来的
   - 四大支柱更像是理论分类

---

## 📝 推荐：使用六模板 + 补充时间转换

### 推荐的 7 个模板（六模板 + 时间转换）

#### 1. **LEFT JOIN + IS NULL（找缺失）** ⭐⭐⭐⭐⭐
```
典型场景：
- 找出从未下单的用户
- 找出缺失的日期
- 找出没有匹配的记录

模板代码：
SELECT a.*
FROM table_a a
LEFT JOIN table_b b ON a.id = b.id
WHERE b.id IS NULL;
```

#### 2. **GROUP BY + HAVING（口径题）** ⭐⭐⭐⭐⭐
```
典型场景：
- 找出订单数 > 10 的用户
- 找出销售额 > 1000 的月份
- 找出活跃天数 >= 3 的用户

模板代码：
SELECT column1, COUNT(*) as cnt
FROM table
GROUP BY column1
HAVING COUNT(*) > threshold;
```

#### 3. **TOP N / 排名（ROW_NUMBER / DENSE_RANK）** ⭐⭐⭐⭐⭐
```
典型场景：
- 每个部门的前三名
- 排名前 N 的客户
- 第 N 高的薪水

模板代码：
SELECT *
FROM (
    SELECT *, ROW_NUMBER() OVER (PARTITION BY group_col ORDER BY sort_col DESC) as rn
    FROM table
) t
WHERE rn <= N;
```

#### 4. **连续段（Gaps & Islands）** ⭐⭐⭐⭐
```
典型场景：
- 连续 3 天活跃的用户
- 连续登录的用户
- 连续增长的月份

模板代码（方法1：date_sub + row_number）：
SELECT user_id
FROM (
    SELECT 
        user_id,
        date,
        ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY date) as rn,
        DATE_SUB(date, INTERVAL rn DAY) as group_id
    FROM table
) t
GROUP BY user_id, group_id
HAVING COUNT(*) >= N;

模板代码（方法2：LAG + SUM）：
SELECT user_id
FROM (
    SELECT 
        user_id,
        date,
        SUM(CASE WHEN DATEDIFF(date, LAG(date) OVER (PARTITION BY user_id ORDER BY date), DAY) = 1 
                 THEN 0 ELSE 1 END) 
        OVER (PARTITION BY user_id ORDER BY date) as group_id
    FROM table
) t
GROUP BY user_id, group_id
HAVING COUNT(*) >= N;
```

#### 5. **Rolling / 累计（SUM OVER ORDER BY）** ⭐⭐⭐⭐
```
典型场景：
- 累计销售额
- 移动平均
- 累计用户数

模板代码：
SELECT 
    date,
    SUM(amount) OVER (ORDER BY date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) as cumulative_sum,
    AVG(amount) OVER (ORDER BY date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) as moving_avg_7d
FROM table;
```

#### 6. **多表业务口径（先 dedup/过滤，再 join，再聚合）** ⭐⭐⭐⭐⭐
```
典型场景：
- 每个用户的最近一次订单金额
- 每个部门的最高薪水的员工
- 每个类别的最畅销产品

模板代码：
WITH deduped AS (
    SELECT *
    FROM (
        SELECT *, ROW_NUMBER() OVER (PARTITION BY key_col ORDER BY sort_col DESC) as rn
        FROM table1
    ) t
    WHERE rn = 1
),
filtered AS (
    SELECT * FROM table2 WHERE condition
)
SELECT 
    a.category,
    SUM(b.amount) as total_amount
FROM deduped a
JOIN filtered b ON a.id = b.id
GROUP BY a.category;
```

#### 7. **时间维度转换（DATE_FORMAT / DATEDIFF）** ⭐⭐⭐⭐
```
典型场景：
- 次日留存率
- 月环比增长
- 每周活跃用户

模板代码：
SELECT 
    DATE_FORMAT(date, '%Y-%m') as month,
    COUNT(DISTINCT user_id) as active_users,
    COUNT(DISTINCT CASE WHEN next_date = DATE_ADD(date, INTERVAL 1 DAY) THEN user_id END) as retained_users
FROM (
    SELECT 
        user_id,
        date,
        LEAD(date) OVER (PARTITION BY user_id ORDER BY date) as next_date
    FROM table
) t
GROUP BY DATE_FORMAT(date, '%Y-%m');
```

---

## 🔍 详细对比分析

### 模板 1: LEFT JOIN + IS NULL（找缺失）

**四大支柱：❌ 未覆盖**

**重要性：⭐⭐⭐⭐⭐**
- 这是 Data Engineer 面试中非常常见的题目
- 例如：找出从未下单的用户、找出缺失的日期等
- **六模板的优势：补充了这个重要场景**

**典型题目：**
- LeetCode 607: Sales Person（找出没有订单的销售）
- LeetCode 1350: Students With Invalid Departments（找出无效部门的学生）
- 各种"找缺失"的题目

---

### 模板 2: GROUP BY + HAVING（口径题）

**四大支柱：⚠️ 部分覆盖（组件 D 主要强调 CASE WHEN + SUM）**

**重要性：⭐⭐⭐⭐⭐**
- 这是最基础的聚合过滤场景
- 四大支柱的组件 D 更强调"条件聚合"（CASE WHEN），而不是"聚合过滤"（HAVING）
- **六模板的优势：更基础，更常用**

**典型题目：**
- LeetCode 182: Duplicate Emails（找出重复的邮箱）
- LeetCode 586: Customer Placing the Largest Number of Orders
- 各种"找出满足条件的组"的题目

---

### 模板 3: TOP N / 排名（ROW_NUMBER / DENSE_RANK）

**四大支柱：✅ 完全覆盖（组件 A）**

**重要性：⭐⭐⭐⭐⭐**
- 这是窗口函数最经典的应用
- 四大支柱和六模板完全一致
- **两者一致：都是核心模板**

**典型题目：**
- LeetCode 177: Nth Highest Salary
- LeetCode 178: Rank Scores
- LeetCode 185: Department Top Three Salaries

---

### 模板 4: 连续段（Gaps & Islands）

**四大支柱：⚠️ 部分覆盖（组件 B：连续性问题）**

**重要性：⭐⭐⭐⭐**
- 四大支柱的组件 B 更抽象（"连续性问题"）
- 六模板更具体（"Gaps & Islands"），给出了两种具体方法
- **六模板的优势：更具体，给出了两种实现方法**

**典型题目：**
- LeetCode 180: Consecutive Numbers
- LeetCode 601: Human Traffic of Stadium
- LeetCode 1454: Active Users

---

### 模板 5: Rolling / 累计（SUM OVER ORDER BY）

**四大支柱：⚠️ 部分覆盖（组件 C 的一部分）**

**重要性：⭐⭐⭐⭐**
- 四大支柱的组件 C 主要强调"时间转换"（DATE_FORMAT, DATEDIFF）
- 六模板的"Rolling / 累计"更强调"窗口函数的累计功能"
- **六模板的优势：更具体地强调累计和移动窗口**

**典型题目：**
- LeetCode 1097: Game Play Analysis V（累计用户数）
- 各种"累计销售额"、"移动平均"的题目

---

### 模板 6: 多表业务口径（先 dedup/过滤，再 join，再聚合）

**四大支柱：❌ 未覆盖**

**重要性：⭐⭐⭐⭐⭐**
- 这是实际业务中最常见的复杂 SQL 模式
- 四大支柱过于强调单表操作，忽略了多表组合
- **六模板的优势：补充了最重要的实际业务场景**

**典型题目：**
- LeetCode 184: Department Highest Salary（先排名，再 join）
- LeetCode 185: Department Top Three Salaries（先排名，再 join）
- 各种"每个组的最大值/最小值对应的记录"的题目

---

### 模板 7: 时间维度转换（DATE_FORMAT / DATEDIFF）

**四大支柱：✅ 完全覆盖（组件 C）**

**重要性：⭐⭐⭐⭐**
- 四大支柱单独列出了这个组件
- 六模板未单独列出，但可能包含在"Rolling / 累计"中
- **四大支柱的优势：单独列出，更清晰**

**典型题目：**
- LeetCode 1098: Unpopular Books（时间过滤）
- LeetCode 1126: Active Businesses（时间聚合）
- 各种"留存率"、"增长率"的题目

---

## 🎯 最终推荐

### 推荐的 SQL 刷题模板（7 个）

```
1. LEFT JOIN + IS NULL（找缺失）⭐⭐⭐⭐⭐
2. GROUP BY + HAVING（口径题）⭐⭐⭐⭐⭐
3. TOP N / 排名（ROW_NUMBER / DENSE_RANK）⭐⭐⭐⭐⭐
4. 连续段（Gaps & Islands）⭐⭐⭐⭐
5. Rolling / 累计（SUM OVER ORDER BY）⭐⭐⭐⭐
6. 多表业务口径（先 dedup/过滤，再 join，再聚合）⭐⭐⭐⭐⭐
7. 时间维度转换（DATE_FORMAT / DATEDIFF）⭐⭐⭐⭐
```

### 学习建议

1. **优先掌握前 6 个模板**（六模板版本）
   - 这 6 个模板覆盖了 90% 的面试题目
   - 特别是模板 1、2、3、6 是必会的

2. **补充学习模板 7**（时间转换）
   - 虽然六模板未单独列出，但时间转换在 Data Engineer 面试中也很常见
   - 建议作为补充学习

3. **重点练习模板 6**（多表业务口径）
   - 这是最复杂的模板，但也是最实用的
   - 实际业务中的 SQL 大多是这种模式

4. **不要死记硬背**
   - 理解每个模板的核心逻辑
   - 能够灵活组合使用
   - 能够根据题目快速识别应该使用哪个模板

---

## 💡 总结

**六模板更全面，更贴近实际面试场景。**

- ✅ 六模板覆盖了四大支柱的所有内容
- ✅ 六模板补充了两个重要场景（找缺失、多表业务）
- ✅ 六模板更具体，给出了实现方法
- ⚠️ 建议在六模板基础上，补充学习时间转换（模板 7）

**建议：以六模板为主，补充时间转换，形成 7 个核心模板。**
