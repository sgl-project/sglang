# Window Functions 窗口函数刷题集合

这个目录包含了经典的窗口函数相关 LeetCode 题目，每道题都包含：
- 题目描述和示例
- 核心思路分析
- 多种解法（包括最优解和对比解法）
- 常见错误分析
- 测试用例

## 📚 题目列表

### 窗口函数基础题（Medium）

1. **01_rank_scores.sql** - Rank Scores (分数排名)
   - 难度: ⭐⭐ Medium
   - 核心: DENSE_RANK() 窗口函数
   - 时间复杂度: O(n log n)
   - 知识点: 连续排名 vs 非连续排名

2. **02_nth_highest_salary.sql** - Nth Highest Salary (第 N 高的薪水)
   - 难度: ⭐⭐ Medium
   - 核心: DENSE_RANK() 窗口函数 + MySQL 函数定义
   - 时间复杂度: O(n log n)
   - 知识点: 窗口函数在函数中的使用

## 🎯 刷题建议

### 优先级排序

1. **必刷（核心基础）**：
   - Rank Scores - 理解窗口函数的基本用法
   - Nth Highest Salary - 学习如何在函数中使用窗口函数

### 核心技巧总结

1. **窗口函数的选择**：
   - `ROW_NUMBER()`: 行号，即使值相同也不同（1,2,3,4,5）
   - `RANK()`: 相同值相同排名，但排名不连续（1,1,3,4,4,6）
   - `DENSE_RANK()`: 相同值相同排名，且排名连续（1,1,2,3,3,4）✅

2. **窗口函数语法**：
   ```sql
   窗口函数名() OVER (
     [PARTITION BY 列名]  -- 可选：分组
     [ORDER BY 列名 [ASC|DESC]]  -- 可选：排序
   )
   ```

3. **MySQL 函数定义**：
   ```sql
   CREATE FUNCTION function_name(param TYPE) RETURNS TYPE
   BEGIN
     RETURN (子查询);
   END
   ```

4. **保留关键字处理**：
   - MySQL 中 `Rank` 是保留关键字，需要用反引号包裹：`` `Rank` ``
   - 其他数据库可能不需要

5. **NULL 值处理**：
   - 如果子查询没有结果，MySQL 函数会自动返回 NULL
   - 不需要显式处理 NULL 情况

## 🚀 运行方式

### MySQL 8.0+（推荐，支持窗口函数）

```sql
-- 直接运行查询
SOURCE 01_rank_scores.sql;

-- 创建函数并调用
SOURCE 02_nth_highest_salary.sql;
SELECT getNthHighestSalary(2);
```

### MySQL 5.7 及以下（不支持窗口函数）

需要使用变体方法（子查询实现），在文件中已提供注释掉的代码。

### 其他数据库

- **PostgreSQL**: 语法类似，但函数定义略有不同
- **SQL Server**: 函数参数使用 `@N` 而不是 `N`
- **Oracle**: 使用 PL/SQL 语法
- **SQLite 3.25.0+**: 支持窗口函数

## 📝 学习笔记

### 窗口函数 vs 聚合函数

- **聚合函数**：将多行合并为一行（GROUP BY）
- **窗口函数**：为每一行计算值，但保留所有行

### 三种排名函数的区别示例

假设有分数：[100, 100, 90, 90, 80]

```sql
-- ROW_NUMBER()
-- 结果: 1, 2, 3, 4, 5 (每行都不同)

-- RANK()
-- 结果: 1, 1, 3, 3, 5 (相同值相同排名，但会跳过)

-- DENSE_RANK() ✅ 大多数题目使用这个
-- 结果: 1, 1, 2, 2, 3 (相同值相同排名，且连续)
```

### 常见陷阱

1. **混淆 RANK() 和 DENSE_RANK()**
   - 题目通常要求连续排名，应该用 `DENSE_RANK()`
   
2. **忘记外层 ORDER BY**
   - 窗口函数中的 ORDER BY 只用于计算，不保证结果顺序
   - 外层查询也需要 ORDER BY 来排序结果

3. **MySQL 保留关键字**
   - `Rank` 是保留关键字，需要用反引号包裹

4. **函数返回值类型**
   - 如果不存在应该返回 `NULL`，不是 `0` 或空字符串

## 🔗 相关资源

- [LeetCode 数据库专题](https://leetcode.com/problemset/database/)
- [MySQL 窗口函数文档](https://dev.mysql.com/doc/refman/8.0/en/window-functions.html)
- [SQL Window Functions 详解](https://www.postgresql.org/docs/current/tutorial-window.html)

## 📌 下一步学习

掌握窗口函数后，可以继续学习：

1. **高级窗口函数**：
   - `LAG()` / `LEAD()` - 前后行数据
   - `FIRST_VALUE()` / `LAST_VALUE()` - 第一/最后值
   - `SUM() OVER()` - 累积求和

2. **PARTITION BY 用法**：
   - 分组计算窗口函数
   - 例如：每个部门的最高薪水

3. **性能优化**：
   - 窗口函数的索引优化
   - 大数据集的性能考虑
