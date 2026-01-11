/*
LeetCode 177: Nth Highest Salary
难度: ⭐⭐ Medium
标签: Database, Window Function, Function

题目描述:
编写一个 SQL 查询，获取 Employee 表中第 n 高的薪水（Salary）。
如果不存在第 n 高的薪水，那么查询应返回 null。

示例:
输入: Employee 表
+----+--------+
| Id | Salary |
+----+--------+
| 1  | 100    |
| 2  | 200    |
| 3  | 300    |
+----+--------+

调用: getNthHighestSalary(2)
输出:
+------------------------+
| getNthHighestSalary(2) |
+------------------------+
| 200                    |
+------------------------+

核心思路:
1. 创建 MySQL 函数 getNthHighestSalary(N)
2. 使用窗口函数 DENSE_RANK() 计算排名（相同薪水相同排名，且连续）
3. 筛选排名为 N 的薪水
4. 如果不存在，自动返回 null

时间复杂度: O(n log n) - 排序
空间复杂度: O(n) - 窗口函数需要存储排序结果
*/

-- ========================================
-- 标准 SQL 实现（推荐，MySQL 8.0+）
-- ========================================
CREATE FUNCTION getNthHighestSalary(N INT) RETURNS INT
BEGIN
  RETURN (
    SELECT DISTINCT Salary
    FROM (
      SELECT 
        Salary,
        DENSE_RANK() OVER (ORDER BY Salary DESC) AS ranking
      FROM 
        Employee
    ) AS ranked
    WHERE ranking = N
  );
END

-- ========================================
-- 变体 1: 使用 LIMIT 1 确保只返回一个值
-- ========================================
-- CREATE FUNCTION getNthHighestSalary(N INT) RETURNS INT
-- BEGIN
--   RETURN (
--     SELECT Salary
--     FROM (
--       SELECT 
--         Salary,
--         DENSE_RANK() OVER (ORDER BY Salary DESC) AS ranking
--       FROM 
--         Employee
--     ) AS ranked
--     WHERE ranking = N
--     LIMIT 1
--   );
-- END

-- ========================================
-- 变体 2: 如果数据库不支持窗口函数（MySQL 5.7 及以下）
-- ========================================
-- CREATE FUNCTION getNthHighestSalary(N INT) RETURNS INT
-- BEGIN
--   DECLARE result INT;
--   SET result = (
--     SELECT DISTINCT Salary
--     FROM Employee e1
--     WHERE (N - 1) = (
--       SELECT COUNT(DISTINCT e2.Salary)
--       FROM Employee e2
--       WHERE e2.Salary > e1.Salary
--     )
--   );
--   RETURN result;
-- END

-- ========================================
-- 变体 3: 使用 DISTINCT + OFFSET（更简洁但不推荐，逻辑不严谨）
-- ========================================
-- CREATE FUNCTION getNthHighestSalary(N INT) RETURNS INT
-- BEGIN
--   DECLARE M INT;
--   SET M = N - 1;
--   RETURN (
--     SELECT Salary
--     FROM (
--       SELECT DISTINCT Salary
--       FROM Employee
--     ) AS distinct_salaries
--     ORDER BY Salary DESC
--     LIMIT M, 1
--   );
-- END

-- ========================================
-- 常见错误
-- ========================================

-- ❌ 错误 1: 使用 RANK() 而不是 DENSE_RANK()
-- CREATE FUNCTION getNthHighestSalary(N INT) RETURNS INT
-- BEGIN
--   RETURN (
--     SELECT Salary FROM (
--       SELECT Salary, RANK() OVER (ORDER BY Salary DESC) AS ranking
--       FROM Employee
--     ) AS ranked WHERE ranking = N LIMIT 1
--   );
-- END
-- 错误原因: RANK() 会产生不连续的排名，例如 [1,1,3,4]，如果 N=2 会找不到

-- ❌ 错误 2: 使用 ROW_NUMBER() 而不是 DENSE_RANK()
-- 错误原因: ROW_NUMBER() 会给每个记录不同排名，即使薪水相同

-- ❌ 错误 3: 排序方向错误（使用 ASC 而不是 DESC）
-- 错误原因: 题目要求"第 N 高"，应该按 DESC 排序

-- ❌ 错误 4: 函数语法错误（缺少 BEGIN...END）
-- CREATE FUNCTION getNthHighestSalary(N INT) RETURNS INT
-- RETURN (SELECT Salary FROM ...);
-- 错误原因: MySQL 函数必须使用 BEGIN...END 块

-- ❌ 错误 5: 返回类型错误（返回 0 而不是 null）
-- DECLARE result INT DEFAULT 0;
-- 错误原因: 题目要求如果不存在应该返回 null，不是 0

-- ========================================
-- 测试用例
-- ========================================

-- 测试用例 1: 基本测试
-- 输入: [(1,100), (2,200), (3,300)]
-- getNthHighestSalary(1) -> 300
-- getNthHighestSalary(2) -> 200
-- getNthHighestSalary(3) -> 100
-- getNthHighestSalary(4) -> null

-- 测试用例 2: 有相同薪水
-- 输入: [(1,100), (2,200), (3,200), (4,300)]
-- getNthHighestSalary(1) -> 300
-- getNthHighestSalary(2) -> 200  (有两个 200，但只返回一个)
-- getNthHighestSalary(3) -> 100
-- getNthHighestSalary(4) -> null

-- 测试用例 3: 所有薪水都相同
-- 输入: [(1,200), (2,200), (3,200)]
-- getNthHighestSalary(1) -> 200
-- getNthHighestSalary(2) -> null

-- 测试用例 4: 只有一条记录
-- 输入: [(1,300)]
-- getNthHighestSalary(1) -> 300
-- getNthHighestSalary(2) -> null

-- 测试用例 5: 空表
-- 输入: []
-- getNthHighestSalary(1) -> null
