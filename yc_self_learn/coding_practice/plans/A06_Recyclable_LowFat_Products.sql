-- ========================================
-- A06: OA 题目 - Recyclable and Low Fat Products（LeetCode 1757）
-- 任务：SQL 条件筛选
-- ========================================

-- ========================================
-- 【第一部分】题干完整复述（只讲题目要你干什么）
-- ========================================

-- 题目背景：
-- 你有一个 Products 表，包含产品信息。
-- 每个产品有 product_id（产品ID）、low_fats（是否低脂）、recyclable（是否可回收）等属性。

-- 输入：
--   表名：Products
--   列：
--     - product_id: 产品ID（主键）
--     - low_fats: 是否低脂（'Y' 表示是，'N' 表示否）
--     - recyclable: 是否可回收（'Y' 表示是，'N' 表示否）

-- 你必须做的事：
--   找出既 low_fats = 'Y' 又 recyclable = 'Y' 的产品
--   返回这些产品的 product_id

-- 输出：
--   返回一个结果集，包含满足条件的 product_id
--   结果可以按任意顺序排列（题目通常不要求排序）

-- ========================================
-- 【第二部分】如何分析题干（怎么避免被单测卡）
-- ========================================

-- 你读这种题，建议按 3 层扫描：

-- 第一层：锁死"表名 + 列名 + 返回列"
--   - 表名必须对：Products（注意大小写，有些数据库区分大小写）
--   - 列名必须对：product_id, low_fats, recyclable（注意下划线）
--   - 返回列必须对：只返回 product_id（不是所有列）
--   - 条件值必须对：'Y'（是字符串，不是布尔值 True）

-- 第二层：把题目要求变成"筛选条件"
--   题干其实就是让你做条件筛选：
--     - 条件1：low_fats = 'Y'
--     - 条件2：recyclable = 'Y'
--     - 两个条件必须同时满足（AND，不是 OR）
--   
--   这种题单测经常会检查：
--     - 条件是否正确（AND 不是 OR）
--     - 值是否正确（'Y' 不是 'y' 或 True）
--     - 是否返回了正确的列（只返回 product_id）

-- 第三层：提取"隐含的测试点"（最容易错的地方）
--   1. 条件连接：
--      - 必须用 AND（两个条件都要满足）
--      - 不能用 OR（只要满足一个就行，这是错的）
--   
--   2. 值的大小写：
--      - 题目明确写的是 'Y'（大写），不是 'y'（小写）
--      - 如果数据中有大小写混用，可能需要用 UPPER() 或 LOWER()
--      - 但通常题目数据是统一的，直接用 'Y' 即可
--   
--   3. 数据类型：
--      - low_fats 和 recyclable 是字符串类型（'Y'/'N'），不是布尔值
--      - 所以要用 = 'Y'，不是 = True
--   
--   4. 返回列：
--      - 只返回 product_id，不是 SELECT *
--      - 题目通常不要求排序，所以不需要 ORDER BY
--   
--   5. 空值处理：
--      - 如果 low_fats 或 recyclable 可能是 NULL
--      - 需要明确：NULL != 'Y'，所以 NULL 不会被选中（这是合理的）
--      - 如果题目要求处理 NULL，可能需要用 ISNULL() 或 COALESCE()

-- ========================================
-- 【第三部分】如何设计（写代码前先搭结构，按题目对齐）
-- ========================================

-- 你可以把 SQL 查询写成标准的 SELECT-FROM-WHERE 结构：

-- 区块 A：SELECT 子句
--   - SELECT product_id
--   - 目的：只返回产品ID

-- 区块 B：FROM 子句
--   - FROM Products
--   - 目的：指定数据来源表

-- 区块 C：WHERE 子句
--   - WHERE low_fats = 'Y' AND recyclable = 'Y'
--   - 目的：筛选同时满足两个条件的产品

-- 可选区块 D：ORDER BY 子句
--   - 如果题目要求排序，可以加 ORDER BY product_id
--   - 但通常题目不要求排序，所以可以省略

-- ========================================
-- 【SQL 代码实现】
-- ========================================

-- 标准 SQL 实现（推荐）
SELECT product_id
FROM Products
WHERE low_fats = 'Y' AND recyclable = 'Y';

-- ========================================
-- 【变体 1】如果需要处理 NULL 值
-- ========================================
-- 如果数据中可能有 NULL，可以用 COALESCE() 或 ISNULL() 处理
-- 但通常题目数据是完整的，不需要这个

-- SELECT product_id
-- FROM Products
-- WHERE COALESCE(low_fats, 'N') = 'Y' 
--   AND COALESCE(recyclable, 'N') = 'Y';

-- ========================================
-- 【变体 2】如果需要排序
-- ========================================
-- 如果题目要求按 product_id 排序

-- SELECT product_id
-- FROM Products
-- WHERE low_fats = 'Y' AND recyclable = 'Y'
-- ORDER BY product_id;

-- ========================================
-- 【变体 3】如果需要去重
-- ========================================
-- 如果数据可能有重复（虽然 product_id 是主键，通常不会重复）

-- SELECT DISTINCT product_id
-- FROM Products
-- WHERE low_fats = 'Y' AND recyclable = 'Y';

-- ========================================
-- 【变体 4】如果值可能是小写
-- ========================================
-- 如果数据中 'Y' 可能是小写 'y'，需要统一处理

-- SELECT product_id
-- FROM Products
-- WHERE UPPER(low_fats) = 'Y' AND UPPER(recyclable) = 'Y';

-- ========================================
-- 【测试用例】
-- ========================================

-- 示例数据：
-- | product_id | low_fats | recyclable |
-- |------------|----------|------------|
-- | 1          | Y        | Y          |  ← 应该返回
-- | 2          | Y        | N          |  ← 不返回（recyclable 不是 Y）
-- | 3          | N        | Y          |  ← 不返回（low_fats 不是 Y）
-- | 4          | N        | N          |  ← 不返回
-- | 5          | Y        | Y          |  ← 应该返回

-- 预期结果：
-- | product_id |
-- |------------|
-- | 1          |
-- | 5          |

-- ========================================
-- 【常见错误】
-- ========================================

-- ❌ 错误 1：用 OR 而不是 AND
-- SELECT product_id
-- FROM Products
-- WHERE low_fats = 'Y' OR recyclable = 'Y';  -- 错误！这会返回只满足一个条件的产品

-- ❌ 错误 2：返回所有列
-- SELECT * FROM Products WHERE ...;  -- 错误！题目只要求返回 product_id

-- ❌ 错误 3：值写错（小写或布尔值）
-- SELECT product_id FROM Products WHERE low_fats = 'y' ...;  -- 错误！应该是 'Y'
-- SELECT product_id FROM Products WHERE low_fats = True ...;  -- 错误！应该是 'Y'

-- ❌ 错误 4：列名写错
-- SELECT product_id FROM Products WHERE low_fat = 'Y' ...;  -- 错误！应该是 low_fats（有 s）

-- ========================================
-- 【不同数据库的兼容性】
-- ========================================

-- MySQL:
-- SELECT product_id
-- FROM Products
-- WHERE low_fats = 'Y' AND recyclable = 'Y';

-- PostgreSQL:
-- SELECT product_id
-- FROM Products
-- WHERE low_fats = 'Y' AND recyclable = 'Y';

-- SQL Server:
-- SELECT product_id
-- FROM Products
-- WHERE low_fats = 'Y' AND recyclable = 'Y';

-- Oracle:
-- SELECT product_id
-- FROM Products
-- WHERE low_fats = 'Y' AND recyclable = 'Y';

-- SQLite:
-- SELECT product_id
-- FROM Products
-- WHERE low_fats = 'Y' AND recyclable = 'Y';

-- 注意：这个查询在所有主流数据库中都是兼容的，没有特殊语法
