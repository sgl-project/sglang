-- Day 1: SQL 组件 A - 排名与 Top N
-- 题目：员工薪资排名 - 找出每个部门薪资排名前三的员工

-- 请在这里写你的 SQL 查询



-- 思路提示：
-- 1. 使用窗口函数 ROW_NUMBER() OVER (PARTITION BY department ORDER BY salary DESC) 计算排名
-- 2. 使用子查询先计算排名，然后筛选 rank <= 3
-- 3. 注意：并列时如何处理？用 ROW_NUMBER() 还是 RANK()？

-- 你的第一版（需要修改）：
select 
    id,
    name,
    department, 
    salary,
    
    select (
        id,
        name, 
        department,
        salary, 
        DENSE_RANK() OVER 
        (Partition by department 
        order by salary DESC
        ) as  rank 
    ) as t
    
from employees
group by department 
order by rank ASC 
limit 3;

-- ========================================
-- 【重要】为什么 DENSE_RANK() 括号里没东西？
-- ========================================

-- 【核心理解】：窗口函数 vs 普通聚合函数的区别
--
-- 普通聚合函数（需要字段参数）：
--   COUNT(id)     → 统计 id 字段的数量
--   SUM(salary)   → 求和 salary 字段的值
--   MAX(salary)   → 找 salary 字段的最大值
--   AVG(salary)   → 计算 salary 字段的平均值
--   这些函数需要"对哪个字段进行计算"
--
-- 窗口函数（不需要字段参数）：
--   RANK()        → 计算排名（不关心具体值，只关心顺序）
--   DENSE_RANK()  → 计算排名（不关心具体值，只关心顺序）
--   ROW_NUMBER()  → 计算行号（不关心具体值，只关心顺序）
--   这些函数不需要"对哪个字段进行计算"
--   它们关心的是"在哪个窗口内，按什么顺序计算排名"
--
-- 【关键区别】：
--   普通聚合函数：对字段的值进行计算 → 需要字段参数
--   窗口函数：对行的顺序进行计算 → 不需要字段参数，但需要排序规则
--
-- 【为什么这样设计？】
--   DENSE_RANK() 的含义："计算这一行在窗口内的排名"
--   - 它不需要知道具体的值是什么
--   - 它只需要知道："在哪个窗口内？"（PARTITION BY）
--   - 它只需要知道："按什么顺序排列？"（ORDER BY）
--   - 然后它就能知道："这一行排第几名"
--
-- 【类比理解】：
--   想象你在排队：
--   - 普通聚合：需要知道每个人有多重 → SUM(weight)
--   - 窗口函数：只需要知道谁在前面 → RANK() OVER (ORDER BY position)
--
--   排名函数不关心具体的薪资是多少，只关心：
--   "在这个部门内，按薪资从高到低，我排第几？"
--
-- 【语法对比】：
--   普通聚合函数：
--     SELECT department, SUM(salary) as total_salary
--     FROM employees
--     GROUP BY department;
--     → SUM(salary)：对 salary 字段求和
--     → 需要 GROUP BY 分组
--     → 结果会被聚合（每个部门只有一行）
--
--   窗口函数：
--     SELECT id, department, salary,
--            DENSE_RANK() OVER (PARTITION BY department ORDER BY salary DESC) as rank
--     FROM employees;
--     → DENSE_RANK()：计算排名（不需要字段参数）
--     → OVER (PARTITION BY ... ORDER BY ...)：定义窗口和排序
--     → 不需要 GROUP BY
--     → 结果保留所有行（每行都有排名）
--
-- 【关键理解】：
--   DENSE_RANK() OVER (PARTITION BY department ORDER BY salary DESC)
--   |
--   |-- DENSE_RANK()：函数名，括号里不需要参数
--   |-- OVER(...)：窗口定义，括号里需要分组和排序信息
--   |    |
--   |    |-- PARTITION BY department：按部门分组（定义窗口）
--   |    |-- ORDER BY salary DESC：按薪资降序排列（定义排序规则）
--   |
--   整体意思是："在每个部门的窗口内，按薪资从高到低，计算排名"

-- ========================================
-- 代码问题分析：
-- ========================================
-- ❌ 问题1：子查询语法错误
--    `select (...) as t` 是错误的
--    不能在 SELECT 列表里放一个返回多列的子查询
--    
-- ❌ 问题2：仍然有 GROUP BY department
--    窗口函数不需要 GROUP BY，它会自动按 PARTITION BY 分组
--    
-- ❌ 问题3：limit 3 只会返回全局前3行
--    应该是每个部门的前3名，需要用 WHERE rank <= 3
--    
-- ❌ 问题4：中文逗号 `，` 应该改为英文逗号 `,`
--    
-- ❌ 问题5：子查询结构不对
--    窗口函数应该直接在外层查询的 SELECT 中使用
--    或者用子查询包裹整个查询，然后外层筛选

-- ========================================
-- 修正版本：
-- ========================================

-- 方案1：直接使用窗口函数（推荐）
SELECT *
FROM (
    SELECT 
        id,
        name,
        department,
        salary,
        DENSE_RANK() OVER (
            PARTITION BY department 
            ORDER BY salary DESC
        ) as rank
    FROM employees
) t
WHERE rank <= 3
ORDER BY department, rank, id;

-- 方案2：如果你坚持用子查询包裹的方式（不推荐，但语法正确）
SELECT 
    t.id,
    t.name,
    t.department,
    t.salary,
    t.rank
FROM (
    SELECT 
        id,
        name,
        department,
        salary,
        DENSE_RANK() OVER (
            PARTITION BY department 
            ORDER BY salary DESC
        ) as rank
    FROM employees
) t
WHERE t.rank <= 3
ORDER BY t.department, t.rank, t.id;

-- ========================================
-- 【重要】为什么窗口函数这样设计？人类思考逻辑 vs SQL执行逻辑
-- ========================================

-- 【人类的思考逻辑】：
-- 步骤1：我想看所有员工的信息（id, name, department, salary）
-- 步骤2：在每个部门内，按薪资从高到低排序
-- 步骤3：给每个人标上排名（第1名、第2名...）
-- 步骤4：只显示排名 <= 3 的人

-- 【传统SQL的问题】（为什么不能用子查询或GROUP BY）：
-- 如果用子查询计算排名：
--   SELECT id, name, department, salary,
--          (SELECT COUNT(*) FROM employees e2 
--           WHERE e2.department = e1.department 
--           AND e2.salary >= e1.salary) as rank
--   这样写：
--   ❌ 性能差（每行都要执行子查询）
--   ❌ 逻辑复杂（需要比较所有行）
--   ❌ 并列排名处理麻烦
--
-- 如果用 GROUP BY：
--   SELECT department, MAX(salary), ... GROUP BY department
--   ❌ GROUP BY 会聚合数据，返回的是"每个部门的汇总"，不是"每个员工"
--   ❌ 无法同时显示员工信息和排名

-- 【窗口函数的设计逻辑】：
-- 窗口函数的核心思想："看数据的时候，同时计算一个窗口内的值"
-- 
-- RANK() OVER (PARTITION BY department ORDER BY salary DESC)
--   |
--   |-- RANK(): 计算排名（这是要做什么）
--   |-- OVER(): 定义"窗口"（在哪里计算）
--   |    |
--   |    |-- PARTITION BY department: "窗口"是按部门划分的
--   |    |   想象：把表格按部门分组，每组是一个"窗口"
--   |    |
--   |    |-- ORDER BY salary DESC: 在每个"窗口"内，按薪资降序排列
--   |    |   想象：每个部门的员工，按薪资从高到低排好队
--   |
--   |-- 整体意思："在每个部门的窗口内，按薪资降序，计算排名"
--
-- 【执行过程理解】：
-- 原始数据：
-- | id | name  | department | salary |
-- |----|-------|------------|--------|
-- | 1  | Alice | IT         | 9000   |  ← IT部门窗口：9000最高，rank=1
-- | 2  | Bob   | IT         | 8500   |  ← IT部门窗口：8500第二，rank=2
-- | 3  | Carol | IT         | 8500   |  ← IT部门窗口：8500并列，rank=2
-- | 4  | Dave  | IT         | 8000   |  ← IT部门窗口：8000第四，rank=4（注意跳号）
-- | 5  | Eve   | HR         | 7500   |  ← HR部门窗口：7500最高，rank=1
-- | 6  | Frank | HR         | 7200   |  ← HR部门窗口：7200第二，rank=2
--
-- SQL执行步骤（想象过程）：
-- 步骤1：SELECT * FROM employees  → 拿到所有员工数据（6行）
-- 步骤2：计算窗口函数 rank → 对每行，看看它在"自己的窗口"里的排名
--         - Alice (IT, 9000): 在IT窗口里，9000最高 → rank=1
--         - Bob (IT, 8500): 在IT窗口里，比9000低，第二 → rank=2
--         - Carol (IT, 8500): 在IT窗口里，和Bob并列 → rank=2
--         - Dave (IT, 8000): 在IT窗口里，第三名，但因为有并列 → rank=4
--         - Eve (HR, 7500): 在HR窗口里，最高 → rank=1
-- 步骤3：WHERE rank <= 3 → 筛选排名<=3的
-- 步骤4：ORDER BY department, rank → 排序输出

-- 【为什么需要子查询？】
-- 因为窗口函数在 SELECT 子句中计算，而 WHERE 子句在 SELECT 之前执行
-- 所以需要：
--   - 内层查询：计算排名（用窗口函数）
--   - 外层查询：筛选 rank <= 3（用 WHERE）
--
-- 这是SQL的执行顺序决定的：
--   FROM → WHERE → GROUP BY → HAVING → SELECT → ORDER BY → LIMIT
--   窗口函数在 SELECT 阶段计算，所以要用子查询才能在 WHERE 阶段使用
--
-- 【更直观的理解方式 - 用Excel类比】：
-- 想象你在Excel里操作：
-- 1. 你有所有员工的数据（6行）
-- 2. 你想添加一列"排名"：
--    - 不是全局排名，而是"在每个部门内的排名"
--    - 相当于：先按部门筛选，然后排序，再排名
-- 3. Excel里你会：
--    - 添加辅助列："部门分组"（相当于PARTITION BY department）
--    - 添加辅助列："排名"（相当于RANK()）
--    - 筛选：只显示排名 <= 3 的行（相当于WHERE rank <= 3）
--
-- 窗口函数就是SQL里的"辅助列计算"，但它更聪明：
--   - 不需要真正分组（GROUP BY会聚合）
--   - 可以在计算排名的同时，保留所有原始行
--   - 每个部门分别计算排名，互不干扰

-- ========================================
-- 改进建议（使用 RANK() 窗口函数）：
-- ========================================

-- 关键修改点：
-- 1. 删除 GROUP BY department（窗口函数不需要 GROUP BY）
-- 2. 在括号里填入：RANK() OVER (PARTITION BY department ORDER BY salary DESC)
-- 3. 删除 () + 1（RANK() 本身就是排名，从1开始）
-- 4. 使用子查询，外层筛选 rank <= 3（而不是用 limit 3，因为 limit 只返回全局前3行）
-- 5. 如果需要并列时按 id 排序，可以在 ORDER BY 里加第二个排序字段

-- 修改后的代码：
SELECT *
FROM (
    SELECT 
        id,
        name,
        department,
        salary,
        RANK() OVER (PARTITION BY department ORDER BY salary DESC) as rank
    FROM employees
) t
WHERE rank <= 3
ORDER BY department, rank, id;

-- ========================================
-- 【全面讲解】DENSE_RANK() OVER 的其他写法和用法
-- ========================================

-- 【一、OVER() 子句的完整语法】：
--   OVER (
--       [PARTITION BY 字段1, 字段2, ...]  -- 可选：分组
--       [ORDER BY 字段1 [ASC|DESC], 字段2 [ASC|DESC], ...]  -- 可选：排序
--       [ROWS BETWEEN ... AND ...]  -- 可选：窗口范围（高级用法）
--   )

-- ========================================
-- 【二、基本用法（按复杂度递增）】
-- ========================================

-- 1. 最简单的用法：只有 ORDER BY（全局排名）
-- SELECT 
--     id,
--     name,
--     salary,
--     DENSE_RANK() OVER (ORDER BY salary DESC) as rank
-- FROM employees;
-- 结果：所有员工按薪资排名，不分部门

-- 2. 常用用法：PARTITION BY + ORDER BY（分组内排名）
-- SELECT 
--     id,
--     name,
--     department,
--     salary,
--     DENSE_RANK() OVER (PARTITION BY department ORDER BY salary DESC) as rank
-- FROM employees;
-- 结果：每个部门内按薪资排名

-- 3. 多字段分组：PARTITION BY 多个字段
-- SELECT 
--     id,
--     name,
--     department,
--     location,
--     salary,
--     DENSE_RANK() OVER (PARTITION BY department, location ORDER BY salary DESC) as rank
-- FROM employees;
-- 结果：每个"部门+地点"组合内按薪资排名

-- 4. 多字段排序：ORDER BY 多个字段
-- SELECT 
--     id,
--     name,
--     department,
--     salary,
--     hire_date,
--     DENSE_RANK() OVER (
--         PARTITION BY department 
--         ORDER BY salary DESC, hire_date ASC
--     ) as rank
-- FROM employees;
-- 结果：每个部门内，先按薪资降序，薪资相同按入职日期升序排名

-- ========================================
-- 【三、窗口函数的不同类型】
-- ========================================

-- 1. 排名函数（不需要参数）
-- SELECT 
--     id,
--     salary,
--     ROW_NUMBER() OVER (ORDER BY salary DESC) as row_num,     -- 连续排名：1,2,3,4,5...
--     RANK() OVER (ORDER BY salary DESC) as rank,               -- 并列会跳号：1,2,2,4,5...
--     DENSE_RANK() OVER (ORDER BY salary DESC) as dense_rank,   -- 并列不跳号：1,2,2,3,4...
--     PERCENT_RANK() OVER (ORDER BY salary DESC) as percent_rank,  -- 百分比排名：0.0-1.0
--     NTILE(4) OVER (ORDER BY salary DESC) as quartile          -- 分桶：分成4组，每组的编号1-4
-- FROM employees;
--
-- 示例数据对比（salary: 9000, 8500, 8500, 8000, 7000）：
-- row_num:   1,   2,   3,   4,   5
-- rank:      1,   2,   2,   4,   5  (并列后跳号)
-- dense_rank:1,   2,   2,   3,   4  (并列后不跳号)
-- percent_rank: 0,  0.25, 0.25, 0.75, 1.0
-- quartile:  1,   1,   2,   3,   4

-- 2. 取值函数（需要字段参数）
-- SELECT 
--     id,
--     salary,
--     department,
--     LAG(salary, 1) OVER (PARTITION BY department ORDER BY salary DESC) as prev_salary,  -- 上一行的值
--     LEAD(salary, 1) OVER (PARTITION BY department ORDER BY salary DESC) as next_salary, -- 下一行的值
--     FIRST_VALUE(salary) OVER (PARTITION BY department ORDER BY salary DESC) as first_salary,  -- 窗口内第一个值
--     LAST_VALUE(salary) OVER (PARTITION BY department ORDER BY salary DESC ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) as last_salary,  -- 窗口内最后一个值
--     NTH_VALUE(salary, 2) OVER (PARTITION BY department ORDER BY salary DESC) as second_salary  -- 窗口内第N个值
-- FROM employees;

-- 3. 聚合函数（作为窗口函数使用）
-- SELECT 
--     id,
--     department,
--     salary,
--     SUM(salary) OVER (PARTITION BY department) as dept_total,      -- 部门总薪资
--     AVG(salary) OVER (PARTITION BY department) as dept_avg,        -- 部门平均薪资
--     COUNT(*) OVER (PARTITION BY department) as dept_count,         -- 部门人数
--     MAX(salary) OVER (PARTITION BY department) as dept_max,        -- 部门最高薪资
--     MIN(salary) OVER (PARTITION BY department) as dept_min         -- 部门最低薪资
-- FROM employees;

-- ========================================
-- 【四、高级用法：窗口范围（ROWS BETWEEN）】
-- ========================================

-- 1. 移动窗口（滑动窗口）
-- SELECT 
--     id,
--     salary,
--     -- 当前行及前2行的平均值（3行滑动窗口）
--     AVG(salary) OVER (
--         ORDER BY id 
--         ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
--     ) as moving_avg_3,
--     
--     -- 当前行前后各1行（3行窗口）
--     AVG(salary) OVER (
--         ORDER BY id 
--         ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING
--     ) as moving_avg_centered,
--     
--     -- 从开头到当前行（累计）
--     SUM(salary) OVER (
--         ORDER BY id 
--         ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
--     ) as running_total
-- FROM employees;
--
-- ROWS BETWEEN 选项说明：
--   UNBOUNDED PRECEDING  - 窗口开始（从第一行开始）
--   n PRECEDING          - 当前行前n行
--   CURRENT ROW          - 当前行
--   n FOLLOWING          - 当前行后n行
--   UNBOUNDED FOLLOWING  - 窗口结束（到最后一行）

-- 2. 范围窗口（RANGE BETWEEN，基于值而不是行数）
-- SELECT 
--     id,
--     salary,
--     -- 薪资在 [salary-1000, salary+1000] 范围内的平均值
--     AVG(salary) OVER (
--         ORDER BY salary 
--         RANGE BETWEEN 1000 PRECEDING AND 1000 FOLLOWING
--     ) as range_avg
-- FROM employees;
--
-- RANGE vs ROWS 的区别：
--   ROWS: 基于行数（前N行、后N行）
--   RANGE: 基于值（前N个单位、后N个单位）

-- ========================================
-- 【五、窗口函数命名（WINDOW 子句）】
-- ========================================

-- 如果多个窗口函数使用相同的 OVER 子句，可以命名窗口：
-- SELECT 
--     id,
--     name,
--     department,
--     salary,
--     DENSE_RANK() OVER dept_window as rank,
--     ROW_NUMBER() OVER dept_window as row_num,
--     AVG(salary) OVER dept_window as dept_avg
-- FROM employees
-- WINDOW dept_window AS (
--     PARTITION BY department 
--     ORDER BY salary DESC
-- );
-- 优点：代码更简洁，避免重复写相同的 OVER 子句

-- ========================================
-- 【六、常见场景和组合用法】
-- ========================================

-- 场景1：Top N 问题（已学）
-- SELECT *
-- FROM (
--     SELECT 
--         *,
--         DENSE_RANK() OVER (PARTITION BY department ORDER BY salary DESC) as rank
--     FROM employees
-- ) t
-- WHERE rank <= 3;

-- 场景2：连续性问题（组件B - 明天会学）
-- SELECT DISTINCT user_id
-- FROM (
--     SELECT 
--         user_id,
--         date,
--         ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY date) as rn,
--         date - INTERVAL rn DAY as group_id
--     FROM user_activity
-- ) t
-- GROUP BY user_id, group_id
-- HAVING COUNT(*) >= 3;
-- 找出连续3天活跃的用户

-- 场景3：留存率计算（组件C - 第3天会学）
-- SELECT 
--     DATE_FORMAT(login_date, '%Y-%m-%d') as date,
--     COUNT(DISTINCT user_id) as total_users,
--     COUNT(DISTINCT CASE 
--         WHEN next_login_date = DATE_ADD(login_date, INTERVAL 1 DAY) 
--         THEN user_id 
--     END) as retained_users
-- FROM (
--     SELECT 
--         user_id,
--         login_date,
--         LEAD(login_date) OVER (PARTITION BY user_id ORDER BY login_date) as next_login_date
--     FROM user_logins
-- ) t
-- GROUP BY DATE_FORMAT(login_date, '%Y-%m-%d');

-- 场景4：同比/环比（增长率计算）
-- SELECT 
--     month,
--     sales,
--     LAG(sales, 1) OVER (ORDER BY month) as prev_month_sales,
--     LAG(sales, 12) OVER (ORDER BY month) as prev_year_sales,
--     (sales - LAG(sales, 1) OVER (ORDER BY month)) * 100.0 / LAG(sales, 1) OVER (ORDER BY month) as month_over_month,
--     (sales - LAG(sales, 12) OVER (ORDER BY month)) * 100.0 / LAG(sales, 12) OVER (ORDER BY month) as year_over_year
-- FROM monthly_sales;

-- ========================================
-- 【七、注意事项和常见错误】
-- ========================================

-- ❌ 错误1：窗口函数不能在 WHERE 中使用
-- SELECT * FROM employees 
-- WHERE DENSE_RANK() OVER (ORDER BY salary DESC) <= 3;  -- 错误！
-- 正确：需要用子查询
-- SELECT * FROM (
--     SELECT *, DENSE_RANK() OVER (ORDER BY salary DESC) as rank
--     FROM employees
-- ) t WHERE rank <= 3;

-- ❌ 错误2：窗口函数不能嵌套
-- SELECT DENSE_RANK() OVER (ORDER BY RANK() OVER (...))  -- 错误！

-- ❌ 错误3：窗口函数中不能使用窗口函数的结果
-- SELECT 
--     DENSE_RANK() OVER (ORDER BY salary DESC) as rank1,
--     RANK() OVER (ORDER BY rank1) as rank2  -- 错误！不能引用 rank1
-- 
-- ✅ 正确：如果需要嵌套，用子查询
-- SELECT 
--     rank1,
--     RANK() OVER (ORDER BY rank1) as rank2
-- FROM (
--     SELECT DENSE_RANK() OVER (ORDER BY salary DESC) as rank1
--     FROM employees
-- ) t;

-- ❌ 错误4：GROUP BY 和窗口函数混用（除非窗口函数不在 GROUP BY 中）
-- SELECT department, MAX(salary), DENSE_RANK() OVER (ORDER BY MAX(salary))
-- FROM employees
-- GROUP BY department;  -- 可能出错，取决于数据库

-- ========================================
-- 【八、不同数据库的差异】
-- ========================================

-- MySQL 8.0+：
--   ✅ 支持窗口函数
--   ✅ 语法：OVER (PARTITION BY ... ORDER BY ...)

-- PostgreSQL：
--   ✅ 完全支持窗口函数
--   ✅ 支持 WINDOW 子句命名窗口

-- SQL Server：
--   ✅ 支持窗口函数（从2012版本开始）
--   ✅ 语法相同

-- SQLite：
--   ✅ 从3.25.0版本开始支持窗口函数

-- Oracle：
--   ✅ 很早就支持窗口函数
--   ✅ 支持更多高级功能（FIRST/LAST, IGNORE NULLS等）
