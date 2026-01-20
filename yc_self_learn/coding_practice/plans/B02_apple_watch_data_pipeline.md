# B02: Apple Watch 健康数据处理 Pipeline 实战案例

## 📋 目录

1. [业务场景概述](#业务场景概述)
2. [数据模型设计](#数据模型设计)
3. [数据处理 Pipeline](#数据处理-pipeline)
4. [SQL 实战查询](#sql-实战查询)
5. [性能优化实践](#性能优化实践)
6. [总结与学习要点](#总结与学习要点)

---

## 📱 业务场景概述

### 场景描述

**Apple Watch 每天收集用户的健康数据**，包括：
- **心率数据**：每分钟记录一次心率
- **步数数据**：每小时记录一次步数
- **运动数据**：每次运动记录（跑步、游泳、健身等）
- **睡眠数据**：每晚记录睡眠时长和质量
- **卡路里数据**：每天记录消耗的卡路里

**业务需求**：
1. 实时监控用户健康状态
2. 生成每日、每周、每月的健康报告
3. 分析用户运动习惯和趋势
4. 识别异常数据（如心率异常）
5. 提供个性化的健康建议

---

## 📊 数据模型设计

### 表结构设计

#### 1. 用户表（users）

```sql
CREATE TABLE users (
    user_id INT PRIMARY KEY,
    name VARCHAR(100),
    age INT,
    gender VARCHAR(10),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 示例数据
INSERT INTO users (user_id, name, age, gender) VALUES
(1, 'Alice', 28, 'Female'),
(2, 'Bob', 35, 'Male'),
(3, 'Charlie', 42, 'Male');
```

---

#### 📚 数据库设计基础知识（第 40-54 行解析）

**在理解表结构之前，我们先学习数据库设计的基础概念：**

##### 1. PRIMARY KEY（主键）—— 如何设计？

**什么是 PRIMARY KEY？**
- **主键**：唯一标识表中每一行记录的列（或列组合）
- **特点**：
  - ✅ **唯一性**：每个值必须唯一，不能重复
  - ✅ **非空**：不能为 NULL
  - ✅ **唯一标识**：通过主键可以唯一确定一行记录

**为什么选择 `user_id` 作为主键？**

```sql
CREATE TABLE users (
    user_id INT PRIMARY KEY,  -- ← 为什么用 user_id 作为主键？
    name VARCHAR(100),
    age INT,
    gender VARCHAR(10)
);
```

**设计原则**：

1. **唯一性** ✅：
   - 每个用户的 `user_id` 必须唯一（不能有两个用户的 id 都是 1）
   - 这样可以唯一标识每个用户

2. **稳定性** ✅：
   - `user_id` 不会改变（一旦分配，不会修改）
   - 名字可能会改变（结婚改姓），但 `user_id` 不会变

3. **简洁性** ✅：
   - `user_id` 是整数，简单高效
   - 比用 `name` 作为主键更好（名字可能有重复）

4. **索引性能** ✅：
   - 整数主键查询速度快（O(log n)）
   - 字符串主键查询相对慢一些

**常见主键选择**：

| 主键类型 | 示例 | 优点 | 缺点 |
|---------|------|------|------|
| **自增整数**（推荐）✅ | `id INT PRIMARY KEY AUTO_INCREMENT` | 简单、高效、自动生成 | 需要额外列 |
| **业务ID** | `user_id INT PRIMARY KEY` | 有业务含义 | 需要保证唯一性 |
| **字符串** | `email VARCHAR(100) PRIMARY KEY` | 有业务含义 | 性能较慢 |
| **UUID** | `id CHAR(36) PRIMARY KEY` | 全局唯一 | 占用空间大 |

**设计建议**：

```sql
-- ✅ 推荐：自增整数主键（最简单、最常用）
CREATE TABLE users (
    id INT PRIMARY KEY AUTO_INCREMENT,  -- 自动生成：1, 2, 3, ...
    user_id INT UNIQUE,  -- 业务ID（如果有需要）
    name VARCHAR(100),
    ...
);

-- ⚠️ 也可以：直接用业务ID作为主键（如果保证唯一）
CREATE TABLE users (
    user_id INT PRIMARY KEY,  -- 直接用业务ID作为主键
    name VARCHAR(100),
    ...
);

-- ❌ 不推荐：用可变字段作为主键
CREATE TABLE users (
    name VARCHAR(100) PRIMARY KEY,  -- 名字可能重复、可能改变
    ...
);
```

---

##### 2. VARCHAR—— 什么是 VARCHAR？

**什么是 VARCHAR？**
- **VARCHAR**：可变长度的字符串类型
- **特点**：
  - ✅ **可变长度**：实际存储空间 = 实际长度（不是固定长度）
  - ✅ **节省空间**：只存储实际字符，不存储空字符
  - ✅ **最大长度**：需要指定最大长度

**VARCHAR vs CHAR**：

| 类型 | 示例 | 存储方式 | 适用场景 |
|------|------|---------|---------|
| **VARCHAR(n)** | `VARCHAR(100)` | 可变长度（实际长度） | 长度不固定的字符串 ✅ |
| **CHAR(n)** | `CHAR(10)` | 固定长度（总是 n 个字符） | 长度固定的字符串（如身份证） |

**示例对比**：

```sql
-- VARCHAR(100)：最大 100 个字符，实际存储根据内容
name VARCHAR(100)
-- 存储 'Alice'：只占用 5 个字符的空间
-- 存储 'Bob'：只占用 3 个字符的空间
-- 存储 'Charlie'：只占用 7 个字符的空间

-- CHAR(10)：总是 10 个字符
code CHAR(10)
-- 存储 'A123'：占用 10 个字符的空间（后面补空格）
-- 存储 'B456'：占用 10 个字符的空间（后面补空格）
```

**为什么用 VARCHAR？**

```sql
CREATE TABLE users (
    user_id INT PRIMARY KEY,
    name VARCHAR(100),     -- ← 为什么用 VARCHAR(100)？
    gender VARCHAR(10)     -- ← 为什么用 VARCHAR(10)？
);
```

**原因**：
1. **长度不固定**：
   - 名字长度不同：'Alice' (5个字符)、'Bob' (3个字符)、'Charlie' (7个字符)
   - 如果用 `CHAR(100)`，'Bob' 会浪费 97 个字符的空间

2. **节省空间**：
   - VARCHAR 只存储实际字符，节省存储空间
   - 对于长度变化大的字段，VARCHAR 更高效

3. **合理设置最大长度**：
   - `name VARCHAR(100)`：名字最长 100 个字符（足够）
   - `gender VARCHAR(10)`：性别最长 10 个字符（Male/Female 足够）

**常见长度设置**：

| 字段 | 类型 | 长度 | 说明 |
|------|------|------|------|
| 名字 | `VARCHAR(50)` | 50 | 一般名字 20-30 字符足够 |
| 邮箱 | `VARCHAR(255)` | 255 | 邮箱标准最大长度 |
| 手机号 | `VARCHAR(20)` | 20 | 国际手机号格式 |
| 地址 | `VARCHAR(500)` | 500 | 详细地址可能较长 |
| 描述 | `TEXT` | 无限制 | 长文本用 TEXT |

---

##### 3. INT NOT NULL—— 什么是 INT NOT NULL？

**什么是 INT？**
- **INT**：整数类型（32位，范围：-2,147,483,648 到 2,147,483,647）
- **特点**：
  - ✅ **整数**：只能存储整数（不能存储小数）
  - ✅ **固定大小**：总是占用 4 字节
  - ✅ **性能好**：整数运算和比较速度快

**什么是 NOT NULL？**
- **NOT NULL**：约束，表示该列不能为空
- **特点**：
  - ✅ **必须填写**：插入数据时必须提供值
  - ✅ **不能为空**：不能插入 NULL 值
  - ✅ **数据完整性**：保证数据完整

**INT vs INT NOT NULL**：

```sql
-- 1. INT（允许 NULL）
age INT
-- 可以插入：age = 28
-- 可以插入：age = NULL  ← 允许为空
-- 查询时：可能需要处理 NULL 值

-- 2. INT NOT NULL（不允许 NULL）
age INT NOT NULL
-- 可以插入：age = 28
-- 不能插入：age = NULL  ← 不允许为空，会报错
-- 查询时：不需要处理 NULL 值（更简单）
```

**为什么用 INT NOT NULL？**

```sql
CREATE TABLE heart_rate (
    id INT PRIMARY KEY AUTO_INCREMENT,
    user_id INT NOT NULL,        -- ← 为什么用 NOT NULL？
    heart_rate INT NOT NULL,     -- ← 为什么用 NOT NULL？
    recorded_at TIMESTAMP NOT NULL  -- ← 为什么用 NOT NULL？
);
```

**原因**：

1. **业务逻辑要求**：
   - `user_id INT NOT NULL`：每条心率数据必须属于某个用户（不能为空）
   - `heart_rate INT NOT NULL`：每条心率数据必须有心率值（不能为空）
   - `recorded_at TIMESTAMP NOT NULL`：每条心率数据必须有记录时间（不能为空）

2. **数据完整性**：
   - 防止插入不完整的数据（缺少关键字段）
   - 保证数据质量

3. **查询简化**：
   - 不需要处理 NULL 值（查询更简单）
   - 避免 NULL 值带来的问题（如 NULL + 1 = NULL）

**常见数据类型**：

| 类型 | 示例 | 范围 | 说明 |
|------|------|------|------|
| **INT** | `age INT` | -2,147,483,648 到 2,147,483,647 | 整数（32位） |
| **BIGINT** | `id BIGINT` | 更大范围 | 大整数（64位） |
| **DECIMAL** | `price DECIMAL(10,2)` | 精确小数 | 金额等需要精确的数据 |
| **FLOAT** | `score FLOAT` | 近似小数 | 一般数值（可能有误差） |
| **VARCHAR** | `name VARCHAR(100)` | 字符串 | 可变长度字符串 |
| **TEXT** | `description TEXT` | 长文本 | 长文本（无长度限制） |
| **TIMESTAMP** | `created_at TIMESTAMP` | 日期时间 | 时间戳 |
| **DATE** | `birth_date DATE` | 日期 | 日期（不含时间） |
| **BOOLEAN** | `is_active BOOLEAN` | TRUE/FALSE | 布尔值 |

**NULL vs NOT NULL 选择建议**：

```sql
-- ✅ 推荐：关键字段用 NOT NULL
user_id INT NOT NULL,        -- 用户ID必须存在
email VARCHAR(255) NOT NULL,  -- 邮箱必须存在
created_at TIMESTAMP NOT NULL -- 创建时间必须存在

-- ⚠️ 可选：可选字段允许 NULL
middle_name VARCHAR(50),      -- 中间名可选（可能有，可能没有）
deleted_at TIMESTAMP,        -- 删除时间可选（未删除时为 NULL）

-- ❌ 不推荐：关键字段允许 NULL（除非有特殊需求）
user_id INT,  -- 用户ID允许为 NULL（不合理）
```

---

#### 📝 表设计要点总结

**基于 users 表的设计要点**：

```sql
CREATE TABLE users (
    user_id INT PRIMARY KEY,              -- 主键：唯一标识用户
    name VARCHAR(100),                    -- 可变字符串：节省空间
    age INT,                              -- 整数：年龄是整数
    gender VARCHAR(10),                   -- 可变字符串：Male/Female
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP  -- 时间戳：自动记录创建时间
);
```

**设计原则**：

1. **主键选择** ✅：
   - 选择唯一、稳定、简洁的字段作为主键
   - 推荐：自增整数 `id INT PRIMARY KEY AUTO_INCREMENT`
   - 或：业务ID `user_id INT PRIMARY KEY`

2. **数据类型选择** ✅：
   - 整数用 `INT` 或 `BIGINT`
   - 字符串用 `VARCHAR(n)`（长度不固定）或 `CHAR(n)`（长度固定）
   - 日期时间用 `TIMESTAMP` 或 `DATE`
   - 长文本用 `TEXT`

3. **NULL vs NOT NULL** ✅：
   - 关键字段用 `NOT NULL`（保证数据完整性）
   - 可选字段允许 `NULL`（如果业务允许为空）

4. **字段命名** ✅：
   - 使用有意义的名称（`user_id` 而不是 `id1`）
   - 遵循命名规范（下划线命名：`user_id` 而不是 `userId`）

---

#### 2. 心率数据表（heart_rate）

```sql
CREATE TABLE heart_rate (
    id INT PRIMARY KEY AUTO_INCREMENT,
    user_id INT NOT NULL,
    heart_rate INT NOT NULL,  -- 心率值（次/分钟）
    recorded_at TIMESTAMP NOT NULL,  -- 记录时间（每分钟一条）
    FOREIGN KEY (user_id) REFERENCES users(user_id),
    INDEX idx_user_time (user_id, recorded_at),
    INDEX idx_recorded_at (recorded_at)
);

-- 示例数据（用户1，某一天的心率数据，每分钟一条）
-- 假设一天有 1440 分钟（24小时 × 60分钟）
-- 这里只展示部分数据
INSERT INTO heart_rate (user_id, heart_rate, recorded_at) VALUES
(1, 72, '2024-01-10 00:01:00'),
(1, 75, '2024-01-10 00:02:00'),
(1, 68, '2024-01-10 00:03:00'),
(1, 140, '2024-01-10 07:00:00'),  -- 运动时心率升高
(1, 145, '2024-01-10 07:01:00'),
(1, 135, '2024-01-10 07:02:00');
```

#### 3. 步数数据表（steps）

```sql
CREATE TABLE steps (
    id INT PRIMARY KEY AUTO_INCREMENT,
    user_id INT NOT NULL,
    steps INT NOT NULL,  -- 步数（每小时累计）
    recorded_at TIMESTAMP NOT NULL,  -- 记录时间（每小时一条）
    FOREIGN KEY (user_id) REFERENCES users(user_id),
    INDEX idx_user_time (user_id, recorded_at)
);

-- 示例数据（用户1，某一天的步数数据，每小时一条）
INSERT INTO steps (user_id, steps, recorded_at) VALUES
(1, 0, '2024-01-10 00:00:00'),  -- 0点：0步
(1, 250, '2024-01-10 01:00:00'),  -- 1点：250步（凌晨走路）
(1, 500, '2024-01-10 02:00:00'),  -- 2点：500步
(1, 3200, '2024-01-10 07:00:00'),  -- 7点：3200步（晨跑）
(1, 4500, '2024-01-10 08:00:00'),  -- 8点：4500步
(1, 8200, '2024-01-10 12:00:00'),  -- 12点：8200步（上午活动）
(1, 12500, '2024-01-10 18:00:00'),  -- 18点：12500步
(1, 15000, '2024-01-10 23:00:00');  -- 23点：15000步（一天总步数）
```

#### 4. 运动数据表（activities）

```sql
CREATE TABLE activities (
    id INT PRIMARY KEY AUTO_INCREMENT,
    user_id INT NOT NULL,
    activity_type VARCHAR(50) NOT NULL,  -- 运动类型（Running, Swimming, Gym等）
    duration_minutes INT NOT NULL,  -- 运动时长（分钟）
    calories_burned INT NOT NULL,  -- 消耗的卡路里
    avg_heart_rate INT,  -- 平均心率
    started_at TIMESTAMP NOT NULL,  -- 开始时间
    ended_at TIMESTAMP NOT NULL,  -- 结束时间
    FOREIGN KEY (user_id) REFERENCES users(user_id),
    INDEX idx_user_date (user_id, started_at),
    INDEX idx_started_at (started_at)
);

-- 示例数据
INSERT INTO activities (user_id, activity_type, duration_minutes, calories_burned, avg_heart_rate, started_at, ended_at) VALUES
(1, 'Running', 30, 300, 140, '2024-01-10 07:00:00', '2024-01-10 07:30:00'),
(1, 'Walking', 60, 200, 95, '2024-01-10 12:00:00', '2024-01-10 13:00:00'),
(1, 'Gym', 45, 400, 120, '2024-01-10 18:00:00', '2024-01-10 18:45:00'),
(2, 'Swimming', 60, 500, 110, '2024-01-10 08:00:00', '2024-01-10 09:00:00'),
(2, 'Running', 45, 450, 145, '2024-01-10 19:00:00', '2024-01-10 19:45:00');
```

#### 5. 睡眠数据表（sleep）

```sql
CREATE TABLE sleep (
    id INT PRIMARY KEY AUTO_INCREMENT,
    user_id INT NOT NULL,
    sleep_start TIMESTAMP NOT NULL,  -- 睡眠开始时间
    sleep_end TIMESTAMP NOT NULL,  -- 睡眠结束时间
    sleep_duration_minutes INT NOT NULL,  -- 睡眠时长（分钟）
    sleep_quality VARCHAR(20),  -- 睡眠质量（Good, Fair, Poor）
    deep_sleep_minutes INT,  -- 深度睡眠时长（分钟）
    light_sleep_minutes INT,  -- 浅度睡眠时长（分钟）
    FOREIGN KEY (user_id) REFERENCES users(user_id),
    INDEX idx_user_date (user_id, sleep_start),
    INDEX idx_sleep_start (sleep_start)
);

-- 示例数据
INSERT INTO sleep (user_id, sleep_start, sleep_end, sleep_duration_minutes, sleep_quality, deep_sleep_minutes, light_sleep_minutes) VALUES
(1, '2024-01-09 23:00:00', '2024-01-10 07:00:00', 480, 'Good', 120, 360),
(1, '2024-01-10 23:30:00', '2024-01-11 07:30:00', 480, 'Fair', 90, 390),
(2, '2024-01-09 22:30:00', '2024-01-10 06:30:00', 480, 'Good', 135, 345);
```

---

## 🔄 数据处理 Pipeline

### Pipeline 架构

```
数据源（Apple Watch）→ 数据收集 → 数据清洗 → 数据存储 → 数据处理 → 数据分析 → 数据输出
```

### 阶段 1：数据收集（Data Collection）

**目标**：从 Apple Watch 收集原始数据

**数据格式**（原始数据，可能包含错误）：
```json
{
  "user_id": 1,
  "heart_rate": 72,
  "recorded_at": "2024-01-10T00:01:00Z",
  "device_id": "AW123456"
}
```

**处理方式**：
- 实时数据流（Streaming）
- 或批量数据（Batch）

---

### 阶段 2：数据清洗（Data Cleaning）

**目标**：清洗和验证数据

**常见问题**：
1. **缺失值**：某些时间点没有数据
2. **异常值**：心率超过正常范围（如 0 或 300）
3. **重复数据**：同一时间点有多条记录
4. **时间错误**：记录时间不合理（如未来时间）

**清洗 SQL 示例**：

```sql
-- 1. 识别异常心率数据（正常范围：40-200 次/分钟）
SELECT 
    user_id,
    heart_rate,
    recorded_at
FROM heart_rate
WHERE heart_rate < 40 OR heart_rate > 200;  -- 异常数据

-- 2. 识别重复数据（同一用户同一时间有多条记录）
SELECT 
    user_id,
    recorded_at,
    COUNT(*) as count
FROM heart_rate
GROUP BY user_id, recorded_at
HAVING COUNT(*) > 1;  -- 重复数据

-- 3. 识别缺失数据（某个时间段没有数据）
-- 假设应该每分钟都有数据，但某些时间点缺失
WITH time_series AS (
    SELECT 
        user_id,
        DATE_FORMAT(recorded_at, '%Y-%m-%d %H:%i:00') as expected_time,
        DATE_FORMAT(recorded_at, '%Y-%m-%d %H:%i:00') as actual_time
    FROM heart_rate
    WHERE recorded_at >= '2024-01-10 00:00:00'
      AND recorded_at < '2024-01-11 00:00:00'
)
SELECT 
    user_id,
    expected_time,
    actual_time
FROM time_series
WHERE expected_time != actual_time;  -- 缺失数据
```

---

### 阶段 3：数据存储（Data Storage）

**目标**：将清洗后的数据存储到数据库

**存储策略**：
1. **实时存储**：数据清洗后立即存储
2. **批量存储**：每小时或每天批量存储
3. **分区存储**：按日期分区（提高查询性能）

**分区示例**：

```sql
-- MySQL 分区（按日期分区）
CREATE TABLE heart_rate (
    id INT PRIMARY KEY AUTO_INCREMENT,
    user_id INT NOT NULL,
    heart_rate INT NOT NULL,
    recorded_at TIMESTAMP NOT NULL
) PARTITION BY RANGE (TO_DAYS(recorded_at)) (
    PARTITION p20240101 VALUES LESS THAN (TO_DAYS('2024-01-02')),
    PARTITION p20240102 VALUES LESS THAN (TO_DAYS('2024-01-03')),
    PARTITION p20240103 VALUES LESS THAN (TO_DAYS('2024-01-04')),
    -- ... 更多分区
);
```

---

### 阶段 4：数据处理（Data Processing）

**目标**：对数据进行聚合和转换

**常见处理**：
1. **时间聚合**：按小时、天、周、月聚合
2. **统计计算**：平均值、最大值、最小值
3. **数据转换**：单位转换、格式转换

---

### 阶段 5：数据分析（Data Analysis）

**目标**：分析数据，生成洞察

**分析需求**：
1. **用户健康报告**：每日、每周、每月报告
2. **趋势分析**：心率趋势、步数趋势
3. **异常检测**：识别异常数据
4. **用户画像**：运动习惯、健康状态

---

## 📈 SQL 实战查询

### 查询 1：用户每日步数统计（基础聚合）

**需求**：统计每个用户每天的步数

```sql
SELECT 
    user_id,
    DATE(recorded_at) as date,
    MAX(steps) as daily_steps  -- 一天的最终步数
FROM steps
WHERE recorded_at >= '2024-01-10'
  AND recorded_at < '2024-01-11'
GROUP BY user_id, DATE(recorded_at)
ORDER BY user_id, date;
```

**结果示例**：
```
user_id | date       | daily_steps
--------|------------|------------
1       | 2024-01-10 | 15000
2       | 2024-01-10 | 12000
```

---

### 查询 2：用户每日平均心率（窗口函数）

**需求**：计算每个用户每天的平均心率

```sql
SELECT 
    user_id,
    DATE(recorded_at) as date,
    AVG(heart_rate) as avg_heart_rate,
    MIN(heart_rate) as min_heart_rate,
    MAX(heart_rate) as max_heart_rate
FROM heart_rate
WHERE recorded_at >= '2024-01-10'
  AND recorded_at < '2024-01-11'
GROUP BY user_id, DATE(recorded_at)
ORDER BY user_id, date;
```

**结果示例**：
```
user_id | date       | avg_heart_rate | min_heart_rate | max_heart_rate
--------|------------|----------------|----------------|---------------
1       | 2024-01-10 | 85.5          | 68            | 145
```

---

### 查询 3：用户运动时长排名（窗口函数）

**需求**：找出每个用户运动时长最长的运动

**使用窗口函数（ROW_NUMBER）**：

```sql
SELECT 
    user_id,
    activity_type,
    duration_minutes,
    calories_burned,
    started_at
FROM (
    SELECT 
        user_id,
        activity_type,
        duration_minutes,
        calories_burned,
        started_at,
        ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY duration_minutes DESC) as rn
    FROM activities
    WHERE DATE(started_at) = '2024-01-10'
) t
WHERE rn = 1;  -- 每个用户最长的运动
```

**结果示例**：
```
user_id | activity_type | duration_minutes | calories_burned | started_at
--------|---------------|------------------|-----------------|-------------------
1       | Gym          | 45              | 400            | 2024-01-10 18:00:00
2       | Swimming     | 60              | 500            | 2024-01-10 08:00:00
```

---

### 查询 4：用户每日卡路里消耗汇总（JOIN + 聚合）

**需求**：统计每个用户每天消耗的总卡路里（来自运动数据）

```sql
SELECT 
    u.user_id,
    u.name,
    DATE(a.started_at) as date,
    SUM(a.calories_burned) as total_calories,
    COUNT(*) as activity_count
FROM users u
LEFT JOIN activities a ON u.user_id = a.user_id
WHERE DATE(a.started_at) = '2024-01-10'
GROUP BY u.user_id, u.name, DATE(a.started_at)
ORDER BY total_calories DESC;
```

**结果示例**：
```
user_id | name    | date       | total_calories | activity_count
--------|---------|------------|----------------|----------------
1       | Alice   | 2024-01-10 | 900           | 3
2       | Bob     | 2024-01-10 | 950           | 2
```

---

### 查询 5：用户连续运动天数（连续性问题 - Gaps & Islands）

**需求**：找出连续运动 3 天以上的用户

**使用连续段（Gaps & Islands）模板**：

```sql
WITH daily_activities AS (
    SELECT DISTINCT
        user_id,
        DATE(started_at) as activity_date
    FROM activities
    WHERE started_at >= '2024-01-01'
      AND started_at < '2024-01-31'
),
consecutive_groups AS (
    SELECT 
        user_id,
        activity_date,
        DATE_SUB(activity_date, INTERVAL ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY activity_date) DAY) as group_id
    FROM daily_activities
),
consecutive_days AS (
    SELECT 
        user_id,
        group_id,
        COUNT(*) as consecutive_days
    FROM consecutive_groups
    GROUP BY user_id, group_id
)
SELECT 
    user_id,
    MAX(consecutive_days) as max_consecutive_days
FROM consecutive_days
GROUP BY user_id
HAVING MAX(consecutive_days) >= 3;  -- 连续运动3天以上
```

**结果示例**：
```
user_id | max_consecutive_days
--------|----------------------
1       | 5
2       | 3
```

---

### 查询 6：用户心率趋势分析（窗口函数 - 移动平均）

**需求**：计算每个用户心率的 7 天移动平均

**使用窗口函数（移动平均）**：

```sql
SELECT 
    user_id,
    DATE(recorded_at) as date,
    AVG(heart_rate) as daily_avg_heart_rate,
    AVG(AVG(heart_rate)) OVER (
        PARTITION BY user_id 
        ORDER BY DATE(recorded_at) 
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) as moving_avg_7d  -- 7天移动平均
FROM heart_rate
WHERE recorded_at >= '2024-01-01'
  AND recorded_at < '2024-01-31'
GROUP BY user_id, DATE(recorded_at)
ORDER BY user_id, date;
```

**结果示例**：
```
user_id | date       | daily_avg_heart_rate | moving_avg_7d
--------|------------|---------------------|---------------
1       | 2024-01-01 | 82.5               | 82.5
1       | 2024-01-02 | 85.0               | 83.75
1       | 2024-01-03 | 88.0               | 85.17
...
1       | 2024-01-10 | 85.5               | 85.2
```

---

### 查询 7：用户健康报告（多表 JOIN + 条件聚合）

**需求**：生成用户每日健康报告（步数、心率、运动、睡眠）

**使用多表 JOIN 和条件聚合**：

```sql
WITH daily_steps AS (
    SELECT 
        user_id,
        DATE(recorded_at) as date,
        MAX(steps) as daily_steps
    FROM steps
    WHERE recorded_at >= '2024-01-10'
      AND recorded_at < '2024-01-11'
    GROUP BY user_id, DATE(recorded_at)
),
daily_heart_rate AS (
    SELECT 
        user_id,
        DATE(recorded_at) as date,
        AVG(heart_rate) as avg_heart_rate,
        MAX(heart_rate) as max_heart_rate,
        MIN(heart_rate) as min_heart_rate
    FROM heart_rate
    WHERE recorded_at >= '2024-01-10'
      AND recorded_at < '2024-01-11'
    GROUP BY user_id, DATE(recorded_at)
),
daily_activities AS (
    SELECT 
        user_id,
        DATE(started_at) as date,
        SUM(calories_burned) as total_calories,
        SUM(duration_minutes) as total_minutes,
        COUNT(*) as activity_count
    FROM activities
    WHERE DATE(started_at) = '2024-01-10'
    GROUP BY user_id, DATE(started_at)
),
daily_sleep AS (
    SELECT 
        user_id,
        DATE(sleep_start) as date,
        sleep_duration_minutes,
        sleep_quality
    FROM sleep
    WHERE DATE(sleep_start) = '2024-01-10'
)
SELECT 
    u.user_id,
    u.name,
    ds.daily_steps,
    dhr.avg_heart_rate,
    dhr.max_heart_rate,
    dhr.min_heart_rate,
    da.total_calories,
    da.total_minutes,
    da.activity_count,
    dsl.sleep_duration_minutes,
    dsl.sleep_quality
FROM users u
LEFT JOIN daily_steps ds ON u.user_id = ds.user_id
LEFT JOIN daily_heart_rate dhr ON u.user_id = dhr.user_id
LEFT JOIN daily_activities da ON u.user_id = da.user_id
LEFT JOIN daily_sleep dsl ON u.user_id = dsl.user_id
ORDER BY u.user_id;
```

**结果示例**：
```
user_id | name   | daily_steps | avg_heart_rate | max_heart_rate | min_heart_rate | total_calories | total_minutes | activity_count | sleep_duration_minutes | sleep_quality
--------|--------|-------------|----------------|----------------|----------------|----------------|---------------|----------------|------------------------|---------------
1       | Alice  | 15000      | 85.5          | 145           | 68            | 900           | 135          | 3              | 480                   | Good
2       | Bob    | 12000      | 78.0          | 145           | 65            | 950           | 105          | 2              | 480                   | Good
```

---

### 查询 8：异常心率检测（窗口函数 + 条件筛选）

**需求**：检测用户心率异常（心率突然升高或降低）

**使用窗口函数（LAG）**：

```sql
WITH heart_rate_with_prev AS (
    SELECT 
        user_id,
        heart_rate,
        recorded_at,
        LAG(heart_rate) OVER (PARTITION BY user_id ORDER BY recorded_at) as prev_heart_rate,
        ABS(heart_rate - LAG(heart_rate) OVER (PARTITION BY user_id ORDER BY recorded_at)) as heart_rate_change
    FROM heart_rate
    WHERE recorded_at >= '2024-01-10'
      AND recorded_at < '2024-01-11'
)
SELECT 
    user_id,
    heart_rate,
    prev_heart_rate,
    heart_rate_change,
    recorded_at
FROM heart_rate_with_prev
WHERE heart_rate_change > 50  -- 心率变化超过 50 次/分钟
ORDER BY user_id, recorded_at;
```

**结果示例**：
```
user_id | heart_rate | prev_heart_rate | heart_rate_change | recorded_at
--------|------------|-----------------|-------------------|-------------------
1       | 140       | 68             | 72               | 2024-01-10 07:00:00
1       | 68        | 145            | 77               | 2024-01-10 07:30:00
```

---

## ⚡ 性能优化实践

### 优化 1：索引优化

**问题**：查询用户某一天的数据很慢

**解决方案**：

```sql
-- 1. 在 JOIN 键上建索引
CREATE INDEX idx_activities_user_id ON activities(user_id);
CREATE INDEX idx_heart_rate_user_id ON heart_rate(user_id);

-- 2. 在时间列上建索引（支持时间范围查询）
CREATE INDEX idx_activities_started_at ON activities(started_at);
CREATE INDEX idx_heart_rate_recorded_at ON heart_rate(recorded_at);

-- 3. 复合索引（支持 WHERE user_id = ? AND date = ?）
CREATE INDEX idx_activities_user_date ON activities(user_id, started_at);
CREATE INDEX idx_heart_rate_user_date ON heart_rate(user_id, recorded_at);
```

---

### 优化 2：覆盖索引

**问题**：查询用户每日步数时，需要回表查询

**解决方案**：

```sql
-- 如果查询只需要 user_id, recorded_at, steps
-- 创建覆盖索引
CREATE INDEX idx_steps_covering ON steps(user_id, recorded_at, steps);
-- 索引包含所有需要的列，不需要回表
```

---

### 优化 3：分区表

**问题**：heart_rate 表有几亿行数据，查询很慢

**解决方案**：

```sql
-- 按日期分区
CREATE TABLE heart_rate (
    id INT PRIMARY KEY AUTO_INCREMENT,
    user_id INT NOT NULL,
    heart_rate INT NOT NULL,
    recorded_at TIMESTAMP NOT NULL,
    INDEX idx_user_date (user_id, recorded_at)
) PARTITION BY RANGE (TO_DAYS(recorded_at)) (
    PARTITION p20240101 VALUES LESS THAN (TO_DAYS('2024-01-02')),
    PARTITION p20240102 VALUES LESS THAN (TO_DAYS('2024-01-03')),
    -- ... 更多分区
);

-- 查询时，数据库自动使用分区剪枝
-- 只查询相关分区，大大提升性能
SELECT * FROM heart_rate 
WHERE recorded_at >= '2024-01-10' 
  AND recorded_at < '2024-01-11';
```

---

### 优化 4：物化视图（Materialized View）

**问题**：每日健康报告查询很慢（需要 JOIN 多个表）

**解决方案**：

```sql
-- 创建物化视图（MySQL 不支持，但可以使用定期更新的表）
CREATE TABLE daily_health_summary (
    user_id INT,
    date DATE,
    daily_steps INT,
    avg_heart_rate DECIMAL(5,2),
    total_calories INT,
    sleep_duration_minutes INT,
    PRIMARY KEY (user_id, date),
    INDEX idx_date (date)
);

-- 定期更新物化视图（每天凌晨更新）
INSERT INTO daily_health_summary (user_id, date, daily_steps, avg_heart_rate, total_calories, sleep_duration_minutes)
SELECT 
    u.user_id,
    CURDATE() as date,
    MAX(s.steps) as daily_steps,
    AVG(h.heart_rate) as avg_heart_rate,
    COALESCE(SUM(a.calories_burned), 0) as total_calories,
    sl.sleep_duration_minutes
FROM users u
LEFT JOIN steps s ON u.user_id = s.user_id AND DATE(s.recorded_at) = CURDATE()
LEFT JOIN heart_rate h ON u.user_id = h.user_id AND DATE(h.recorded_at) = CURDATE()
LEFT JOIN activities a ON u.user_id = a.user_id AND DATE(a.started_at) = CURDATE()
LEFT JOIN sleep sl ON u.user_id = sl.user_id AND DATE(sl.sleep_start) = CURDATE()
GROUP BY u.user_id, CURDATE(), sl.sleep_duration_minutes;

-- 查询时直接使用物化视图（很快）
SELECT * FROM daily_health_summary WHERE date = '2024-01-10';
```

---

## 📝 总结与学习要点

### Pipeline 完整流程

1. **数据收集**：从 Apple Watch 收集原始数据
2. **数据清洗**：清洗和验证数据（去重、异常检测等）
3. **数据存储**：存储到数据库（考虑分区）
4. **数据处理**：聚合和转换数据
5. **数据分析**：生成报告和洞察

### SQL 技术要点

**本案例涵盖的技术**：

1. ✅ **基础聚合**（GROUP BY, SUM, AVG, COUNT）
   - 查询 1：每日步数统计
   - 查询 2：每日平均心率

2. ✅ **窗口函数**（ROW_NUMBER, LAG, 移动平均）
   - 查询 3：运动时长排名
   - 查询 6：心率趋势分析
   - 查询 8：异常心率检测

3. ✅ **JOIN 操作**（LEFT JOIN, INNER JOIN）
   - 查询 4：每日卡路里消耗
   - 查询 7：每日健康报告

4. ✅ **连续性问题**（Gaps & Islands）
   - 查询 5：连续运动天数

5. ✅ **性能优化**（索引、分区、物化视图）
   - 优化 1：索引优化
   - 优化 2：覆盖索引
   - 优化 3：分区表
   - 优化 4：物化视图

### 学习建议

1. **逐步练习**：
   - 先理解表结构设计
   - 再练习基础查询（查询 1-2）
   - 然后练习复杂查询（查询 3-8）
   - 最后学习性能优化

2. **理解业务逻辑**：
   - 理解每个查询的业务需求
   - 思考为什么这样设计
   - 理解 SQL 如何实现业务逻辑

3. **性能优化思考**：
   - 思考为什么慢？
   - 如何优化？
   - 优化后的性能如何？

4. **实际应用**：
   - 可以基于这个场景扩展更多查询
   - 可以设计更多业务场景
   - 可以实践性能优化

---

## 🎯 扩展练习

### 练习 1：用户运动习惯分析

**需求**：分析用户最喜欢的运动类型

```sql
-- 你的 SQL 代码
```

### 练习 2：用户健康趋势分析

**需求**：分析用户过去 7 天的健康趋势（心率、步数）

```sql
-- 你的 SQL 代码
```

### 练习 3：异常检测

**需求**：检测睡眠质量下降的用户（连续 3 天睡眠质量为 Poor）

```sql
-- 你的 SQL 代码
```

---

## 📚 相关资源

- [SQL 窗口函数文档](https://dev.mysql.com/doc/refman/8.0/en/window-functions.html)
- [MySQL 分区文档](https://dev.mysql.com/doc/refman/8.0/en/partitioning.html)
- [数据库索引优化指南](../B01_sql_interview_prep.md)
