# ETL 详解：数据工程师必知

## 🎯 ETL 是什么？

### 简单来说

```
ETL = Extract（提取） + Transform（转换） + Load（加载）

就是把数据从"原始状态"变成"可用状态"的过程。
```

### 用生活例子理解

```
想象你在整理房间：

Extract（提取）：
  - 从各个地方收集东西（衣柜、抽屉、书架）
  - 对应：从各个数据源收集数据（数据库、API、文件）

Transform（转换）：
  - 分类整理（衣服放衣柜、书放书架）
  - 清洗处理（扔掉垃圾、修复破损）
  - 对应：数据清洗、格式转换、数据聚合

Load（加载）：
  - 放到合适的位置（整理好的衣柜、书架）
  - 对应：存入数据仓库，供后续使用
```

---

## 📊 ETL 的三个步骤详解

### 1. Extract（提取）- 从数据源获取数据

**实际工作场景**：

```
数据源类型：
  ✅ 数据库（MySQL、PostgreSQL、MongoDB）
  ✅ API（REST API、GraphQL）
  ✅ 文件（CSV、JSON、Parquet）
  ✅ 消息队列（Kafka、RabbitMQ）
  ✅ 云存储（S3、GCS、Azure Blob）

实际例子（Apple App Store）：
  - 从 App Store 数据库提取用户下载数据
  - 从支付系统 API 提取购买数据
  - 从日志文件提取用户行为数据
```

**代码示例**：

```python
# Extract：从多个数据源提取数据

# 1. 从数据库提取
import pandas as pd
from sqlalchemy import create_engine

engine = create_engine('postgresql://user:pass@localhost/db')
df_users = pd.read_sql('SELECT * FROM users', engine)

# 2. 从 API 提取
import requests

response = requests.get('https://api.example.com/data')
api_data = response.json()

# 3. 从文件提取
df_logs = pd.read_csv('user_logs.csv')
df_json = pd.read_json('data.json')
```

### 2. Transform（转换）- 清洗和转换数据

**实际工作场景**：

```
转换操作：
  ✅ 数据清洗（去重、处理缺失值、处理异常值）
  ✅ 数据格式转换（字符串转日期、单位转换）
  ✅ 数据聚合（按时间、地区、用户分组）
  ✅ 数据关联（JOIN 多个数据源）
  ✅ 数据计算（计算新字段、统计指标）

实际例子（Apple App Store）：
  - 清洗：去除重复的下载记录
  - 转换：将时间戳转换为日期格式
  - 聚合：按日期、地区统计下载量
  - 关联：关联用户信息和应用信息
  - 计算：计算用户留存率、应用评分
```

**代码示例**：

```python
# Transform：清洗和转换数据

# 1. 数据清洗
# 去重
df = df.drop_duplicates()

# 处理缺失值
df['age'] = df['age'].fillna(df['age'].mean())  # 用平均值填充
df = df.dropna(subset=['user_id'])  # 删除关键字段缺失的行

# 处理异常值
df = df[df['price'] > 0]  # 删除价格为负的记录
df = df[df['age'].between(0, 120)]  # 删除年龄异常的记录

# 2. 数据格式转换
df['created_at'] = pd.to_datetime(df['created_at'])
df['price_usd'] = df['price_cny'] / 7.0  # 货币转换

# 3. 数据聚合
daily_stats = df.groupby('date').agg({
    'downloads': 'sum',
    'revenue': 'sum',
    'users': 'nunique'
}).reset_index()

# 4. 数据关联
df_merged = df_users.merge(
    df_downloads, 
    on='user_id', 
    how='left'
)

# 5. 数据计算
df['total_spent'] = df['price'] * df['quantity']
df['is_premium'] = df['total_spent'] > 100
```

**SQL 示例**：

```sql
-- Transform：用 SQL 转换数据

-- 1. 数据清洗
SELECT DISTINCT user_id, app_id, download_date
FROM raw_downloads
WHERE download_date IS NOT NULL
  AND app_id IS NOT NULL;

-- 2. 数据格式转换
SELECT 
    user_id,
    DATE(download_timestamp) AS download_date,
    EXTRACT(HOUR FROM download_timestamp) AS download_hour
FROM raw_downloads;

-- 3. 数据聚合
SELECT 
    DATE(download_timestamp) AS date,
    COUNT(*) AS total_downloads,
    COUNT(DISTINCT user_id) AS unique_users,
    SUM(price) AS total_revenue
FROM raw_downloads
GROUP BY DATE(download_timestamp);

-- 4. 数据关联
SELECT 
    u.user_id,
    u.user_name,
    d.app_id,
    d.download_date,
    a.app_name
FROM users u
JOIN downloads d ON u.user_id = d.user_id
JOIN apps a ON d.app_id = a.app_id;

-- 5. 数据计算
SELECT 
    user_id,
    SUM(price) AS total_spent,
    CASE 
        WHEN SUM(price) > 100 THEN 'Premium'
        ELSE 'Regular'
    END AS user_type
FROM purchases
GROUP BY user_id;
```

### 3. Load（加载）- 存入目标系统

**实际工作场景**：

```
目标系统：
  ✅ 数据仓库（Snowflake、BigQuery、Redshift）
  ✅ 数据库（PostgreSQL、MySQL）
  ✅ 数据湖（S3、HDFS）
  ✅ 分析工具（Tableau、Looker）

实际例子（Apple App Store）：
  - 将清洗后的数据存入 Snowflake 数据仓库
  - 提供给 Tableau 做报表
  - 提供给 API 服务查询
```

**代码示例**：

```python
# Load：存入目标系统

# 1. 存入数据仓库（Snowflake）
from snowflake.connector import connect

conn = connect(
    user='user',
    password='pass',
    account='account',
    warehouse='warehouse',
    database='database',
    schema='schema'
)

cursor = conn.cursor()
cursor.execute("""
    INSERT INTO app_store_stats 
    (date, downloads, revenue, users)
    VALUES (?, ?, ?, ?)
""", (date, downloads, revenue, users))

# 2. 存入数据库（PostgreSQL）
df_cleaned.to_sql(
    'app_store_stats',
    engine,
    if_exists='append',
    index=False
)

# 3. 存入文件（Parquet）
df_cleaned.to_parquet('app_store_stats.parquet')

# 4. 存入数据湖（S3）
import boto3

s3 = boto3.client('s3')
df_cleaned.to_parquet('s3://bucket/app_store_stats.parquet')
```

---

## 🔄 ETL 的完整流程示例

### 实际案例：Apple App Store 用户行为分析

```
需求：
  - 分析 App Store 的用户下载和购买行为
  - 数据源：多个系统（App Store、支付系统、用户系统）
  - 目标：存入数据仓库，供分析师查询

ETL 流程：

1. Extract（提取）
   - 从 App Store 数据库提取下载数据
   - 从支付系统 API 提取购买数据
   - 从用户系统提取用户信息

2. Transform（转换）
   - 清洗：去除重复记录、处理缺失值
   - 转换：时间戳转日期、货币转换
   - 聚合：按日期、地区、应用类型统计
   - 关联：关联用户信息、应用信息
   - 计算：计算留存率、转化率

3. Load（加载）
   - 存入 Snowflake 数据仓库
   - 提供给 Tableau 做报表
   - 提供给 API 服务查询
```

**完整代码示例**：

```python
# ETL 完整流程：Apple App Store 用户行为分析

import pandas as pd
from sqlalchemy import create_engine
import requests
from datetime import datetime

# ========== Extract（提取）==========
print("Step 1: Extract - 从数据源提取数据")

# 1. 从数据库提取下载数据
engine = create_engine('postgresql://user:pass@localhost/appstore')
df_downloads = pd.read_sql('''
    SELECT user_id, app_id, download_timestamp, country
    FROM downloads
    WHERE download_timestamp >= CURRENT_DATE - INTERVAL '7 days'
''', engine)

# 2. 从 API 提取购买数据
response = requests.get('https://api.payment.apple.com/purchases')
purchases_data = response.json()
df_purchases = pd.DataFrame(purchases_data)

# 3. 从文件提取用户信息
df_users = pd.read_csv('users.csv')

print(f"提取完成：下载数据 {len(df_downloads)} 条，购买数据 {len(df_purchases)} 条")

# ========== Transform（转换）==========
print("Step 2: Transform - 清洗和转换数据")

# 1. 数据清洗
# 去重
df_downloads = df_downloads.drop_duplicates()
df_purchases = df_purchases.drop_duplicates()

# 处理缺失值
df_downloads = df_downloads.dropna(subset=['user_id', 'app_id'])
df_purchases = df_purchases.fillna({'price': 0})

# 处理异常值
df_purchases = df_purchases[df_purchases['price'] >= 0]

# 2. 数据格式转换
df_downloads['download_date'] = pd.to_datetime(df_downloads['download_timestamp']).dt.date
df_purchases['purchase_date'] = pd.to_datetime(df_purchases['purchase_timestamp']).dt.date

# 3. 数据关联
df_merged = df_downloads.merge(
    df_users,
    on='user_id',
    how='left'
).merge(
    df_purchases,
    on=['user_id', 'app_id'],
    how='left'
)

# 4. 数据聚合
daily_stats = df_merged.groupby('download_date').agg({
    'user_id': 'nunique',
    'app_id': 'count',
    'price': 'sum'
}).reset_index()

daily_stats.columns = ['date', 'unique_users', 'total_downloads', 'total_revenue']

# 5. 数据计算
daily_stats['avg_revenue_per_user'] = daily_stats['total_revenue'] / daily_stats['unique_users']
daily_stats['conversion_rate'] = (df_merged['price'].notna().sum() / len(df_merged)) * 100

print(f"转换完成：生成 {len(daily_stats)} 条统计记录")

# ========== Load（加载）==========
print("Step 3: Load - 存入目标系统")

# 1. 存入数据仓库
daily_stats.to_sql(
    'app_store_daily_stats',
    engine,
    if_exists='append',
    index=False
)

# 2. 存入文件（备份）
daily_stats.to_parquet('app_store_daily_stats.parquet')

print("加载完成：数据已存入数据仓库和文件")

print("ETL 流程完成！")
```

---

## 🛠️ ETL 工具和框架

### 常用 ETL 工具

#### 1. Python 库

```
数据处理：
  - Pandas（最常用）
  - NumPy
  - PySpark（大数据处理）

数据库连接：
  - SQLAlchemy
  - psycopg2（PostgreSQL）
  - pymongo（MongoDB）

API 调用：
  - requests
  - httpx

文件处理：
  - pandas（CSV、JSON、Parquet）
  - openpyxl（Excel）
```

#### 2. 调度工具

```
Airflow（最常用）：
  - 调度 ETL 作业
  - 监控作业状态
  - 处理依赖关系

Luigi：
  - 类似 Airflow
  - 更轻量级

Prefect：
  - 现代化的调度工具
  - 更好的错误处理
```

#### 3. 大数据处理

```
Spark：
  - 处理大规模数据
  - 分布式计算

Hadoop：
  - 大数据生态系统
  - HDFS 存储

Flink：
  - 流处理
  - 实时 ETL
```

---

## 📅 ETL 在实际工作中的使用

### 典型的一天

```
早上：
  - 检查昨晚的 ETL 作业是否成功
  - 如果有失败，查看日志，修复问题

上午：
  - 开发新的 ETL 作业
  - 写 Extract 代码（从数据源提取）
  - 写 Transform 代码（清洗和转换）
  - 写 Load 代码（存入目标系统）

下午：
  - 优化现有 ETL 作业性能
  - 处理数据质量问题
  - 支持数据分析师的需求
```

### 实际工作占比

```
ETL 开发：40%
  - 写 Extract 代码
  - 写 Transform 代码
  - 写 Load 代码

ETL 维护：30%
  - 监控作业状态
  - 修复失败作业
  - 优化性能

数据质量：20%
  - 检查数据质量
  - 处理数据异常
  - 更新数据清洗逻辑

其他：10%
  - 文档编写
  - 团队协作
```

---

## 🎯 ETL vs ELT

### 传统 ETL（Extract → Transform → Load）

```
流程：
  数据源 → 提取 → 转换（在 ETL 工具中） → 加载到数据仓库

优点：
  ✅ 数据在加载前已经清洗好
  ✅ 数据仓库存储的是干净的数据

缺点：
  ❌ 转换逻辑在 ETL 工具中，不够灵活
  ❌ 数据仓库的计算能力没有被充分利用
```

### 现代 ELT（Extract → Load → Transform）

```
流程：
  数据源 → 提取 → 加载到数据仓库 → 转换（在数据仓库中）

优点：
  ✅ 利用数据仓库的强大计算能力
  ✅ 转换逻辑更灵活（用 SQL）
  ✅ 可以保留原始数据（数据湖）

缺点：
  ❌ 数据仓库需要存储原始数据
  ❌ 需要更多的存储空间
```

### 实际选择

```
小规模数据：
  - 使用 ETL（传统方式）
  - 在 ETL 工具中转换

大规模数据：
  - 使用 ELT（现代方式）
  - 在数据仓库中转换（Snowflake、BigQuery）
```

---

## 💡 ETL 最佳实践

### 1. 数据质量检查

```
在 Transform 阶段：
  ✅ 检查数据完整性（有没有缺失值？）
  ✅ 检查数据准确性（数据是否符合预期？）
  ✅ 检查数据一致性（不同数据源的数据是否一致？）

代码示例：
  if df.isnull().sum().sum() > 0:
      raise ValueError("数据有缺失值！")
  
  if df['price'].min() < 0:
      raise ValueError("价格不能为负！")
```

### 2. 错误处理

```
处理异常情况：
  ✅ 数据源不可用（API 失败、数据库连接失败）
  ✅ 数据格式错误（JSON 解析失败、日期格式错误）
  ✅ 数据质量问题（缺失值、异常值）

代码示例：
  try:
      df = pd.read_csv('data.csv')
  except FileNotFoundError:
      print("文件不存在，使用备用数据源")
      df = pd.read_csv('backup_data.csv')
```

### 3. 增量更新

```
不要每次都全量更新：
  ✅ 只处理新增的数据
  ✅ 只处理更新的数据
  ✅ 提高 ETL 性能

代码示例：
  # 只处理今天的数据
  df_new = df[df['date'] == datetime.today().date()]
  
  # 只处理更新的数据
  df_updated = df[df['updated_at'] > last_run_time]
```

### 4. 监控和日志

```
记录 ETL 执行情况：
  ✅ 记录处理的数据量
  ✅ 记录处理时间
  ✅ 记录错误信息

代码示例：
  import logging
  
  logging.basicConfig(level=logging.INFO)
  logger = logging.getLogger(__name__)
  
  logger.info(f"开始处理 {len(df)} 条数据")
  # ... ETL 处理 ...
  logger.info(f"处理完成，成功 {success_count} 条，失败 {fail_count} 条")
```

---

## 🎯 总结

### ETL 的核心

```
ETL = Extract（提取） + Transform（转换） + Load（加载）

就是把数据从"原始状态"变成"可用状态"的过程。
```

### 实际工作

```
Data Engineer 的 80% 工作都是 ETL：
  ✅ 从数据源提取数据
  ✅ 清洗和转换数据
  ✅ 存入数据仓库

需要的技能：
  ✅ SQL（数据转换、查询）
  ✅ Python（ETL 脚本、数据处理）
  ✅ 数据管道工具（Airflow、Spark）
  ✅ 数据质量思维
```

### 不需要的技能

```
❌ 复杂的算法（动态规划、图算法）
❌ LeetCode Hard
❌ 机器学习模型训练
❌ 深度学习
```

---

**记住：ETL 是 Data Engineer 的核心工作，就是把数据从"原始状态"变成"可用状态"。**

**你的 SQL 和 Python 技能，已经足够做 ETL 了！** 🚀

