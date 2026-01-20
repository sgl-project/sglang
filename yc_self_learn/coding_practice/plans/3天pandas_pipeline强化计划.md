# 3 天 pandas + Pipeline 强化学习计划

## 📅 总体目标

**目标**：3 天内掌握 pandas 基础，同时理解 ETL Pipeline 概念

**策略**：
- 每天 4-6 小时（白天 + 晚上）
- **边学边练**：看概念 → 写代码 → 做练习
- **实战导向**：用实际题目练习（A02、A03、A04）
- **Pipeline 思维**：每做一个练习都思考"如何改造成 pipeline"

---

## 🎯 Day 1: pandas 基础 + 数据提取（Extract）

### **时间安排**
- **上午**（2-3 小时）：pandas 基础操作
- **下午**（2-3 小时）：数据读取和清洗
- **晚上**（1-2 小时）：练习 + 总结

### **学习内容**

#### **上午：pandas 基础操作**

**1. 基本数据结构（30 分钟）**

```python
import pandas as pd
import numpy as np

# Series（一维数组）
s = pd.Series([1, 2, 3, 4, 5], name='numbers')
print(s)
print(s.head())
print(s.describe())

# DataFrame（二维表格）
df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35],
    'salary': [50000, 60000, 70000]
})
print(df)
print(df.head())
print(df.info())
print(df.describe())
```

**2. 数据查看和选择（30 分钟）**

```python
# 查看数据
df.head()        # 前 5 行
df.tail()        # 后 5 行
df.info()        # 数据类型和缺失值
df.describe()    # 统计摘要
df.shape         # 行数和列数
df.columns       # 列名
df.dtypes        # 数据类型

# 选择数据
df['name']                    # 选择一列（Series）
df[['name', 'age']]           # 选择多列（DataFrame）
df[0:3]                       # 选择前 3 行
df.loc[0]                     # 选择第一行
df.loc[0:2, 'name']           # 选择前 3 行的 name 列
df.iloc[0:3, 0:2]             # 用位置索引选择
```

**3. 数据筛选（30 分钟）**

```python
# 条件筛选（重要！）
df[df['age'] > 25]                           # 年龄大于 25
df[(df['age'] > 25) & (df['salary'] > 50000)]  # 多个条件（AND）
df[(df['age'] > 30) | (df['salary'] > 60000)]  # 多个条件（OR）

# 使用 isin
df[df['name'].isin(['Alice', 'Bob'])]

# 使用 query（可选，更简洁）
df.query('age > 25 and salary > 50000')
```

**4. 数据排序（30 分钟）**

```python
# 排序
df.sort_values('age')                    # 升序
df.sort_values('age', ascending=False)   # 降序
df.sort_values(['age', 'salary'], ascending=[True, False])  # 多列排序

# 重置索引
df = df.sort_values('age').reset_index(drop=True)
```

**实践练习（30 分钟）**：
- 用你之前的数据项目，用 pandas 重做一遍基本操作
- 或者做 A06（SQL 版本）的 pandas 版本练习

---

#### **下午：数据读取和清洗（Extract + Transform）**

**1. 数据读取（30 分钟）**

```python
# 读取 CSV（最常用）
df = pd.read_csv('data.csv')

# 读取 Excel
df = pd.read_excel('data.xlsx')

# 读取数据库（SQLite 示例）
import sqlite3
conn = sqlite3.connect('database.db')
df = pd.read_sql_query("SELECT * FROM table_name", conn)
conn.close()

# 或者用 SQLAlchemy
from sqlalchemy import create_engine
engine = create_engine('sqlite:///database.db')
df = pd.read_sql_query("SELECT * FROM table_name", engine)
```

**2. 缺失值处理（30 分钟）**

```python
# 检查缺失值
df.isna()                    # 返回布尔 DataFrame
df.isna().sum()              # 每列缺失值数量
df.isna().sum() / len(df)    # 每列缺失值比例

# 处理缺失值
df.dropna()                  # 删除包含缺失值的行
df.dropna(subset=['column']) # 删除特定列的缺失值
df.fillna(0)                 # 用 0 填充
df.fillna({'col1': 0, 'col2': 'unknown'})  # 用不同值填充不同列
df.fillna(method='ffill')    # 用前一个值填充
df.fillna(method='bfill')    # 用后一个值填充
df.fillna(df.mean())         # 用均值填充（数值列）
```

**3. 数据类型转换（30 分钟）**

```python
# 查看数据类型
df.dtypes

# 转换数据类型
df['age'] = df['age'].astype(int)
df['date'] = pd.to_datetime(df['date'])
df['value'] = pd.to_numeric(df['value'], errors='coerce')  # errors='coerce' 会把无法转换的值变成 NaN

# 批量转换
df = df.convert_dtypes()  # 自动推断最佳类型
```

**4. 数据清洗实践（30 分钟）**

```python
# 完整的清洗流程
def clean_data(df):
    """清洗数据的 Pipeline 函数"""
    # 1. 检查数据
    print(f"原始数据: {len(df)} 行, {len(df.columns)} 列")
    print(f"缺失值: {df.isna().sum().sum()} 个")
    
    # 2. 删除重复行
    df = df.drop_duplicates()
    print(f"删除重复后: {len(df)} 行")
    
    # 3. 处理缺失值
    # 数值列用均值填充
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    
    # 文本列用众数填充（或删除）
    text_cols = df.select_dtypes(include=['object']).columns
    for col in text_cols:
        mode_value = df[col].mode()[0] if not df[col].mode().empty else 'unknown'
        df[col] = df[col].fillna(mode_value)
    
    # 4. 类型转换
    # 自动转换（如果需要）
    
    # 5. 验证数据
    print(f"清洗后: {len(df)} 行, {df.isna().sum().sum()} 个缺失值")
    
    return df
```

**实践练习（1 小时）**：
- **练习 1**：用 A04 的数据清洗部分练习
  - 读取数据
  - 处理缺失值
  - 类型转换
  
- **练习 2**：设计一个 Extract + Clean 的 Pipeline 函数
  ```python
  def extract_and_clean(source_file):
      """Extract 和 Clean 的组合 Pipeline"""
      # Extract
      df = pd.read_csv(source_file)
      
      # Clean
      df = clean_data(df)
      
      return df
  ```

---

#### **晚上：练习 + 总结**

**练习（1 小时）**：
1. 用 A04 的第一部分（数据清洗）练习
2. 设计一个简单的 Extract + Clean Pipeline
3. 思考：如何让这个 Pipeline 更稳定、更易维护？

**总结（30 分钟）**：
- 回顾今天学的 pandas 操作
- 理解 Extract 和 Clean 的概念
- 准备明天的学习内容

---

## 🎯 Day 2: pandas 进阶 + 数据转换（Transform）

### **时间安排**
- **上午**（2-3 小时）：数据合并和分组聚合
- **下午**（2-3 小时）：数据转换和特征工程
- **晚上**（1-2 小时）：练习 + 总结

### **学习内容**

#### **上午：数据合并和分组聚合**

**1. 数据合并（1 小时）**

```python
# 连接（类似 SQL 的 JOIN）
df1 = pd.DataFrame({'key': ['A', 'B', 'C'], 'value1': [1, 2, 3]})
df2 = pd.DataFrame({'key': ['B', 'C', 'D'], 'value2': [4, 5, 6]})

# Inner Join（只保留两边都有的 key）
pd.merge(df1, df2, on='key', how='inner')

# Left Join（保留左边的所有行）
pd.merge(df1, df2, on='key', how='left')

# Right Join（保留右边的所有行）
pd.merge(df1, df2, on='key', how='right')

# Outer Join（保留所有行）
pd.merge(df1, df2, on='key', how='outer')

# 如果列名不同
df1.merge(df2, left_on='key1', right_on='key2', how='inner')

# 多列合并
pd.merge(df1, df2, on=['key1', 'key2'], how='inner')

# 拼接（类似 SQL 的 UNION）
pd.concat([df1, df2], axis=0)  # 垂直拼接（增加行）
pd.concat([df1, df2], axis=1)  # 水平拼接（增加列）
```

**2. 分组聚合（1 小时）**

```python
# 基本分组
df.groupby('category')['value'].mean()           # 每个类别的平均值
df.groupby('category')['value'].sum()            # 每个类别的总和
df.groupby('category')['value'].count()          # 每个类别的计数
df.groupby('category')['value'].min()            # 每个类别的最小值
df.groupby('category')['value'].max()            # 每个类别的最大值

# 多列分组
df.groupby(['category', 'subcategory'])['value'].mean()

# 多列聚合
df.groupby('category').agg({
    'value1': 'mean',
    'value2': ['sum', 'count'],
    'value3': lambda x: x.max() - x.min()  # 自定义函数
})

# 分组后应用多个函数
df.groupby('category')['value'].agg(['mean', 'std', 'min', 'max'])

# 分组后过滤
df.groupby('category').filter(lambda x: x['value'].mean() > 100)

# 分组后转换（不改变行数）
df['group_mean'] = df.groupby('category')['value'].transform('mean')
```

**3. 窗口函数（30 分钟）**

```python
# 滚动窗口（类似 SQL 的窗口函数）
df['rolling_mean'] = df['value'].rolling(window=3).mean()  # 3 行滚动平均
df['rolling_sum'] = df['value'].rolling(window=3).sum()

# 累计
df['cumsum'] = df['value'].cumsum()        # 累计和
df['cummax'] = df['value'].cummax()        # 累计最大值
df['cummin'] = df['value'].cummin()        # 累计最小值

# 排名
df['rank'] = df['value'].rank()            # 排名（并列会跳号）
df['dense_rank'] = df['value'].rank(method='dense')  # 密集排名（并列不跳号）
```

**实践练习（1 小时）**：
- **练习 1**：用 A04 的数据合并部分练习
  - 三个 DataFrame 的合并（INNER JOIN + LEFT JOIN）
  
- **练习 2**：用 A02 的数据统计部分练习
  - 分组聚合统计

---

#### **下午：数据转换和特征工程（Transform）**

**1. 数据转换（1 小时）**

```python
# 应用函数
df['new_col'] = df['value'].apply(lambda x: x * 2)
df['new_col'] = df.apply(lambda row: row['col1'] + row['col2'], axis=1)  # 行级别

# 使用 map（替换值）
df['category'] = df['category'].map({'A': 1, 'B': 2, 'C': 3})

# 使用 replace（替换值）
df['category'] = df['category'].replace({'A': 1, 'B': 2, 'C': 3})

# 分箱（binning）
df['age_group'] = pd.cut(df['age'], bins=[0, 18, 35, 60, 100], labels=['Child', 'Young', 'Middle', 'Old'])
df['age_group'] = pd.qcut(df['age'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])  # 按分位数分箱

# 字符串操作
df['name_upper'] = df['name'].str.upper()
df['name_length'] = df['name'].str.len()
df['email_domain'] = df['email'].str.split('@').str[1]
```

**2. 特征工程（1 小时）**

```python
# 创建新特征
df['total'] = df['price'] * df['quantity']
df['ratio'] = df['col1'] / df['col2']
df['difference'] = df['col1'] - df['col2']

# 时间特征
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['day_of_week'] = df['date'].dt.dayofweek

# One-Hot 编码
pd.get_dummies(df['category'], prefix='category')

# 数据透视表
pd.pivot_table(df, values='value', index='category', columns='subcategory', aggfunc='mean')
```

**3. Transform Pipeline 设计（30 分钟）**

```python
def transform_data(df):
    """数据转换 Pipeline 函数"""
    # 1. 创建新特征
    df['total'] = df['col1'] + df['col2']
    
    # 2. 分箱
    df['age_group'] = pd.cut(df['age'], bins=[0, 18, 35, 60, 100], labels=['Child', 'Young', 'Middle', 'Old'])
    
    # 3. 分组聚合（创建聚合特征）
    df['group_mean'] = df.groupby('category')['value'].transform('mean')
    
    # 4. 时间特征
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
    
    return df
```

**实践练习（1 小时）**：
- **练习 1**：用 A04 的 Transform 部分练习
  - 创建 tenureBinned 列（分箱）
  - 其他数据转换
  
- **练习 2**：设计一个完整的 Transform Pipeline 函数
  ```python
  def extract_transform_load(source_file, destination_file):
      """完整的 ETL Pipeline"""
      # Extract
      df = pd.read_csv(source_file)
      
      # Transform
      df = clean_data(df)
      df = transform_data(df)
      
      # Load
      df.to_csv(destination_file, index=False)
      
      return df
  ```

---

#### **晚上：练习 + 总结**

**练习（1 小时）**：
1. 用 A04 的完整流程练习（Extract + Transform）
2. 设计一个简单的 ETL Pipeline 类
3. 思考：如何让 Pipeline 更模块化、更易扩展？

**总结（30 分钟）**：
- 回顾今天学的 pandas 操作
- 理解 Transform 的概念
- 理解 Pipeline 的模块化设计
- 准备明天的学习内容

---

## 🎯 Day 3: 完整 Pipeline + 实战项目

### **时间安排**
- **上午**（2-3 小时）：完整 ETL Pipeline 设计和实现
- **下午**（2-3 小时）：实战项目（重做 A02、A04）
- **晚上**（1-2 小时）：总结 + 查漏补缺

### **学习内容**

#### **上午：完整 ETL Pipeline 设计和实现**

**1. Pipeline 类设计（1 小时）**

```python
import pandas as pd
import numpy as np
import logging
from typing import Optional, Callable

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataExtractor:
    """数据提取器"""
    def extract(self, source: str) -> pd.DataFrame:
        """从数据源提取数据"""
        try:
            logger.info(f"Extracting data from {source}")
            df = pd.read_csv(source)
            logger.info(f"Extracted {len(df)} rows, {len(df.columns)} columns")
            return df
        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            raise

class DataCleaner:
    """数据清洗器"""
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """清洗数据"""
        logger.info(f"Cleaning data: {len(df)} rows")
        
        # 记录清洗前的数据量
        before = len(df)
        
        # 删除重复行
        df = df.drop_duplicates()
        
        # 处理缺失值（根据实际情况调整）
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        
        text_cols = df.select_dtypes(include=['object']).columns
        for col in text_cols:
            mode_value = df[col].mode()[0] if not df[col].mode().empty else 'unknown'
            df[col] = df[col].fillna(mode_value)
        
        # 记录清洗后的数据量
        after = len(df)
        logger.info(f"Cleaned data: {before} -> {after} rows")
        
        return df

class DataTransformer:
    """数据转换器"""
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """转换数据"""
        logger.info(f"Transforming data: {len(df)} rows")
        
        # 根据实际需求添加转换逻辑
        # 例如：创建新特征、分箱、分组聚合等
        
        return df

class DataLoader:
    """数据加载器"""
    def load(self, df: pd.DataFrame, destination: str):
        """加载数据到目标"""
        try:
            logger.info(f"Loading data to {destination}")
            df.to_csv(destination, index=False)
            logger.info(f"Loaded {len(df)} rows to {destination}")
        except Exception as e:
            logger.error(f"Loading failed: {e}")
            raise

class ETLPipeline:
    """ETL Pipeline"""
    def __init__(self, extractor=None, cleaner=None, transformer=None, loader=None):
        self.extractor = extractor or DataExtractor()
        self.cleaner = cleaner or DataCleaner()
        self.transformer = transformer or DataTransformer()
        self.loader = loader or DataLoader()
    
    def run(self, source: str, destination: str) -> pd.DataFrame:
        """运行完整的 ETL Pipeline"""
        try:
            logger.info("Starting ETL Pipeline")
            
            # Extract
            df = self.extractor.extract(source)
            
            # Transform (Clean + Transform)
            df = self.cleaner.clean(df)
            df = self.transformer.transform(df)
            
            # Load
            self.loader.load(df, destination)
            
            logger.info("ETL Pipeline completed successfully")
            return df
        except Exception as e:
            logger.error(f"ETL Pipeline failed: {e}")
            raise
```

**2. Pipeline 使用示例（30 分钟）**

```python
# 使用 Pipeline
if __name__ == "__main__":
    pipeline = ETLPipeline()
    result = pipeline.run('input.csv', 'output.csv')
    print(result.head())
```

**3. Pipeline 扩展和定制（30 分钟）**

```python
# 自定义 Cleaner
class CustomDataCleaner(DataCleaner):
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """自定义清洗逻辑"""
        # 先调用父类的清洗方法
        df = super().clean(df)
        
        # 添加自定义清洗逻辑
        # 例如：删除特定条件的行、替换特定值等
        
        return df

# 使用自定义 Pipeline
pipeline = ETLPipeline(cleaner=CustomDataCleaner())
result = pipeline.run('input.csv', 'output.csv')
```

**实践练习（1 小时）**：
- **练习**：设计一个简单的 ETL Pipeline 类
- 测试：用你的数据测试 Pipeline
- 思考：如何让 Pipeline 更灵活、更易扩展？

---

#### **下午：实战项目（重做 A02、A04）**

**1. 重做 A02（pandas 版本，1 小时）**

**题目回顾**：房价数据分析和线性回归

**任务**：
- 用 pandas 完成数据分析部分（summary_list）
- 用 sklearn 完成线性回归部分（regression_list）
- 思考：如何改造成 Pipeline？

**代码框架**：
```python
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def analyse_and_fit_lrm(file_path):
    """房价数据分析和线性回归"""
    
    # Extract
    df = pd.read_csv(file_path)
    
    # ========== Summary List ==========
    # 1. Tax 统计量（Bathroom=2 & Bedroom=4）
    df_tax = df[(df['Bathroom'] == 2) & (df['Bedroom'] == 4)].dropna(subset=['Tax'])
    stats_vector = np.array([
        df_tax['Tax'].mean(),
        df_tax['Tax'].std(),
        df_tax['Tax'].median(),
        df_tax['Tax'].min(),
        df_tax['Tax'].max()
    ])
    
    # 2. Space > 800，按 Price 降序排序
    df_filtered = df[df['Space'] > 800].sort_values('Price', ascending=False)
    
    # 3. Lot >= 80% 分位数的数量
    q80 = df['Lot'].dropna().quantile(0.8)
    num_obs = (df['Lot'] >= q80).sum()
    
    summary_list = {
        'statistics': stats_vector,
        'data_frame': df_filtered,
        'number_of_observations': int(num_obs)
    }
    
    # ========== Regression List ==========
    # 1. 拟合线性回归
    df_clean = df.dropna()
    feature_columns = [col for col in df_clean.columns if col != 'Price']
    X = df_clean[feature_columns].values
    y = df_clean['Price'].values
    
    model = LinearRegression(fit_intercept=True)
    model.fit(X, y)
    
    # 2. 提取参数
    params = pd.Series(
        np.concatenate([[model.intercept_], model.coef_]),
        index=['Intercept'] + feature_columns
    )
    
    # 3. 预测新数据
    new_house = pd.DataFrame({
        'Bedroom': [3],
        'Space': [1500],
        'Room': [8],
        'Lot': [40],
        'Tax': [1000],
        'Bathroom': [2],
        'Garage': [1],
        'Condition': [0]
    })
    price_pred = float(model.predict(new_house[feature_columns].values)[0])
    
    regression_list = {
        'model_parameters': params,
        'price_prediction': price_pred
    }
    
    return {
        'summary_list': summary_list,
        'regression_list': regression_list
    }
```

**思考 Pipeline**：
```python
# 如何改造成 Pipeline？
class HouseDataPipeline:
    def extract(self, file_path):
        return pd.read_csv(file_path)
    
    def transform_summary(self, df):
        # Summary 分析逻辑
        pass
    
    def transform_regression(self, df):
        # 回归分析逻辑
        pass
    
    def run(self, file_path):
        df = self.extract(file_path)
        summary = self.transform_summary(df)
        regression = self.transform_regression(df)
        return {'summary_list': summary, 'regression_list': regression}
```

**2. 重做 A04（完整 Pipeline，1.5 小时）**

**题目回顾**：电信客户数据探索性分析

**任务**：
- 用 pandas 完成所有功能
- 改造成完整的 ETL Pipeline
- 思考：如何让 Pipeline 更模块化？

**代码框架**：
```python
def explanatory_analysis(charges_data_path, personal_data_path, plan_data_path):
    """数据探索性分析 Pipeline"""
    
    # Extract
    charges_data = pd.read_csv(charges_data_path)
    personal_data = pd.read_csv(personal_data_path)
    plan_data = pd.read_csv(plan_data_path)
    
    # Transform - Clean charges_data
    # 1. 类型转换
    numeric_cols = ['tenure', 'monthlyCharges', 'totalCharges']
    for col in numeric_cols:
        if col in charges_data.columns:
            charges_data[col] = pd.to_numeric(charges_data[col], errors='coerce')
    
    # 2. 计算 trimmed mean
    monthly_non_null = charges_data['monthlyCharges'].dropna().sort_values()
    n = len(monthly_non_null)
    k = int(np.floor(0.1 * n))
    trimmed = monthly_non_null[k:n-k] if 2*k < n else monthly_non_null
    monthly_charges_mean = int(round(trimmed.mean()))
    
    # 3. 填充缺失值
    charges_data['monthlyCharges'] = charges_data['monthlyCharges'].fillna(monthly_charges_mean)
    charges_data['totalCharges'] = charges_data['totalCharges'].fillna(
        charges_data['monthlyCharges'] * charges_data['tenure']
    )
    
    # 4. 分箱
    charges_data['tenureBinned'] = pd.cut(
        charges_data['tenure'],
        bins=[0, 24, 48, 60, np.inf],
        labels=['group1', 'group2', 'group3', 'group4'],
        right=True,
        include_lowest=True
    )
    
    charges_data_updated = charges_data.copy()
    
    # 5. 计算 churn_pct
    churn_pct = int(round((charges_data_updated['churn'] == 'Yes').mean() * 100))
    
    # Transform - Merge
    merged_cp = pd.merge(charges_data_updated, personal_data, on='customerID', how='inner')
    data_merged = pd.merge(merged_cp, plan_data, on='customerID', how='left')
    
    # Transform - Analysis
    pct_age_above_60 = int(round((data_merged['age'] > 60).mean() * 100))
    internet_service_counts = data_merged['internetService'].value_counts(dropna=True).to_dict()
    
    return {
        'monthly_charges_mean': monthly_charges_mean,
        'charges_data_updated': charges_data_updated,
        'churn_pct': churn_pct,
        'data_merged': data_merged,
        'pct_age_above_60': pct_age_above_60,
        'internet_service_counts': internet_service_counts
    }
```

**思考 Pipeline**：
```python
# 改造成 Pipeline 类
class TelecomDataPipeline:
    def extract(self, charges_path, personal_path, plan_path):
        return (
            pd.read_csv(charges_path),
            pd.read_csv(personal_path),
            pd.read_csv(plan_path)
        )
    
    def transform_charges(self, charges_df):
        # Charges 数据清洗和转换
        pass
    
    def transform_merge(self, charges_df, personal_df, plan_df):
        # 数据合并
        pass
    
    def transform_analysis(self, merged_df):
        # 数据分析
        pass
    
    def run(self, charges_path, personal_path, plan_path):
        charges, personal, plan = self.extract(charges_path, personal_path, plan_path)
        charges_updated = self.transform_charges(charges)
        merged = self.transform_merge(charges_updated, personal, plan)
        analysis = self.transform_analysis(merged)
        return analysis
```

---

#### **晚上：总结 + 查漏补缺**

**1. 知识点总结（30 分钟）**

**pandas 核心操作**：
- ✅ 数据读取：`pd.read_csv()`, `pd.read_excel()`, `pd.read_sql()`
- ✅ 数据查看：`head()`, `info()`, `describe()`, `shape`
- ✅ 数据选择：`[]`, `loc`, `iloc`, `query()`
- ✅ 数据筛选：`df[condition]`, `isin()`, `&`, `|`
- ✅ 数据排序：`sort_values()`, `reset_index()`
- ✅ 缺失值处理：`dropna()`, `fillna()`, `isna()`
- ✅ 类型转换：`astype()`, `pd.to_numeric()`, `pd.to_datetime()`
- ✅ 数据合并：`merge()`, `concat()`, `join()`
- ✅ 分组聚合：`groupby()`, `agg()`, `transform()`, `filter()`
- ✅ 数据转换：`apply()`, `map()`, `replace()`, `cut()`, `qcut()`
- ✅ 窗口函数：`rolling()`, `cumsum()`, `rank()`
- ✅ 数据保存：`to_csv()`, `to_excel()`, `to_sql()`

**Pipeline 核心概念**：
- ✅ Extract（提取）：从数据源读取数据
- ✅ Transform（转换）：清洗、转换、聚合数据
  - Clean（清洗）：处理缺失值、类型转换、去重
  - Transform（转换）：创建新特征、分箱、分组聚合
- ✅ Load（加载）：保存到目标位置（文件、数据库）
- ✅ 模块化设计：每个步骤独立，易于维护和扩展
- ✅ 错误处理：try-except、日志记录
- ✅ 数据验证：验证数据质量、异常检测

**2. 常见问题和解决方案（30 分钟）**

**问题 1**：如何快速上手 pandas？
- **答案**：从"10 minutes to pandas"开始，边学边练，重点掌握常用操作

**问题 2**：如何设计 Pipeline？
- **答案**：按照 ETL 流程（Extract → Transform → Load）分解，每个步骤独立函数

**问题 3**：如何让 Pipeline 更稳定？
- **答案**：添加错误处理、日志记录、数据验证

**问题 4**：如何让 Pipeline 更易维护？
- **答案**：模块化设计、代码注释、文档说明

**3. 查漏补缺（30 分钟）**

**检查清单**：
- [ ] 是否掌握了 pandas 基本操作？
- [ ] 是否能独立完成数据清洗？
- [ ] 是否能独立完成数据合并？
- [ ] 是否能独立完成分组聚合？
- [ ] 是否能设计简单的 Pipeline？
- [ ] 是否能改造成 Pipeline 类？

**如果还有薄弱环节**：
- 针对性地补充学习
- 多做相关练习
- 思考实际应用场景

**4. 下一步计划（30 分钟）**

**巩固练习**：
- 每天做 1-2 道练习题
- 复习之前做的题目
- 思考如何改造成 Pipeline

**深入学习**（如果时间允许）：
- 学习更高级的 pandas 操作
- 学习更复杂的 Pipeline 设计
- 学习数据库操作（SQLite、MySQL）

---

## 📚 学习资源

### **pandas**

1. **官方文档**：
   - [10 minutes to pandas](https://pandas.pydata.org/docs/user_guide/10min.html)（必看）
   - [pandas 用户指南](https://pandas.pydata.org/docs/user_guide/)（参考）

2. **实践项目**：
   - 做 A02（pandas 版本）
   - 做 A04（完整 Pipeline）

3. **代码模板**：
   - 收集常用 pandas 操作的代码模板
   - 面试时快速套用

### **Pipeline**

1. **概念理解**：
   - 看 A04 和 A05 的代码，理解 Pipeline 结构
   - 理解 ETL 的基本流程

2. **实践项目**：
   - 把之前的数据项目改造成 Pipeline 形式
   - 设计一个简单的 ETL Pipeline

---

## ✅ 每日检查清单

### **Day 1 检查清单**
- [ ] 掌握了 pandas 基本操作（查看、选择、筛选、排序）
- [ ] 能独立读取和清洗数据
- [ ] 理解了 Extract 和 Clean 的概念
- [ ] 能设计简单的 Extract + Clean Pipeline

### **Day 2 检查清单**
- [ ] 掌握了数据合并（merge、concat）
- [ ] 掌握了分组聚合（groupby、agg）
- [ ] 理解了 Transform 的概念
- [ ] 能设计简单的 Transform Pipeline

### **Day 3 检查清单**
- [ ] 能设计完整的 ETL Pipeline 类
- [ ] 能独立完成 A02（pandas 版本）
- [ ] 能独立完成 A04（完整 Pipeline）
- [ ] 理解了 Pipeline 的模块化设计

---

## 💪 最后的话

**记住**：
- 你不需要全部精通，**掌握常用操作即可**
- 重点是**能快速上手做项目**，不是背所有 API
- **Pipeline 思维**比工具更重要（逻辑比语法重要）

**建议**：
- 每天**边学边练**，不是只看文档
- 每做一个练习都**思考如何改造成 Pipeline**
- 遇到问题**先查文档**，再问别人

**加油！3 天后你会发现自己已经掌握了 pandas 基础！** 🚀
