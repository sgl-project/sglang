# ========================================
# A05: OA 题目 - CouncilsJob（England councils ETL 聚合题）
# 任务：使用 PySpark 进行数据提取、转换和加载（ETL）
# ========================================

# ========================================
# 【第一部分】题干完整复述（只讲题目要你干什么）
# ========================================

# 题目背景：
# 你需要处理英格兰议会（councils）相关的数据，包括：
#   - 4 种类型的议会数据（district_councils, london_boroughs, metropolitan_districts, unitary_authorities）
#   - 房产平均价格数据（property_avg_price.csv）
#   - 房产销售数量数据（property_sales_volume.csv）
# 
# 你的任务是创建一个 PySpark ETL 作业，将这些数据合并成一个统一的数据集。

# 输入：
#   你需要创建一个类：CouncilsJob
#   类需要实现以下方法：
#     - __init__(self, ...): 初始化 SparkSession，设置 input_directory
#     - extract_councils(self): 读取 4 个 council CSV 文件，添加 council_type 列，合并
#     - extract_avg_price(self): 读取 property_avg_price.csv，选择并重命名列
#     - extract_sales_volume(self): 读取 property_sales_volume.csv，选择并重命名列
#     - transform(self, councils_df, avg_price_df, sales_volume_df): 用 council 作为键进行 left join
#     - run(self): 调用所有 extract 方法，然后调用 transform，返回最终 DataFrame

# 数据文件结构：
#   - base_path = f"{input_directory}/england_councils"
#     - district_councils.csv
#     - london_boroughs.csv
#     - metropolitan_districts.csv
#     - unitary_authorities.csv
#   - property_avg_price.csv（包含 local_authority, avg_price_nov_2019）
#   - property_sales_volume.csv（包含 local_authority, sales_volume_sep_2019）

# 你必须按步骤做这些事：
#   1. extract_councils:
#      - 读取 4 个 council CSV 文件
#      - 每个文件添加 council_type 列（对应类型名称）
#      - 选择 council, county, council_type 三列
#      - 用 unionByName 合并 4 个 DataFrame
#   
#   2. extract_avg_price:
#      - 读取 property_avg_price.csv
#      - 选择并重命名：local_authority → council
#      - 返回包含 council, avg_price_nov_2019 的 DataFrame
#   
#   3. extract_sales_volume:
#      - 读取 property_sales_volume.csv
#      - 选择并重命名：local_authority → council
#      - 返回包含 council, sales_volume_sep_2019 的 DataFrame
#   
#   4. transform:
#      - 用 council 作为键，对三个 DataFrame 进行 left join
#      - 选择最终需要的列：council, county, council_type, avg_price_nov_2019, sales_volume_sep_2019
#   
#   5. run:
#      - 调用 extract_councils(), extract_avg_price(), extract_sales_volume()
#      - 调用 transform() 传入三个 DataFrame
#      - 返回最终合并后的 DataFrame

# 输出：
#   run() 方法返回一个 PySpark DataFrame，包含：
#     - council: 议会名称
#     - county: 县/郡
#     - council_type: 议会类型
#     - avg_price_nov_2019: 2019年11月平均价格
#     - sales_volume_sep_2019: 2019年9月销售数量

# ========================================
# 【第二部分】如何分析题干（怎么避免被单测卡）
# ========================================

# 你读这种题，建议按 3 层扫描：

# 第一层：锁死"类结构 + 方法签名"
#   - 类名必须对：CouncilsJob
#   - 方法名必须对：extract_councils, extract_avg_price, extract_sales_volume, transform, run
#   - 方法参数必须对：transform 需要 3 个 DataFrame 参数
#   - 返回类型必须对：extract 方法返回 DataFrame，run 返回 DataFrame

# 第二层：把题目要求变成"ETL 流水线"
#   题干其实就是让你做一条 ETL pipeline（提取-转换-加载），顺序非常明确：
#     1. Extract（提取）：
#        - extract_councils: 从 4 个 CSV 提取议会数据
#        - extract_avg_price: 从 1 个 CSV 提取价格数据
#        - extract_sales_volume: 从 1 个 CSV 提取销售数据
#     2. Transform（转换）：
#        - 用 council 作为键进行 left join
#        - 选择需要的列
#     3. Load（加载）：
#        - run() 方法返回最终结果
#   
#   这种题单测经常会检查：
#     - 你是不是在正确的阶段做某个操作（比如 council_type 必须在 extract 阶段添加）
#     - join 的顺序和类型（left join，不是 inner join）
#     - 列名是否正确（council 不是 local_authority）

# 第三层：提取"隐含的测试点"（最容易错的地方）
#   1. unionByName vs union:
#      - 必须用 unionByName，因为不同 DataFrame 的列顺序可能不同
#      - union 要求列顺序一致，unionByName 按列名匹配
#   
#   2. council_type 的值：
#      - district_councils.csv → "district_councils"
#      - london_boroughs.csv → "london_boroughs"
#      - metropolitan_districts.csv → "metropolitan_districts"
#      - unitary_authorities.csv → "unitary_authorities"
#      - 注意：是文件名去掉 .csv，不是其他格式
#   
#   3. 列名重命名：
#      - local_authority → council（必须重命名，否则 join 会失败）
#      - 注意：avg_price_df 和 sales_volume_df 都需要重命名
#   
#   4. join 类型：
#      - 必须用 left join（保留 councils_df 的所有行）
#      - 如果 avg_price 或 sales_volume 没有对应数据，会显示 null
#   
#   5. 列选择顺序：
#      - transform 方法中 select 的列顺序要符合题目要求
#      - 通常顺序：council, county, council_type, avg_price_nov_2019, sales_volume_sep_2019

# ========================================
# 【第三部分】如何设计（写代码前先搭结构，按题目对齐）
# ========================================

# 你可以把整个类设计成标准的 ETL 模式，每个方法只干一件事：

# 类初始化（__init__）：
#   - 创建 SparkSession（如果还没有）
#   - 设置 input_directory（默认 "data"）
#   - 目的：为后续所有方法提供 SparkSession 和路径配置

# 方法 A：extract_councils（提取议会数据）
#   - 定义 4 个文件路径
#   - 定义 4 个 council_type 值
#   - 循环读取 4 个 CSV，每个都：
#     - 读取 CSV（header=true, inferSchema=true）
#     - 添加 council_type 列（用 F.lit()）
#     - 选择 council, county, council_type
#   - 用 unionByName 合并 4 个 DataFrame
#   - 返回合并后的 DataFrame
#   - 目的：统一不同来源的议会数据，添加类型标识

# 方法 B：extract_avg_price（提取价格数据）
#   - 读取 property_avg_price.csv
#   - 重命名 local_authority → council
#   - 选择 council, avg_price_nov_2019
#   - 返回 DataFrame
#   - 目的：提取并标准化价格数据

# 方法 C：extract_sales_volume（提取销售数据）
#   - 读取 property_sales_volume.csv
#   - 重命名 local_authority → council
#   - 选择 council, sales_volume_sep_2019
#   - 返回 DataFrame
#   - 目的：提取并标准化销售数据

# 方法 D：transform（转换/合并数据）
#   - 接收 3 个 DataFrame 参数
#   - 用 council 作为键进行两次 left join
#   - 选择最终需要的列
#   - 返回合并后的 DataFrame
#   - 目的：将三个数据源合并成一个统一的数据集

# 方法 E：run（执行完整流程）
#   - 调用 extract_councils()
#   - 调用 extract_avg_price()
#   - 调用 extract_sales_volume()
#   - 调用 transform() 传入三个 DataFrame
#   - 返回最终结果
#   - 目的：执行完整的 ETL 流程

# ========================================
# 【代码实现部分】
# ========================================

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import *
from pyspark.sql import DataFrame


class CouncilsJob:
    """
    英格兰议会数据 ETL 作业类
    
    功能：
    - 提取 4 种类型的议会数据
    - 提取房产平均价格数据
    - 提取房产销售数量数据
    - 合并所有数据源
    """
    
    def __init__(self, spark: SparkSession = None, input_directory: str = "data"):
        """
        初始化 CouncilsJob
        
        参数:
            spark: SparkSession 对象（如果为 None，会创建一个新的）
            input_directory: 输入数据目录路径（默认 "data"）
        """
        # 如果传入的 spark 为 None，创建一个新的 SparkSession
        if spark is None:
            self.spark = SparkSession.builder \
                .appName("CouncilsJob") \
                .getOrCreate()
        else:
            self.spark = spark
        
        # 设置输入目录路径
        self.input_directory = input_directory
    
    def extract_councils(self) -> DataFrame:
        """
        提取议会数据
        
        读取 4 个 council CSV 文件：
        - district_councils.csv
        - london_boroughs.csv
        - metropolitan_districts.csv
        - unitary_authorities.csv
        
        每个文件：
        1. 读取 CSV（header=true, inferSchema=true）
        2. 添加 council_type 列（对应文件名）
        3. 选择 council, county, council_type 三列
        
        最后用 unionByName 合并 4 个 DataFrame
        
        返回:
            合并后的 DataFrame，包含 council, county, council_type 三列
        """
        # 基础路径
        base_path = f"{self.input_directory}/england_councils"
        
        # 定义 4 个文件路径和对应的 council_type
        council_files = [
            ("district_councils.csv", "district_councils"),
            ("london_boroughs.csv", "london_boroughs"),
            ("metropolitan_districts.csv", "metropolitan_districts"),
            ("unitary_authorities.csv", "unitary_authorities")
        ]
        
        # 存储 4 个 DataFrame
        dataframes = []
        
        # 循环读取每个文件
        for filename, council_type in council_files:
            # 构建完整路径
            path = f"{base_path}/{filename}"
            
            # 读取 CSV 文件
            # option("header", "true"): 第一行是列名
            # option("inferSchema", "true"): 自动推断数据类型
            df = (self.spark.read
                  .option("header", "true")
                  .option("inferSchema", "true")
                  .csv(path)
                  .withColumn("council_type", F.lit(council_type))  # 添加 council_type 列
                  .select("council", "county", "council_type"))      # 选择需要的列
            
            dataframes.append(df)
        
        # 用 unionByName 合并 4 个 DataFrame
        # unionByName 按列名匹配，不要求列顺序一致
        # union 要求列顺序一致，所以这里必须用 unionByName
        councils_df = dataframes[0]
        for df in dataframes[1:]:
            councils_df = councils_df.unionByName(df)
        
        return councils_df
    
    def extract_avg_price(self) -> DataFrame:
        """
        提取房产平均价格数据
        
        读取 property_avg_price.csv，包含：
        - local_authority: 议会名称
        - avg_price_nov_2019: 2019年11月平均价格
        
        操作：
        1. 读取 CSV
        2. 重命名 local_authority → council（为了后续 join）
        3. 选择 council, avg_price_nov_2019
        
        返回:
            DataFrame，包含 council, avg_price_nov_2019 两列
        """
        # 构建文件路径
        path = f"{self.input_directory}/property_avg_price.csv"
        
        # 读取 CSV 文件
        avg_price_df = (self.spark.read
                        .option("header", "true")
                        .option("inferSchema", "true")
                        .csv(path)
                        .withColumnRenamed("local_authority", "council")  # 重命名列
                        .select("council", "avg_price_nov_2019"))        # 选择需要的列
        
        return avg_price_df
    
    def extract_sales_volume(self) -> DataFrame:
        """
        提取房产销售数量数据
        
        读取 property_sales_volume.csv，包含：
        - local_authority: 议会名称
        - sales_volume_sep_2019: 2019年9月销售数量
        
        操作：
        1. 读取 CSV
        2. 重命名 local_authority → council（为了后续 join）
        3. 选择 council, sales_volume_sep_2019
        
        返回:
            DataFrame，包含 council, sales_volume_sep_2019 两列
        """
        # 构建文件路径
        path = f"{self.input_directory}/property_sales_volume.csv"
        
        # 读取 CSV 文件
        sales_volume_df = (self.spark.read
                           .option("header", "true")
                           .option("inferSchema", "true")
                           .csv(path)
                           .withColumnRenamed("local_authority", "council")  # 重命名列
                           .select("council", "sales_volume_sep_2019"))      # 选择需要的列
        
        return sales_volume_df
    
    def transform(self, councils_df: DataFrame, avg_price_df: DataFrame, sales_volume_df: DataFrame) -> DataFrame:
        """
        转换/合并数据
        
        用 council 作为键，对三个 DataFrame 进行 left join：
        1. councils_df LEFT JOIN avg_price_df on council
        2. 上一步结果 LEFT JOIN sales_volume_df on council
        
        然后选择最终需要的列
        
        参数:
            councils_df: 议会数据 DataFrame
            avg_price_df: 平均价格数据 DataFrame
            sales_volume_df: 销售数量数据 DataFrame
        
        返回:
            合并后的 DataFrame，包含：
            - council: 议会名称
            - county: 县/郡
            - council_type: 议会类型
            - avg_price_nov_2019: 2019年11月平均价格
            - sales_volume_sep_2019: 2019年9月销售数量
        """
        # 第一次 join：councils_df LEFT JOIN avg_price_df
        # on="council" 表示用 council 列作为 join 键
        # how="left" 表示左连接（保留 councils_df 的所有行）
        joined_df = (councils_df
                     .join(avg_price_df, on="council", how="left")
                     .join(sales_volume_df, on="council", how="left")  # 第二次 join
                     .select("council", "county", "council_type",
                             "avg_price_nov_2019", "sales_volume_sep_2019"))  # 选择最终需要的列
        
        return joined_df
    
    def run(self) -> DataFrame:
        """
        执行完整的 ETL 流程
        
        步骤：
        1. 调用 extract_councils() 提取议会数据
        2. 调用 extract_avg_price() 提取价格数据
        3. 调用 extract_sales_volume() 提取销售数据
        4. 调用 transform() 合并所有数据
        5. 返回最终结果
        
        返回:
            合并后的最终 DataFrame
        """
        # 提取数据
        councils_df = self.extract_councils()
        avg_price_df = self.extract_avg_price()
        sales_volume_df = self.extract_sales_volume()
        
        # 转换/合并数据
        result_df = self.transform(councils_df, avg_price_df, sales_volume_df)
        
        return result_df


# ========================================
# 使用示例（测试用，题目中不需要）
# ========================================
# 注意：题目要求不要在文件里手动调用函数，单元测试会自动调用
#
# if __name__ == "__main__":
#     # 创建 SparkSession
#     spark = SparkSession.builder \
#         .appName("CouncilsJob") \
#         .getOrCreate()
#     
#     # 创建 CouncilsJob 实例
#     job = CouncilsJob(spark=spark, input_directory="data")
#     
#     # 执行 ETL 流程
#     result = job.run()
#     
#     # 查看结果
#     result.show(10)
#     result.printSchema()
#     
#     # 停止 SparkSession
#     spark.stop()
# ========================================
