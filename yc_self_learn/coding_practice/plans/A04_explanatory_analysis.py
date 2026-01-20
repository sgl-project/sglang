# ========================================
# A04: OA 题目 - 数据探索性分析（Explanatory Analysis）
# 任务：数据清洗、缺失值处理、数据合并和统计分析
# ========================================

# ========================================
# 【第一部分】题干完整复述（只讲题目要你干什么）
# ========================================

# 题目背景：
# 你有三份电信客户数据集，每一行是一个 customer（客户），每个 dataset 存了不同属性。
# 你的任务是做一套"解释性分析 + 数据清洗 + 合并"，并返回一个指定结构的结果。

# 输入：
# 你需要写一个函数：
#   函数名：explanatory_analysis(...)
#   传入 3 个参数（都是路径）：
#     - charges_data_path
#     - personal_data_path
#     - plan_data_path

# 数据概览（题目给了每个表的列）：
#   - charges_data：与费用/合同/是否流失有关（包含 customerID、tenure、monthlyCharges、totalCharges、churn 等）
#   - personal_data：个人信息（包含 customerID、gender、partner、dependents、age 等）
#   - plan_data：套餐/服务信息（包含 customerID、internetService 等很多服务字段）

# 你必须按步骤做这些事（题目逐条要求）：
#   1. 读入三份数据
#   2. charges_data 里 monthlyCharges 和 totalCharges 有缺失，需要处理
#   3. 用 monthlyCharges 的 trimmed average（截尾均值）填补 monthlyCharges 的缺失：
#      - 计算 trimmed average 时：去掉最大 10% 和最小 10%
#      - 只使用非空（non-empty）的 monthlyCharges 值
#      - 计算出的 trimmed average 四舍五入到整数（nearest integer）
#   4. 用 monthlyCharges * tenure 来填补 totalCharges 的缺失
#   5. 新增一列 tenureBinned：把 tenure 离散分箱成 4 组：
#      - group1：(0, 24]
#      - group2：(24, 48]
#      - group3：(48, 60]
#      - group4：(60, Inf]
#   6. 计算 churn rate（流失率）：
#      - "churned customers 的百分比"
#      - 结果四舍五入到整数（例如 0.124 → 12）
#   7. 做两次 join（按 customerID）：
#      - 第一次：charges_data_updated 和 personal_data 做 join，使结果只保留共同 customerID（= inner join）
#      - 第二次：把第一次 join 的结果再和 plan_data 做 join，保留第一次结果的所有行（= left join）
#   8. 在合并后的数据上，计算 age > 60 的客户百分比，并四舍五入到整数（例如 0.678 → 68）
#   9. 在合并后的数据上，对 internetService 列做 value counts，生成一个 dict：
#      - key = internetService 的各个取值
#      - value = 每个取值出现次数

# 输出（非常关键：单测会严格检查 key 名）：
#   函数必须返回一个 dictionary，包含以下 key（名字必须完全一致）：
#     - monthly_charges_mean：整数，monthlyCharges 的 trimmed average（按题意计算并取整）
#     - charges_data_updated：DataFrame，填补缺失 + 新增 tenureBinned 后的 charges 表
#     - churn_pct：整数，流失率百分比（取整）
#     - data_merged：DataFrame，三表 join 后的数据
#     - pct_age_above_60：整数，合并数据中 age>60 的百分比（取整）
#     - internet_service_counts：dict，internetService 的计数结果

# ========================================
# 【第二部分】如何分析题干（怎么避免被单测卡）
# ========================================

# 你读这种题，建议按 3 层扫描：

# 第一层：锁死"函数签名 + 返回结构"
#   - 函数名必须对：explanatory_analysis
#   - 参数必须是 3 个路径（字符串）
#   - 返回必须是 dict
#   - dict 的 key 必须一字不差（这通常是 Codility mandatory test 的第一关）
#     例如：monthly_charges_mean（不是 monthlyChargesMean 或 monthly_charges_avg）

# 第二层：把题目要求变成"流水线步骤"
#   题干其实就是让你做一条 pipeline（流水线），顺序非常明确：
#     1. 先算 trimmed mean → 再填 monthlyCharges
#     2. 再填 totalCharges
#     3. 再分箱 tenureBinned
#     4. 再算 churn_pct
#     5. 再 join（inner → left）
#     6. 再在 merged 上算 pct_age_above_60 和 internet_service_counts
#   
#   这种题单测经常会检查：
#     - 你是不是在正确的数据阶段计算某个指标
#     - 比如 age>60 必须在 merged 后算，而不是 personal_data 上算
#     - 比如 internetService counts 必须在 merged 后算，而不是 plan_data 上算

# 第三层：提取"隐含的测试点"（最容易错的地方）
#   1. trimmed average 的定义：
#      - 去掉两端 10% 后求均值，然后取整
#      - "non-empty observations"：说明 trimmed mean 只用非缺失值
#      - 取整规则：题干写的是 round to nearest integer，不是 floor/ceil
#   
#   2. 分箱边界：
#      - 给的是 (0,24], (24,48], (48,60], (60,Inf]
#      - 注意右闭（right=True），即 24 属于 group1，48 属于 group2
#      - 注意 include_lowest=True，即 0 也要包含在 group1 中
#   
#   3. join 类型：
#      - 第一次是只保留共同 customerID（inner join）
#      - 第二次保留第一次结果全部行（left join）
#      - 顺序不能错：先 inner 再 left
#   
#   4. 计算指标的位置：
#      - churn_pct 在 charges_data_updated 上算（不是 merged 上）
#      - pct_age_above_60 在 data_merged 上算（不是 personal_data 上）
#      - internetService counts 在 data_merged 上算（不是 plan_data 上）
#   
#   5. 数据类型：
#      - 数值列可能有字符串或空字符串，需要先转成 numeric
#      - 使用 pd.to_numeric(errors="coerce") 处理

# ========================================
# 【第三部分】如何设计（写代码前先搭结构，按题目对齐）
# ========================================

# 你可以把整个函数写成 6 个"区块"，每个区块只干一件事，最后统一组装 results。
# 这样不会乱、也不容易漏 key。

# 区块 A：读取与类型准备
#   - 读 3 个 csv → 得到 3 个 DataFrame
#   - 把 charges_data 里会参与运算的列（tenure/monthlyCharges/totalCharges）转成 numeric
#     防止字符串/空值
#   - 目的：后面要排序/截尾/乘法/分箱，类型不对会崩。

# 区块 B：trimmed mean（只负责算 monthly_charges_mean）
#   - 从 monthlyCharges 里取出非空值
#   - 排序
#   - 砍掉两端 10%
#   - 求均值
#   - round → 得到整数 monthly_charges_mean
#   - 目的：这一步是独立的"统计量"，也是后续填补的依据。

# 区块 C：补缺失 + 新增 tenureBinned（构造 charges_data_updated）
#   - 用 monthly_charges_mean 填 monthlyCharges 的 NA
#   - 用 monthlyCharges * tenure 填 totalCharges 的 NA
#   - 用 tenure 做分箱 → 新列 tenureBinned（group1~4）
#   - 最终输出一个更新版表：charges_data_updated
#   - 目的：题目明确要返回更新后的 charges 表。

# 区块 D：计算 churn_pct（在 charges_data_updated 上）
#   - 用 churn 列计算 "Yes 的比例"
#   - *100 再 round → churn_pct
#   - 目的：题目要求 churn rate 作为百分比整数返回。

# 区块 E：两次 join 得到 data_merged（顺序不能错）
#   - charges_data_updated INNER JOIN personal_data on customerID
#   - 上一步结果 LEFT JOIN plan_data on customerID
#   - 输出：data_merged
#   - 目的：这是后面 age>60 和 internetService counts 的统一数据基础。

# 区块 F：merged 上的两个指标
#   - pct_age_above_60：在 data_merged 上算 age>60 的比例 → *100 → round
#   - internet_service_counts：对 data_merged['internetService'] 做 counts → dict

# 最后：组装 results（严格按题目 key）
#   - 把 A~F 的产物按题目指定 key 填进 dict 并 return。

# ========================================
# 【代码实现部分】
# ========================================

import pandas as pd
import numpy as np

# ========================================
# 主函数：explanatory_analysis
# ========================================
# 输入：
#   - charges_data_path: charges 数据 CSV 文件路径
#   - personal_data_path: personal 数据 CSV 文件路径
#   - plan_data_path: plan 数据 CSV 文件路径
#
# 输出：
#   字典（dict），必须包含以下 6 个键（顺序和命名必须完全一致）：
#   1. monthly_charges_mean (int): monthlyCharges 的 10% trimmed mean（四舍五入取整）
#   2. charges_data_updated (DataFrame): 对 charges_data 做完缺失处理 + 新列 tenureBinned 的更新表
#   3. churn_pct (int): churn == "Yes" 的百分比（四舍五入取整）
#   4. data_merged (DataFrame): charges_updated inner join personal，再 left join plan 的结果
#   5. pct_age_above_60 (int): 在 data_merged 上 age > 60 的百分比（四舍五入取整）
#   6. internet_service_counts (dict): 在 data_merged 上 internetService 各类别计数
# ========================================

def explanatory_analysis(charges_data_path, personal_data_path, plan_data_path):
    """
    数据探索性分析函数
    
    参数:
        charges_data_path: charges 数据 CSV 文件路径（字符串）
        personal_data_path: personal 数据 CSV 文件路径（字符串）
        plan_data_path: plan 数据 CSV 文件路径（字符串）
    
    返回:
        字典，包含 6 个键值对（见上方说明）
    """
    
    # =========================================================
    # Step 1) 读入三张表
    # =========================================================
    # 从给定路径读入三个 CSV 文件，得到三个 pandas DataFrame
    # read_csv() 会自动识别列名和数据类型，缺失值会被读入为 NaN
    # 空字符串会被读入为字符串 ""，需要后续处理
    
    charges_data = pd.read_csv(charges_data_path)
    personal_data = pd.read_csv(personal_data_path)
    plan_data = pd.read_csv(plan_data_path)
    
    
    # =========================================================
    # Step 2) 数据类型清洗：把数值列转成 numeric
    # =========================================================
    # 目的：后面要做均值、乘法、分箱，如果列是字符串会出错
    # 处理：把数值列（tenure, monthlyCharges, totalCharges）转成 numeric
    # 注意：列里可能有 missing / 空字符串，必须正确处理
    
    # 需要转换的数值列
    numeric_columns = ["tenure", "monthlyCharges", "totalCharges"]
    
    for col in numeric_columns:
        if col in charges_data.columns:
            # pd.to_numeric() 会把字符串转成数字
            # errors="coerce" 的意思：无法转数字的值（比如 ""、非数字字符串）会变成 NaN
            # 这样后续处理缺失值时可以统一处理
            charges_data[col] = pd.to_numeric(charges_data[col], errors="coerce")
    
    
    # =========================================================
    # Step 3) 计算 monthlyCharges 的 10% trimmed mean
    # =========================================================
    # 题目要求：计算 monthlyCharges 的 10% trimmed mean（两端各砍 10%）
    # 
    # Trimmed mean 的计算步骤：
    #   1) 只用非空值（去掉 NaN）
    #   2) 排序
    #   3) 两端各去掉 10% 的样本
    #   4) 对剩余部分求均值
    #   5) 四舍五入取整
    
    # 提取 monthlyCharges 的非空值
    monthly_non_null = charges_data["monthlyCharges"].dropna().to_numpy()
    
    # 排序（从小到大）
    monthly_non_null.sort()
    
    # 计算两端各砍掉的数量
    n = len(monthly_non_null)  # 总样本数
    k = int(np.floor(0.1 * n))  # 两端各砍掉的数量（向下取整）
    
    # 防止极端情况：数据太少，砍完反而没有值
    # 如果 2*k >= n，说明砍完没有数据了，就用全部数据
    if 2 * k < n:
        # 正常情况：去掉前 k 个和后 k 个，保留中间部分
        trimmed = monthly_non_null[k:n - k]
    else:
        # 极端情况：数据太少，不砍，用全部数据
        trimmed = monthly_non_null
    
    # 计算 trimmed mean
    monthly_charges_mean_value = float(trimmed.mean())
    
    # 题目要求：四舍五入取整
    monthly_charges_mean = int(round(monthly_charges_mean_value))
    
    
    # =========================================================
    # Step 4) 用 monthly_charges_mean 填补 monthlyCharges 缺失值
    # =========================================================
    # 题目要求：处理 missing values
    # monthlyCharges 缺失 -> 用 trimmed mean 填充
    
    # fillna() 会用指定值填充所有 NaN
    charges_data["monthlyCharges"] = charges_data["monthlyCharges"].fillna(monthly_charges_mean)
    
    
    # =========================================================
    # Step 5) 用 monthlyCharges * tenure 填补 totalCharges 缺失值
    # =========================================================
    # 题目要求：totalCharges 缺失 -> 用 monthlyCharges * tenure 计算
    # 
    # 注意：
    # - 我们已经在 Step 4 填补了 monthlyCharges，所以这里 monthlyCharges 没有 NaN
    # - 但 tenure 可能还有 NaN，所以需要处理
    # - 如果 tenure 是 NaN，那么 monthlyCharges * tenure 也是 NaN，这样 fillna() 就不会填充
    # - 所以我们需要用 fillna() 配合 lambda 函数，或者先处理 tenure 的 NaN
    
    # 方法1：直接用 fillna() 配合计算（推荐）
    # 如果 totalCharges 是 NaN，就用 monthlyCharges * tenure 填充
    # 如果 tenure 也是 NaN，那么结果还是 NaN（这是合理的，因为无法计算）
    charges_data["totalCharges"] = charges_data["totalCharges"].fillna(
        charges_data["monthlyCharges"] * charges_data["tenure"]
    )
    
    # 注意：如果题目要求更严格（比如 tenure 缺失也要处理），可能需要：
    # charges_data["tenure"] = charges_data["tenure"].fillna(0)  # 或其他默认值
    # 然后再计算 totalCharges
    
    
    # =========================================================
    # Step 6) 根据 tenure 分桶生成 tenureBinned
    # =========================================================
    # 题目要求：按 tenure 分箱产生一个新列 tenureBinned
    # 
    # 分箱规则：
    #   group1: (0, 24]   - 0 到 24 个月（包含 24）
    #   group2: (24, 48]  - 24 到 48 个月（包含 48）
    #   group3: (48, 60]  - 48 到 60 个月（包含 60）
    #   group4: (60, inf] - 60 个月以上（包含 60 以上）
    #
    # 注意：
    #   - right=True 表示右闭区间（(0,24] 表示 0 < x <= 24）
    #   - include_lowest=True 表示包含最小值（0 也会被包含）
    #   - np.inf 表示正无穷（60 以上）
    
    # 定义分箱边界和标签
    bins = [0, 24, 48, 60, np.inf]
    labels = ["group1", "group2", "group3", "group4"]
    
    # 使用 pd.cut() 进行分箱
    charges_data["tenureBinned"] = pd.cut(
        charges_data["tenure"],
        bins=bins,
        labels=labels,
        right=True,          # 右闭区间 (0,24]...
        include_lowest=True  # 把最小值也包含进去
    )
    
    # 注意：如果 tenure 有 NaN，分箱后 tenureBinned 也会是 NaN（这是合理的）
    
    # 题目要求：输出"更新后的 charges_data"
    # 创建一个副本，避免修改原数据（虽然这里不需要，但更安全）
    charges_data_updated = charges_data.copy()
    
    
    # =========================================================
    # Step 7) 计算 churn_pct：churn == "Yes" 的百分比（整数）
    # =========================================================
    # 题目要求：计算 churn rate（百分比）并四舍五入
    # 
    # 计算步骤：
    #   1) (charges_data_updated["churn"] == "Yes") 返回布尔 Series（True/False）
    #   2) .mean() 计算 True 的比例（True=1, False=0，所以 mean() 就是 True 的比例）
    #   3) * 100 转成百分比
    #   4) round() 四舍五入
    #   5) int() 转成整数
    
    churn_pct = int(round((charges_data_updated["churn"] == "Yes").mean() * 100))
    
    # 注意：如果 churn 列有缺失值，缺失值会被视为 False（因为 "Yes" != NaN），
    # 这通常是合理的（缺失值不算作 "Yes"）
    
    
    # =========================================================
    # Step 8) 两次 join 得到 data_merged
    # =========================================================
    # 题目要求：按 customerID 做两次 join
    #   1) charges_data_updated INNER JOIN personal_data
    #      - inner：只保留两边都有的 customerID（交集）
    #   2) 上一步结果 LEFT JOIN plan_data
    #      - left：保留上一步所有行，plan_data 缺的部分补 NaN
    
    # 第一次 join：charges_data_updated INNER JOIN personal_data
    # on="customerID" 表示按 customerID 列进行 join
    # how="inner" 表示内连接（只保留两边都有的 customerID）
    merged_cp = pd.merge(
        charges_data_updated,
        personal_data,
        on="customerID",
        how="inner"
    )
    
    # 第二次 join：merged_cp LEFT JOIN plan_data
    # how="left" 表示左连接（保留左边表的所有行，右边表缺的部分补 NaN）
    data_merged = pd.merge(
        merged_cp,
        plan_data,
        on="customerID",
        how="left"
    )
    
    # 注意：
    # - 第一次用 inner join，所以 merged_cp 只包含两边都有的 customerID
    # - 第二次用 left join，所以 data_merged 包含 merged_cp 的所有行
    # - 如果 plan_data 中没有某个 customerID，对应的列会是 NaN
    
    
    # =========================================================
    # Step 9) pct_age_above_60：在 merged 数据中 age > 60 的百分比
    # =========================================================
    # 题目要求：在 data_merged 上计算 "age > 60 的百分比"
    # 
    # 计算步骤：
    #   1) (data_merged["age"] > 60) 返回布尔 Series
    #   2) .mean() 计算 True 的比例
    #   3) * 100 转成百分比
    #   4) round() 四舍五入
    #   5) int() 转成整数
    
    pct_age_above_60 = int(round((data_merged["age"] > 60).mean() * 100))
    
    # 注意：如果 age 有缺失值，缺失值会被视为 False（因为 NaN > 60 是 False），
    # 这通常是合理的（缺失值不算作 > 60）
    
    
    # =========================================================
    # Step 10) internet_service_counts：internetService 各类别计数（dict）
    # =========================================================
    # 题目要求：计算 internetService 各类别的计数，返回字典
    # 
    # 计算步骤：
    #   1) data_merged["internetService"].value_counts() 计算各类别计数
    #   2) dropna=True 表示不把 NaN 作为一个类别统计（如果 dropna=False，NaN 也会被统计）
    #   3) .to_dict() 转成字典（类别名 -> 计数）
    
    internet_service_counts = data_merged["internetService"].value_counts(dropna=True).to_dict()
    
    # 注意：
    # - value_counts() 返回的是 Series，索引是类别名，值是计数
    # - to_dict() 转成字典，键是类别名，值是计数
    # - dropna=True 表示忽略 NaN（不统计 NaN 的数量）
    # - 如果 dropna=False，NaN 也会作为一个类别被统计（键是 NaN）
    
    
    # =========================================================
    # Step 11) 按题目规定 key 名返回结果 dict
    # =========================================================
    # 题目要求：返回的字典必须包含以下 6 个键（顺序和命名必须完全一致）
    # 单测会检查 key 名是否完全一致，所以必须严格按照题目要求命名
    
    results = {
        "monthly_charges_mean": monthly_charges_mean,              # int
        "charges_data_updated": charges_data_updated,               # DataFrame
        "churn_pct": churn_pct,                                     # int
        "data_merged": data_merged,                                 # DataFrame
        "pct_age_above_60": pct_age_above_60,                       # int
        "internet_service_counts": internet_service_counts,         # dict
    }
    
    return results


# ========================================
# 使用示例（测试用，题目中不需要）
# ========================================
# 注意：题目要求不要在文件里手动调用函数，单元测试会自动调用
#
# if __name__ == "__main__":
#     # 假设 CSV 文件路径
#     charges_path = "charges_data.csv"
#     personal_path = "personal_data.csv"
#     plan_path = "plan_data.csv"
#     
#     # 调用函数
#     result = explanatory_analysis(charges_path, personal_path, plan_path)
#     
#     # 查看结果
#     print("Monthly charges mean:", result["monthly_charges_mean"])
#     print("\nCharges data updated:")
#     print(result["charges_data_updated"].head())
#     print("\nChurn percentage:", result["churn_pct"])
#     print("\nData merged shape:", result["data_merged"].shape)
#     print("\nPercentage age above 60:", result["pct_age_above_60"])
#     print("\nInternet service counts:", result["internet_service_counts"])
# ========================================
