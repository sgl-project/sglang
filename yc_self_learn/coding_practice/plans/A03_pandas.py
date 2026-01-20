# ========================================
# A03: OA 题目 - BNSF Data Engineer (Python/pandas 版本)
# 任务：房价数据分析和线性回归
# ========================================

# 题目要求：
# 1. 对房价数据做统计分析（summary）
# 2. 拟合线性回归模型并用模型做预测
#
# 数据字段：
# - Price: 房价（因变量/label）
# - Bedroom: 卧室数
# - Space: 房屋面积（平方英尺）
# - Room: 房间数
# - Lot: 地块宽度
# - Tax: 年税
# - Bathroom: 卫生间数
# - Garage: 车库停车位数
# - Condition: 房屋状况（1=好，0=其他/差）
#
# ⚠️ 注意：数据中可能有缺失值 NaN，需要正确处理

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# ========================================
# 主函数：analyse_and_fit_lrm
# ========================================
# 输入：file_path (CSV 文件路径)
# 输出：字典（dict），包含：
#   - summary_list: 包含 statistics, data_frame, number_of_observations
#   - regression_list: 包含 model_parameters, price_prediction
# ========================================

def analyse_and_fit_lrm(file_path):
    """
    分析和拟合线性回归模型
    
    参数:
        file_path: CSV 文件路径（字符串）
    
    返回:
        字典，包含：
        - summary_list: 字典，包含 statistics, data_frame, number_of_observations
        - regression_list: 字典，包含 model_parameters, price_prediction
    """
    
    # =========================
    # 0) 读取数据
    # =========================
    # 从给定路径读入 CSV，得到一个 pandas DataFrame
    # read_csv() 会自动识别列名和数据类型，缺失值会被读入为 NaN
    df = pd.read_csv(file_path)
    
    # ============================================================
    # 1) summary_list: 做 3 件事
    #    (a) 统计 Tax 的 mean/sd/median/min/max（特定筛选条件）
    #    (b) 输出 Space > 800 的数据，并按 Price 降序排序
    #    (c) 计算 Lot >= 80% 分位数(q80) 的观测数
    # ============================================================
    
    # -------------------------
    # 1(a) Bathroom=2 & Bedroom=4 的 Tax 统计量
    # -------------------------
    # 步骤：
    # 1. 筛选出 Bathroom=2 且 Bedroom=4 的行
    # 2. 去掉 Tax 字段的缺失值（NaN）
    # 3. 提取 Tax 列作为 Series
    # 4. 计算统计量：mean, sd, median, min, max
    
    # 先筛选出满足条件的行
    df_tax = df[(df['Bathroom'] == 2) & (df['Bedroom'] == 4)]
    
    # 去掉 Tax 的缺失值（NaN）
    # dropna() 会删除包含 NaN 的行，subset=['Tax'] 表示只检查 Tax 列
    df_tax = df_tax.dropna(subset=['Tax'])
    
    # 把 Tax 拿出来作为 Series
    tax_values = df_tax['Tax']
    
    # 依次计算：mean / sd / median / min / max
    # pandas Series 的统计函数会自动忽略 NaN
    # 注意：pandas 的 std() 默认使用样本标准差（ddof=1），与 R 的 sd() 一致
    # numpy 的 std() 默认使用总体标准差（ddof=0），如果要用 numpy，需要指定 ddof=1
    stats_vector = np.array([
        tax_values.mean(),    # 平均值
        tax_values.std(),     # 标准差（样本标准差，ddof=1）
        tax_values.median(),  # 中位数
        tax_values.min(),     # 最小值
        tax_values.max()      # 最大值
    ])
    
    # 题目要求：不需要名字，所以用 numpy array（而不是 pandas Series）
    # numpy array 默认没有名字，满足题目要求
    
    
    # -------------------------
    # 1(b) Space > 800 且按 Price 降序排序
    # -------------------------
    # 步骤：
    # 1. 筛选出 Space > 800 的行
    # 2. 按 Price 降序排序（ascending=False 表示降序）
    # 3. 返回完整的 DataFrame
    
    # 筛选 Space > 800 的行
    df_filtered = df[df['Space'] > 800]
    
    # 按 Price 降序排序
    # ascending=False 表示降序，inplace=False 表示不修改原 DataFrame（返回新 DataFrame）
    df_filtered = df_filtered.sort_values(by='Price', ascending=False)
    
    # 重置索引（可选，但通常建议这样做，避免索引混乱）
    df_filtered = df_filtered.reset_index(drop=True)
    
    
    # -------------------------
    # 1(c) Lot >= 80% 分位数(q80) 的数量
    # -------------------------
    # 步骤：
    # 1. 去掉 Lot 的 NaN，避免 quantile 被 NaN 干扰
    # 2. 计算 80% 分位数（第 4 个 5-quantile）
    #    5-quantile 分位数点：0%, 20%, 40%, 60%, 80%, 100%
    #    第 4 个 = 80% 分位数（索引从 0 开始，所以是第 5 个值，q=0.8）
    # 3. 统计 Lot >= q80 的行数（处理 NaN）
    
    # 先去掉 Lot 的 NaN，避免 quantile 被 NaN 干扰
    lot_clean = df['Lot'].dropna()
    
    # 第 4 个 5-quantile = 80% 分位数
    # pandas quantile() 返回的是 Series（如果只有一个值）或标量（如果只有一个值）
    # 需要用 float() 或 .item() 转成纯 Python 数字
    q80 = float(lot_clean.quantile(0.8))
    
    # 统计 Lot >= q80 的行数
    # 使用条件筛选：df['Lot'] >= q80 会返回布尔 Series（NaN 会被视为 False）
    # sum() 会计算 True 的数量（True=1, False=0）
    # 或者用 len() 配合布尔筛选
    num_obs = (df['Lot'] >= q80).sum()
    
    # 如果担心 NaN 的影响，也可以用：
    # num_obs = ((df['Lot'] >= q80) & (df['Lot'].notna())).sum()
    # 但题目已经要求处理 NA，所以直接用 sum() 即可（NaN >= q80 返回 False，不影响计数）
    
    
    # -------------------------
    # 组装 summary_list（必须是字典，且包含 3 个键）
    # -------------------------
    # 题目要求：summary_list 必须包含 3 个元素
    # - statistics: 长度为 5 的 numpy array（或 list）
    # - data_frame: pandas DataFrame
    # - number_of_observations: int（计数）
    
    summary_list = {
        'statistics': stats_vector,
        'data_frame': df_filtered,
        'number_of_observations': int(num_obs)  # 转成 int，确保是整数
    }
    
    
    # ============================================================
    # 2) regression_list: 拟合线性回归 + 输出参数 + 对新样本预测
    # ============================================================
    
    # -------------------------
    # 2(a) 拟合线性回归：Price ~ 其他所有变量
    # -------------------------
    # 步骤：
    # 1. 处理缺失值：去掉包含任何 NaN 的行（用 dropna()）
    #    这样确保模型训练时数据完整
    # 2. 准备特征（X）和目标（y）
    #    X: 所有其他变量（Bedroom, Space, Room, Lot, Tax, Bathroom, Garage, Condition）
    #    y: Price（因变量）
    # 3. 用 LinearRegression() 拟合线性回归模型
    
    # 题目提示：有 NaN，需要正确处理
    # 最简单稳妥：直接去掉包含任何 NaN 的行（与很多作业评测一致）
    # dropna() 会删除包含任何 NaN 的行
    df_clean = df.dropna()
    
    # 准备特征（X）和目标（y）
    # X: 所有其他变量（除了 Price 以外的所有列）
    # y: Price（因变量）
    
    # 获取所有列名
    feature_columns = [col for col in df_clean.columns if col != 'Price']
    
    # 提取特征（X）和目标（y）
    X = df_clean[feature_columns].values  # .values 转为 numpy array
    y = df_clean['Price'].values          # .values 转为 numpy array
    
    # 用 LinearRegression() 拟合线性回归模型
    # fit_intercept=True 表示包含截距项（默认值）
    model = LinearRegression(fit_intercept=True)
    model.fit(X, y)  # 拟合模型
    
    # -------------------------
    # 2(b) 提取参数，并把截距名改成 Intercept
    # -------------------------
    # 步骤：
    # 1. 提取模型系数（coef_）和截距（intercept_）
    # 2. 组装参数向量，第一个元素是截距（Intercept）
    # 3. 其余参数名要和变量名一致（Bedroom, Space, Room, Lot, Tax, Bathroom, Garage, Condition）
    
    # 提取模型系数和截距
    # model.coef_ 是一个 numpy array，包含所有特征的系数（按特征顺序）
    # model.intercept_ 是一个标量，包含截距项
    coefficients = model.coef_  # 特征系数（按 feature_columns 的顺序）
    intercept = model.intercept_  # 截距项
    
    # 组装参数向量：第一个元素是截距，其余是各变量的系数
    # 使用 numpy array，方便后续操作
    params_array = np.concatenate([[intercept], coefficients])
    
    # 创建参数名列表：第一个是 'Intercept'，其余是变量名
    param_names = ['Intercept'] + feature_columns
    
    # 创建 pandas Series（带名字的向量），方便访问和显示
    # 注意：题目要求是 "numeric vector"，在 Python 中可以是 numpy array 或 pandas Series
    # 如果题目明确要求带名字，用 Series；如果不需要名字，用 numpy array
    # 根据题目要求，参数需要命名，所以用 Series
    params = pd.Series(params_array, index=param_names)
    
    # 验证：params 现在应该包含：
    # - Intercept: 截距项
    # - Bedroom, Space, Room, Lot, Tax, Bathroom, Garage, Condition: 各变量的系数
    # 总共 9 个参数（1个截距 + 8个变量）
    
    # 注意：如果题目要求返回纯 numpy array（不带名字），可以用：
    # params = params_array
    # 但题目明确要求参数需要命名（Intercept 和其他变量名），所以保留 Series
    
    
    # -------------------------
    # 2(c) 按题目给定的新房子特征做预测
    # -------------------------
    # 步骤：
    # 1. 创建新房子特征的数据（new_house）
    #    变量名必须和训练模型时使用的变量名完全一致
    # 2. 用 model.predict() 对 new_house 预测 Price
    # 3. 转成纯 numeric（numpy array 的标量或 Python float）
    
    # 题目给定的新房子特征：
    # Bedroom=3, Space=1500, Room=8, Lot=40, Tax=1000, Bathroom=2, Garage=1, Condition=0
    # 注意：列名必须和训练数据一致，顺序也要一致（按照 feature_columns 的顺序）
    
    new_house_dict = {
        'Bedroom': 3,
        'Space': 1500,
        'Room': 8,
        'Lot': 40,
        'Tax': 1000,
        'Bathroom': 2,
        'Garage': 1,
        'Condition': 0
    }
    
    # 按照 feature_columns 的顺序提取值，确保顺序一致
    new_house_array = np.array([[new_house_dict[col] for col in feature_columns]])
    # 注意：predict() 需要 2D array（行数=1，列数=特征数），所以用 [[...]]
    
    # 使用模型对 new_house 预测 Price
    # predict() 返回的是 numpy array，即使只有一个值也是 array
    pred = model.predict(new_house_array)
    
    # 转成纯 numeric（题目要 numeric value）
    # .item() 或 float() 可以把单元素 array 转成 Python float
    price_pred = float(pred[0])  # 或者 price_pred = pred.item()
    
    
    # -------------------------
    # 组装 regression_list（必须是字典，且包含 2 个键）
    # -------------------------
    # 题目要求：regression_list 必须包含 2 个元素
    # - model_parameters: pandas Series（带名字的向量，9个参数）
    # - price_prediction: float（预测值）
    
    regression_list = {
        'model_parameters': params,  # pandas Series（带名字）
        'price_prediction': price_pred  # float
    }
    
    
    # ============================================================
    # 3) 返回最终 result（包含 summary_list 和 regression_list）
    # ============================================================
    # 题目要求：返回一个字典，包含：
    # - summary_list: 字典，包含 statistics, data_frame, number_of_observations
    # - regression_list: 字典，包含 model_parameters, price_prediction
    
    result = {
        'summary_list': summary_list,
        'regression_list': regression_list
    }
    
    return result


# ========================================
# 使用示例（测试用，题目中不需要）
# ========================================
# 注意：题目要求不要在文件里手动调用函数，单元测试会自动调用
#
# if __name__ == "__main__":
#     # 假设 CSV 文件路径为 "house_prices.csv"
#     result = analyse_and_fit_lrm("house_prices.csv")
#     
#     # 查看结果
#     print("Statistics:", result['summary_list']['statistics'])
#     print("\nFiltered DataFrame:")
#     print(result['summary_list']['data_frame'].head())
#     print("\nNumber of observations:", result['summary_list']['number_of_observations'])
#     print("\nModel parameters:")
#     print(result['regression_list']['model_parameters'])
#     print("\nPrice prediction:", result['regression_list']['price_prediction'])
# ========================================
