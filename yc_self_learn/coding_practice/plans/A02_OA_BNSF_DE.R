# ========================================
# A02: OA 题目 - BNSF Data Engineer
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
# ⚠️ 注意：数据中可能有缺失值 NA，需要正确处理

library(dplyr)

# ========================================
# 主函数：analyse_and_fit_lrm
# ========================================
# 输入：file_path (CSV 文件路径)
# 输出：命名 list，包含：
#   - summary_list: 包含 statistics, data_frame, number_of_observations
#   - regression_list: 包含 model_parameters, price_prediction
# ========================================

analyse_and_fit_lrm <- function(file_path) {
  
  # =========================
  # 0) 读取数据
  # =========================
  # 从给定路径读入 CSV，得到一个 data.frame
  # read.csv() 会自动识别列名和数据类型
  df <- read.csv(file_path)
  
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
  # 2. 去掉 Tax 字段的缺失值（NA）
  # 3. 提取 Tax 列作为向量
  # 4. 计算统计量：mean, sd, median, min, max
  
  df_tax <- df %>%
    filter(Bathroom == 2, Bedroom == 4) %>%
    filter(!is.na(Tax))  # 去掉 Tax 的 NA
  
  # 把 Tax 拿出来作为向量
  tax_values <- df_tax$Tax
  
  # 依次计算：mean / sd / median / min / max
  # 注意：这里 tax_values 已经剔除了 NA，所以不用再写 na.rm=TRUE
  # 但为了代码健壮性，还是加上 na.rm=TRUE 更安全
  stats_vector <- c(
    mean(tax_values, na.rm = TRUE),
    sd(tax_values, na.rm = TRUE),
    median(tax_values, na.rm = TRUE),
    min(tax_values, na.rm = TRUE),
    max(tax_values, na.rm = TRUE)
  )
  
  # 题目要求：不需要名字，所以把 names 清空
  # 虽然已经是无名向量，但显式清除更保险
  names(stats_vector) <- NULL
  
  
  # -------------------------
  # 1(b) Space > 800 且按 Price 降序排序
  # -------------------------
  # 步骤：
  # 1. 筛选出 Space > 800 的行
  # 2. 按 Price 降序排序（desc() 表示降序）
  # 3. 返回完整的数据框
  
  df_filtered <- df %>%
    filter(Space > 800) %>%
    arrange(desc(Price))  # desc() 表示降序
  
  
  # -------------------------
  # 1(c) Lot >= 80% 分位数(q80) 的数量
  # -------------------------
  # 步骤：
  # 1. 去掉 Lot 的 NA，避免 quantile 被 NA 干扰
  # 2. 计算 80% 分位数（第 4 个 5-quantile）
  #    5-quantile 分位数点：0%, 20%, 40%, 60%, 80%, 100%
  #    第 4 个 = 80% 分位数（索引从 0 开始，所以是第 5 个值，probs=0.8）
  # 3. 统计 Lot >= q80 的行数（处理 NA）
  
  # 先去掉 Lot 的 NA，避免 quantile 被 NA 干扰
  lot_clean <- df$Lot[!is.na(df$Lot)]
  
  # 第 4 个 5-quantile = 80% 分位数
  # quantile() 返回的是命名向量，需要用 as.numeric() 转成纯数字
  q80 <- as.numeric(quantile(lot_clean, probs = 0.8))
  
  # 统计 Lot >= q80 的行数
  # sum() 会计算 TRUE 的数量（TRUE=1, FALSE=0）
  # na.rm=TRUE 会忽略 NA（NA >= q80 会返回 NA，被忽略）
  num_obs <- sum(df$Lot >= q80, na.rm = TRUE)
  
  
  # -------------------------
  # 组装 summary_list（必须是命名 list，且长度=3）
  # -------------------------
  # 题目要求：summary_list 必须包含 3 个命名元素
  # - statistics: 长度为 5 的 numeric vector
  # - data_frame: data frame
  # - number_of_observations: numeric（计数）
  
  summary_list <- list(
    statistics = stats_vector,
    data_frame = df_filtered,
    number_of_observations = num_obs
  )
  
  
  # ============================================================
  # 2) regression_list: 拟合线性回归 + 输出参数 + 对新样本预测
  # ============================================================
  
  # -------------------------
  # 2(a) 拟合线性回归：Price ~ 其他所有变量
  # -------------------------
  # 步骤：
  # 1. 处理缺失值：去掉包含任何 NA 的行（用 na.omit()）
  #    这样确保模型训练时数据完整
  # 2. 用 lm() 拟合线性回归模型
  #    公式：Price ~ . 表示 Price 对除了 Price 以外的所有列做回归
  # 3. 模型会自动处理所有 predictor 变量（Bedroom, Space, Room, Lot, Tax, Bathroom, Garage, Condition）
  
  # 题目提示：有 NA，需要正确处理
  # 最简单稳妥：直接去掉包含任何 NA 的行（与很多作业评测一致）
  # na.omit() 会删除包含任何 NA 的行
  df_clean <- na.omit(df)
  
  # 用所有其它列预测 Price
  # Price ~ . 表示：Price 对除 Price 以外的所有列做回归
  # 这是 R 中的简写，等价于 Price ~ Bedroom + Space + Room + Lot + Tax + Bathroom + Garage + Condition
  model <- lm(Price ~ ., data = df_clean)
  
  
  # -------------------------
  # 2(b) 提取参数，并把截距名改成 Intercept
  # -------------------------
  # 步骤：
  # 1. 用 coef() 提取模型参数（系数）
  # 2. R 默认把截距叫 "(Intercept)"，题目要求叫 "Intercept"
  # 3. 参数向量第一个元素必须是 Intercept，其余参数名要和变量名一致
  
  # 提取模型系数（参数）
  params <- coef(model)
  
  # R 默认把截距叫 "(Intercept)"，题目要求叫 "Intercept"
  # 修改名称
  names(params)[names(params) == "(Intercept)"] <- "Intercept"
  
  # 验证：params 现在应该包含：
  # - Intercept: 截距项
  # - Bedroom, Space, Room, Lot, Tax, Bathroom, Garage, Condition: 各变量的系数
  # 总共 9 个参数（1个截距 + 8个变量）
  
  
  # -------------------------
  # 2(c) 按题目给定的新房子特征做预测
  # -------------------------
  # 步骤：
  # 1. 创建新房子特征的数据框（new_house）
  #    变量名必须和训练模型时使用的变量名完全一致
  # 2. 用 predict() 对 new_house 预测 Price
  # 3. 转成纯 numeric（去掉预测结果的属性）
  
  # 题目给定的新房子特征：
  # Bedroom=3, Space=1500, Room=8, Lot=40, Tax=1000, Bathroom=2, Garage=1, Condition=0
  # 注意：列名必须和训练数据一致
  new_house <- data.frame(
    Bedroom   = 3,
    Space     = 1500,
    Room      = 8,
    Lot       = 40,
    Tax       = 1000,
    Bathroom  = 2,
    Garage    = 1,
    Condition = 0
  )
  
  # 使用模型对 new_house 预测 Price
  # predict() 会返回一个带名字的向量，需要转成纯 numeric
  pred <- predict(model, newdata = new_house)
  
  # 转成纯 numeric（去掉属性，题目要 numeric value）
  price_pred <- as.numeric(pred)
  
  
  # -------------------------
  # 组装 regression_list（必须是命名 list，且长度=2）
  # -------------------------
  # 题目要求：regression_list 必须包含 2 个命名元素
  # - model_parameters: 长度为 9 的 numeric vector（1个截距 + 8个变量系数）
  # - price_prediction: numeric（预测值）
  
  regression_list <- list(
    model_parameters = params,
    price_prediction = price_pred
  )
  
  
  # ============================================================
  # 3) 返回最终 result（包含 summary_list 和 regression_list）
  # ============================================================
  # 题目要求：返回一个命名 list，包含：
  # - summary_list: 包含 statistics, data_frame, number_of_observations
  # - regression_list: 包含 model_parameters, price_prediction
  
  result <- list(
    summary_list = summary_list,
    regression_list = regression_list
  )
  
  return(result)
}

# ========================================
# 使用示例（测试用，题目中不需要）
# ========================================
# 注意：题目要求不要在文件里手动调用函数，单元测试会自动调用
#
# # 假设 CSV 文件路径为 "house_prices.csv"
# result <- analyse_and_fit_lrm("house_prices.csv")
#
# # 查看结果
# # result$summary_list$statistics       # 统计量向量
# # result$summary_list$data_frame       # 筛选后的数据框
# # result$summary_list$number_of_observations  # 观测数
# # result$regression_list$model_parameters     # 模型参数
# # result$regression_list$price_prediction     # 预测值
# ========================================
