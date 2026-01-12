# A02: 完整答案（Answer Only）

## R 代码实现

```r
library(dplyr)

analyse_and_fit_lrm <- function(file_path) {
  
  # =========================
  # 0) 读取数据
  # =========================
  df <- read.csv(file_path)
  
  # ============================================================
  # 1) summary_list: 做 3 件事
  # ============================================================
  
  # -------------------------
  # 1(a) Bathroom=2 & Bedroom=4 的 Tax 统计量
  # -------------------------
  df_tax <- df %>%
    filter(Bathroom == 2, Bedroom == 4) %>%
    filter(!is.na(Tax))
  
  tax_values <- df_tax$Tax
  
  stats_vector <- c(
    mean(tax_values, na.rm = TRUE),
    sd(tax_values, na.rm = TRUE),
    median(tax_values, na.rm = TRUE),
    min(tax_values, na.rm = TRUE),
    max(tax_values, na.rm = TRUE)
  )
  
  names(stats_vector) <- NULL
  
  # -------------------------
  # 1(b) Space > 800 且按 Price 降序排序
  # -------------------------
  df_filtered <- df %>%
    filter(Space > 800) %>%
    arrange(desc(Price))
  
  # -------------------------
  # 1(c) Lot >= 80% 分位数(q80) 的数量
  # -------------------------
  lot_clean <- df$Lot[!is.na(df$Lot)]
  q80 <- as.numeric(quantile(lot_clean, probs = 0.8))
  num_obs <- sum(df$Lot >= q80, na.rm = TRUE)
  
  # -------------------------
  # 组装 summary_list
  # -------------------------
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
  df_clean <- na.omit(df)
  model <- lm(Price ~ ., data = df_clean)
  
  # -------------------------
  # 2(b) 提取参数，并把截距名改成 Intercept
  # -------------------------
  params <- coef(model)
  names(params)[names(params) == "(Intercept)"] <- "Intercept"
  
  # -------------------------
  # 2(c) 按题目给定的新房子特征做预测
  # -------------------------
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
  
  pred <- predict(model, newdata = new_house)
  price_pred <- as.numeric(pred)
  
  # -------------------------
  # 组装 regression_list
  # -------------------------
  regression_list <- list(
    model_parameters = params,
    price_prediction = price_pred
  )
  
  # ============================================================
  # 3) 返回最终 result
  # ============================================================
  result <- list(
    summary_list = summary_list,
    regression_list = regression_list
  )
  
  return(result)
}
```
