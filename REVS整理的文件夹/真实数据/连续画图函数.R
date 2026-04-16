library(ggplot2)
library(dplyr)

# 参数设定
csv_path <- "C:/Users/锡/Desktop/REVS论文/真实数据/student（连续）（base=0.86378）.csv"
dataset_name <- "Student Preformance "
baseline <- 0.86378

#csv_path <- "C:/Users/锡/Desktop/REVS论文/真实数据/crime（base=0.58692）.csv"
#dataset_name <- "Communities and Crime"
#baseline <- 0.58692

# 读取数据
df_raw <- read.csv(csv_path, stringsAsFactors = FALSE, check.names = FALSE)

# 提取列索引
method_cols <- seq(2, ncol(df_raw), by = 3)
mean_cols <- seq(3, ncol(df_raw), by = 3)
sd_cols <- seq(4, ncol(df_raw), by = 3)

methods <- as.character(df_raw[1, method_cols])
methods <- methods[!is.na(methods) & methods != ""]

df_long <- data.frame()
for(i in seq_along(methods)) {
  method_name <- methods[i]
  temp_df <- data.frame(
    n_features = as.numeric(df_raw[, 1]),
    method = method_name,
    mean = as.numeric(df_raw[, mean_cols[i]]),
    sd = as.numeric(df_raw[, sd_cols[i]])
  )
  df_long <- rbind(df_long, temp_df)
}

df_long <- df_long %>%
  filter(!is.na(n_features), !is.na(mean)) %>%
  filter(n_features >= 1 & n_features <= 30)

# 添加一个代表"All Features"的虚拟行
df_all_features <- data.frame(
  n_features = 1,
  method = "All Features",
  mean = baseline,
  sd = 0
)
df_long <- rbind(df_long, df_all_features)

y_min <- min(c(df_long$mean - df_long$sd, baseline), na.rm = TRUE) 
y_max <- max(c(df_long$mean + df_long$sd, baseline), na.rm = TRUE)

# 定义颜色和图例标签，确保"All Features"是第一个
color_methods <- c("All Features", methods)  # All Features放在第一位
# 定义颜色，确保"All Features"对应黑色
color_values <- c("black", "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", 
                  "#8c564b", "#e377c2", "#bcbd22", "#FDBF6F")

# 绘图
p <- ggplot(df_long, aes(x = n_features, y = mean, color = method)) +
  geom_line(linetype = "dashed", size = 1, alpha = 0.8) +
  geom_point(size = 2) +
  geom_errorbar(aes(ymin = mean - sd, ymax = mean + sd), width = 0.3, alpha = 0.6) +
  # 为水平线添加颜色映射
  geom_hline(yintercept = baseline, linetype = "dashed", size = 1, 
             aes(color = "All Features")) +
  scale_x_continuous(breaks = seq(1, 30, 2), limits = c(1, 30)) +
  #scale_y_continuous(limits = c(y_min, y_max)) +
  scale_y_continuous(limits = c(0, 0.9)) +
  scale_color_manual(
    values = color_values,
    breaks = color_methods,
    labels = color_methods
  ) +
  labs(title = dataset_name, x = "Number of Features", y = "Adjusted R²") +
  theme_bw() +
  theme(
    # 标题设置
    plot.title = element_text(hjust = 0.5, size = 16, face = "bold"),
    
    # 坐标轴标签设置
    axis.title.y = element_text(size = 16, face = "bold"),  # y轴标签与标题一致
    axis.title.x = element_text(size = 12),                 # x轴标签保持原样
    
    # 背景和边框设置
    panel.background = element_rect(fill = "white", color = NA),  # 面板背景白色
    plot.background = element_rect(fill = "white", color = NA),   # 绘图区背景白色
    panel.border = element_rect(color = "black", fill = NA, size = 1.5),  # 保留黑色边框
    
    # 去除网格线
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    
    # 图例设置
    legend.position = c(0.98, 0.02),
    legend.justification = c(1, 0),
    legend.background = element_rect(fill = "white", color = "black"),
    legend.title = element_blank(),
    
    # 坐标轴刻度文字
    axis.text = element_text(size = 10)
  )

print(p)