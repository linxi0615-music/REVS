#分类情况
# 参数设定
class_path <- "C:/Users/锡/Desktop/REVS论文/真实数据/二分类（最后）（base=0.805）.csv";baseline <- 0.805;dataset_name <- "Wiki4he"

library(ggplot2)
library(dplyr)

# 参数设定
#class_path <- "C:/Users/锡/Desktop/REVS论文/真实数据/5class版本二（未挑选）（base=0.4937）.csv";baseline <- 0.4937;dataset_name <- "Student Preformance "

# 读取数据
df_raw <- read.csv(class_path, stringsAsFactors = FALSE, check.names = FALSE)

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

# 添加一个代表"All Features"的虚拟行，确保图例中有这个条目
df_all_features <- data.frame(
  n_features = 1,
  method = "All Features",
  mean = baseline,
  sd = 0
)
df_long <- rbind(df_long, df_all_features)

y_min <- min(c(df_long$mean - df_long$sd, baseline), na.rm = TRUE) - 0.01
y_max <- max(c(df_long$mean + df_long$sd, baseline), na.rm = TRUE) + 0.01

color_methods <- c("All Features", methods)  # 共6个元素
# 提供6个颜色值（All Features用黑色，其余用不同颜色）
color_values <- c("black", "#d62728", "#ff7f0e", "#2ca02c", "#e377c2", "#1f77b4")

# 在绘图前自动计算 y 轴范围
# 考虑所有点的均值 ± 标准差，以及 baseline
y_min_all <- min(c(df_long$mean - df_long$sd, baseline), na.rm = TRUE)
y_max_all <- max(c(df_long$mean + df_long$sd, baseline), na.rm = TRUE)

# 添加一些边距，使图形更美观（可选项）
y_margin <- (y_max_all - y_min_all) * 0.05  # 5% 的边距
y_min_auto <- y_min_all - y_margin
y_max_auto <- y_max_all + y_margin

# 绘图
p <- ggplot(df_long, aes(x = n_features, y = mean, color = method)) +
  geom_line(linetype = "dashed", size = 1, alpha = 0.8) +
  geom_point(size = 0.8) +
  geom_errorbar(aes(ymin = mean - sd, ymax = mean + sd), width = 0.3, alpha = 0.6) +
  # 为水平线添加颜色映射
  geom_hline(yintercept = baseline, linetype = "dashed", size = 1, 
             aes(color = "All Features")) +
  scale_y_continuous(limits = c(y_min_auto, y_max_auto)) +  # 自动范围
  scale_color_manual(
    values = color_values,
    breaks = color_methods,
    labels = color_methods
  ) +
  labs(title = dataset_name, x = "Number of Features", y = "Accuracy") +
  theme_bw() +
  theme(
    # 标题设置
    plot.title = element_text(hjust = 0.5, size = 16, face = "bold"),
    
    # 坐标轴标签设置
    axis.title.y = element_text(size = 16, face = "bold"),
    axis.title.x = element_text(size = 12),
    
    # 背景和边框设置
    panel.background = element_rect(fill = "white", color = NA),
    plot.background = element_rect(fill = "white", color = NA),
    panel.border = element_rect(color = "black", fill = NA, size = 1.5),
    
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