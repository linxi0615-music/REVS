library(ggplot2)
library(dplyr)
library(patchwork)

# ==================== 定义绘图函数 ====================

# 回归任务绘图函数
plot_regression <- function(csv_path, dataset_name, baseline, y_label = "Adjusted R²") {
  
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
  
  # 添加代表"All Features"的虚拟行
  df_all_features <- data.frame(
    n_features = 1,
    method = "All Features",
    mean = baseline,
    sd = 0
  )
  df_long <- rbind(df_long, df_all_features)
  
  # 自动计算纵轴范围
  y_min_all <- min(df_long$mean - df_long$sd, baseline, na.rm = TRUE)
  y_max_all <- max(df_long$mean + df_long$sd, baseline, na.rm = TRUE)
  y_margin <- (y_max_all - y_min_all) * 0.05
  y_min_auto <- y_min_all - y_margin
  y_max_auto <- y_max_all + y_margin
  
  # 定义颜色（确保 All Features 是第一个）
  color_methods <- c("All Features", methods)
  color_values <- c("black", "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", 
                    "#8c564b", "#e377c2", "#bcbd22", "#FDBF6F")
  
  # 绘图
  p <- ggplot(df_long, aes(x = n_features, y = mean, color = method)) +
    geom_line(linetype = "dashed", size = 0.8, alpha = 0.8) +
    geom_point(size = 0.6) +
    geom_errorbar(aes(ymin = mean - sd, ymax = mean + sd), width = 0.3, alpha = 0.6, linewidth = 0.3) +
    geom_hline(yintercept = baseline, linetype = "dashed", size = 0.8, color = "black") +
    scale_x_continuous(breaks = seq(1, 30, 5), limits = c(1, 30)) +
    scale_y_continuous(limits = c(y_min_auto, y_max_auto)) +
    scale_color_manual(
      values = color_values,
      breaks = color_methods,
      labels = color_methods
    ) +
    labs(title = dataset_name, x = "Number of Features", y = y_label) +
    theme_bw() +
    theme(
      plot.title = element_text(hjust = 0.5, size = 11, face = "bold"),
      axis.title.y = element_text(size = 9, face = "bold"),
      axis.title.x = element_text(size = 8),
      axis.text = element_text(size = 7),
      panel.background = element_rect(fill = "white", color = NA),
      plot.background = element_rect(fill = "white", color = NA),
      panel.border = element_rect(color = "black", fill = NA, size = 0.8),
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      legend.position = "bottom",
      legend.direction = "horizontal",
      legend.background = element_rect(fill = "white", color = "black", linewidth = 0.3),
      legend.title = element_blank(),
      legend.text = element_text(size = 7),
      legend.key.size = unit(0.25, "cm"),
      legend.key.width = unit(0.25, "cm"),
      legend.spacing.x = unit(0.1, "cm"),
      # 使用 ggplot2::margin 明确指定函数
      legend.margin = ggplot2::margin(t = 2, r = 2, b = 2, l = 2),
      plot.margin = ggplot2::margin(t = 5, r = 5, b = 15, l = 5)
    )
  
  return(p)
}

# 分类任务绘图函数
plot_classification <- function(csv_path, dataset_name, baseline, y_label = "Accuracy") {
  
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
  
  # 添加代表"All Features"的虚拟行
  df_all_features <- data.frame(
    n_features = 1,
    method = "All Features",
    mean = baseline,
    sd = 0
  )
  df_long <- rbind(df_long, df_all_features)
  
  # 自动计算纵轴范围
  y_min_all <- min(df_long$mean - df_long$sd, baseline-0.2, na.rm = TRUE)
  y_max_all <- max(df_long$mean + df_long$sd, baseline, na.rm = TRUE)
  y_margin <- (y_max_all - y_min_all) * 0.05
  y_min_auto <- y_min_all - y_margin
  y_max_auto <- y_max_all + y_margin
  
  # 定义颜色
  color_methods <- c("All Features", methods)
  color_values <- c("black", "#d62728", "#ff7f0e", "#2ca02c", "#e377c2", "#1f77b4")
  
  # 绘图
  p <- ggplot(df_long, aes(x = n_features, y = mean, color = method)) +
    geom_line(linetype = "dashed", size = 0.8, alpha = 0.8) +
    geom_point(size = 0.6) +
    geom_errorbar(aes(ymin = mean - sd, ymax = mean + sd), width = 0.3, alpha = 0.6, linewidth = 0.3) +
    geom_hline(yintercept = baseline, linetype = "dashed", size = 0.8, color = "black") +
    scale_x_continuous(breaks = seq(1, 30, 5), limits = c(1, 30)) +
    scale_y_continuous(limits = c(y_min_auto, y_max_auto)) +
    scale_color_manual(
      values = color_values,
      breaks = color_methods,
      labels = color_methods
    ) +
    labs(title = dataset_name, x = "Number of Features", y = y_label) +
    theme_bw() +
    theme(
      plot.title = element_text(hjust = 0.5, size = 11, face = "bold"),
      axis.title.y = element_text(size = 9, face = "bold"),
      axis.title.x = element_text(size = 8),
      axis.text = element_text(size = 7),
      panel.background = element_rect(fill = "white", color = NA),
      plot.background = element_rect(fill = "white", color = NA),
      panel.border = element_rect(color = "black", fill = NA, size = 0.8),
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      legend.position = "bottom",
      legend.direction = "horizontal",
      legend.background = element_rect(fill = "white", color = "black", linewidth = 0.3),
      legend.title = element_blank(),
      legend.text = element_text(size = 7),
      legend.key.size = unit(0.25, "cm"),
      legend.key.width = unit(0.25, "cm"),
      legend.spacing.x = unit(0.1, "cm"),
      # 使用 ggplot2::margin 明确指定函数
      legend.margin = ggplot2::margin(t = 2, r = 2, b = 2, l = 2),
      plot.margin = ggplot2::margin(t = 5, r = 5, b = 15, l = 5)
    )
  
  return(p)
}

# ==================== 定义四个数据路径 ====================

# 回归任务 - 两个数据集
reg_student_path <- "C:/Users/锡/Desktop/REVS论文/真实数据/student（连续）（base=0.86378）.csv"
reg_student_name <- "Student Performance (Regression)"
reg_student_baseline <- 0.86378

reg_crime_path <- "C:/Users/锡/Desktop/REVS论文/真实数据/crime（base=0.58692）.csv"
reg_crime_name <- "Communities and Crime (Regression)"
reg_crime_baseline <- 0.58692

# 分类任务 - 两个数据集
class_binary_path <- "C:/Users/锡/Desktop/REVS论文/真实数据/二分类（最后）（base=0.805）.csv"
class_binary_name <- "Wiki4he (Binary Classification)"
class_binary_baseline <- 0.805

class_multi_path <- "C:/Users/锡/Desktop/REVS论文/真实数据/5class版本二（未挑选）（base=0.4937）.csv"
class_multi_name <- "Student Performance (Multi-class)"
class_multi_baseline <- 0.4937

# ==================== 生成四个图形 ====================

p1 <- plot_regression(reg_student_path, reg_student_name, reg_student_baseline, y_label = "Adjusted R²")
p2 <- plot_regression(reg_crime_path, reg_crime_name, reg_crime_baseline, y_label = "Adjusted R²")
p3 <- plot_classification(class_binary_path, class_binary_name, class_binary_baseline, y_label = "Accuracy")
p4 <- plot_classification(class_multi_path, class_multi_name, class_multi_baseline, y_label = "Accuracy")

# ==================== 组合成 2×2 布局 ====================

combined_plot <- (p1 + p2) / (p3 + p4) +
  plot_annotation(
    theme = theme(
      plot.title = element_text(hjust = 0.5, size = 14, face = "bold")
    )
  ) &
  theme(
    legend.position = "bottom",
    legend.direction = "horizontal"
  )

# 显示图形
print(combined_plot)

# 保存图形
ggsave("combined_plot.pdf", combined_plot, width = 10, height = 9, dpi = 300)