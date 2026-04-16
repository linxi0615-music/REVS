library(randomForest)
library(glmnet)
library(ggplot2)
library(gridExtra)
library(dplyr)
library(reshape2)
library(caret)      # 用于RFE
library(e1071)      # 用于SVM

# =============================================================================
# 生成三分类数据集的函数（保持不变：信号变量=5个，公式不变系数不同）
# =============================================================================
generate_multiclass_data <- function(n = 300, p = 1000, n_signal = 5, sigma = 0.3) {
  X <- matrix(rnorm(n * p), nrow = n, ncol = p)
  signal_vars <- 1:n_signal  # 前5个为真实信号特征
  
  # 线性组合
  linear_comb_0 <- 3.2 * X[, 1] + 2.8 * X[, 2] - 1.5 * X[, 3] +
    2.1 * X[, 4] - 1.8 * X[, 5] + rnorm(n, 0, sigma) + 2
  linear_comb_1 <- -2.5 * X[, 1] + 1.9 * X[, 2] + 2.3 * X[, 3] +
    -1.7 * X[, 4] + 2.0 * X[, 5] + rnorm(n, 0, sigma) + 0
  linear_comb_2 <- -1.8 * X[, 1] - 2.2 * X[, 2] + 1.2 * X[, 3] +
    -2.4 * X[, 4] - 1.5 * X[, 5] + rnorm(n, 0, sigma) - 2
  
  # 非线性项
  nonlinear_0 <- 0.8 * sin(pi * X[, 1] * X[, 2]) + 0.5 * (X[, 3]^2)
  nonlinear_1 <- 0.6 * cos(pi * X[, 3] * X[, 4]) + 0.4 * (X[, 5]^2)
  nonlinear_2 <- 0.7 * sin(pi * X[, 5] * X[, 1]) + 0.3 * (X[, 2]^2)
  
  # 合并分数
  score_0 <- linear_comb_0 + nonlinear_0
  score_1 <- linear_comb_1 + nonlinear_1
  score_2 <- linear_comb_2 + nonlinear_2
  
  # Softmax概率
  scores <- cbind(score_0, score_1, score_2)
  exp_scores <- exp(scores)
  probs <- exp_scores / rowSums(exp_scores)
  
  # 生成标签
  Y <- apply(probs, 1, function(p) sample(0:2, 1, prob = p))
  
  return(list(
    X = X,
    Y = Y,
    signal_vars = signal_vars,  # 信号变量：1-5
    true_vars_0 = 1:5,          # 类别0真实相关变量
    true_vars_1 = 1:5,          # 类别1真实相关变量
    true_vars_2 = 1:5           # 类别2真实相关变量
  ))
}

# =============================================================================
# 辅助函数：移除零方差和高度相关特征（保持不变）
# =============================================================================
preprocess_features <- function(X) {
  # 移除零方差特征
  variances <- apply(X, 2, var, na.rm = TRUE)
  X <- X[, variances > 1e-6, drop = FALSE]
  
  # 移除高度相关特征（避免共线性）
  if (ncol(X) > 1) {
    cor_mat <- cor(X)
    diag(cor_mat) <- 0
    high_cor <- which(abs(cor_mat) > 0.9, arr.ind = TRUE)
    if (nrow(high_cor) > 0) {
      to_remove <- unique(high_cor[, 2])
      X <- X[, -to_remove, drop = FALSE]
    }
  }
  X
}

# =============================================================================
# reward函数（仅保留RF分支）
# =============================================================================
reward <- function(Y, X, gammas, type = "rf", niter, ntree, binary_reward) {
  if (sum(gammas) == 0) return(numeric(0))
  
  # 提取并预处理特征
  X_selected <- X[, gammas == 1, drop = FALSE]
  X_selected <- preprocess_features(X_selected)
  n_selected_feat <- ncol(X_selected)
  if (n_selected_feat == 0) {
    warning("所有选中的特征均被预处理移除，返回空奖励")
    return(numeric(0))
  }
  
  # 仅保留RF逻辑
  if (type == "rf") {
    # 训练随机森林（多分类，Y转为因子）
    result <- randomForest::randomForest(
      x = X_selected,
      y = factor(Y, levels = unique(Y)),
      ntree = ntree,
      importance = TRUE
    )
    
    # 提取特征重要性（Gini系数减少量）
    imp <- randomForest::importance(result, type = 2)
    
    # 计算奖励并对齐原始特征长度
    reward <- numeric(length(gammas))
    if (binary_reward) {
      threshold <- median(imp, na.rm = TRUE)
      threshold <- ifelse(is.na(threshold), 0, threshold)
      reward[gammas == 1][1:ncol(X_selected)] <- as.numeric(imp >= threshold)
    } else {
      max_imp <- max(imp, na.rm = TRUE)
      max_imp <- ifelse(max_imp == 0, 1, max_imp)
      reward[gammas == 1][1:ncol(X_selected)] <- imp / max_imp
    }
    
    reward[is.na(reward)] <- 0
    return(reward)
  }
  
  stop("Unsupported selector type: ", type)  # 仅RF有效，基本不触发
}

# =============================================================================
# TVS函数（仅支持RF选择器）
# =============================================================================
TVS <- function(Y, X, selector = "rf", topq, maxnrep, stop_crit = 100, 
                fix_stop, niter, ntree, a = 1, b = 1, binary_reward = TRUE) {
  
  # 仅允许RF选择器
  selector <- match.arg(selector, choices = "rf")
  
  start_time <- Sys.time()
  p <- ncol(X)
  n <- nrow(X)
  round <- 1
  k <- 1
  
  # 初始化Beta分布参数（A=成功次数，B=失败次数）
  A <- matrix(a, p, maxnrep * round)
  B <- matrix(b, p, maxnrep * round)
  model_diff <- rep(1, stop_crit)  # 模型变化量（判断收敛）
  model <- rep(0, p)               # 当前选中的特征模型
  iter <- 1
  
  converge_time <- NA  # 收敛时间
  
  for (i in (1:maxnrep)) {
    # 1. 从Beta分布采样特征选中概率
    Aprev <- A[, (k-1)*maxnrep + i]
    Bprev <- B[, (k-1)*maxnrep + i]
    thetas <- rbeta(p, Aprev, Bprev)
    
    # 2. 确定当前选中的特征（gammas=1表示选中）
    if (missing(topq)) {
      gammas <- as.numeric(thetas > 0.5)  # 概率>0.5选中
    } else {
      gammas <- numeric(p)
      index <- order(thetas, decreasing = TRUE)
      gammas[index[1:topq]] <- 1  # 选Top-q个特征
    }
    
    # 3. 调用reward函数获取特征奖励
    rbin <- reward(Y, X, gammas, type = selector, niter, ntree = ntree,
                   binary_reward = binary_reward)
    
    # 4. 处理奖励（二值化抽样，增强随机性）
    if (length(rbin) > 0 && max(rbin, na.rm = TRUE) > 0) {
      rbin_prob <- rbin / max(rbin, na.rm = TRUE)
      rbin_prob[is.na(rbin_prob)] <- 0
      rbin <- rbinom(sum(gammas), 1, rbin_prob)
    } else {
      rbin <- rep(0, sum(gammas))
    }
    
    # 5. 更新Beta分布参数
    Anew <- Aprev
    Anew[gammas == 1] <- Anew[gammas == 1] + rbin  # 选中且有奖励：A+1
    Bnew <- Bprev
    Bnew[gammas == 1] <- Bnew[gammas == 1] + 1 - rbin  # 选中无奖励：B+1
    
    # 6. 判断模型收敛（特征选中状态无变化）
    model_new <- ((Anew/(Anew + Bnew)) > 0.5)
    model_diff[iter %% stop_crit + 1] <- sum(abs(model - model_new))
    model <- model_new
    
    # 7. 保存更新后的参数
    if (i < maxnrep) {
      A[, (k-1)*maxnrep + i + 1] <- Anew
      B[, (k-1)*maxnrep + i + 1] <- Bnew
    }
    
    iter <- iter + 1
    
    # 收敛则提前停止
    if (sum(model_diff) == 0) {
      converge_time <- Sys.time() - start_time
      break
    }
  }
  
  # 未收敛则取总运行时间
  if (is.na(converge_time)) {
    converge_time <- Sys.time() - start_time
  }
  if (!missing(fix_stop)) {
    converge_time <- Sys.time() - start_time
  }
  
  # 返回TVS结果
  r <- list(A = A[, 1:(iter-1)], B = B[, 1:(iter-1)], 
            model_diff = model_diff, time = converge_time, type = 'offline',
            selector = selector)
  class(r) <- 'TVS'
  return(r)
}

# =============================================================================
# 绘图函数：特征包含概率演化图（保持不变，适配单方法）
# =============================================================================
plot_inclusion_probs_multiclass <- function(tvs_result, title, top_n = 10, max_time = NULL) {
  prob <- tvs_result$A/(tvs_result$A + tvs_result$B)
  p <- nrow(prob)
  n_time <- ncol(prob)
  
  if (!is.null(max_time)) n_time <- min(n_time, max_time)
  
  # 获取Top-N特征
  final_probs <- prob[, n_time]
  top_indices <- order(final_probs, decreasing = TRUE)[1:min(top_n, p)]
  
  # 设置绘图参数
  par(mar = c(5, 5, 4, 2) + 0.1)
  
  # 绘制空图框架
  plot(1, type = "n", 
       xlim = c(1, n_time), 
       ylim = c(0, 1),
       xlab = "TVS Iteration", 
       ylab = "Inclusion Probability",
       main = title,
       cex.lab = 1.2,
       cex.axis = 1.1,
       cex.main = 1.3)
  
  # 绘制非Top特征（黑色细线）
  for (i in setdiff(1:p, top_indices)) {
    lines(1:n_time, prob[i, 1:n_time], col = "black", lwd = 0.3, alpha = 0.5)
  }
  
  # 绘制Top特征（彩色粗线）
  colors <- c("red", "darkred", "firebrick", "coral", "orangered",
              "tomato", "indianred", "brown", "maroon", "darkorange")
  for (i in 1:length(top_indices)) {
    var_idx <- top_indices[i]
    color_idx <- ((i-1) %% length(colors)) + 1
    lines(1:n_time, prob[var_idx, 1:n_time], 
          col = colors[color_idx], lwd = 2)
    
    # 右侧添加特征索引标签
    text(n_time, prob[var_idx, n_time], 
         labels = var_idx, 
         col = colors[color_idx], cex = 0.8, pos = 4)
  }
  
  # 添加0.5阈值线（判断是否选中特征）
  abline(h = 0.5, lty = 2, col = "blue", lwd = 1.5)
  
  # 添加图例
  legend("topright", 
         legend = c(paste("Top", top_n, "Features"), "Other Features"),
         col = c("red", "black"), 
         lwd = c(2, 0.5),
         bty = "n",
         cex = 0.9)
  
  return(list(
    top_indices = top_indices,
    top_probs = final_probs[top_indices]
  ))
}

# =============================================================================
# 扩展性能评估函数（增加FDP、power等指标）
# =============================================================================
calculate_metrics <- function(selected_vars, true_signal_vars, p) {
  # 计算混淆矩阵元素
  true_positives <- sum(selected_vars %in% true_signal_vars)
  false_positives <- sum(!(selected_vars %in% true_signal_vars))
  false_negatives <- sum(!(true_signal_vars %in% selected_vars))
  true_negatives <- p - length(union(selected_vars, true_signal_vars))
  
  # 计算指标
  precision <- ifelse(length(selected_vars) > 0, true_positives / length(selected_vars), 0)
  recall <- ifelse(length(true_signal_vars) > 0, true_positives / length(true_signal_vars), 0)
  f1_score <- ifelse((precision + recall) > 0, 2 * precision * recall / (precision + recall), 0)
  
  # Power (统计功效) = 召回率
  power <- recall
  
  # FDP (False Discovery Proportion)
  fdp <- ifelse(length(selected_vars) > 0, false_positives / length(selected_vars), 0)
  
  # 汉明距离
  true_selection <- rep(0, p)
  true_selection[true_signal_vars] <- 1
  pred_selection <- rep(0, p)
  pred_selection[selected_vars] <- 1
  hamming_distance <- sum(true_selection != pred_selection)
  
  return(list(
    precision = precision,
    recall = recall,
    f1_score = f1_score,
    power = power,
    fdp = fdp,
    hamming_distance = hamming_distance,
    true_positives = true_positives,
    false_positives = false_positives,
    false_negatives = false_negatives,
    num_selected = length(selected_vars)
  ))
}

evaluate_performance <- function(method_result, true_signal_vars, p, computation_time = NULL) {
  if (inherits(method_result, 'TVS')) {
    # 对于TVS，使用最后一次迭代的概率
    final_probs <- method_result$A[, ncol(method_result$A)] / 
      (method_result$A[, ncol(method_result$A)] + method_result$B[, ncol(method_result$B)])
    
    # 选中概率>0.5的特征
    selected_vars <- which(final_probs > 0.5)
    
    # 计算各种指标
    metrics <- calculate_metrics(selected_vars, true_signal_vars, p)
    metrics$selected_vars <- selected_vars
    metrics$computation_time <- as.numeric(method_result$time, units = "secs")
  } else if (is.list(method_result) && "selected_vars" %in% names(method_result)) {
    # 对于其他方法，直接使用选中的变量
    metrics <- calculate_metrics(method_result$selected_vars, true_signal_vars, p)
    metrics$selected_vars <- method_result$selected_vars
    metrics$computation_time <- computation_time
  } else {
    stop("不支持的结果类型")
  }
  
  return(metrics)
}

# =============================================================================
# 基准方法实现
# =============================================================================

# SVM特征选择方法
evaluate_svm <- function(X, Y, true_signal_vars) {
  start_time <- Sys.time()
  
  tryCatch({
    # 为特征命名
    colnames(X) <- paste0("V", 1:ncol(X))
    
    # 使用线性SVM进行特征选择
    svm_model <- svm(x = X, y = factor(Y), kernel = "linear", 
                     type = "C-classification", scale = TRUE)
    
    # 获取线性SVM的权重系数
    svm_weights <- t(svm_model$coefs) %*% svm_model$SV
    
    # 计算特征重要性（权重的绝对值）
    feature_importance <- abs(svm_weights)
    
    # 选择重要性大于阈值的变量（选择前10%的特征）
    threshold <- quantile(feature_importance, 0.9)
    selected_features <- which(feature_importance >= threshold)
    
    # 转换为原始索引
    selected_indices <- as.numeric(gsub("V", "", colnames(X)[selected_features]))
    
    computation_time <- as.numeric(Sys.time() - start_time, units = "secs")
    
    return(list(
      selected_vars = selected_indices,
      computation_time = computation_time
    ))
    
  }, error = function(e) {
    cat("SVM 错误:", e$message, "\n")
    # 返回默认结果
    return(list(
      selected_vars = integer(0),
      computation_time = as.numeric(Sys.time() - start_time, units = "secs")
    ))
  })
}

# 逻辑回归
evaluate_logistic_regression <- function(X, Y, true_signal_vars) {
  start_time <- Sys.time()
  
  # 标准化特征
  X_scaled <- scale(X)
  
  # 使用cv.glmnet进行L1正则化逻辑回归
  cv_fit <- cv.glmnet(X_scaled, Y, alpha = 1, family = "multinomial")
  
  # 获取系数
  coef_list <- coef(cv_fit, s = "lambda.1se")
  
  # 合并所有类别的非零特征
  selected_features <- c()
  for (coef_matrix in coef_list) {
    nonzero_features <- which(abs(coef_matrix[-1, 1]) > 1e-6)  # 去除截距项
    selected_features <- union(selected_features, nonzero_features)
  }
  
  computation_time <- as.numeric(Sys.time() - start_time, units = "secs")
  
  return(list(
    selected_vars = selected_features,
    computation_time = computation_time
  ))
}

# 随机森林
evaluate_random_forest <- function(X, Y, true_signal_vars, n_estimators = 100, threshold_percentile = 90) {
  start_time <- Sys.time()
  
  # 训练随机森林
  rf <- randomForest(x = X, y = factor(Y), ntree = n_estimators, importance = TRUE)
  
  # 使用特征重要性阈值选择特征
  importance_values <- importance(rf, type = 2)
  importance_threshold <- quantile(importance_values, threshold_percentile/100)
  selected_features <- which(importance_values >= importance_threshold)
  
  computation_time <- as.numeric(Sys.time() - start_time, units = "secs")
  
  return(list(
    selected_vars = selected_features,
    computation_time = computation_time
  ))
}

# =============================================================================
# 任务1：50次运行统计分析
# =============================================================================
run_multiple_experiments <- function(n_runs = 50) {
  cat("开始进行", n_runs, "次TVS实验...\n")
  
  all_results <- list()
  
  for (run in 1:n_runs) {
    if (run %% 10 == 0) {
      cat("完成", run, "/", n_runs, "次实验\n")
    }
    
    # 设置种子确保可重现性
    set.seed(2024 + run)
    
    # 生成数据 (n=300, p=1400)
    data <- generate_multiclass_data(n = 100, p = 2000, n_signal = 5)
    X <- data$X
    Y <- data$Y
    true_signal_vars <- data$signal_vars
    
    # 运行TVS
    tvs_result <- TVS(
      Y = Y,
      X = X,
      selector = "rf",          # 仅RF选择器
      maxnrep = 1000,             # 最大迭代次数
      stop_crit = 100,           # 收敛判断窗口
      niter = 300,               # RF模型训练参数（与reward函数兼容）
      ntree = 85,               # RF树数量（平衡性能与速度）
      binary_reward = TRUE      # 二值化奖励（1=重要，0=不重要）
    )
    
    # 评估性能
    metrics <- evaluate_performance(tvs_result, true_signal_vars, ncol(X))
    metrics$run <- run
    all_results[[run]] <- metrics
  }
  
  return(all_results)
}

analyze_multiple_runs <- function(results) {
  # 转换为数据框
  metrics_df <- do.call(rbind, lapply(results, function(x) {
    data.frame(
      precision = x$precision,
      recall = x$recall,
      f1_score = x$f1_score,
      power = x$power,
      fdp = x$fdp,
      hamming_distance = x$hamming_distance,
      num_selected = x$num_selected,
      computation_time = x$computation_time,
      run = x$run
    )
  }))
  
  # 计算统计量
  metrics_to_analyze <- c('precision', 'recall', 'f1_score', 'power', 'fdp', 
                          'hamming_distance', 'num_selected', 'computation_time')
  
  stats <- list()
  for (metric in metrics_to_analyze) {
    stats[[metric]] <- list(
      mean = mean(metrics_df[[metric]], na.rm = TRUE),
      sd = sd(metrics_df[[metric]], na.rm = TRUE),
      min = min(metrics_df[[metric]], na.rm = TRUE),
      max = max(metrics_df[[metric]], na.rm = TRUE),
      median = median(metrics_df[[metric]], na.rm = TRUE)
    )
  }
  
  return(list(stats = stats, df = metrics_df))
}

plot_multiple_runs_mean_sd <- function(results_analysis) {
  stats <- results_analysis$stats
  
  # 创建数据框用于绘图
  plot_data <- data.frame(
    Metric = names(stats),
    Mean = sapply(stats, function(x) x$mean),
    SD = sapply(stats, function(x) x$sd)
  )
  
  # 设置指标显示名称
  metric_names <- c(
    'precision' = 'Precision',
    'recall' = 'Recall', 
    'f1_score' = 'F1 Score',
    'power' = 'Power',
    'fdp' = 'FDP',
    'hamming_distance' = 'Hamming Distance',
    'num_selected' = 'Number Selected',
    'computation_time' = 'Computation Time (s)'
  )
  
  plot_data$Metric_Name <- metric_names[plot_data$Metric]
  
  # 创建均值与标准差柱状图
  p <- ggplot(plot_data, aes(x = Metric_Name, y = Mean, fill = Metric_Name)) +
    geom_bar(stat = "identity", alpha = 0.8) +
    geom_errorbar(aes(ymin = pmax(Mean - SD, 0), ymax = Mean + SD), 
                  width = 0.2, position = position_dodge(0.9)) +
    labs(title = "TVS方法50次运行指标均值与标准差",
         x = "性能指标",
         y = "均值 ± 标准差") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1),
          legend.position = "none",
          plot.title = element_text(hjust = 0.5, size = 14)) +
    geom_text(aes(label = paste0(round(Mean, 3), " ± ", round(SD, 3))), 
              vjust = -0.5, size = 3) +
    scale_fill_brewer(palette = "Set3")
  
  print(p)
  
  # 打印统计结果
  cat("\n=== TVS方法50次运行统计结果 ===\n")
  for (metric in names(stats)) {
    cat("\n", metric_names[metric], ":\n")
    cat("  均值:", round(stats[[metric]]$mean, 4), "\n")
    cat("  标准差:", round(stats[[metric]]$sd, 4), "\n")
    cat("  最小值:", round(stats[[metric]]$min, 4), "\n")
    cat("  最大值:", round(stats[[metric]]$max, 4), "\n")
    cat("  中位数:", round(stats[[metric]]$median, 4), "\n")
  }
}

# =============================================================================
# 任务2：多方法对比
# =============================================================================

# =============================================================================
# 主程序执行
# =============================================================================

main <- function() {
  # 任务1：50次运行统计分析
  cat("=", rep("=", 49), "\n", sep = "")
  cat("任务1：TVS方法50次运行统计分析\n")
  cat("=", rep("=", 49), "\n", sep = "")
  
  # 运行50次实验
  multiple_results <- run_multiple_experiments(n_runs = 50)
  
  # 分析结果
  results_analysis <- analyze_multiple_runs(multiple_results)
  
  #绘制均值方差柱状图
  plot_multiple_runs_mean_sd(results_analysis)
  

}

# 运行主程序
if (interactive()) {
  main()
}