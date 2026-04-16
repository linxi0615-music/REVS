# ============================================================================
# 完整实验代码：TVS_stream 方法评估
# ============================================================================

library(BART)

# ============================================================================
# 1. 数据生成函数
# ============================================================================
generate_data <- function(n = 1000, p = 100, n_signal = 5, noise_sd = 1) {
  sigma <- 1
  X <- matrix(runif(n * p), n, p)
  Y <- 10 * sin(pi * X[, 1] * X[, 2]) +
    20 * (X[, 3] - 0.5)^2 +
    10 * X[, 4] +
    5 * X[, 5] +
    rnorm(n, 0, sigma)
  return(list(X = X, Y = Y, signal_vars = 1:5))
}

# ============================================================================
# 2. 奖励函数
# ============================================================================
reward <- function(Y, X, gammas, type, niter, ntree, binary_reward) {
  if (type == "bart") {
    # 检查选中的特征数量
    selected_idx <- which(gammas == 1)
    if (length(selected_idx) == 0) {
      return(numeric(0))
    }
    
    X_sel <- X[, gammas == 1, drop = FALSE]
    
    # 检查方差
    var_check <- apply(X_sel, 2, var)
    if (any(var_check < 1e-8)) {
      valid_cols <- var_check >= 1e-8
      if (sum(valid_cols) == 0) {
        return(rep(0, length(selected_idx)))
      }
      X_sel <- X_sel[, valid_cols, drop = FALSE]
      selected_idx <- selected_idx[valid_cols]
    }
    
    out <- capture.output(
      result <- BART::wbart(
        X_sel,
        Y,
        sparse = TRUE,
        ndpost = niter,
        printevery = 10000,
        ntree = ntree
      )
    )
    
    if (binary_reward) {
      reward <- as.numeric(result$varcount[niter, ] != 0)
    } else {
      reward <- as.numeric(result$varcount.mean)
      reward <- as.numeric(reward > 1)
    }
    
    # 返回完整长度的奖励向量
    full_reward <- numeric(sum(gammas))
    full_reward[1:length(reward)] <- reward
    
    return(full_reward)
  }
  
  if (type == "SSL") {
    result <- SSLASSO::SSLASSO(
      X[, gammas == 1],
      Y,
      lambda0 = seq(0.1, 100, length = 10),
      lambda1 = 0.1
    )
    reward <- as.numeric(abs(result$beta[, 10]))
    return(reward)
  }
}

# ============================================================================
# 3. TVS_stream 主函数
# ============================================================================
TVS_stream <- function(Y, X, selector = 'bart', topq, niter = 1000, ntree = 10,
                       a = 1, b = 1, binary_reward = TRUE, batchsize,
                       fix_stop, maxround = 100, stop_crit = 100) {
  
  if (!missing(fix_stop)) {
    stopifnot(fix_stop <= maxround)
    round <- fix_stop
  } else {
    round <- 1
  }
  
  converge_iter <- stop_crit
  start_time <- Sys.time()
  
  p <- ncol(X)
  n <- nrow(X)
  nrep <- floor(n / batchsize)
  
  A <- matrix(a, p, nrep * maxround)
  B <- matrix(b, p, nrep * maxround)
  
  con_criteria <- rep(1, converge_iter)
  iter <- 1
  converged <- FALSE
  First_convergence <- TRUE
  
  iter_converge <- NA
  k_converge <- NA
  converge_time <- NA
  iter_fixed <- NA
  time_fixed <- NA
  
  for (k in 1:maxround) {
    permute <- sample(1:n)
    Yperm <- Y[permute]
    Xperm <- X[permute, ]
    
    for (i in 1:nrep) {
      Aprev <- A[, (k - 1) * nrep + i]
      Bprev <- B[, (k - 1) * nrep + i]
      
      thetas <- rbeta(p, Aprev, Bprev)
      
      if (missing(topq)) {
        gammas <- as.numeric(thetas > 0.5)
      } else {
        gammas <- numeric(p)
        index <- order(thetas, decreasing = TRUE)
        gammas[index[1:topq]] <- 1
      }
      
      # 确保至少选择一个特征
      if (sum(gammas) == 0) {
        gammas[which.max(thetas)] <- 1
      }
      
      # 收集奖励
      index <- (i - 1) * batchsize + (1:batchsize)
      Ysub <- Yperm[index]
      Xsub <- Xperm[index, ]
      
      rbin <- reward(Ysub, Xsub, gammas, type = selector,
                     niter, ntree = ntree, binary_reward = binary_reward)
      
      if (length(rbin) > 0 && max(rbin) > 0) {
        rbin <- rbinom(sum(gammas), 1, rbin / max(rbin))
      } else {
        rbin <- rep(0, sum(gammas))
      }
      
      # 更新
      Anew <- Aprev
      Anew[gammas == 1] <- Anew[gammas == 1] + rbin
      Bnew <- Bprev
      Bnew[gammas == 1] <- Bnew[gammas == 1] + 1 - rbin
      
      # 记录收敛
      con_criteria[(iter %% converge_iter) + 1] <- 
        sum(abs((Anew / (Anew + Bnew) > 0.5) - (Aprev / (Aprev + Bprev) > 0.5)))
      
      if (sum(con_criteria) == 0) {
        converged <- TRUE
      }
      
      iter <- iter + 1
      
      if (i < nrep | k < maxround) {
        A[, (k - 1) * nrep + i + 1] <- Anew
        B[, (k - 1) * nrep + i + 1] <- Bnew
      }
    }
    
    # 检查收敛条件
    if (converged) {
      if (First_convergence) {
        First_convergence <- FALSE
        iter_converge <- iter
        k_converge <- k
        converge_time <- Sys.time() - start_time
      }
      
      if (k > round) {
        break
      }
    }
    
    if (k == round) {
      iter_fixed <- iter
      time_fixed <- Sys.time() - start_time
      
      if (converged) {
        break
      }
    }
  }
  
  # 最大轮次未收敛
  if (k == maxround) {
    iter_fixed <- iter
    k_converge <- k
    converge_time <- Sys.time() - start_time
  }
  
  if (First_convergence) {
    First_convergence <- FALSE
  }
  
  r <- list(
    A = A, 
    B = B, 
    iter_converge = ifelse(is.na(iter_converge), NA, iter_converge - 1),
    round_converge = k_converge, 
    iter_fixed = ifelse(is.na(iter_fixed), NA, iter_fixed - 1),
    time_fixed = time_fixed, 
    time_converge = converge_time,
    type = 'online'
  )
  
  class(r) <- 'TVS_stream'
  return(r)
}

# ============================================================================
# 4. 评估函数
# ============================================================================
evaluate_performance <- function(tvs_result, true_signal_vars, p) {
  # 确定使用最后一次迭代的概率
  last_iter <- if (!is.null(tvs_result$iter_converge) && !is.na(tvs_result$iter_converge)) {
    min(tvs_result$iter_converge + 1, ncol(tvs_result$A))
  } else {
    ncol(tvs_result$A)
  }
  
  # 计算最终特征包含概率
  final_probs <- tvs_result$A[, last_iter] / 
    (tvs_result$A[, last_iter] + tvs_result$B[, last_iter])
  
  # 选中概率>0.5的特征
  selected_vars <- which(final_probs > 0.5)
  
  # 计算评估指标
  true_positives <- sum(selected_vars %in% true_signal_vars)
  false_positives <- sum(!(selected_vars %in% true_signal_vars))
  false_negatives <- sum(!(true_signal_vars %in% selected_vars))
  
  # FDP (False Discovery Proportion)
  fdp <- ifelse(length(selected_vars) > 0, 
                false_positives / length(selected_vars), 0)
  
  # Precision
  precision <- ifelse(length(selected_vars) > 0, 
                      true_positives / length(selected_vars), 0)
  
  # Recall (Power)
  recall <- ifelse(length(true_signal_vars) > 0,
                   true_positives / length(true_signal_vars), 0)
  
  # F1 Score
  f1_score <- ifelse(precision + recall > 0, 
                     2 * precision * recall / (precision + recall), 0)
  
  # Hamming Distance
  true_selection <- rep(0, p)
  true_selection[true_signal_vars] <- 1
  estimated_selection <- rep(0, p)
  estimated_selection[selected_vars] <- 1
  hamming_distance <- sum(true_selection != estimated_selection)
  
  # Time
  time_taken <- if (!is.null(tvs_result$time_converge)) {
    as.numeric(tvs_result$time_converge, units = "secs")
  } else if (!is.null(tvs_result$time_fixed)) {
    as.numeric(tvs_result$time_fixed, units = "secs")
  } else {
    NA
  }
  
  return(list(
    fdp = fdp,
    precision = precision,
    recall = recall,
    f1_score = f1_score,
    hamming_distance = hamming_distance,
    time = time_taken,
    selected_vars = selected_vars,
    num_selected = length(selected_vars)
  ))
}

# ============================================================================
# 5. 打印结果函数
# ============================================================================
print_summary <- function(all_results) {
  cat("\n", paste(rep("=", 80), collapse = ""), "\n")
  cat("TVS_stream 方法性能评估（10次重复实验）\n")
  cat(paste(rep("=", 80), collapse = ""), "\n\n")
  
  metrics <- c("fdp", "precision", "recall", "f1_score", "hamming_distance", "time")
  metric_names <- c("FDP", "Precision", "Recall", "F1 Score", "Hamming Distance", "Time (seconds)")
  
  for (i in 1:length(metrics)) {
    metric <- metrics[i]
    metric_name <- metric_names[i]
    
    values <- sapply(all_results, function(x) x[[metric]])
    mean_val <- mean(values, na.rm = TRUE)
    sd_val <- sd(values, na.rm = TRUE)
    
    cat(sprintf("%-25s: %.4f ± %.4f\n", metric_name, mean_val, sd_val))
  }
  
  cat("\n", paste(rep("=", 80), collapse = ""), "\n")
  
  # 额外统计
  num_selected <- sapply(all_results, function(x) x$num_selected)
  cat(sprintf("\n平均选中特征数: %.2f ± %.2f\n", 
              mean(num_selected), sd(num_selected)))
  cat(sprintf("真实信号特征数: %d\n", length(all_results[[1]]$selected_vars)))
}

# ============================================================================
# 6. 主实验函数
# ============================================================================
run_experiment <- function(
    n_rep = 50,
    n = 1000,
    p = 100,
    niter = 100,
    ntree = 50,
    batchsize = 200,
    maxround = 50
) {
  
  cat("开始实验...\n")
  cat("参数设置:\n")
  cat(sprintf("  重复次数: %d\n", n_rep))
  cat(sprintf("  样本数 n: %d\n", n))
  cat(sprintf("  特征数 p: %d\n", p))
  cat(sprintf("  BART迭代数: %d\n", niter))
  cat(sprintf("  BART树数: %d\n", ntree))
  cat(sprintf("  批次大小: %d\n", batchsize))
  cat(sprintf("  最大轮数: %d\n\n", maxround))
  
  all_results <- list()
  
  for (rep in 1:n_rep) {
    cat(sprintf(">>> 运行第 %d/%d 次实验...\n", rep, n_rep))
    
    # 设置种子
    set.seed(2024 + rep * 100)
    
    # 生成数据
    data <- generate_data(n = n, p = p)
    
    # 运行 TVS_stream
    tvs_result <- TVS_stream(
      Y = data$Y,
      X = data$X,
      selector = 'bart',
      niter = niter,
      ntree = ntree,
      a = 1,
      b = 1,
      binary_reward = TRUE,
      batchsize = batchsize,
      maxround = maxround,
      stop_crit = 10
    )
    
    # 评估性能
    perf <- evaluate_performance(tvs_result, data$signal_vars, p)
    
    all_results[[rep]] <- perf
    
    cat(sprintf("    选中特征数: %d, F1: %.4f, 时间: %.2fs\n", 
                perf$num_selected, perf$f1_score, perf$time))
  }
  
  # 打印总结
  print_summary(all_results)
  
  return(invisible(all_results))
}

# ============================================================================
# 7. 运行实验
# ============================================================================
results <- run_experiment(
  n_rep = 50,
  n = 1000,
  p = 100,
  niter = 300,
  ntree = 50,
  batchsize = 50,
  maxround = 50
)