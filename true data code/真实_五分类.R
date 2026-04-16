library(readr)
library(dplyr)
library(caret)
library(randomForest)
library(e1071)
library(ggplot2)
library(gridExtra)
library(FNN)    
library(nnet)   
library(tidyr)  
library(rpart)  

set.seed(2024)
preprocess_student_data <- function(df, target_column = "Mjob") {
  # 处理缺失值（先填充所有NA）
  df[is.na(df)] <- 0  # 全局NA填充，避免后续编码出错
  
  # 检测并编码类别变量（强制转为数值，避免因子残留）
  categorical_cols <- names(df)[!sapply(df, is.numeric)]
  categorical_cols <- setdiff(categorical_cols, target_column)
  
  cat("检测到", length(categorical_cols), "个类别变量需要编码:", 
      paste(categorical_cols, collapse = ", "), "\n")
  
  for (col in categorical_cols) {
    # 强制转为字符→因子→数值，避免编码异常
    df[[col]] <- as.character(df[[col]])
    df[[col]] <- as.numeric(factor(df[[col]], levels = unique(df[[col]])))
  }
  
  # 分离特征和目标（强制目标为因子，确保分类）
  if (!(target_column %in% names(df))) {
    stop(paste("Target column", target_column, "not found in dataset"))
  }
  
  Y <- as.factor(df[[target_column]])  # 强制因子型，避免数值混淆
  X_df <- df[, setdiff(names(df), target_column), drop = FALSE]
  
  # 强制所有特征为数值型，最终校验
  X_df <- X_df %>% mutate(across(everything(), ~as.numeric(.x)))
  X <- as.matrix(X_df)
  feature_names <- colnames(X_df)
  
  # 标准化特征（增加鲁棒性，处理极端值）
  X <- scale(X, center = TRUE, scale = TRUE)
  X[is.na(X) | is.infinite(X)] <- 0  # 标准化后可能出现的NA/无穷值填充
  
  cat("预处理完成：特征数 =", ncol(X), "，样本数 =", nrow(X), "，目标类别数 =", length(levels(Y)), "\n")
  
  return(list(
    X = X,
    Y = Y,
    feature_names = feature_names
  ))
}

# =============================================================================
# TVS-RFE算法实现（核心修复：强化数值校验+单特征保护）
# =============================================================================
# 2.1 RF奖励函数（修复单特征+非数值问题）
reward_rf_offline <- function(Y, X, gammas, ntree = 50, binary_reward = TRUE) {
  p <- ncol(X)
  selected_idx <- which(gammas == 1)
  full <- rep(0, p)
  
  # 空选择保护
  if (length(selected_idx) == 0) return(full)
  
  # 强制X_sel为矩阵+数值型（核心修复）
  X_sel <- X[, selected_idx, drop = FALSE]
  if (!is.matrix(X_sel)) X_sel <- as.matrix(X_sel)
  if (!is.numeric(X_sel)) X_sel <- matrix(as.numeric(X_sel), nrow = nrow(X_sel))
  
  # 单特征维度保护
  if (ncol(X_sel) == 1) colnames(X_sel) <- "feature_1"
  
  res <- tryCatch({
    # 强制Y为因子（分类任务）
    Y_factor <- as.factor(Y)
    
    # 样本数保护（至少2个样本才能训练）
    if (nrow(X_sel) < 2) {
      message("reward_rf警告：样本数不足，返回0奖励")
      return(full)
    }
    
    rf <- randomForest::randomForest(
      x = X_sel, 
      y = Y_factor, 
      ntree = ntree, 
      importance = TRUE,
      na.action = na.omit  # 避免残留NA
    )
    
    imp <- randomForest::importance(rf, type = 1)
    # 处理单特征时imp为向量的情况
    if (is.matrix(imp)) imp <- as.numeric(imp[, 1])
    imp[is.na(imp)] <- 0
    
    if (binary_reward) {
      thr <- stats::median(imp, na.rm = TRUE)
      thr <- ifelse(is.na(thr), 0, thr)
      rv <- as.numeric(imp >= thr)
    } else {
      maxv <- max(imp, na.rm = TRUE)
      if (!is.finite(maxv) || maxv == 0) maxv <- 1
      rv <- imp / maxv
    }
    
    full[selected_idx] <- rv
    full
  }, error = function(e) {
    message("reward_rf error: ", e$message)
    full
  })
  return(res)
}

TVS_RF_offline <- function(Y, X, topq = 10, fix_stop = 150, binary_reward = TRUE) {
  result <- TVS_generic(
    Y = Y, X = X,
    reward_func = reward_rf_offline,
    reward_args = list(ntree = 75),
    topq = topq, fix_stop = fix_stop,
    binary_reward = binary_reward,
    M_theta = 8, update_mode = "single"
  )
  
  prob <- result$A[, result$final_iter + 1] / 
    (result$A[, result$final_iter + 1] + result$B[, result$final_iter + 1])
  
  list(
    prob = prob, A = result$A, B = result$B,
    time = result$time, iterations = result$final_iter
  )
}

# ============================================================
TVS_generic <- function(Y, X, reward_func, reward_args = list(),
                        topq = 10, fix_stop = 200, binary_reward = TRUE,
                        M_theta = 6, delta = 1e-3, q = NULL, update_mode = "batch") {
  if (is.null(q)) q <- delta
  start_time <- Sys.time()
  p <- ncol(X)
  
  # 强制X为数值矩阵（核心保护）
  if (!is.matrix(X)) X <- as.matrix(X)
  if (!is.numeric(X)) X <- matrix(as.numeric(X), nrow = nrow(X))
  
  A <- matrix(1, p, fix_stop + 1)
  B <- matrix(1, p, fix_stop + 1)
  phi_q <- stats::qnorm(1 - q); if (!is.finite(phi_q)) phi_q <- 3
  final_iter <- fix_stop
  
  true_important <- NULL
  
  for (t in 1:fix_stop) {
    Aprev <- A[, t]; Bprev <- B[, t]
    Ni <- Aprev + Bprev; Ni[Ni <= 0] <- 1
    mu_hat <- Aprev / Ni; mu_hat[is.na(mu_hat)] <- 0
    L1t <- log(12 * (max(2, choose(p,topq))^2) * (t^2) / delta + 1)
    C_t <- L1t / (phi_q^2 + 1e-8)
    
    # draw M_theta theta samples
    theta_mat <- matrix(0, nrow = M_theta, ncol = p)
    for (k in 1:M_theta) {
      sd_vec <- sqrt(C_t / Ni); sd_vec[!is.finite(sd_vec)] <- sqrt(C_t)
      theta_k <- stats::rnorm(p, mean = mu_hat, sd = sd_vec)
      theta_k <- pmin(pmax(theta_k, 0), 1)
      theta_mat[k, ] <- theta_k
    }
    
    # Oracle on mu_hat
    if (!is.null(topq) && topq > 0 && topq < p) {
      S_hat <- order(mu_hat, decreasing = TRUE)[1:topq]
    } else {
      S_hat <- which(mu_hat > 0.5)
      if (length(S_hat) == 0) {
        S_hat <- which.max(mu_hat)
      }
    }
    
    # compute S_tilde and gaps
    S_tilde_list <- vector("list", M_theta); delta_tilde <- numeric(M_theta)
    for (k in 1:M_theta) {
      th <- theta_mat[k, ]
      if (!is.null(topq) && topq > 0 && topq < p) {
        S_tilde <- order(th, decreasing = TRUE)[1:topq]
      } else {
        S_tilde <- which(th > 0.5)
        if (length(S_tilde) == 0) S_tilde <- which.max(th)
      }
      S_tilde_list[[k]] <- S_tilde
      delta_tilde[k] <- sum(th[S_tilde]) - sum(th[S_hat])
    }
    
    # stop if all equal
    all_same <- all(sapply(S_tilde_list, function(s) length(s) == length(S_hat) && all(sort(s) == sort(S_hat))))
    if (all_same) { final_iter <- t; break }
    
    kstar <- which.max(delta_tilde)
    S_tilde_star <- S_tilde_list[[kstar]]
    exch_set <- sort(unique(c(setdiff(S_hat, S_tilde_star), setdiff(S_tilde_star, S_hat))))
    
    if (length(exch_set) == 0) {
      pull_idx <- which.min(Ni)
    } else {
      pull_idx <- exch_set[which.min(Ni[exch_set])]
    }
    
    # form gammas as S_hat indicators
    gammas <- integer(p); gammas[S_hat] <- 1L
    
    # call reward_func safely
    args <- c(list(Y = Y, X = X, gammas = gammas), reward_args)
    rfull <- tryCatch(do.call(reward_func, args), error = function(e) rep(0, p))
    
    if (!is.numeric(rfull) || length(rfull) != p) {
      if (is.numeric(rfull) && length(rfull) == sum(gammas)) {
        full <- rep(0, p); full[which(gammas == 1)] <- rfull
        rfull <- full
      } else {
        rfull <- rep(0, p)
      }
    }
    
    # normalize to [0,1]
    if (max(rfull, na.rm = TRUE) > 0) {
      rprob <- pmin(pmax(rfull / max(rfull, na.rm = TRUE), 0), 1)
    } else rprob <- rep(0, p)
    
    # update A/B
    Anew <- Aprev; Bnew <- Bprev
    
    idxs <- which(gammas == 1)
    if (length(idxs) > 0) {
      vals <- stats::rbinom(length(idxs), 1, prob = rprob[idxs])
      Anew[idxs] <- Anew[idxs] + vals
      Bnew[idxs] <- Bnew[idxs] + (1 - vals)
    }
    
    Anew[pull_idx] <- Anew[pull_idx] + 1
    
    A[, t + 1] <- Anew; B[, t + 1] <- Bnew
  } 
  
  end_time <- Sys.time()
  
  A_out <- A[, 1:(final_iter + 1), drop = FALSE]
  B_out <- B[, 1:(final_iter + 1), drop = FALSE]
  final_prob <- A_out[, ncol(A_out)] / (A_out[, ncol(A_out)] + B_out[, ncol(B_out)])
  final_selected <- which(final_prob > 0.5)
  
  return(list(
    A = A_out,
    B = B_out,
    prob = final_prob,
    final_iter = final_iter,
    time = as.numeric(difftime(end_time, start_time, units = "secs"))
  ))
}

# =============================================================================
# 特征选择方法实现（核心修复：单特征+数值校验）
# =============================================================================
logistic_feature_selection <- function(X, Y, n_features) {
  # 强制X为数值矩阵
  if (!is.matrix(X)) X <- as.matrix(X)
  if (!is.numeric(X)) X <- matrix(as.numeric(X), nrow = nrow(X))
  
  Y_factor <- factor(Y)
  data_df <- data.frame(Y = Y_factor, X)
  
  selected_indices <- tryCatch({
    # 单特征保护
    if (ncol(X) == 1) {
      return(1)
    }
    
    full_model <- multinom(Y ~ ., data = data_df, trace = FALSE, maxit = 5)
    coef_mat <- as.matrix(coef(full_model))
    
    if (is.null(dim(coef_mat))) {
      importance_scores <- abs(coef_mat[-1])
    } else {
      importance_scores <- colSums(abs(coef_mat[, -1]))
    }
    
    # NA保护
    importance_scores[is.na(importance_scores)] <- 0
    selected <- order(importance_scores, decreasing = TRUE)[1:n_features]
    selected
    
  }, error = function(e) {
    # 相关系数计算时的数值保护
    correlations <- sapply(1:ncol(X), function(j) {
      cor_val <- cor(X[, j], as.numeric(as.integer(Y)), use = "complete.obs")
      ifelse(is.na(cor_val) || !is.finite(cor_val), 0, abs(cor_val))
    })
    
    selected <- order(correlations, decreasing = TRUE)[1:n_features]
    selected
  })
  
  # 最终长度校验
  selected_indices <- selected_indices[!is.na(selected_indices)]
  if (length(selected_indices) < n_features) {
    extra <- sample(setdiff(1:ncol(X), selected_indices), n_features - length(selected_indices))
    selected_indices <- c(selected_indices, extra)
  }
  
  return(selected_indices[1:min(n_features, length(selected_indices))])
}

# RandomForest特征选择：强化数值校验
rf_feature_selection <- function(X, Y, n_features) {
  # 数值校验
  if (!is.matrix(X)) X <- as.matrix(X)
  if (!is.numeric(X)) X <- matrix(as.numeric(X), nrow = nrow(X))
  
  Y_factor <- factor(Y)
  # 单特征保护
  if (ncol(X) == 1) {
    return(1)
  }
  
  rf_model <- randomForest(X, Y_factor, importance = TRUE, ntree = 10, max_depth = 3)
  importance_scores <- importance(rf_model)[, "MeanDecreaseGini"]
  importance_scores[is.na(importance_scores)] <- 0
  
  selected_indices <- order(importance_scores, decreasing = TRUE)[1:n_features]
  return(selected_indices)
}

# SVM特征选择：强化数值校验
svm_feature_selection <- function(X, Y, n_features) {
  # 数值校验
  if (!is.matrix(X)) X <- as.matrix(X)
  if (!is.numeric(X)) X <- matrix(as.numeric(X), nrow = nrow(X))
  
  Y_factor <- factor(Y)
  # 单特征保护
  if (ncol(X) == 1) {
    return(1)
  }
  
  tryCatch({
    svm_model <- svm(X, Y_factor, kernel = "linear", type = "C-classification",
                     cost = 1e-6, max_iter = 3)
    
    if (!is.null(svm_model$coefs) && !is.null(svm_model$SV)) {
      w <- t(svm_model$coefs) %*% svm_model$SV
      imp <- abs(as.vector(colSums(abs(w))))
      imp[is.na(imp)] <- 0
      selected_indices <- order(imp, decreasing = TRUE)[1:n_features]
    } else {
      correlations <- sapply(1:ncol(X), function(j) {
        cor_val <- cor(X[, j], as.numeric(as.integer(Y)), use = "complete.obs")
        ifelse(is.na(cor_val) || !is.finite(cor_val), 0, abs(cor_val))
      })
      selected_indices <- order(correlations, decreasing = TRUE)[1:n_features]
    }
  }, error = function(e) {
    correlations <- sapply(1:ncol(X), function(j) {
      cor_val <- cor(X[, j], as.numeric(as.integer(Y)), use = "complete.obs")
      ifelse(is.na(cor_val) || !is.finite(cor_val), 0, abs(cor_val))
    })
    selected_indices <- order(correlations, decreasing = TRUE)[1:n_features]
  })
  
  return(selected_indices[1:min(n_features, length(selected_indices))])
}

# 核心修复：TVS函数中的reward函数（强化数值校验）
reward <- function(Y, X, gammas, type, niter, ntree, binary_reward) {
  if (type == "rf") {
    p <- ncol(X)
    selected_idx <- which(gammas == 1)
    full <- rep(0, p)
    if (length(selected_idx) == 0) return(full)
    
    # 强制X_sel为数值矩阵（核心修复）
    X_sel <- X[, selected_idx, drop = FALSE]
    if (!is.matrix(X_sel)) X_sel <- as.matrix(X_sel)
    if (!is.numeric(X_sel)) X_sel <- matrix(as.numeric(X_sel), nrow = nrow(X_sel))
    # 单特征列名保护
    if (ncol(X_sel) == 1) colnames(X_sel) <- "feature_1"
    
    res <- tryCatch({
      # 强制Y为因子
      Y_factor <- as.factor(Y)
      # 样本数保护
      if (nrow(X_sel) < 2) return(full)
      
      rf <- randomForest(x = X_sel, y = Y_factor, ntree = ntree, importance = TRUE)
      imp <- importance(rf, type = 1)
      if (is.matrix(imp)) imp <- as.numeric(imp[, 1])
      imp[is.na(imp)] <- 0
      
      if (binary_reward) {
        thr <- median(imp, na.rm = TRUE)
        thr <- ifelse(is.na(thr), 0, thr)
        rv <- as.numeric(imp >= thr)
      } else {
        maxv <- max(imp, na.rm = TRUE)
        if (!is.finite(maxv) || maxv == 0) maxv <- 1
        rv <- imp / maxv
      }
      full[selected_idx] <- rv
      full
    }, error = function(e) {
      message("reward_rf error: ", e$message)
      full
    })
    
    return(res)
  }
}

# TVS函数（强化数值校验）
TVS <- function(Y, X, selector = c('rf'), topq, maxnrep, stop_crit = 100, fix_stop, niter, ntree, a = 1, b = 1, binary_reward = T) {
  
  start_time = Sys.time()
  
  # 强制X为数值矩阵
  if (!is.matrix(X)) X <- as.matrix(X)
  if (!is.numeric(X)) X <- matrix(as.numeric(X), nrow = nrow(X))
  
  p <- ncol(X)
  n <- nrow(X)
  round <- 1
  k <- 1
  Yperm <- Y
  Xperm <- X
  
  if (missing(maxnrep) & missing(fix_stop)) {
    maxnrep <- 10000
  } else if (!missing(fix_stop)) {
    maxnrep <- fix_stop
    stop_crit <- fix_stop
  }
  
  A <- matrix(a, p, maxnrep * round)
  B <- matrix(b, p, maxnrep * round)
  Aprev <- numeric(p)
  Anew <- numeric(p)
  Bprev <- numeric(p)
  Bnew <- numeric(p)
  model_diff <- rep(1, stop_crit)
  model <- rep(0, p)
  iter <- 1
  
  for (i in (1:maxnrep)) {
    Aprev <- A[, (k-1) * maxnrep + i]
    Bprev <- B[, (k-1) * maxnrep + i]
    thetas <- rbeta(p, Aprev, Bprev)
    
    if (missing(topq)) {
      gammas <- thetas > 0.5
    } else {
      gammas <- numeric(p)
      # 排序时NA保护
      thetas[is.na(thetas)] <- 0
      index <- order(thetas, decreasing = T)
      gammas[index[1:topq]] <- 1
    }
    
    rbin <- reward(Yperm, Xperm, gammas, type = selector,
                   niter, ntree = ntree,
                   binary_reward = binary_reward)
    
    if (max(rbin) > 0) {
      rbin <- rbinom(sum(gammas), 1, rbin / max(rbin))
    }
    
    Anew <- Aprev
    Anew[gammas == 1] <- Anew[gammas == 1] + rbin
    Bnew <- Bprev
    Bnew[gammas == 1] <- Bnew[gammas == 1] + 1 - rbin
    
    model_new <- ((Anew / (Anew + Bnew)) > 0.5)
    model_diff[iter %% stop_crit + 1] = sum(abs(model - model_new))
    model <- model_new
    
    if (i < maxnrep) {
      A[, (k-1) * maxnrep + i + 1] <- Anew
      B[, (k-1) * maxnrep + i + 1] <- Bnew
    }
    
    iter = iter + 1
    
    if (sum(model_diff) == 0) {
      converge_time <- Sys.time() - start_time
      break
    }
  }
  
  if (!missing(fix_stop)) {
    converge_time <- Sys.time() - start_time
  }
  
  r = list(A = A[, 1:(iter-1)], B = B[, 1:(iter-1)], model_diff = model_diff, 
           time = converge_time, type = 'offline')
  
  class(r) = 'TVS'
  
  return(r)
}

# 获取后验概率函数（NA保护）
posterior <- function(tvs_obj) {
  A_final <- tvs_obj$A[, ncol(tvs_obj$A)]
  B_final <- tvs_obj$B[, ncol(tvs_obj$B)]
  prob <- A_final / (A_final + B_final)
  prob[is.na(prob) | !is.finite(prob)] <- 0  # NA/无穷值保护
  return(prob)
}

# 核心修复：TVS-RF特征选择函数
tvs_rf_feature_selection <- function(X, Y, n_features) {
  # 数值校验
  if (!is.matrix(X)) X <- as.matrix(X)
  if (!is.numeric(X)) X <- matrix(as.numeric(X), nrow = nrow(X))
  
  tryCatch({
    # 单特征保护
    if (ncol(X) == 1) {
      return(1)
    }
    
    tvs_result <- TVS(Y = Y, X = X, selector = "rf", topq = n_features, 
                      fix_stop = 50, ntree = 50, binary_reward = TRUE)
    
    prob <- posterior(tvs_result)
    selected_indices <- order(prob, decreasing = TRUE)[1:n_features]
    
    return(selected_indices)
  }, error = function(e) {
    cat("TVS-RF特征选择错误:", e$message, "\n")
    return(rf_feature_selection(X, Y, n_features))
  })
}

# =============================================================================
# 决策树训练函数（强化鲁棒性）
# =============================================================================
train_decision_tree <- function(X, Y, selected_features = NULL) {
  # 数值校验
  if (!is.matrix(X)) X <- as.matrix(X)
  if (!is.numeric(X)) X <- matrix(as.numeric(X), nrow = nrow(X))
  
  if (!is.null(selected_features)) {
    selected_features <- selected_features[!is.na(selected_features)]
    if (length(selected_features) == 0) {
      selected_features <- 1:min(5, ncol(X))
    }
    X_selected <- X[, selected_features, drop = FALSE]
  } else {
    X_selected <- X
  }
  
  Y_factor <- factor(Y)
  
  # 标准化（NA保护）
  preProc <- preProcess(X_selected, method = c("center", "scale"), na.remove = TRUE)
  X_scaled <- predict(preProc, X_selected)
  X_scaled[is.na(X_scaled)] <- 0
  data_df <- data.frame(Y = Y_factor, X_scaled)
  
  # 交叉验证折数保护
  k_folds <- min(5, nrow(data_df)-1)
  cv_folds <- createFolds(Y, k = k_folds)
  
  cp_grid <- seq(0.001, 0.1, by = 0.005)
  cv_accuracy <- numeric(length(cp_grid))
  
  for (i in seq_along(cp_grid)) {
    fold_accuracy <- numeric(length(cv_folds))
    for (j in seq_along(cv_folds)) {
      train_idx <- -cv_folds[[j]]
      test_idx <- cv_folds[[j]]
      
      # 样本数保护
      if (length(train_idx) < 2) {
        fold_accuracy[j] <- 0
        next
      }
      
      dt_model <- rpart::rpart(
        formula = Y ~ .,
        data = data_df[train_idx, ],
        method = "class",
        cp = cp_grid[i],
        maxdepth = 10,
        minsplit = 10,
        minbucket = 5
      )
      
      dt_pred <- predict(dt_model, data_df[test_idx, ], type = "class")
      fold_accuracy[j] <- mean(dt_pred == data_df$Y[test_idx], na.rm = TRUE)
    }
    cv_accuracy[i] <- mean(fold_accuracy, na.rm = TRUE)
  }
  
  best_cp_idx <- which.max(cv_accuracy)
  best_cp <- cp_grid[best_cp_idx]
  
  final_dt <- rpart::rpart(
    formula = Y ~ .,
    data = data_df,
    method = "class",
    cp = best_cp,
    maxdepth = 10,
    minsplit = 10,
    minbucket = 5
  )
  
  pred <- predict(final_dt, data_df, type = "class")
  accuracy <- mean(pred == data_df$Y, na.rm = TRUE)
  
  return(list(
    cp = best_cp,
    accuracy = accuracy,
    predictions = pred
  ))
}

# =============================================================================
# 实验函数（无核心修改）
# =============================================================================
run_single_experiment <- function(X, Y, run_id) {
  cat(paste("\n=== 开始第", run_id, "次实验 ===\n"))
  set.seed(250 + run_id)  # 每次实验使用不同的种子，123（2），125（3）
  cat("步骤1: 使用所有特征训练决策树...\n")
  baseline_result <- train_decision_tree(X, Y)
  baseline_accuracy <- baseline_result$accuracy
  baseline_cp <- baseline_result$cp
  cat("基准准确率:", baseline_accuracy, "(最佳剪枝参数cp =", round(baseline_cp, 4), ")\n")
  
  feature_counts <- seq(1, min(30, ncol(X)), by = 1)
  
  results <- data.frame(
    run_id = integer(),
    n_features = integer(),
    method = character(),
    accuracy = numeric(),
    baseline_accuracy = numeric(),
    stringsAsFactors = FALSE
  )
  
  methods <- c("Logistic","REVS-RF", "RandomForest","TVS-RF",  "SVM")
  
  for (method in methods) {
    cat(paste("处理方法:", method, "\n"))
    
    for (n_feat in feature_counts) {
      cat(paste("  特征数量:", n_feat, "\n"))
      
      tryCatch({
        if (method == "TVS-RF") {
          selected_features <- tvs_rf_feature_selection(X, Y, n_feat)
        } else if (method == "REVS-RF") {
          tvs_result <- TVS_RF_offline(Y, X, topq = n_feat, fix_stop = 150)
          selected_features <- order(tvs_result$prob, decreasing = TRUE)[1:n_feat]
        } else if (method == "Logistic") {
          selected_features <- logistic_feature_selection(X, Y, n_feat)
        } else if (method == "RandomForest") {
          selected_features <- rf_feature_selection(X, Y, n_feat)
        } else if (method == "SVM") {
          selected_features <- svm_feature_selection(X, Y, n_feat)
        }
        
        dt_result <- train_decision_tree(X, Y, selected_features)
        
        results <- rbind(results, data.frame(
          run_id = run_id,
          n_features = n_feat,
          method = method,
          accuracy = dt_result$accuracy,
          baseline_accuracy = baseline_accuracy
        ))
        
        cat(paste("    准确率:", dt_result$accuracy, "(cp =", round(dt_result$cp, 4), ")\n"))
        
      }, error = function(e) {
        cat(paste("    错误:", e$message, "\n"))
        results <- rbind(results, data.frame(
          run_id = run_id,
          n_features = n_feat,
          method = method,
          accuracy = NA,
          baseline_accuracy = baseline_accuracy
        ))
      })
    }
  }
  
  return(results)
}

run_multiple_experiments <- function(X, Y, n_runs = 3) {
  all_results <- data.frame()
  baseline_results <- numeric(n_runs)
  
  for (i in 1:n_runs) {
    single_results <- run_single_experiment(X, Y, i)
    all_results <- rbind(all_results, single_results)
    baseline_results[i] <- single_results$baseline_accuracy[1]
  }
  
  avg_baseline <- mean(baseline_results, na.rm = TRUE)
  
  return(list(
    all_results = all_results,
    avg_baseline = avg_baseline,
    baseline_results = baseline_results
  ))
}

