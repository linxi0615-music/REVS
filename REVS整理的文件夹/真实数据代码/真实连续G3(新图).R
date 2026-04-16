# 回归
library(readr)
library(dplyr)
library(rpart)  
library(randomForest)
library(glmnet)
library(BART)
library(SSLASSO)
library(SIS)
library(ggplot2)
library(caret) 


preprocess_student_data <- function(df, target_column = "G3") {
  # 处理缺失值
  numeric_cols <- sapply(df, is.numeric)
  df[numeric_cols] <- lapply(df[numeric_cols], function(x) {
    x[is.na(x)] <- median(x, na.rm = TRUE)
    return(x)
  })
  
  # 检测并编码类别变量
  categorical_cols <- names(df)[!sapply(df, is.numeric)]
  categorical_cols <- setdiff(categorical_cols, target_column)
  
  cat("检测到", length(categorical_cols), "个类别变量需要编码:", 
      paste(categorical_cols, collapse = ", "), "\n")
  
  for (col in categorical_cols) {
    df[[col]] <- as.numeric(as.factor(df[[col]]))
  }
  
  # 分离特征和目标
  if (!(target_column %in% names(df))) {
    stop(paste("Target column", target_column, "not found in dataset"))
  }
  
  Y <- df[[target_column]]
  X <- as.matrix(df[, setdiff(names(df), target_column)])
  feature_names <- setdiff(names(df), target_column)
  
  # 标准化特征
  X <- scale(X)
  
  return(list(
    X = X,
    Y = Y,
    feature_names = feature_names
  ))
}

reward_rf <-function(Y, X, gammas, ntree = 50, binary_reward = TRUE) {
  p <- ncol(X)
  selected_idx <- which(gammas == 1)
  full <- rep(0, p)
  if (length(selected_idx) == 0) return(full)
  X_sel <- X[, selected_idx, drop = FALSE]
  res <- tryCatch({
    rf <- randomForest::randomForest(x = X_sel, y = Y, ntree = ntree, importance = TRUE)
    imp <- randomForest::importance(rf, type = 1)
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

reward_bart <- function(Y, X, gammas, ntree = 100, binary_reward = TRUE) {
  p <- ncol(X)
  selected_idx <- which(gammas == 1)
  full <- rep(0, p)
  if (length(selected_idx) == 0) return(full)
  X_sel <- X[, selected_idx, drop = FALSE]
  
  res <- tryCatch({
    out <- capture.output(
      bart_model <- BART::wbart(x.train = X_sel, y.train = Y, 
                                ndpost = 100, ntree = ntree, 
                                sparse = TRUE, printevery = 10000)
    )
    imp <- bart_model$varcount.mean
    if (is.null(imp)) imp <- colMeans(bart_model$varcount)
    imp[is.na(imp)] <- 0
    if (binary_reward) {
      thr <- stats::median(imp, na.rm = TRUE)
      rv <- as.numeric(imp >= thr)
    } else {
      maxv <- max(imp, na.rm = TRUE)
      if (!is.finite(maxv) || maxv == 0) maxv <- 1
      rv <- imp / maxv
    }
    full[selected_idx] <- rv
    full
  }, error = function(e) {
    message("reward_bart error: ", e$message)
    full
  })
  return(res)
}

# =============================================================================
# REVS通用框架（保持不变）
# =============================================================================
REVS_generic <- function(Y, X, reward_func, reward_args = list(),
                         topq = 10, fix_stop = 200, binary_reward = TRUE,
                         M_theta = 6, delta = 1e-3, q = NULL)  {
  if (is.null(q)) q <- delta
  start_time <- Sys.time()
  p <- ncol(X)
  A <- matrix(1, p, fix_stop + 1)
  B <- matrix(1, p, fix_stop + 1)
  phi_q <- stats::qnorm(1 - q); if (!is.finite(phi_q)) phi_q <- 3
  final_iter <- fix_stop
  
  for (t in 1:fix_stop) {
    Aprev <- A[, t]; Bprev <- B[, t]
    Ni <- Aprev + Bprev; Ni[Ni <= 0] <- 1
    mu_hat <- Aprev / Ni; mu_hat[is.na(mu_hat)] <- 0
    L1t <- log(12 * (max(2, choose(p,topq))^2) * (t^2) / delta + 1)
    C_t <- L1t / (phi_q^2 + 1e-8)
    
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
    
    if (max(rfull, na.rm = TRUE) > 0) {
      rprob <- pmin(pmax(rfull / max(rfull, na.rm = TRUE), 0), 1)
    } else rprob <- rep(0, p)
    
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
reward2 <- function(Y, X, gammas, type, niter, ntree, binary_reward) { 
  selected_indices <- which(gammas == 1) 
  if (length(selected_indices) == 0) { 
    return(rep(0, length(gammas))) 
  } 
  X_selected <- X[, selected_indices, drop = FALSE] 
  
  if (type == "bart") { 
    out <- capture.output( 
      result <- BART::wbart( 
        X_selected, 
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
    full_reward <- rep(0, length(gammas)) 
    full_reward[selected_indices] <- reward 
    return(full_reward) 
    
  } else if (type == "rf") {
    # 修复RF部分
    rf_model <- randomForest::randomForest(X_selected, Y, ntree = ntree, importance = TRUE)
    
    # 获取重要性分数（%IncMSE）
    importance_matrix <- randomForest::importance(rf_model, type = 1)
    
    # 确保是数值向量
    if (is.matrix(importance_matrix) && ncol(importance_matrix) >= 1) {
      importance_values <- as.numeric(importance_matrix[, 1])
    } else {
      importance_values <- as.numeric(importance_matrix)
    }
    
    # 安全处理NA值
    importance_values[is.na(importance_values)] <- 0
    
    if (binary_reward) {
      # 使用中位数作为阈值
      threshold <- ifelse(length(importance_values) > 0, 
                          median(importance_values, na.rm = TRUE), 
                          0)
      reward <- as.numeric(importance_values > threshold)
    } else {
      # 归一化到[0,1]
      max_val <- max(importance_values, na.rm = TRUE)
      if (max_val > 0) {
        reward <- importance_values / max_val
      } else {
        reward <- rep(0, length(importance_values))
      }
    }
    
    full_reward <- rep(0, length(gammas)) 
    full_reward[selected_indices] <- reward 
    return(full_reward)
  }
}


reward <- function(Y, X, gammas, type, niter, ntree, binary_reward, reward_func) { 
  return(reward_func(Y, X, gammas, type, niter, ntree, binary_reward)) 
} 

TVS <- function(Y, X, selector = c('bart', 'rf'), reward_func, topq, maxnrep, 
                stop_crit = 100, fix_stop, niter, ntree, a = 1, b = 1, 
                binary_reward = TRUE) { 
  start_time <- Sys.time() 
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
  model_diff <- rep(1, stop_crit) 
  model <- rep(0, p) 
  iter <- 1 
  converge_time <- NA 
  
  for (i in (1:maxnrep)) { 
    # Pick a subset 
    Aprev <- A[, (k-1)*maxnrep + i] 
    Bprev <- B[, (k-1)*maxnrep + i] 
    thetas <- rbeta(p, Aprev, Bprev) 
    
    if (missing(topq)) { 
      gammas <- as.numeric(thetas > 0.5) 
    } else { 
      gammas <- numeric(p) 
      index <- order(thetas, decreasing = TRUE) 
      gammas[index[1:min(topq, p)]] <- 1 
    } 
    
    # Collect reward 
    rbin_full <- reward_func(Yperm, Xperm, gammas, type = selector, 
                             niter, ntree = ntree, binary_reward = binary_reward) 
    
    # 确保rbin_full是数值向量
    if (is.list(rbin_full)) {
      rbin_full <- unlist(rbin_full)
    }
    if (is.matrix(rbin_full)) {
      rbin_full <- as.numeric(rbin_full)
    }
    
    # 确保长度正确
    if (length(rbin_full) != length(gammas)) {
      rbin_full <- rep(0, length(gammas))
    }
    
    # 标准化奖励概率 
    max_rbin <- max(rbin_full, na.rm = TRUE)
    if (max_rbin > 0 && !is.infinite(max_rbin)) {
      rbin_prob <- rbin_full / max_rbin
      rbin_prob[is.na(rbin_prob)] <- 0
      rbin_prob <- pmin(pmax(rbin_prob, 0), 1)
    } else {
      rbin_prob <- rep(0, length(gammas))
    }
    
    # 对选中的特征进行二项抽样 
    if (sum(gammas) > 0) {
      rbin_sampled <- rbinom(sum(gammas), 1, rbin_prob[gammas == 1])
    } else {
      rbin_sampled <- numeric(0)
    }
    
    # Update 
    Anew <- Aprev
    Bnew <- Bprev
    
    # 安全更新参数 
    if (sum(gammas) > 0 && length(rbin_sampled) == sum(gammas)) {
      Anew[gammas == 1] <- Anew[gammas == 1] + rbin_sampled 
      Bnew[gammas == 1] <- Bnew[gammas == 1] + 1 - rbin_sampled 
    } 
    
    # 计算模型差异 
    model_new <- as.numeric((Anew/(Anew + Bnew)) > 0.5) 
    model_new[is.na(model_new)] <- 0 
    current_index <- ((iter - 1) %% stop_crit) + 1 
    model_diff[current_index] <- sum(abs(model - model_new), na.rm = TRUE) 
    model <- model_new 
    
    if (i < maxnrep) { 
      A[, (k-1)*maxnrep + i + 1] <- Anew 
      B[, (k-1)*maxnrep + i + 1] <- Bnew 
    } 
    
    iter <- iter + 1 
    
    # Check convergence 
    if (iter > 10 && sum(model_diff, na.rm = TRUE) == 0) { 
      converge_time <- Sys.time() - start_time 
      break 
    } 
  } 
  
  if (is.na(converge_time)) { 
    converge_time <- Sys.time() - start_time 
  } 
  
  if (!missing(fix_stop)) { 
    converge_time <- Sys.time() - start_time 
  } 
  
  # 安全截取结果 
  final_iter <- min(iter - 1, ncol(A))
  if (final_iter < 1) final_iter <- 1
  
  r <- list(A = A[, 1:final_iter, drop = FALSE], 
            B = B[, 1:final_iter, drop = FALSE], 
            model_diff = model_diff, 
            time = converge_time, 
            type = 'offline') 
  class(r) <- 'TVS' 
  return(r) 
} 
posterior_iter <- function(result, iter) { 
  prob <- result$A[, iter] / (result$A[, iter] + result$B[, iter]) 
  prob[is.na(prob)] <- 0 
  return(prob) 
} 
posterior <- function(result, type = c('converge', 'fixed')) { 
  if (result$type == 'online') { 
    if (type == 'converge') { 
      iter <- result$iter_converge 
    } else { 
      iter <- result$iter_fixed 
    } 
  } else { 
    iter <- ncol(result$A) 
  } 
  return(posterior_iter(result, iter)) 
}

# =============================================================================
# 特征选择方法（保持不变）
# =============================================================================
revs_rf_feature_selection <- function(X, Y, n_features) {
  tryCatch({
    result <- REVS_generic(
      Y = Y, X = X,
      reward_func = reward_rf,
      reward_args = list(ntree = 50, binary_reward = TRUE),
      topq = n_features,
      fix_stop = 300,
      M_theta = 6
    )
    selected <- order(result$prob, decreasing = TRUE)[1:n_features]
    return(selected)
  }, error = function(e) {
    message("REVS-RF error: ", e$message)
    correlations <- abs(cor(X, Y))
    return(order(correlations, decreasing = TRUE)[1:n_features])
  })
}

revs_bart_feature_selection <- function(X, Y, n_features) {
  tryCatch({
    result <- REVS_generic(
      Y = Y, X = X,
      reward_func = reward_bart,
      reward_args = list(ntree = 20, binary_reward = TRUE),
      topq = n_features,
      fix_stop = 100,
      M_theta = 6
    )
    selected <- order(result$prob, decreasing = TRUE)[1:n_features]
    return(selected)
  }, error = function(e) {
    message("REVS-BART error: ", e$message)
    correlations <- abs(cor(X, Y))
    return(order(correlations, decreasing = TRUE)[1:n_features])
  })
}

tvs_rf_feature_selection <- function(X, Y, n_features) {
  tryCatch({
    result <- TVS(Y = Y, X = X, selector = "rf", reward_func = reward2,
                  topq = n_features, fix_stop = 50, niter = 100, ntree = 20, 
                  binary_reward = TRUE)
    prob <- posterior(result)
    selected <- order(prob, decreasing = TRUE)[1:n_features]
    return(selected)
  }, error = function(e) {
    message("TVS-RF error: ", e$message)
    correlations <- abs(cor(X, Y))
    return(order(correlations, decreasing = TRUE)[1:n_features])
  })
}

tvs_bart_feature_selection <- function(X, Y, n_features) {
  tryCatch({
    result <- TVS(Y = Y, X = X, selector = "bart", reward_func = reward2,
                  topq = n_features, fix_stop = 50, niter = 100, ntree = 20, 
                  binary_reward = TRUE)
    prob <- posterior(result)
    selected <- order(prob, decreasing = TRUE)[1:n_features]
    return(selected)
  }, error = function(e) {
    message("TVS-BART error: ", e$message)
    correlations <- abs(cor(X, Y))
    return(order(correlations, decreasing = TRUE)[1:n_features])
  })
}

bart_feature_selection <- function(X, Y, n_features) {
  tryCatch({
    fit <- BART::wbart(x.train = X, y.train = Y, ndpost = 100, ntree = 20, printevery = 10000)
    importance <- fit$varcount.mean
    importance[is.na(importance)] <- 0
    return(order(importance, decreasing = TRUE)[1:n_features])
  }, error = function(e) {
    message("BART error: ", e$message)
    correlations <- abs(cor(X, Y))
    return(order(correlations, decreasing = TRUE)[1:n_features])
  })
}

# DART特征选择
dart_feature_selection <- function(X, Y, n_features) {
  tryCatch({
    fit <- BART::wbart(x.train = X, y.train = Y, ndpost = 100, ntree = 20, 
                       sparse = TRUE, printevery = 10000)
    if (!is.null(fit$varcount.mean)) {
      importance <- fit$varcount.mean
    } else if (!is.null(fit$varcount)) {
      importance <- colMeans(fit$varcount)
    } else {
      stop("无法提取变量重要性")
    }
    importance[is.na(importance)] <- 0
    return(order(importance, decreasing = TRUE)[1:n_features])
  }, error = function(e) {
    message("DART error: ", e$message)
    correlations <- abs(cor(X, Y))
    return(order(correlations, decreasing = TRUE)[1:n_features])
  })
}

lasso_feature_selection <- function(X, Y, n_features) {
  tryCatch({
    fit <- glmnet::cv.glmnet(X, Y, alpha = 1, family = "gaussian", lambda = seq(1, 10, 0.5))
    coef_vals <- as.numeric(coef(fit, s = "lambda.max"))[-1]
    importance <- abs(coef_vals)
    return(order(importance, decreasing = TRUE)[1:n_features])
  }, error = function(e) {
    correlations <- abs(cor(X, Y))
    return(order(correlations, decreasing = TRUE)[1:n_features])
  })
}

sslasso_feature_selection <- function(X, Y, n_features) {
  tryCatch({
    fit <- SSLASSO::SSLASSO(X, Y, lambda0 = seq(0.1, 100, length = 10), lambda1 = 10)
    importance <- abs(fit$beta[, ncol(fit$beta)])
    return(order(importance, decreasing = TRUE)[1:n_features])
  }, error = function(e) {
    message("SSLASSO error: ", e$message)
    correlations <- abs(cor(X, Y))
    return(order(correlations, decreasing = TRUE)[1:n_features])
  })
}

sis_feature_selection <- function(X, Y, n_features) {
  tryCatch({
    sis_result <- SIS::SIS(X, Y, family = "gaussian", tune = "bic")
    selected <- sis_result$ix
    
    if (length(selected) >= n_features) {
      selected <- sample(selected, n_features)
      return(selected)
    } else {
      remaining <- n_features - length(selected)
      correlations <- abs(cor(X, Y))
      available <- setdiff(1:ncol(X), selected)
      additional <- available[order(correlations[available], decreasing = FALSE)[1:remaining]]
      selected <- c(selected, additional)
      return(selected)
    }
  }, error = function(e) {
    message("SIS error: ", e$message)
    correlations <- abs(cor(X, Y))
    return(order(correlations, decreasing = TRUE)[1:n_features])
  })
}

# =============================================================================
# 决策树训练函数（保持不变）
# =============================================================================
# =============================================================================
# 决策树训练函数（修改为计算调整R方）
# =============================================================================
train_decision_tree_regression <- function(X, Y, selected_features = NULL) {
  if (!is.null(selected_features)) {
    X_selected <- X[, selected_features, drop = FALSE]
    p <- length(selected_features)  # 使用的特征数量
  } else {
    X_selected <- X
    p <- ncol(X)  # 所有特征数量
  }
  
  n <- nrow(X_selected)  # 样本数量
  
  preProc <- preProcess(X_selected, method = c("center", "scale"))
  X_scaled <- predict(preProc, X_selected)
  data_df <- data.frame(Y = Y, X_scaled)
  
  set.seed(Sys.time())
  cv_folds <- createFolds(Y, k = 5)
  
  cp_grid <- seq(0.001, 0.1, by = 0.005)
  cv_rmse <- numeric(length(cp_grid))
  
  for (i in seq_along(cp_grid)) {
    fold_rmse <- numeric(length(cv_folds))
    for (j in seq_along(cv_folds)) {
      train_idx <- -cv_folds[[j]]
      test_idx <- cv_folds[[j]]
      
      dt_model <- rpart::rpart(
        formula = Y ~ .,
        data = data_df[train_idx, ],
        method = "anova",
        cp = cp_grid[i],
        maxdepth = 10,
        minsplit = 10,
        minbucket = 5
      )
      
      dt_pred <- predict(dt_model, data_df[test_idx, ])
      fold_rmse[j] <- sqrt(mean((Y[test_idx] - dt_pred)^2))
    }
    cv_rmse[i] <- mean(fold_rmse)
  }
  
  best_cp_idx <- which.min(cv_rmse)
  best_cp <- cp_grid[best_cp_idx]
  
  final_dt <- rpart::rpart(
    formula = Y ~ .,
    data = data_df,
    method = "anova",
    cp = best_cp,
    maxdepth = 10,
    minsplit = 10,
    minbucket = 5
  )
  
  # 计算原始R²
  pred <- predict(final_dt, data_df)
  ss_res <- sum((Y - pred)^2)
  ss_tot <- sum((Y - mean(Y))^2)
  r_squared <- 1 - (ss_res / ss_tot)
  
  # ============================================
  # 新增：计算调整R方
  # ============================================
  adjusted_r_squared <- 1 - ((1 - r_squared) * (n - 1) / (n - p - 1))
  
  # 防止负值（当模型比均值预测还差时）
  adjusted_r_squared <- max(0, adjusted_r_squared)
  
  return(list(
    cp = best_cp,
    r_squared = r_squared,            # 原始R方
    adjusted_r_squared = adjusted_r_squared,  # 新增：调整R方
    predictions = pred,
    n_samples = n,                    # 新增：样本数
    n_features = p                    # 新增：特征数
  ))
}
# =============================================================================
# 主实验函数（修复run_multiple_experiments）
# =============================================================================
# =============================================================================
# 主实验函数（修改为记录调整R方）
# =============================================================================
run_single_experiment <- function(X, Y) {
  cat("\n========== 开始特征选择实验 ==========\n\n")
  
  cat("步骤1: 使用所有特征训练决策树作为基准...\n")
  baseline_result <- train_decision_tree_regression(X, Y)
  baseline_r2 <- baseline_result$r_squared
  baseline_adj_r2 <- baseline_result$adjusted_r_squared  # 新增
  cat("基准R² (所有", ncol(X), "个特征):", round(baseline_r2, 4), 
      " | 调整R²:", round(baseline_adj_r2, 4), "\n\n")
  
  feature_counts <- seq(1, min(30, ncol(X)), by = 1)
  
  # 修改结果数据框，增加调整R方列
  results <- data.frame(
    n_features = integer(),
    method = character(),
    r_squared = numeric(),
    adjusted_r_squared = numeric(),  # 新增列
    stringsAsFactors = FALSE
  )
  
  methods <- c("REVS-RF", "TVS-RF", "TVS-BART","REVS-BART", "BART", "DART", "Lasso", "SSLASSO", "SIS")
  
  for (method in methods) {
    cat(paste("\n===== 处理方法:", method, "=====\n"))
    
    for (n_feat in feature_counts) {
      cat(paste("  特征数量:", sprintf("%2d", n_feat), "..."))
      
      tryCatch({
        if (method == "REVS-RF") {
          selected_features <- revs_rf_feature_selection(X, Y, n_feat)
        } else if (method == "REVS-BART") {
          selected_features <- revs_bart_feature_selection(X, Y, n_feat)
        } else if (method == "TVS-RF") {
          selected_features <- tvs_rf_feature_selection(X, Y, n_feat)
        } else if (method == "TVS-BART") {
          selected_features <- tvs_bart_feature_selection(X, Y, n_feat)
        } else if (method == "Lasso") {
          selected_features <- lasso_feature_selection(X, Y, n_feat)
        } else if (method == "SSLASSO") {
          selected_features <- sslasso_feature_selection(X, Y, n_feat)
        } else if (method == "SIS") {
          selected_features <- sis_feature_selection(X, Y, n_feat)
        } else if (method == "BART") {
          selected_features <- bart_feature_selection(X, Y, n_feat)
        } else if (method == "DART") {
          selected_features <- dart_feature_selection(X, Y, n_feat)
        }
        
        tree_result <- train_decision_tree_regression(X, Y, selected_features)
        
        results <- rbind(results, data.frame(
          n_features = n_feat,
          method = method,
          r_squared = tree_result$r_squared,
          adjusted_r_squared = tree_result$adjusted_r_squared  # 新增
        ))
        
        cat(" R² =", sprintf("%.4f", tree_result$r_squared),
            " | 调整R² =", sprintf("%.4f", tree_result$adjusted_r_squared), "\n")
        
      }, error = function(e) {
        cat(" 错误:", e$message, "\n")
        results <- rbind(results, data.frame(
          n_features = n_feat,
          method = method,
          r_squared = NA,
          adjusted_r_squared = NA  # 新增
        ))
      })
    }
  }
  
  return(list(
    results = results, 
    baseline_r2 = baseline_r2,
    baseline_adj_r2 = baseline_adj_r2  # 新增
  ))
}
run_multiple_experiments <- function(X, Y, n_runs = 3) {
  all_results <- data.frame()  # 空数据框存储所有实验结果
  baseline_results <- numeric(n_runs)  # 存储每次实验的基准R²
  
  for (i in 1:n_runs) {
    cat(paste("\n================ 第", i, "次实验 ================\n"))
    single_results <- run_single_experiment(X, Y)
    # 关键修复：拼接single_results$results（数据框），而非single_results（列表）
    all_results <- rbind(all_results, single_results$results)  
    baseline_results[i] <- single_results$baseline_r2
  }
  
  avg_baseline <- mean(baseline_results, na.rm = TRUE)
  
  return(list(
    all_results = all_results,
    avg_baseline = avg_baseline,
    baseline_results = baseline_results
  ))
}

