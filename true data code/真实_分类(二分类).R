#分类（二分类）
#特征数量: 52 
#样本数量: 913 
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

set.seed(2025)

load_wiki4he_data_classification <- function(file_path) {
  tryCatch({
    data <- read.csv(file_path, stringsAsFactors = FALSE, na.strings = c("", "NA", "?"))
    cat("数据集形状:", dim(data), "\n")
    
    if (!"PhD" %in% names(data)) {
      stop("数据集中未找到'PhD'列,请检查数据格式")
    }
    
    cat("PhD类别分布:\n")
    print(table(data$PhD, useNA = "always"))
    
    # 先处理PhD的NA值
    data$PhD <- ifelse(is.na(data$PhD), 
                          names(sort(table(data$PhD, useNA = "no"), decreasing = TRUE)[1]),
                          data$PhD)
    data <- data %>%
      dplyr::mutate(PhD_encoded = as.numeric(factor(PhD)))
    
    cat("PhD标签编码完成\n")
    
    # 提取特征并强制转为数值型
    X <- data %>%
      dplyr::select(-PhD, -PhD_encoded) %>%
      mutate(across(everything(), ~as.numeric(as.character(.x)))) %>%  # 统一转为数值
      as.matrix()
    
    Y <- data$PhD_encoded
    
    # 填充特征NA（仅数值型）
    if (any(is.na(X))) {
      cat("警告：特征中存在缺失值，用中位数填充\n")
      for (j in 1:ncol(X)) {
        med_val <- median(X[, j], na.rm = TRUE)
        X[is.na(X[, j]), j] <- ifelse(is.finite(med_val), med_val, 0)
      }
    }
    
    # 确保无NA
    X[is.na(X)] <- 0
    Y[is.na(Y)] <- as.numeric(names(sort(table(Y, useNA = "no"), decreasing = TRUE)[1]))
    
    cat("特征数量:", ncol(X), "\n")
    cat("样本数量:", nrow(X), "\n")
    cat("目标变量PhD类别数:", length(unique(Y)), "\n")
    cat("PhD类别分布:\n")
    print(table(Y))
    
    return(list(X = X, Y = Y))
    
  }, error = function(e) {
    cat("加载数据失败:", e$message, "\n")
    stop("无法加载wiki4he数据集")
  })
}

reward_rf_offline <- function(Y, X, gammas, ntree = 50, binary_reward = TRUE) {
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

TVS_generic <- function(Y, X, reward_func, reward_args = list(),
                        topq = 10, fix_stop = 200, binary_reward = TRUE,
                        M_theta = 6, delta = 1e-3, q = NULL, update_mode = "batch") {
  if (is.null(q)) q <- delta
  start_time <- Sys.time()
  p <- ncol(X)
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
    
    theta_mat <- matrix(0, nrow = M_theta, ncol = p)
    for (k in 1:M_theta) {
      sd_vec <- sqrt(C_t / Ni); sd_vec[!is.finite(sd_vec)] <- sqrt(C_t)
      theta_k <- stats::rnorm(p, mean = mu_hat, sd = sd_vec)
      theta_k <- pmin(pmax(theta_k, 0), 1)
      theta_mat[k, ] <- theta_k
    }
    
    if (!is.null(topq) && topq > 0 && topq < p) {
      S_hat <- order(mu_hat, decreasing = TRUE)[1:topq]
    } else {
      S_hat <- which(mu_hat > 0.5)
      if (length(S_hat) == 0) {
        S_hat <- which.max(mu_hat)
      }
    }
    
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
    
    gammas <- integer(p); gammas[S_hat] <- 1L
    
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
# TVS算法实现（无修改）
# =============================================================================
TVS_RF_offline <- function(Y, X, topq = 10, fix_stop = 150, binary_reward = TRUE) {
  result <- TVS_generic(
    Y = Y, X = X,
    reward_func = reward_rf_offline,
    reward_args = list(ntree = 20),
    topq = topq, fix_stop = fix_stop,
    binary_reward = binary_reward,
    M_theta = 6, update_mode = "single"
  )
  
  prob <- result$A[, result$final_iter + 1] / 
    (result$A[, result$final_iter + 1] + result$B[, result$final_iter + 1])
  
  list(
    prob = prob, A = result$A, B = result$B,
    time = result$time, iterations = result$final_iter
  )
}

logistic_feature_selection <- function(X, Y, n_features) {
  # 强制转为因子，确保分类
  Y_factor <- factor(Y)
  # 确保X是数据框且无NA
  X_df <- as.data.frame(X) %>% mutate(across(everything(), ~ifelse(is.na(.x), 0, .x)))
  data_df <- data.frame(Y = Y_factor, X_df)
  
  selected_indices <- tryCatch({
    # 单特征时直接返回该特征，避免multinom报错
    if (ncol(X) == 1) {
      return(1)
    }
    
    # 拟合逻辑回归（增加鲁棒性）
    full_model <- multinom(Y ~ ., data = data_df, trace = FALSE, maxit = 10, MaxNWts = 10000)
    coef_mat <- as.matrix(coef(full_model))
    
    # 计算重要性得分（全NA保护）
    if (is.null(dim(coef_mat))) {
      importance_scores <- abs(coef_mat[-1])
    } else {
      importance_scores <- colSums(abs(coef_mat[, -1]))
    }
    # 替换NA为0
    importance_scores[is.na(importance_scores)] <- 0
    
    # 排序（处理全0情况）
    if (all(importance_scores == 0)) {
      selected <- sample(1:ncol(X), n_features)  # 随机选
    } else {
      selected <- order(importance_scores, decreasing = TRUE)[1:n_features]
    }
    selected
    
  }, error = function(e) {
    # 备选方案：相关系数（增加NA保护）
    cor_vals <- sapply(1:ncol(X), function(j) {
      cor_val <- cor(X[, j], as.numeric(as.integer(Y)), use = "complete.obs")
      ifelse(is.na(cor_val) || !is.finite(cor_val), 0, abs(cor_val))
    })
    
    # 处理全0相关系数
    if (all(cor_vals == 0)) {
      selected <- sample(1:ncol(X), n_features)
    } else {
      selected <- order(cor_vals, decreasing = TRUE)[1:n_features]
    }
    selected
  })
  
  # 最终保护：确保返回有效索引
  selected_indices <- selected_indices[!is.na(selected_indices)]
  if (length(selected_indices) < n_features) {
    # 补充随机索引
    extra <- sample(setdiff(1:ncol(X), selected_indices), n_features - length(selected_indices))
    selected_indices <- c(selected_indices, extra)
  }
  
  return(selected_indices[1:min(n_features, length(selected_indices))])
}

# =============================================================================
# 其他特征选择函数（增强鲁棒性）
# =============================================================================
rf_feature_selection <- function(X, Y, n_features) {
  Y_factor <- factor(Y)
  # 单特征保护
  if (ncol(X) == 1) {
    return(1)
  }
  rf_model <- randomForest(X, Y_factor, importance = TRUE, ntree = 10, max_depth = 3)
  importance_scores <- importance(rf_model)[, "MeanDecreaseGini"]
  importance_scores[is.na(importance_scores)] <- 0
  
  if (all(importance_scores == 0)) {
    selected <- sample(1:ncol(X), n_features)
  } else {
    selected <- order(importance_scores, decreasing = TRUE)[1:n_features]
  }
  return(selected)
}

# =============================================================================
# TVS辅助函数（无修改）
# =============================================================================
reward <- function(Y, X, gammas, type, niter, ntree, binary_reward) {
  if (type == "rf") {
    p <- ncol(X)
    selected_idx <- which(gammas == 1)
    full <- rep(0, p)
    if (length(selected_idx) == 0) return(full)
    
    X_sel <- X[, selected_idx, drop = FALSE]
    res <- tryCatch({
      rf <- randomForest(x = X_sel, y = Y, ntree = ntree, importance = TRUE)
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

TVS <- function(Y, X, selector = c('rf'), topq, maxnrep, stop_crit = 100, fix_stop, niter, ntree, a = 1, b = 1, binary_reward = T) {
  
  start_time = Sys.time()
  
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

posterior <- function(tvs_obj) {
  A_final <- tvs_obj$A[, ncol(tvs_obj$A)]
  B_final <- tvs_obj$B[, ncol(tvs_obj$B)]
  return(A_final / (A_final + B_final))
}

tvs_rf_feature_selection <- function(X, Y, n_features) {
  tryCatch({
    tvs_result <- TVS(Y = Y, X = X, selector = "rf", topq = n_features, 
                      fix_stop = 50, ntree = 75, binary_reward = TRUE)
    
    prob <- posterior(tvs_result)
    prob[is.na(prob)] <- 0
    
    selected_indices <- order(prob, decreasing = TRUE)[1:n_features]
    return(selected_indices)
  }, error = function(e) {
    cat("TVS-RF特征选择错误:", e$message, "\n")
    return(rf_feature_selection(X, Y, n_features))
  })
}

# =============================================================================
# 决策树训练函数（增强鲁棒性）
# =============================================================================
train_decision_tree <- function(X, Y, selected_features = NULL) {
  # 处理空特征
  if (!is.null(selected_features)) {
    selected_features <- selected_features[!is.na(selected_features)]
    if (length(selected_features) == 0) {
      selected_features <- 1:min(5, ncol(X))  # 默认选前5个
    }
    X_selected <- X[, selected_features, drop = FALSE]
  } else {
    X_selected <- X
  }
  
  Y_factor <- factor(Y)
  
  # 标准化（处理NA）
  preProc <- preProcess(X_selected, method = c("center", "scale"), na.remove = TRUE)
  X_scaled <- predict(preProc, X_selected)
  X_scaled[is.na(X_scaled)] <- 0
  data_df <- data.frame(Y = Y_factor, X_scaled)
  
  # 交叉验证（样本数保护）set.seed(123)
  cv_folds <- createFolds(Y, k = min(3, nrow(data_df)-1))  # 避免折数超过样本数
  
  cp_grid <- seq(0.001, 0.1, by = 0.005)
  cv_accuracy <- numeric(length(cp_grid))
  
  for (i in seq_along(cp_grid)) {
    fold_accuracy <- numeric(length(cv_folds))
    for (j in seq_along(cv_folds)) {
      train_idx <- -cv_folds[[j]]
      test_idx <- cv_folds[[j]]
      
      # 训练决策树（参数更保守）
      dt_model <- rpart::rpart(
        formula = Y ~ .,
        data = data_df[train_idx, ],
        method = "class",
        cp = cp_grid[i],
        maxdepth = 5,    # 降低树深度
        minsplit = 5,    # 降低最小分裂样本数
        minbucket = 2    # 降低叶节点样本数
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
    maxdepth = 5,
    minsplit = 5,
    minbucket = 2
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
  set.seed(2020 + run_id)  # 每次实验使用不同的种子，三次是2025，两次2020
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
  
  methods <- c("TVS-RF", "REVS-RF","Logistic",  "RandomForest", "SVM")
  
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


