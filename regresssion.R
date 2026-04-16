generate_data <- function(n = 3000, p = 300, n_signal = 5, noise_sd = 1) {
  sigma=1
  X <- matrix(runif(n * p), n, p)
  Y <- 10 * sin(pi * X[, 1] * X[, 2]) +
    20 * (X[, 3] - 0.5)^2 +
    10 * X[, 4] +
    5 * X[, 5] +
    rnorm(n, 0, sigma)
  return(list(X = X, Y = Y, signal = 1:5))
}

generate_data1 <- function(n = 1000, p = 100) {
  e <- rnorm(n)
  X <- matrix(0, n, p)
  for (i in 1:n) {
    for (j in 1:p) {
      z_ij <- rnorm(1)
      X[i, j] <- (e[i] + z_ij) / 2
    }
  }
  X <- pnorm(X)
  Y <- numeric(n)
  for (i in 1:n) {
    f0 <- (10 * X[i, 2]) / (1 + X[i, 1]^2) + 5 * sin(X[i, 3] * X[i, 4]) + 2 * X[i, 5]
    Y[i] <- f0 + rnorm(1, 0, sqrt(0.5))
  }
  signal_vars <- 1:5
  return(list(X = X, Y = Y, signal = signal_vars))
}
library(BART)
library(randomForest)
library(glmnet)
library(SSLASSO)
library(SIS)
library(horseshoe)
library(caret)    # 用于 rfe
# library(e1071) # 若 caret 报错需要 e1071，自动依赖
reward_rf <- function(Y, X, gammas, ntree = 50, binary_reward = TRUE) {
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

# BART new reward (M independent runs)
reward_bart_new <- function(Y, X, gammas, niter = 100, ntree = 30, M = 4, binary_reward = TRUE) {
  p <- ncol(X)
  selected_idx <- which(gammas == 1)
  full <- rep(0, p)
  if (length(selected_idx) == 0) return(full)
  X_sel <- X[, selected_idx, drop = FALSE]
  incl_mat <- matrix(0, nrow = M, ncol = length(selected_idx))
  for (m in 1:M) {
    tryCatch({
      fit <- BART::wbart(x.train = X_sel, y.train = Y, ndpost = niter, ntree = ntree, printevery = 10000)
      vc <- fit$varcount
      if (!is.null(vc) && nrow(vc) >= 1) {
        row_idx <- min(nrow(vc), niter)
        incl_mat[m, ] <- as.numeric(vc[row_idx, ] > 0)
      } else if (!is.null(fit$varcount.mean)) {
        incl_mat[m, ] <- as.numeric(fit$varcount.mean > 0)
      } else {
        incl_mat[m, ] <- 0
      }
    }, error = function(e) {
      message("reward_bart_new error (m=", m, "): ", e$message)
      incl_mat[m, ] <- 0
    })
  }
  freq <- colMeans(incl_mat)
  if (binary_reward) {
    thr <- stats::median(freq, na.rm = TRUE)
    thr <- ifelse(is.na(thr), 0, thr)
    rv <- as.numeric(freq >= thr)
  } else {
    maxf <- max(freq, na.rm = TRUE)
    if (!is.finite(maxf) || maxf == 0) maxf <- 1
    rv <- freq / maxf
  }
  full[selected_idx] <- rv
  return(full)
}

# BART reward
reward_bart <- function(Y, X, gammas, niter = 100, ntree = 30, binary_reward = TRUE) {
  p <- ncol(X)
  selected_idx <- which(gammas == 1)
  full <- rep(0, p)
  if (length(selected_idx) == 0) return(full)
  X_sel <- X[, selected_idx, drop = FALSE]
  res <- tryCatch({
    fit <- BART::wbart(x.train = X_sel, y.train = Y, ndpost = niter, ntree = ntree, printevery = 10000)
    if (!is.null(fit$varcount.mean)) {
      imp <- fit$varcount.mean
    } else if (!is.null(fit$varcount)) {
      imp <- colMeans(fit$varcount)
    } else {
      imp <- rep(0, length(selected_idx))
    }
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
    message("reward_bart error: ", e$message)
    full
  })
  return(res)
}



TVS_generic <- function(Y, X, reward_func, reward_args = list(),
                        topq = 20, fix_stop = 200, binary_reward = TRUE,
                        M_theta = 6, delta = 1e-3, q = NULL, update_mode = "batch") {
  if (is.null(q)) q <- delta
  start_time <- Sys.time()
  p <- ncol(X)
  A <- matrix(1, p, fix_stop + 1)
  B <- matrix(1, p, fix_stop + 1)
  phi_q <- stats::qnorm(1 - q); if (!is.finite(phi_q)) phi_q <- 3
  final_iter <- fix_stop
  
  # 假设真实的重要特征（这里需要根据你的实际情况修改）
  # 如果没有真实标签，这些指标将无法计算
  true_important <- NULL  # 你需要在这里指定真实的重要特征索引
  
  for (t in 1:fix_stop) {
    Aprev <- A[, t]; Bprev <- B[, t]
    Ni <- Aprev + Bprev; Ni[Ni <= 0] <- 1
    mu_hat <- Aprev / Ni; mu_hat[is.na(mu_hat)] <- 0
    L1t <- log(12 * (max(2, p)^2) * (t^2) / delta + 1)
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
    
    # update A/B - 按照你的要求修改
    Anew <- Aprev; Bnew <- Bprev
    
    # 第一步：根据 gammas == 1 更新A/B
    idxs <- which(gammas == 1)
    if (length(idxs) > 0) {
      vals <- stats::rbinom(length(idxs), 1, prob = rprob[idxs])
      Anew[idxs] <- Anew[idxs] + vals
      Bnew[idxs] <- Bnew[idxs] + (1 - vals)
    }
    
    # 第二步：对 pull_idx 执行额外的更新
    Anew[pull_idx] <- Anew[pull_idx] + 1
    
    A[, t + 1] <- Anew; B[, t + 1] <- Bnew
  } # end loop
  
  end_time <- Sys.time()
  
  # trim matrices to final_iter
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

# 5) TVS_old (original) - fixed indexing and robust

reward <- function(Y, X, gammas, type, niter, ntree, binary_reward) {
  if (type == "bart") {
    out <- capture.output(
      result <- BART::wbart(
        X[, gammas == 1, drop = FALSE],
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
    return(reward)
  }
  
  if (type == "SSL") {
    result <- SSLASSO::SSLASSO(
      X[, gammas == 1, drop = FALSE],
      Y,
      lambda0 = seq(0.1, 100, length = 10),
      lambda1 = 0.1
    )
    reward <- as.numeric(abs(result$beta[, 10]))
    return(reward)
  }
}

TVS <- function(Y, X, selector = c('bart','SSL'), topq, maxnrep, stop_crit = 100, 
                fix_stop, niter, ntree, a = 1, b = 1, binary_reward = TRUE) {
  
  start_time = Sys.time()
  p <- ncol(X)
  n <- nrow(X)
  round <- 1
  k <- 1
  
  # 初始化参数
  A <- matrix(a, p, maxnrep * round)
  B <- matrix(b, p, maxnrep * round)
  model_diff <- rep(1, stop_crit)
  model <- rep(0, p)
  iter <- 1
  
  for (i in (1:maxnrep)) {
    # 选择变量子集
    Aprev <- A[, (k-1)*maxnrep + i]
    Bprev <- B[, (k-1)*maxnrep + i]
    thetas <- rbeta(p, Aprev, Bprev)
    
    if (missing(topq)) {
      gammas <- thetas > 0.5
    } else {
      gammas <- numeric(p)
      index <- order(thetas, decreasing = TRUE)
      gammas[index[1:min(topq, p)]] <- 1
    }
    
    # 如果没有选中任何变量，随机选择一些
    if (sum(gammas) == 0) {
      gammas[sample(p, min(5, p))] <- 1
    }
    
    # 收集奖励
    rbin <- reward(Y, X, gammas, type = selector, niter, ntree = ntree,
                   binary_reward = binary_reward)
    
    if (max(rbin) > 0) {
      rbin <- rbinom(sum(gammas), 1, rbin/max(rbin))
    }
    
    # 更新参数
    Anew <- Aprev
    Anew[gammas == 1] <- Anew[gammas == 1] + rbin
    Bnew <- Bprev
    Bnew[gammas == 1] <- Bnew[gammas == 1] + 1 - rbin
    
    model_new <- ((Anew/(Anew + Bnew)) > 0.5)
    model_diff[iter %% stop_crit + 1] = sum(abs(model - model_new))
    model <- model_new
    
    if (i < maxnrep) {
      A[, (k-1)*maxnrep + i + 1] <- Anew
      B[, (k-1)*maxnrep + i + 1] <- Bnew
    }
    
    iter = iter + 1
    
    # 检查收敛条件
    if (sum(model_diff) == 0) {
      break
    }
  }
  
  converge_time <- Sys.time() - start_time
  final_prob <- A[, ncol(A)]/(A[, ncol(A)] + B[, ncol(B)])
  selected_vars <- which(final_prob > 0.5)
  
  r = list(A = A[, 1:(iter-1)], B = B[, 1:(iter-1)], 
           model_diff = model_diff, time = as.numeric(converge_time, units = "secs"), 
           selected_vars = selected_vars, type = 'offline', prob = final_prob)
  class(r) = 'TVS'
  return(r)
}

# ================================================================
# 5️⃣ 性能评估函数

evaluate_selection <- function(prob, true_idx) {
  sel <- as.numeric(prob > 0.5)
  truth <- rep(0, length(prob)); truth[true_idx] <- 1
  TP <- sum(sel==1 & truth==1)
  FP <- sum(sel==1 & truth==0)
  FN <- sum(sel==0 & truth==1)
  Precision <- ifelse((TP+FP)==0, 0, TP/(TP+FP))
  Recall <- ifelse((TP+FN)==0, 0, TP/(TP+FN))
  F1 <- ifelse((Precision+Recall)==0, 0, 2*Precision*Recall/(Precision+Recall))
  FDP <- ifelse((TP+FP)==0, 0, FP/(TP+FP))
  Hamming <- sum(abs(sel-truth))
  data.frame(FDP, Precision, Recall, F1, Hamming)
}

# ================================================================
run_bart <- function(X, Y, topq = 10, niter = 1000, ntree = 50) {
  start_time <- Sys.time()
  res <- tryCatch({
    fit <- BART::wbart(x.train = X, y.train = Y, ndpost = niter, ntree = ntree, printevery = 10000, sparse = FALSE)
    imp <- fit$varcount.mean
    if (is.null(imp)) imp <- colMeans(fit$varcount)
    # 取 topq 个变量作为被选择
    top_idx <- order(imp, decreasing = TRUE)[1:min(topq, length(imp))]
    list(selected_vars = sort(unique(top_idx)), time = as.numeric(difftime(Sys.time(), start_time, units = "secs")))
  }, error = function(e) {
    list(selected_vars = integer(0), time = as.numeric(difftime(Sys.time(), start_time, units = "secs")))
  })
  return(res)
}

run_dart <- function(X, Y, topq = 10, niter = 1000, ntree = 50) {
  start_time <- Sys.time()
  res <- tryCatch({
    fit <- BART::wbart(x.train = X, y.train = Y, ndpost = niter, ntree = ntree, sparse = TRUE, printevery = 10000)
    imp <- if (!is.null(fit$varcount.mean)) fit$varcount.mean else colMeans(fit$varcount)
    top_idx <- order(imp, decreasing = TRUE)[1:min(topq, length(imp))]
    list(selected_vars = sort(unique(top_idx)), time = as.numeric(difftime(Sys.time(), start_time, units = "secs")))
  }, error = function(e) {
    list(selected_vars = integer(0), time = as.numeric(difftime(Sys.time(), start_time, units = "secs")))
  })
  return(res)
}

# 修改 LASSO 函数 - 选择系数非零的变量
run_lasso <- function(X, Y, topq = 5) {
  start_time <- Sys.time()
  res <- tryCatch({
    cv_fit <- glmnet::cv.glmnet(X, Y, alpha = 1, family = "gaussian")
    lasso_fit <- glmnet::glmnet(X, Y, alpha = 1, lambda = cv_fit$lambda.min, family = "gaussian")
    beta <- as.vector(coef(lasso_fit))[-1]  # 获取系数（去掉截距）
    
    # 修改：选择所有系数非零的变量，不再限制topq
    sel_idx <- which(abs(beta) > 1e-8)  # 选择系数绝对值大于阈值（非零）的变量
    
    list(selected_vars = sort(unique(sel_idx)), 
         time = as.numeric(difftime(Sys.time(), start_time, units = "secs")))
  }, error = function(e) {
    list(selected_vars = integer(0), 
         time = as.numeric(difftime(Sys.time(), start_time, units = "secs")))
  })
  return(res)
}

# 修改 SSLASSO 函数 - 选择系数非零的变量
run_sslasso <- function(X, Y, topq = 5) {
  start_time <- Sys.time()
  res <- tryCatch({
    fit <- SSLASSO::SSLASSO(X, Y, lambda0 = seq(0.1, 100, length = 10), lambda1 = 0.1)
    beta <- fit$beta[, ncol(fit$beta)]  # 获取系数向量
    
    # 修改：选择所有系数非零的变量，不再限制topq
    sel_idx <- which(abs(beta) > 1e-8)  # 选择系数绝对值大于阈值（非零）的变量
    
    list(selected_vars = sort(unique(sel_idx)), 
         time = as.numeric(difftime(Sys.time(), start_time, units = "secs")))
  }, error = function(e) {
    list(selected_vars = integer(0), 
         time = as.numeric(difftime(Sys.time(), start_time, units = "secs")))
  })
  return(res)
}
run_sis <- function(X, Y, topq = 5) {
  start_time <- Sys.time()
  p <- ncol(X)
  
  # 每个变量与 Y 做独立回归，得到相关性强度
  corr <- apply(X, 2, function(x) abs(cor(x, Y)))
  
  # 取前 topq
  sel_idx <- order(corr, decreasing = TRUE)[1:topq]
  
  time_used <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))
  list(selected_vars = sort(sel_idx), time = time_used)
}

# ================================================================
# 6️⃣ compare_methods：九方法对比（包含 TVS 系列与 独立方法）
# ================================================================
compare_methods <- function(n_runs = 1, n = 3000, p = 200, topq = 10, 
                            niter = 200, ntree = 75, batchsize = 100) {
  metrics_names <- c("FDP", "Precision", "Recall", "F1", "Hamming", "Time")
  method_list <- c("TVS-RF", "TVS-BART(new)", "TVS-BART(old)",
                   "BART", "DART", "LASSO", "SSLASSO",
                   "SIS")
  
  
  # 初始化存放矩阵
  res_mats <- lapply(method_list, function(x) matrix(NA, n_runs, length(metrics_names)))
  names(res_mats) <- method_list
  
  for (run in 1:n_runs) {
    cat("=== run", run, "===\n")
    set.seed(2023 + run)
    d <- generate_data(n, p); X <- d$X; Y <- d$Y; signal_vars <- d$signal
    
    fix_stop=150
    binary_reward=TRUE
    # 2. TVS-RF
    cat("运行 TVS-RF...\n")
    rf_res <- TVS_generic(Y, X, reward_rf, reward_args = list(ntree = ntree, binary_reward = binary_reward),
                          topq = topq, fix_stop = fix_stop, binary_reward = binary_reward, M_theta = 6, update_mode = "batch")
    ev_rf <- evaluate_selection(rf_res$prob, signal_vars)
    res_mats[["TVS-RF"]][run, 1:5] <- unlist(ev_rf); res_mats[["TVS-RF"]][run, 6] <- rf_res$time
    
    set.seed(2023 + run)
    d <- generate_data(n, p); X <- d$X; Y <- d$Y; signal_vars <- d$signal
    
    # 3. TVS-BART(new)
    cat("运行 TVS-BART(new)...\n")
    bart_new <- TVS_generic(Y, X, reward_bart, reward_args = list(niter = niter, ntree = ntree, binary_reward = binary_reward),
                            topq = topq, fix_stop = fix_stop, binary_reward = binary_reward, M_theta = 6, update_mode = "single")
    ev_bart_new <- evaluate_selection(bart_new$prob, signal_vars)
    res_mats[["TVS-BART(new)"]][run, 1:5] <- unlist(ev_bart_new); res_mats[["TVS-BART(new)"]][run, 6] <- bart_new$time
    
    set.seed(2023 + run)
    d <- generate_data(n, p); X <- d$X; Y <- d$Y; signal_vars <- d$signal
    
    
    # 4. TVS-BART(old)
    cat("运行 TVS-BART(old)...\n")
    old_res <- TVS(Y, X, selector = "bart", maxnrep = 100, fix_stop = fix_stop,
                   niter = niter, ntree = ntree, binary_reward = binary_reward)
    ev_old <- evaluate_selection(old_res$prob, signal_vars)
    res_mats[["TVS-BART(old)"]][run, 1:5] <- unlist(ev_old); res_mats[["TVS-BART(old)"]][run, 6] <- old_res$time
    
    set.seed(2023 + run)
    d <- generate_data(n, p); X <- d$X; Y <- d$Y; signal_vars <- d$signal
    
    
    # 5. 独立 BART（不使用 TVS_stream_generic）
    cat("运行 独立 BART...\n")
    sb <- run_bart(X, Y, topq = topq, niter = niter, ntree = ntree)
    prob_bart <- rep(0, p); prob_bart[sb$selected_vars] <- 1
    ev_sb <- evaluate_selection(prob_bart, signal_vars)
    res_mats[["BART"]][run, 1:5] <- unlist(ev_sb); res_mats[["BART"]][run, 6] <- sb$time
    
    set.seed(2023 + run)
    d <- generate_data(n, p); X <- d$X; Y <- d$Y; signal_vars <- d$signal
    
    
    # 6. 独立 DART
    cat("运行 独立 DART...\n")
    sdart <- run_dart(X, Y, topq = topq, niter = niter, ntree = ntree)
    prob_dart <- rep(0, p); prob_dart[sdart$selected_vars] <- 1
    ev_sdart <- evaluate_selection(prob_dart, signal_vars)
    res_mats[["DART"]][run, 1:5] <- unlist(ev_sdart); res_mats[["DART"]][run, 6] <- sdart$time
    
    set.seed(2023 + run)
    d <- generate_data(n, p); X <- d$X; Y <- d$Y; signal_vars <- d$signal
    
    
    # 7. 独立 LASSO
    cat("运行 独立 LASSO...\n")
    slasso <- run_lasso(X, Y, topq = topq)
    prob_lasso <- rep(0, p); prob_lasso[slasso$selected_vars] <- 1
    ev_lasso <- evaluate_selection(prob_lasso, signal_vars)
    res_mats[["LASSO"]][run, 1:5] <- unlist(ev_lasso); res_mats[["LASSO"]][run, 6] <- slasso$time
    
    set.seed(2023 + run)
    d <- generate_data(n, p); X <- d$X; Y <- d$Y; signal_vars <- d$signal
    
    
    # 8. 独立 SSLASSO
    cat("运行 独立 SSLASSO...\n")
    sss <- run_sslasso(X, Y, topq = topq)
    prob_sss <- rep(0, p); prob_sss[sss$selected_vars] <- 1
    ev_sss <- evaluate_selection(prob_sss, signal_vars)
    res_mats[["SSLASSO"]][run, 1:5] <- unlist(ev_sss); res_mats[["SSLASSO"]][run, 6] <- sss$time
    
    set.seed(2023 + run)
    d <- generate_data(n, p); X <- d$X; Y <- d$Y; signal_vars <- d$signal
    
    # SIS
    cat("运行 SIS...\n")
    ssis <- run_sis(X, Y, topq = topq)
    prob_sis <- rep(0, p); prob_sis[ssis$selected_vars] <- 1
    ev_sis <- evaluate_selection(prob_sis, signal_vars)
    res_mats[["SIS"]][run, 1:5] <- unlist(ev_sis)
    res_mats[["SIS"]][run, 6] <- ssis$time
    
  }
  
  return(res_mats)
}

# ================================================================
# 7️⃣ 绘图函数（6 指标柱状图）
# ================================================================
plot_combined_metrics <- function(results, save_path = "tvs_rfe_comparison_regression.pdf") {
  metrics <- c("FDP", "Precision", "Recall", "F1", "Hamming", "Time")
  method_names <- names(results)
  n_methods <- length(method_names)
  
  colors <- rainbow(n_methods)
  names(colors) <- method_names
  
  pdf(save_path, width = 15, height = 10)
  par(mfrow = c(2, 3), mar = c(8, 4, 3, 1), oma = c(2, 2, 3, 1))
  
  for (k in seq_along(metrics)) {
    m <- k  # 用数字索引代替字符串
    metric_name <- metrics[k]
    
    means <- sapply(method_names, function(method) {
      data <- results[[method]][, m]
      data_clean <- data[!is.na(data) & is.finite(data)]
      if (length(data_clean) == 0) return(0)
      mean(data_clean)
    })
    
    sds <- sapply(method_names, function(method) {
      data <- results[[method]][, m]
      data_clean <- data[!is.na(data) & is.finite(data)]
      if (length(data_clean) <= 1) return(0)
      sd(data_clean)
    })
    
    y_max <- max(means + sds, na.rm = TRUE)
    y_lim <- c(0, y_max * 1.25)
    
    bp <- barplot(means, main = metric_name, ylab = "Value",
                  col = colors, ylim = y_lim,
                  las = 2, cex.names = 0.7)
    
    arrows(bp, pmax(means - sds, 0), bp, means + sds,
           length = 0.05, angle = 90, code = 3)
  }
  
  mtext("TVS + Baselines Performance Comparison (Regression)",
        outer = TRUE, cex = 1.3, font = 2, line = 1)
  
  legend("bottom", legend = method_names, fill = colors, ncol = 3, bty = "n", inset = -0.08, xpd = TRUE)
  dev.off()
  message("✅ 图像已保存至: ", save_path)
}

# ================================================================
# 8️⃣ 运行对比实验（示例）
# ================================================================
cat("开始运行 TVS-RFE 对比实验（示例参数）...\n")

# 建议先用较小规模做测试
results <- compare_methods(n_runs = 30, n = 1000, p = 200, topq = 10, niter = 300, ntree = 75, batchsize = 300)

# 生成并保存图
plot_combined_metrics(results, save_path = "tvs_offline_comparison_regression.pdf")

# 打印摘要
summary_method <- function(mat) {
  clean_mat <- mat
  clean_mat[!is.finite(clean_mat)] <- NA
  mu <- colMeans(clean_mat, na.rm = TRUE)
  sdv <- apply(clean_mat, 2, sd, na.rm = TRUE)
  out <- rbind(mean = mu, sd = sdv)
  return(round(out, 4))
}

method_names <- names(results)
for (method in method_names) {
  cat("\n=== Summary:", method, "===\n")
  print(summary_method(results[[method]]))
}
