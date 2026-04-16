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

REVS_generic <- function(Y, X, reward_func, reward_args = list(),
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
