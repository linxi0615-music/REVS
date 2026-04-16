
# Thompson Variable Selection for large streaming dataset

TVS_stream<-function(Y, X, selector ='bart', topq, niter= 1000, ntree = 10,
                     a = 1, b =1 , binary_reward = TRUE, batchsize,
                     fix_stop, maxround = 100, stop_crit =100){
  #' TVS_stream
  #' 'Online' TVS which converges only after fix number of rounds \strong{and} the convergence criteria has been met
  #' @param Y The response, have to be continuous
  #' @param X The matrix of variables
  #' @param selector support BART selector 'bart' or SSLASSO selector 'SSLASSO'
  #' @param topq An optional system. selecting variable with posterior probability greater than 0.5, we have top q variables
  #' @param stop_crit There is a stopping critria if the model has not changed for `stop_crit` iterations, the algorithm is deemed to reach convergence
  #' @param fix_stop If `fix_stop` is not missing, then the the algorithm stops after fixed number of stops and the stop_crit is met
  #' @param niter The number of MCMC iterations in BART
  #' @param ntree The number fo trees in BART
  #' @param a initial prior a
  #' @param b initial prior b
  #' @param binary_reward Whether to binarize the reward or choose a reward based on a bernoulli flip
  #' @param batchsize The size of each batch
  #' @param maxround the max number of round to go through the data
  #' @return A class 'TVS' object containing the following:
  #' \itemize{
  #' \item `A`, `B` The TVS A and B each row is `a_i` and `b_i` across time
  #' \item `iter_converge` The iteration number when the algorithm reach convergence
  #' \item `round_converge` The round number when the algorithm reach convergence
  #' \item `iter_fixed` The iteration number when the algorithm reach convergence
  #' \item `time_fixed` Time taken for the algorithm to run fix number of rounds if fix_stop is missing, it is time taken to run one round
  #' \item `time_converge` Time taken for the algorithm to reach convergence
  #' \item `type` either 'offline' or 'online' the type of algorithm used, in this case 'online'
  #' }
  #' @export
  #' @examples
  #' set.seed(2020)
  #' X <- matrix(rnorm(1000000),10000,100)
  #' Y <- 2 + X[,1]+ 3*X[,2] + rnorm(10000)
  #' TVS_output <- TVS_stream(Y,X,selector= 'bart', niter= 1000, ntree = 10,
  #' a = 1, b =1 , binary_reward = TRUE, batchsize = 500,
  #' fix_stop = 5, maxround = 100, stop_crit =100)
  #' posterior(TVS_output)


  #list(A=A,B=B, iter_converge= (iter_converge-1), round_converge = k_converge, iter_fixed = (iter_fixed-1),
  #    time_fixed = time_fixed, time_converge = converge_time,  type = 'online')



  if(!missing(fix_stop)){ #if the user want to have a fix number of iteration included, then he/she should have fix_stop
    #The fix stop will be round.

    stopifnot(fix_stop <= maxround)

    round <- fix_stop

  }else{ #if not, the round will be 1 and since convergence round is at least 1, we are done.

    round <- 1
  }


  converge_iter <- stop_crit #the number of iteration for which the model do not change is initially called converge_iter,
                             #then for consistency with TVS, it is renamed stop_crit so this is done to have least demage on the code

  start_time = Sys.time() #start timing

	p<-ncol(X)

	n<-nrow(X)

	nrep<-floor(n/batchsize)

	A<-matrix(a,p,nrep*maxround)

	B<-matrix(b,p,nrep*maxround)

	Aprev<-numeric(p)

	Anew<-numeric(p)

	Bprev<-numeric(p)

	Bnew<-numeric(p)

	con_criteria <- rep(1,converge_iter)

	iter <- 1

	#Because we need to end it at the round than at iteration itself and I want to compute both converge iter and fix iter
	#at the same time, I use to

	converged <- FALSE # converge is FALSE

	First_convergence <- TRUE #It is true because the convergence has not been met, once it has converged, it will be FALSE
	#This stop the system from writing over results if fix_round (fix_k) > converge_round (converge_k)

	for (k in (1:maxround)){

		permute<-sample(1:n)

		Yperm<-Y[permute]

		Xperm<-X[permute,]

	  for (i in (1:nrep)){

		  Aprev<-A[,(k-1)*nrep+i]

		  Bprev<-B[,(k-1)*nrep+i]

		  thetas<-rbeta(p,Aprev,Bprev)

		  if (missing(topq)){

			  gammas<-as.numeric(thetas> 0.5)

			}
		  else{

			  gammas<-numeric(p)

			  index<-order(thetas,decreasing=T)

			  gammas[index[1:topq]]<-1

			}

		  # Collect reward

		  index<-(i-1)*batchsize+(1:batchsize)

		  Ysub<-Yperm[index]

		  Xsub<-Xperm[index,]

		  rbin<-reward(Ysub,Xsub,gammas,type=selector,
					 niter,ntree=ntree,binary_reward=binary_reward)

		  if(max(rbin)>0){

		    rbin<-rbinom(sum(gammas),1,rbin/max(rbin))

			}

		  # Update

		  Anew<-Aprev

		  Anew[gammas==1]<-Anew[gammas==1]+rbin

		  Bnew<-Bprev

		  Bnew[gammas==1]<-Bnew[gammas==1]+1-rbin

		  #record convergence

		  con_criteria[(iter%%converge_iter)+1] <- sum(abs((Anew/(Anew+Bnew)>0.5)-(Aprev/(Aprev+Bprev)>0.5)))


		  if(sum(con_criteria) ==0){

		    converged = TRUE

		  }

		  iter = iter + 1

		  if(i < nrep | k< maxround){

		    A[,(k-1)*nrep+i+1]<-Anew

		    B[,(k-1)*nrep+i+1]<-Bnew

		  }

	  }

		#check for convergence condition.

		if(converged){

		  if(First_convergence){ #this is when the convergence is meet for the first time. If there is no first
		    #convergence, the convergence can be met but fix_k > converge_k so each time I pass a round,
		    #the result can be written over and therefore the result is not desirable.

		    First_convergence = FALSE #This is to avoid the result being written over when fix_k > converge_k

		    iter_converge <- iter

		    k_converge = k

		    converge_time =Sys.time() - start_time # this record the number of rounds

		  }

		  if(k > round){

		    break #converge_k > fix_k, then this condition kicks in and we are done.

		  }

		}
		if(k == round){ #This is when system has done fix_k number of time.

		  iter_fixed = iter

		  time_fixed = Sys.time() - start_time

		  if(converged){ #converge_k <= fix_k

		    break

		  }

		}

	}
	#The iteration has reach maxround at this point if it did not break.
	if(k == maxround){ #This is when algorithm never converged

	  iter_fixed = iter

	  k_converge = k

	  converge_time =Sys.time() - start_time # this record the number of rounds

	}
	if(First_convergence){ #It is to ensure that

	  #This is when algorithm never converged

	  First_convergence = FALSE

	}

	r <- list(A=A,B=B, iter_converge= (iter_converge-1), round_converge = k_converge, iter_fixed = (iter_fixed-1),
	          time_fixed = time_fixed, time_converge = converge_time,  type = 'online')

  class(r) <- 'TVS_stream'

  return(r)
}
# 根据图片内容创建的Thompson变量选择示例
set.seed(2025)

# 生成模拟数据
n <- 20000  # 样本数
p <- 1000   # 变量数
e <- rnorm(n)  # 共同因子

# 创建具有约0.5相关性的变量
X <- matrix(0, n, p)
for(i in 1:n) {
  for(j in 1:p) {
    z_ij <- rnorm(1)
    X[i,j] <- (e[i] + z_ij)/2  # 创建相关性约为0.5的变量
  }
}

# 确保变量在[0,1]范围内
X <- pnorm(X)  # 将值转换到[0,1]区间

# 生成响应变量Y，使用图片中的非线性函数
Y <- numeric(n)
for(i in 1:n) {
  f0 <- (10*X[i,2])/(1+X[i,1]^2) + 5*sin(X[i,3]*X[i,4]) + 2*X[i,5]
  Y[i] <- f0 + rnorm(1, 0, sqrt(0.5))  # 噪声方差为0.5
}

# 应用Thompson变量选择算法
tvs_result <- TVS_stream(
  Y = Y, 
  X = X, 
  selector = 'bart',  # 使用BART选择器
  niter = 1000,       # MCMC迭代次数
  ntree = 10,         # 树的数量
  a = 1,              # 先验参数a
  b = 1,              # 先验参数b
  binary_reward = TRUE, 
  batchsize = 500,    # 每批数据大小
  fix_stop = 5,       # 固定5轮后检查收敛
  maxround = 100,     # 最大轮数
  stop_crit = 100     # 收敛标准
)

# 修改可视化函数，添加max_time参数
visualize_top <- function(result, col, top_n = 20, max_time = 50) {
  prob <- result$A/(result$A+result$B)
  
  # 只选择前top_n个变量
  prob_subset <- prob[1:top_n, ]
  
  # 限制显示的时间点数量
  time_points <- min(ncol(prob_subset), max_time)
  prob_subset <- prob_subset[, 1:time_points, drop = FALSE]
  
  col_subset <- col[1:top_n]
  
  # 创建绘图区域
  plot(1, type = "n", ylim = c(0,1), xlim = c(1, time_points), 
       ylab = "Inclusion Probability", xlab = "Time")
  
  # 添加点线
  for(i in 1:top_n) {
    # 对于前5个变量（真实重要的变量），使用更粗的线条
    if(i <= 5) {
      lines(prob_subset[i, ], col = col_subset[i], lwd = 2)
    } else {
      lines(prob_subset[i, ], col = col_subset[i])
    }
  }
  
  # 找出后验概率大于0.5的变量
  select <- prob_subset[, ncol(prob_subset)] > 0.5
  
  cat("被选择的变量（前", top_n, "个中，在时间点", time_points, "）:\n")
  if(any(select)) {
    cat(paste((1:top_n)[select], collapse = ", "), "\n")
  } else {
    cat("没有变量的后验概率超过0.5\n")
  }
  
  # 添加0.5参考线
  abline(h = 0.5, lty = 2, lwd = 2, col = "gray")
  
  # 添加标题
  title("TVS: Inclusion Probability Plot")
  
  # 为前5个变量添加标签
  for(i in 1:min(5, top_n)) {
    text(x = time_points, y = prob_subset[i, time_points], 
         labels = i, adj = c(-0.3, 0.5))
  }
}
# 设置图形参数
par(mar=c(4,4,2,1))

# 可视化结果
# 使用rainbow调色板，但为了与示例图更相似，我们可以修改颜色设置
# 为前5个变量使用红色系（真实重要的变量）
# 为其他变量使用黑色系

# 创建自定义颜色向量
custom_colors <- rep("black", p)
custom_colors[1:5] <- c("#FF0000", "#FF3333", "#FF6666", "#FF9999", "#FFCCCC")

# 调用可视化函数（仅前20个变量）
# 可视化结果，限制为前300个时间点
visualize_top(tvs_result, col = custom_colors, top_n = 1000, max_time =100)
# 如果需要可视化所有变量，可以取消注释下面的代码visualize_top(tvs_result, col=custom_colors, top_n = 1000, max_time = 200)

# 输出性能指标
true_important <- 1:5
post_probs <- tvs_result$A[,ncol(tvs_result$A)] / (tvs_result$A[,ncol(tvs_result$A)] + tvs_result$B[,ncol(tvs_result$B)])
important_vars <- which(post_probs > 0.5)

cat("\n算法性能评估:\n")
cat("识别的重要变量:", important_vars, "\n")
cat("真实重要变量: 1, 2, 3, 4, 5\n")
cat("收敛轮数:", tvs_result$round_converge, "\n")
cat("收敛迭代次数:", tvs_result$iter_converge, "\n")


