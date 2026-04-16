### Reward Function ####


reward<-function(Y,X,gammas,type,niter,ntree,binary_reward){

  #' reward function
  #' @import BART
  #' @import SSLASSO
  #' @description This is an intermediate function that allocates rewards.
  #' @param Y The response, have to be continuous
  #' @param X The matrix of variables
  #' @param gammas The indicator of selected variables
  #' @param type have to be either 'bart' or 'SSL' The indicator of which selection mechanism has to be used
  #' @param niter Number of inner MCMC iterations in BART
  #' @param ntree Number of Trees in BART
  #' @param binary_reward Whether to binarize the reward
  #' @return A list of rewards for each selected variables
  #'

	 if (type=="bart"){

		 out<-capture.output(result<-BART::wbart(
					  X[,gammas==1],
					  Y,
					  sparse=TRUE,
					  ndpost=niter,
					  printevery=10000,
					  ntree=ntree))


		 if (binary_reward){

			reward<-as.numeric(result$varcount[niter,]!=0)

			} else	{

			reward<-as.numeric(result$varcount.mean)

			reward<-as.numeric(reward>1)


					}


		return(reward)

			}


	if (type=="SSL"){

		result<-SSLASSO::SSLASSO(X[,gammas==1],
					 	Y,
					 	lambda0=seq(0.1,100,length=10),
					 	lambda1=0.1)

		reward<-as.numeric(abs(result$beta[,10]))

		return(reward)

	}

}

### Thompson Variable Selection offline ###

TVS<-function(Y, X, selector = c('bart','SSL'), topq, maxnrep, stop_crit =100, fix_stop, niter, ntree, a=1, b=1, binary_reward =T){

  #' TVS
  #' @import BART
  #' @import SSLASSO
  #' @importFrom graphics abline plot points text title
  #' @importFrom stats rbeta rbinom
  #' @importFrom utils capture.output
  #' @description This is offline TVS algorithm
  #' @param Y The response, have to be continuous
  #' @param X The matrix of variables
  #' @param selector support BART selector 'bart' or SSLASSO selector 'SSLASSO'
  #' @param topq An optional system. selecting variable with posterior probability greater than 0.5, we have top q variables
  #' @param maxnrep The maximum number of iterations
  #' @param stop_crit There is a stopping critria if the model has not changed for `stop_crit` number of times, system converge
  #' @param fix_stop If fix stop is not missing, then the the algorithm stops after fixed number of stops
  #' @param niter The number of MCMC iterations in BART
  #' @param ntree The number fo trees in BART
  #' @param a initial prior a
  #' @param b initial prior b
  #' @param binary_reward Whether to binarize the reward or choose a reward based on a bernoulli flip
  #' @return A class 'TVS' object containing the following:
  #' \itemize{
  #' \item `A`, `B` (The TVS A and B each row is `a_i` and `b_i` across time
  #' \item `model_diff` is hamming distance of model at time t and time t-1
  #' \item `time` Time taken for the algorithm to run
  #' \item `type` either 'offline' or 'online' the type of algorithm used
  #' }
  #' @export
  #' @examples
  #' set.seed(2020)
  #' #Friedman Example
  #' X <- matrix(runif(10000),100,100)
  #' Y <- 10 * sin(pi* X[ ,1] * X[,2]) +20 * (X[,3] -.5)^2 + 10 * X[ ,4] + 5 * X[,5] + rnorm(100,sd=1)
  #' TVS_output <- TVS(Y,X,selector= 'bart', maxnrep = 10000, stop_crit = 100, niter = 500, ntree = 10)
  #' posterior(TVS_output)

  start_time = Sys.time()

	p<-ncol(X)

	n<-nrow(X)

	round<-1 #There use to be a round there but since there is no round anymore, I just set round to be 1

	k<- 1 #similarly, the current round is 1

	Yperm<- Y

	Xperm<- X

	if (missing(maxnrep) & missing(fix_stop)){ #maximum number of iteration set at 10000

		maxnrep<-10000

	}else if(!missing(fix_stop)){

	  #This is simple trick I used to make sure that the algorithm stop at iter = fix_stop
	  #The maxnrep is fix_stop means that the algorithm runs at most fix_stop
	  #However, to converge, the algorithm has to at least run stop_crit number of times
	  #so setting stop_crit <- fix_stop ensures that the algorithm will stop at iter=fix_stop

	  maxnrep <- fix_stop

	  stop_crit <- fix_stop

	  #This means that the iteration will end at fix_stop since maxnrep <=stop_criteria

	}

	A<-matrix(a,p,maxnrep*round)

	B<-matrix(b,p,maxnrep*round)

	Aprev<-numeric(p)

	Anew<-numeric(p)

	Bprev<-numeric(p)

	Bnew<-numeric(p)

	model_diff <- rep(1,stop_crit) #initialize everything as 1 so that it will pass the initial stop_crit number of turns

	model<- rep(0,p)

	iter <- 1

	 for (i in (1:maxnrep)){

		  #print(i)

		  # Pick a subset

		  Aprev<-A[,(k-1)*maxnrep+i]

		  Bprev<-B[,(k-1)*maxnrep+i]

		  thetas<-rbeta(p,Aprev,Bprev)

		  if (missing(topq)){

			  #gammas<-rbinom(p,1,thetas)
  			gammas <- thetas >0.5


			}else{

			  gammas<-numeric(p)

			  index<-order(thetas,decreasing=T)

			  gammas[index[1:topq]]<-1

			}

		  # Collect reward

		  rbin<-reward(Yperm,Xperm,gammas,type=selector,
					 niter,ntree=ntree,
			   		 binary_reward=binary_reward)

		  if(max(rbin)>0){

		  rbin<-rbinom(sum(gammas),1,rbin/max(rbin))

			}

		  # Update

		  Anew<-Aprev

		  Anew[gammas==1]<-Anew[gammas==1]+rbin

		  Bnew<-Bprev

		  Bnew[gammas==1]<-Bnew[gammas==1]+1-rbin

		  model_new <- ((Anew/(Anew+Bnew))>0.5)

		  model_diff[iter%%stop_crit+1] = sum(abs(model-model_new)) #differences in model

		  model <- model_new

		  if (i < maxnrep){

		  A[,(k-1)*maxnrep+i+1]<-Anew

		  B[,(k-1)*maxnrep+i+1]<-Bnew


		  }

		  iter = iter + 1

		  #check if the criteria has been met

		  if(sum(model_diff) == 0){

		    converge_time <- Sys.time() - start_time # this record the number of rounds

		    break

		    }

	  }
  if(!missing(fix_stop)){

    converge_time <- Sys.time() - start_time

  }


	r = list(A=A[,1:(iter-1)],B=B[,1:(iter-1)], model_diff = model_diff, time = converge_time, type = 'offline')

	class(r) = 'TVS'

	return(r)

}

###  Plotting ###


visualize<-function(result,col){

  #' visualize
  #' visualize the TVS output across time
  #' @param result output from the TVS or TVS_stream algorithm
  #' @param col The row of color number allocated to each variable.
  #' @return A graph with the output visualized
  #' @export

	prob<-result$A/(result$A+result$B)

	p<-nrow(prob)

	plot(prob[1,],ylim=c(0,1),col=col[1],type="l",ylab="Inclusion Probability",xlab="Time")

	for(i in (2:p)){

	points(prob[i,],ylim=c(0,1),col=col[i],type="l")

	}

	select<- prob[,ncol(prob)]>0.5

	print((1:p)[select])

	abline(h=0.5,lty=2,lwd=2,col="gray")

	title("TVS: Inclusion Probability Plot")

	text(x=ncol(prob),y=prob[,ncol(prob)],paste(1:p))

}

posterior_iter<- function(result,iter){
  #' posterior
  #' Compute the posterior probability of each variable after `iter` number of iterations
  #' @param result from the TVS or TVS_stream algorithm
  #' @param iter The iteration number
  #' @return a column of posterior probability
  #' @export
  #'

  prob<-result$A/(result$A+result$B)

  return(prob[,iter])

}

posterior<-function(result ,type = c('converge', 'fixed')){
  #' posterior
  #' posterior probability computation based on results from TVS
  #' @param result output from the TVS or TVS_stream algorithm
  #' @param type A parameter for 'online' case to check if the user wants the output after'converged' after fix number of iterations
  #' @return a column of posterior probability
  #' @export

  if(result$type =='online'){

    if(type == 'converge'){

      iter = result$iter_converge

    }else{

      iter = result$iter_fixed

    }

  }else{

    iter = ncol(result$A)

  }

  return(posterior_iter(result, iter))


}


# 生成数据
X <- matrix(runif(10000), 100, 100)
Y <- 10*sin(pi*X[,1]*X[,2]) + 20*(X[,3]-.5)^2 + rnorm(100)

# 运行算法
TVS_output <- TVS(Y, X, 
                  selector = 'bart',
                  maxnrep = 10000,
                  stop_crit = 100,
                  niter = 500,
                  ntree = 10)
p=10000
# 创建自定义颜色向量
custom_colors <- rep("black", p)
custom_colors[1:5] <- c("#FF0000", "#FF3333", "#FF6666", "#FF9999", "#FFCCCC")
TVS_output1 <- TVS(Y,X,selector= 'bart', maxnrep = 10000, stop_crit = 100, niter = 500, ntree = 10)
posterior(TVS_output1)
visualize(TVS_output,col=custom_colors)
# 可视化结果
#visualize(TVS_output, col = rainbow(ncol(X)))