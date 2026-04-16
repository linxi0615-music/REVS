# REVS
REVS  is a novel feature selection algorithm that formulates variable selection as a pure exploration problem in Combinatorial Multi-Armed Bandits (CMAB). The algorithm replaces BART with Random Forest as the reward evaluator, making it applicable to both regression and classification tasks.
# REVS: Reinforced Exploration Variable Selection
This repository is the official implementation of the paper REVS: A Reinforced Exploration Algorithm for Variable Selection via Thompson Sampling with Upper Confidence Bound-Augmented Variance。该算法将变量选择建模为组合多臂老虎机（Combinatorial Multi-Armed Bandit）的纯探索问题，在Thompson采样中融合UCB启发的不确定性增强策略。

**Requirements** 
REVS.R: the proposed algorithm in this paper.The core algorithm implementation of REVS includes a general framework and RF/BART reward functions

function：reward_bart or reward_rf: the reward of REVS

TVS：TVS (Thompson Variable Selection),the algorithm in ''Variable Selection via Thompson Sampling (Yi Liu∗and Veronika Roˇckov´a† February 15, 2021)" .
## 📦 Software and main packages
randomForest
BART :BART Reward evaluator 
glmnet
LASSO
SSLASSO:Spike-and-Slab LASSO 
SIS:Sure Independence Screening 
rpart:Decision tree estimator 
dplyr : data processing
ggplot2 : visual
caret : model training and evaluation 
readr :higher Data reading

## 💻Reproducibility
Simulation Experiments

Set the working directory of R to the directory containing the .R files in the REVS folder. Then run the three different scenarios—regression_friedman, regression_nonlinear, and classification_multiclass—to obtain friedman_results, nonlinear_results, and classification_results, respectively. These result files are used to generate the figures and tables in the paper.

Real Data

This study includes raw data from the open-source UCI repository, specifically the Communities and Crime, Wiki4He, and Student Performance datasets. We have preprocessed these datasets. If you have access to the original UCI data, you can run the code in the files to obtain the data versions used in this paper (for example, through different data processing functions to obtain different results such as communities_processed, wiki4he_processed, and student_processed). The data are then saved as CSV files, which are used for generating figures and tables.

Figures and Tables

After saving the datasets, run the corresponding file to generate all figures and tables presented in the paper.
