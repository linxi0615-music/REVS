import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
import time
from scipy.special import softmax


# ============================================================
# 1. 数据生成函数 (R版本转Python)
# ============================================================
def generate_multiclass_data(n=300, p=1000, n_signal=5, sigma=0.3, random_state=None):
    """
    生成多分类数据

    Parameters:
    -----------
    n : int, 样本数
    p : int, 特征数
    n_signal : int, 真实信号特征数
    sigma : float, 噪声标准差
    random_state : int, 随机种子

    Returns:
    --------
    dict: 包含X, Y, signal_vars的字典
    """
    if random_state is not None:
        np.random.seed(random_state)

    # 生成特征矩阵
    X = np.random.randn(n, p)
    signal_vars = list(range(n_signal))  # 前n_signal个为真实信号特征

    # 线性组合
    linear_comb_0 = (3.2 * X[:, 0] + 2.8 * X[:, 1] - 1.5 * X[:, 2] +
                     2.1 * X[:, 3] - 1.8 * X[:, 4] + np.random.randn(n) * sigma + 2)
    linear_comb_1 = (-2.5 * X[:, 0] + 1.9 * X[:, 1] + 2.3 * X[:, 2] -
                     1.7 * X[:, 3] + 2.0 * X[:, 4] + np.random.randn(n) * sigma + 0)
    linear_comb_2 = (-1.8 * X[:, 0] - 2.2 * X[:, 1] + 1.2 * X[:, 2] -
                     2.4 * X[:, 3] - 1.5 * X[:, 4] + np.random.randn(n) * sigma - 2)

    # 非线性项
    nonlinear_0 = 0.8 * np.sin(np.pi * X[:, 0] * X[:, 1]) + 0.5 * (X[:, 2] ** 2)
    nonlinear_1 = 0.6 * np.cos(np.pi * X[:, 2] * X[:, 3]) + 0.4 * (X[:, 4] ** 2)
    nonlinear_2 = 0.7 * np.sin(np.pi * X[:, 4] * X[:, 0]) + 0.3 * (X[:, 1] ** 2)

    # 合并分数
    score_0 = linear_comb_0 + nonlinear_0
    score_1 = linear_comb_1 + nonlinear_1
    score_2 = linear_comb_2 + nonlinear_2

    # Softmax概率
    scores = np.column_stack([score_0, score_1, score_2])
    exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    # 生成标签
    Y = np.array([np.random.choice(3, p=probs[i]) for i in range(n)])

    return {
        'X': X,
        'Y': Y,
        'signal_vars': signal_vars
    }


def logistic_cv_feature_selection(Y, X, cv=5, selection_threshold='auto',
                                  fit_intercept=False, random_state=42):
    """
    使用带交叉验证的L1逻辑回归进行特征选择（优化版）

    Parameters:
    -----------
    Y : array-like, 标签
    X : array-like, 特征矩阵
    cv : int, 交叉验证折数
    selection_threshold : str or float, 选择阈值
        - 'auto': 自动选择（保留系数绝对值大于1e-6的特征）
        - 'mean': 保留大于均值的特征
        - 'median': 保留大于中位数的特征
        - float: 自定义阈值
    fit_intercept : bool, 是否拟合截距
    random_state : int, 随机种子

    Returns:
    --------
    dict: 包含特征重要性prob、selected_features、selected_idx和执行时间time的字典
    """
    start_time = time.time()

    # 创建LogisticRegressionCV模型
    logistic_cv = LogisticRegressionCV(
        cv=cv,
        fit_intercept=fit_intercept,
        penalty='l1',
        solver='saga',
        max_iter=10000,  # 增大最大迭代次数确保收敛
        Cs=15,  # 增加候选正则化参数数量
        scoring='neg_log_loss',  # 多分类使用负对数损失
        random_state=random_state,
        n_jobs=-1,
        verbose=0
    )

    # 训练模型
    logistic_cv.fit(X, Y)

    # 获取系数
    coef = logistic_cv.coef_

    # 处理多分类和二分类的系数
    if len(coef.shape) > 1:
        # 多分类：取所有类别系数的绝对值最大值（或平均值，根据需求）
        prob = np.max(np.abs(coef), axis=0)
        # 可选：也可以使用平均值
        # prob = np.mean(np.abs(coef), axis=0)
    else:
        # 二分类
        prob = np.abs(coef)

    # ============================================================
    # 优化的非零系数选择逻辑
    # ============================================================

    # 处理浮点精度问题：将接近0的系数视为0
    eps = 1e-10
    prob_clean = np.where(prob < eps, 0, prob)

    # 获取非零系数
    non_zero_idx = np.where(prob_clean > 0)[0]
    non_zero_coef = prob_clean[non_zero_idx]

    # 根据阈值策略选择特征
    if selection_threshold == 'auto':
        # 默认策略：保留所有非零系数（考虑浮点精度）
        selected_idx = non_zero_idx
        threshold_used = 1e-6

    elif selection_threshold == 'mean':
        # 保留大于非零系数均值的特征
        if len(non_zero_coef) > 0:
            mean_val = np.mean(non_zero_coef)
            selected_idx = non_zero_idx[non_zero_coef > mean_val]
            threshold_used = mean_val
        else:
            selected_idx = np.array([])
            threshold_used = 0

    elif selection_threshold == 'median':
        # 保留大于非零系数中位数的特征
        if len(non_zero_coef) > 0:
            median_val = np.median(non_zero_coef)
            selected_idx = non_zero_idx[non_zero_coef > median_val]
            threshold_used = median_val
        else:
            selected_idx = np.array([])
            threshold_used = 0

    elif selection_threshold == 'quantile_75':
        # 保留大于75%分位数的特征
        if len(non_zero_coef) > 0:
            q75_val = np.percentile(non_zero_coef, 75)
            selected_idx = non_zero_idx[non_zero_coef > q75_val]
            threshold_used = q75_val
        else:
            selected_idx = np.array([])
            threshold_used = 0

    elif selection_threshold == 'top_k':
        # 需要额外传入k值，这里默认k=10
        k = getattr(logistic_cv_feature_selection, 'top_k', 10)
        if len(non_zero_coef) > 0:
            sorted_idx = non_zero_idx[np.argsort(non_zero_coef)[::-1]]
            selected_idx = sorted_idx[:min(k, len(sorted_idx))]
            threshold_used = non_zero_coef[selected_idx[-1]] if len(selected_idx) > 0 else 0
        else:
            selected_idx = np.array([])
            threshold_used = 0

    elif isinstance(selection_threshold, (int, float)):
        # 自定义数值阈值
        selected_idx = non_zero_idx[non_zero_coef > selection_threshold]
        threshold_used = selection_threshold

    else:
        # 默认：保留所有非零系数
        selected_idx = non_zero_idx
        threshold_used = 1e-6

    # 构建选择掩码（长度为p）
    selected_mask = np.zeros(len(prob), dtype=bool)
    selected_mask[selected_idx] = True

    end_time = time.time()

    # 打印选择信息（可选）
    print(f"    原始非零系数数: {len(non_zero_idx)}")
    print(f"    阈值策略: {selection_threshold} (阈值={threshold_used:.6f})")
    print(f"    最终选择特征数: {len(selected_idx)}")

    return {
        'prob': prob,  # 所有特征的重要性分数
        'selected_mask': selected_mask,  # 选择掩码
        'selected_idx': selected_idx,  # 选中的特征索引
        'selected_coef': prob[selected_idx],  # 选中的特征系数
        'threshold_used': threshold_used,  # 使用的阈值
        'non_zero_count': len(non_zero_idx),  # 原始非零系数数量
        'time': end_time - start_time
    }


# ============================================================
# 便捷函数：带top_k参数的选择
# ============================================================
def logistic_cv_top_k(Y, X, k=10, cv=5, fit_intercept=False, random_state=42):
    """
    使用逻辑回归选择top-k个最重要的特征

    Parameters:
    -----------
    Y : array-like, 标签
    X : array-like, 特征矩阵
    k : int, 要选择的特征数量
    cv : int, 交叉验证折数
    fit_intercept : bool, 是否拟合截距
    random_state : int, 随机种子

    Returns:
    --------
    dict: 包含特征重要性prob、selected_features、selected_idx的字典
    """
    # 先训练模型
    start_time = time.time()

    logistic_cv = LogisticRegressionCV(
        cv=cv,
        fit_intercept=fit_intercept,
        penalty='l1',
        solver='saga',
        max_iter=10000,
        Cs=15,
        random_state=random_state,
        n_jobs=-1
    )
    logistic_cv.fit(X, Y)

    # 获取系数
    coef = logistic_cv.coef_
    if len(coef.shape) > 1:
        prob = np.max(np.abs(coef), axis=0)
    else:
        prob = np.abs(coef)

    # 选择top-k个特征
    top_k_idx = np.argsort(prob)[::-1][:k]
    selected_mask = np.zeros(len(prob), dtype=bool)
    selected_mask[top_k_idx] = True

    end_time = time.time()

    return {
        'prob': prob,
        'selected_mask': selected_mask,
        'selected_idx': top_k_idx,
        'selected_coef': prob[top_k_idx],
        'time': end_time - start_time
    }


# ============================================================
# 更新评估函数以使用新的返回格式
# ============================================================
def evaluate_selection_v2(selection_result, true_idx):
    """
    评估特征选择效果（适配新版本）

    Parameters:
    -----------
    selection_result : dict, logistic_cv_feature_selection的返回结果
    true_idx : list, 真实信号特征的索引

    Returns:
    --------
    dict: 包含FDP, Precision, Recall, F1, Hamming的字典
    """
    sel = selection_result['selected_mask'].astype(int)

    # 构建真实标记向量
    truth = np.zeros(len(sel))
    truth[true_idx] = 1

    # 计算混淆矩阵元素
    TP = np.sum((sel == 1) & (truth == 1))
    FP = np.sum((sel == 1) & (truth == 0))
    FN = np.sum((sel == 0) & (truth == 1))

    # 计算各项指标
    Precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    Recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    F1 = 2 * Precision * Recall / (Precision + Recall) if (Precision + Recall) > 0 else 0
    FDP = FP / (TP + FP) if (TP + FP) > 0 else 0
    Hamming = np.sum(np.abs(sel - truth))

    return {
        'FDP': FDP,
        'Precision': Precision,
        'Recall': Recall,
        'F1': F1,
        'Hamming': Hamming,
        'Time': selection_result['time'],
        'N_Selected': np.sum(sel)
    }


# ============================================================
# 使用示例
# ============================================================

# 4. 主实验循环
# ============================================================
def run_experiment(num_runs=3, n=1000, p=100, n_signal=5):
    """
    运行多次实验

    Parameters:
    -----------
    num_runs : int, 重复运行次数
    n : int, 样本数
    p : int, 特征数
    n_signal : int, 真实信号特征数

    Returns:
    --------
    dict: 所有实验结果
    """
    all_results = {}

    for run in range(1, num_runs + 1):
        print(f"\n===== 第 {run} 次运行 =====")

        # 生成数据
        data = generate_multiclass_data(
            n=n,
            p=p,
            n_signal=n_signal,
            sigma=0.3,
            random_state=2023 + run
        )
        X = data['X']
        Y = data['Y']
        signal_vars = data['signal_vars']  # 真实信号特征：0到n_signal-1

        print(f"数据生成完成: X形状 {X.shape}, Y类别分布: {np.bincount(Y)}")

        # 存储本次运行的结果
        run_results = {}

        # ---------------------- 逻辑回归特征选择 ----------------------
        print("运行Logistic回归特征选择...")
        logistic_result = logistic_cv_feature_selection(Y, X, cv=5)

        # 评估特征选择效果
        eval_result = evaluate_selection(logistic_result['prob'], signal_vars)
        eval_result['Time'] = logistic_result['time']

        # 输出选择的特征数量
        n_selected = np.sum(np.abs(logistic_result['prob']) > 1e-4)
        print(f"  选择的特征数: {n_selected}/{p}")
        print(f"  Precision: {eval_result['Precision']:.4f}, Recall: {eval_result['Recall']:.4f}")
        print(f"  F1: {eval_result['F1']:.4f}, Hamming距离: {eval_result['Hamming']}")

        run_results['LogisticCV'] = eval_result

        # 保存本次运行结果
        all_results[f'run_{run}'] = run_results

    return all_results


# ============================================================
# 5. 结果汇总与分析
# ============================================================
def summarize_results(all_results):

    num_runs = len(all_results)
    method_names = list(all_results['run_1'].keys())
    metrics = ['FDP', 'Precision', 'Recall', 'F1', 'Hamming', 'Time']

    # 收集所有结果
    results_matrix = {method: {metric: [] for metric in metrics} for method in method_names}

    for run_key, run_result in all_results.items():
        for method in method_names:
            for metric in metrics:
                results_matrix[method][metric].append(run_result[method][metric])

    # 计算均值和标准差
    mean_results = pd.DataFrame(index=method_names, columns=metrics)
    std_results = pd.DataFrame(index=method_names, columns=metrics)

    for method in method_names:
        for metric in metrics:
            values = results_matrix[method][metric]
            mean_results.loc[method, metric] = np.mean(values)
            std_results.loc[method, metric] = np.std(values)

    # 创建详细结果（均值 ± 标准差）
    detailed_results = pd.DataFrame(index=method_names)
    for metric in metrics:
        detailed_results[metric] = [
            f"{mean_results.loc[method, metric]:.4f} ± {std_results.loc[method, metric]:.4f}"
            for method in method_names
        ]

    return mean_results, std_results, detailed_results


# ============================================================
# 6. 打印详细结果
# ============================================================
def print_results(mean_results, std_results, detailed_results):
    """
    打印结果汇总
    """
    print("\n" + "=" * 60)
    print("多次运行结果汇总（均值 ± 标准差）")
    print("=" * 60)

    print("\n--- 详细结果 ---")
    print(detailed_results.to_string())

    print("\n--- 均值结果 ---")
    print(mean_results.round(4).to_string())

    print("\n--- 标准差结果 ---")
    print(std_results.round(4).to_string())

    # 输出最佳结果
    print("\n--- 最佳性能指标 ---")
    best_f1_method = mean_results['F1'].idxmax()
    best_f1_value = mean_results.loc[best_f1_method, 'F1']
    print(f"最佳F1分数: {best_f1_method} = {best_f1_value:.4f}")

    best_precision_method = mean_results['Precision'].idxmax()
    best_precision_value = mean_results.loc[best_precision_method, 'Precision']
    print(f"最佳Precision: {best_precision_method} = {best_precision_value:.4f}")

    best_recall_method = mean_results['Recall'].idxmax()
    best_recall_value = mean_results.loc[best_recall_method, 'Recall']
    print(f"最佳Recall: {best_recall_method} = {best_recall_value:.4f}")

    min_hamming_method = mean_results['Hamming'].idxmin()
    min_hamming_value = mean_results.loc[min_hamming_method, 'Hamming']
    print(f"最小Hamming距离: {min_hamming_method} = {min_hamming_value:.4f}")


if __name__ == "__main__":
    # 生成示例数据
    data = generate_multiclass_data(n=1000, p=100, n_signal=5, random_state=42)
    X, Y = data['X'], data['Y']
    signal_vars = data['signal_vars']

    # 方法1：自动选择非零系数
    print("=" * 50)
    print("方法1：自动选择非零系数")
    result1 = logistic_cv_feature_selection(Y, X, selection_threshold='auto')
    eval1 = evaluate_selection_v2(result1, signal_vars)
    print(f"  Precision: {eval1['Precision']:.4f}, Recall: {eval1['Recall']:.4f}, F1: {eval1['F1']:.4f}")
    print(f"  选中特征: {result1['selected_idx']}")

