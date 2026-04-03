import numpy as np
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB, quicksum
#由电网公司自行进行调配，或者让CDP直接自己
def compute_K(rho, Fr_center_mean):
    rho = max(float(rho), 1e-9)
    moment = max(float(Fr_center_mean), 1e-12)
    inner = (1.0 + rho + np.log(moment)) / (2.0 * rho)
    return 2.0 * np.sqrt(max(inner, 0.0))

def compute_eplsilon(K, partial, r):
    return K * np.sqrt(2 / r * np.log(1 / (1 - partial)))

def find_optimal_rho(Fr_center_mean, rho_low=1e-6, rho_high=10.0, tol=1e-6, max_iter=100):
    """
    用二分搜索法找到最小化 K(rho, F) 的 rho
    Args:
        Fr_centered: 零均值化后的误差样本
        rho_low: 搜索下限 (必须 >0)
        rho_high: 搜索上限
        tol: 收敛容忍度
        max_iter: 最大迭代次数
    Returns:
        optimal_rho: 最优的 rho
        min_K: 最小的 K(rho, F) 值
    """
    for _ in range(max_iter):
        rho_mid1 = rho_low + (rho_high - rho_low) / 3
        rho_mid2 = rho_high - (rho_high - rho_low) / 3
        # 计算 K(rho) 的值
        K1 = compute_K(rho_mid1, Fr_center_mean)
        K2 = compute_K(rho_mid2, Fr_center_mean)
        # 更新搜索区间
        if K1 < K2:
            rho_high = rho_mid2
        else:
            rho_low = rho_mid1
        # 检查收敛
        if rho_high - rho_low < tol:
            break
    optimal_rho = (rho_low + rho_high) / 2
    min_K = compute_K(optimal_rho, Fr_center_mean)
    return optimal_rho, min_K

def wasserstein_ball(Fr_samples, epsilon, n_test_samples=1000):
    """
    检查生成的测试样本是否在Wasserstein球内（适用于1维样本集）
    
    Args:
        Fr_samples: 历史样本 (1, n_samples) 或 (n_samples,)
        epsilon: Wasserstein球半径
        n_test_samples: 测试样本数量
    
    Returns:
        bool: 是否所有维度的距离都满足条件
        distances: 各维度的Wasserstein距离
    """
    
    # 生成测试样本（添加噪声）
    noise = np.random.normal(0, 0.2, size=n_test_samples)
    F_test_samples = Fr_samples[:n_test_samples] + noise
    
    # 计算Wasserstein距离（单变量）
    dist = wasserstein_distance(Fr_samples, F_test_samples)
    
    # 返回结果（单维度）
    return dist <= epsilon, [dist]


def generate_distribution_in_ball(Fr_samples, epsilon, n_samples=1000):
    """
    生成Wasserstein球内的新分布（适用于1维样本集）
    
    Args:
        Fr_samples: 历史样本 (1, n_samples) 或 (n_samples,)
        epsilon: Wasserstein球半径
        n_samples: 生成样本数
    
    Returns:
        new_samples: 新样本 (n_samples,)
    """
    
    # 生成扰动（单变量）
    noise = np.random.normal(0, epsilon/2, size=n_samples)  # 直接使用epsilon控制噪声幅度
    new_samples = Fr_samples[:n_samples] + noise
    
    return new_samples

def extract_96_elements(new_samples):
    """
    从new_samples中随机提取96个元素组成一个列表
    
    Args:
        new_samples: 生成的样本 (n_samples, 96)
        
    Returns:
        list: 由96个随机元素组成的列表
    """
    # 将new_samples展平为1维数组
    flattened_samples = new_samples.flatten()
    
    # 随机选择96个元素（可重复）
    selected_elements = np.random.choice(flattened_samples, size=96)
    
    # 返回为列表形式
    return selected_elements.tolist()

def find_worst_case_distributions1(Fr_samples, lambda_VPPO_CDP_up, lambda_VPPO_CDP_down, epsilon, p_max):
    """
    在Wasserstein球内找到使成本最大的两个分布 delta_1 和 delta_2
    Args:
        Fr_samples: 历史样本 (1000, )
        price: 价格向量 (96,)
        epsilon1, epsilon2: Wasserstein球半径
    Returns:
        delta_1, delta_2: 最坏情况分布 (96,)
        max_cost: 最大成本
    """
    n_dim = 96
    #n_samples = Fr_samples.shape[0]
    model = gp.Model("Worst_Case_Distributions")

    #n_perturbations = 200  # 生成1000个扰动样本集

    # 生成多个扰动样本集
    #perturbed_samples_list = [generate_distribution_in_ball(Fr_samples, epsilon, n_samples=1000)
    #for _ in range(n_perturbations)]
    
    # 1. 定义变量
    delta_W = model.addVars(n_dim, lb=-GRB.INFINITY, name="delta_W")  # 96维分布
    delta_PV = model.addVars(n_dim, lb=-GRB.INFINITY, name="delta_PV")  # 96维分布
    delta_sum = model.addVars(n_dim, lb=-GRB.INFINITY, name="delta_sum") # 96维分布

    model.addConstrs(
    (delta_sum[i] == delta_W[i] + delta_PV[i]
    for i in range(n_dim)),
    name="delta_sum_add"
    )

    C_VPPO_delta = 0

    # 大M值（根据问题规模调整）
    M = 1e6  # 足够大的常数
    is_nonnegative = model.addVars(n_dim, vtype=GRB.BINARY, name="is_nonnegative")

# 4. 使用大M法建模条件逻辑
    for i in range(n_dim):
    # 如果delta_sum[i] >= 0，则is_nonnegative[i] = 1
        model.addConstr(delta_sum[i] <= M * is_nonnegative[i], name=f"force_nonneg_{i}")
    
    # 如果delta_sum[i] < 0，则is_nonnegative[i] = 0
        model.addConstr(delta_sum[i] >= -M * (1 - is_nonnegative[i]), name=f"force_neg_{i}")
    
    # 成本计算
        #C_VPPO_delta += ((1 - is_nonnegative[i]) * lambda_VPPO_CDP_up[i] * delta_sum[i] 
        #            - is_nonnegative[i] * lambda_VPPO_CDP_down[i] * delta_sum[i])

    lage_W = model.addVar(lb=0, name="lage_W")  # 拉格朗日乘子
    lage_PV = model.addVar(lb=0, name="lage_PV")  # 拉格朗日乘子

    model.addConstr(lage_W <= 1e5, name="lagrange_W_bound") 
    model.addConstr(lage_PV <= 1e5, name="lagrange_PV_bound")
    
    # 2. 定义辅助变量（用于线性化Wasserstein约束）
    transport_cost_W = model.addVars(n_dim, lb = 0, ub = 2, name="transport_cost_W")  # W(delta_W, Fr)
    transport_cost_PV = model.addVars(n_dim, lb = 0, ub = 2, name="transport_cost_PV")  # W(delta_PV, Fr)
    
    # 3. 目标函数：最大化成本 - lambda1*(transport_cost1 - epsilon1) - lambda2*(transport_cost2 - epsilon2)
    model.setObjective(
        gp.quicksum(p_max * delta_W[i] + p_max * delta_PV[i]  
        + ((1 - is_nonnegative[i]) * lambda_VPPO_CDP_up[i] * delta_sum[i] 
        - is_nonnegative[i] * lambda_VPPO_CDP_down[i] * delta_sum[i])
        - lage_W * (gp.quicksum(transport_cost_W) - epsilon) 
        - lage_PV * (gp.quicksum(transport_cost_PV) - epsilon) for i in range(n_dim)),
        GRB.MAXIMIZE
    )

    for i in range(28):
    # 使用样本的第i%1000个分位数(循环使用样本)
        model.addConstr(delta_PV[i] == 0, name="NO_SUN_1")
    for i in range(89, 96):
        model.addConstr(delta_PV[i] == 0, name="NO_SUN_2")
    
# 对于1D情况，可以用排序后的分位数约束
    sorted_Fr = np.sort(Fr_samples)

# 近似约束：每个 delta_W[i] 必须在 [F^{-1}(alpha), F^{-1}(1-alpha)] 范围内
# 其中 alpha 取决于 epsilon（这里简化处理）
    alpha = epsilon
    lower_bound = sorted_Fr[int(alpha * len(sorted_Fr))]
    upper_bound = sorted_Fr[int((1 - alpha) * len(sorted_Fr))]

    model.addConstrs(delta_W[i] >= lower_bound for i in range(n_dim))
    model.addConstrs(delta_W[i] <= upper_bound for i in range(n_dim))

    model.addConstrs(delta_PV[i] >= lower_bound for i in range(n_dim))
    model.addConstrs(delta_PV[i] <= upper_bound for i in range(n_dim))

    # 计算排序后的Fr样本(已排序)
    sorted_Fr = np.sort(Fr_samples)

# 确保transport_cost与delta变量的关系
    for i in range(n_dim):
    # 使用样本的第i%1000个分位数(循环使用样本)
        fr_val = sorted_Fr[i % len(sorted_Fr)]
        model.addConstr(transport_cost_W[i] >= 0.1 * (delta_W[i] - fr_val), name=f"transport_W_pos_{i}")
        model.addConstr(transport_cost_W[i] >= 0.1 * (fr_val - delta_W[i]), name=f"transport_W_neg_{i}")
    
    # 同样约束PV
        model.addConstr(transport_cost_PV[i] >= 0.1 * (delta_PV[i] - fr_val), name=f"transport_PV_pos_{i}")
        model.addConstr(transport_cost_PV[i] >= 0.1 * (fr_val - delta_PV[i]), name=f"transport_PV_neg_{i}")
    
    # 5. 求解
    model.Params.NonConvex = 2 
    model.optimize()

    #model.computeIIS()
    #model.write("modelwasser.ilp")
    
    if model.status == GRB.OPTIMAL:
        print("球求解成功")
        delta1_opt = np.array([delta_W[i].X for i in range(n_dim)])
        delta2_opt = np.array([delta_PV[i].X for i in range(n_dim)])
        P_delta_opt = np.array([delta_sum[i].X for i in range(n_dim)])
        nonneg_index = np.array([is_nonnegative[i].X for i in range(n_dim)])
        worst_cost = model.ObjVal
        return delta1_opt, delta2_opt, P_delta_opt, worst_cost, nonneg_index
    
    if model.status == GRB.Status.INFEASIBLE:
        print("模型不可行")
    elif model.status == GRB.Status.UNBOUNDED:
        print("模型无界")
    elif model.status == GRB.Status.INF_OR_UNBD:
        print("模型不可行或无界")
    else:
        print(f"优化失败，状态码: {model.status}")

def find_worst_case_distributions(Fr_samples, lambda_VPPO_CDP_up, lambda_VPPO_CDP_down, epsilon, p_max):
    n_dim = 96
    
    m = gp.Model("worst_case_distribution")
    
    # 决策变量 - 添加合理的上下界
    delta_W = m.addVars(n_dim, lb=-1000, ub=1000, name="delta_W")
    delta_PV = m.addVars(n_dim, lb=-1000, ub=1000, name="delta_PV")
    delta_sum = m.addVars(n_dim, lb=-2000, ub=2000, name="delta_sum")
    
    # 对偶变量
    lage_W = m.addVar(lb=0, ub=100, name="lambda_W")
    lage_PV = m.addVar(lb=0, ub=100, name="lambda_PV")
    
    # 二进制变量
    is_nonnegative = m.addVars(n_dim, vtype=GRB.BINARY, name="is_nonnegative")
    
    # 运输成本变量 - 使用绝对值约束
    transport_cost_W = m.addVars(n_dim, lb=0, ub=1000, name="transport_cost_W")
    transport_cost_PV = m.addVars(n_dim, lb=0, ub=1000, name="transport_cost_PV")
    
    # 定义 delta_sum
    for i in range(n_dim):
        m.addConstr(delta_sum[i] == delta_W[i] + delta_PV[i], name=f"delta_sum_constr_{i}")
    
    # 修正运输成本约束 - 使用绝对值
    for i in range(n_dim):
        # transport_cost_W[i] >= |delta_W[i]|
        m.addConstr(transport_cost_W[i] >= delta_W[i], name=f"transport_cost_W_pos_{i}")
        m.addConstr(transport_cost_W[i] >= -delta_W[i], name=f"transport_cost_W_neg_{i}")
        
        # transport_cost_PV[i] >= |delta_PV[i]|
        m.addConstr(transport_cost_PV[i] >= delta_PV[i], name=f"transport_cost_PV_pos_{i}")
        m.addConstr(transport_cost_PV[i] >= -delta_PV[i], name=f"transport_cost_PV_neg_{i}")
    
    # 修正大M法约束
    M = 1000  # 减小M值
    for i in range(n_dim):
        # 如果 is_nonnegative[i] = 1, 则 delta_sum[i] >= 0
        m.addConstr(delta_sum[i] >= -M * (1 - is_nonnegative[i]), name=f"nonneg_constr1_{i}")
        
        # 如果 is_nonnegative[i] = 0, 则 delta_sum[i] <= -1e-6
        m.addConstr(delta_sum[i] <= M * is_nonnegative[i] - 1e-6, name=f"nonneg_constr2_{i}")
    
    # 添加Wasserstein球约束
    m.addConstr(gp.quicksum(transport_cost_W[i] for i in range(n_dim)) <= epsilon, name="wasserstein_W")
    m.addConstr(gp.quicksum(transport_cost_PV[i] for i in range(n_dim)) <= epsilon, name="wasserstein_PV")
    
    transport_cost_W_sum = gp.quicksum(transport_cost_W[i] for i in range(n_dim))
    transport_cost_PV_sum = gp.quicksum(transport_cost_PV[i] for i in range(n_dim))    

    # 目标函数
    objective = gp.quicksum(
        p_max * delta_W[i] + p_max * delta_PV[i] + 
        ((1 - is_nonnegative[i]) * lambda_VPPO_CDP_up[i] - 
         is_nonnegative[i] * lambda_VPPO_CDP_down[i]) * delta_sum[i]
        for i in range(n_dim)) - lage_W * (transport_cost_W_sum - epsilon) - lage_PV * (transport_cost_PV_sum - epsilon)
    
    m.setObjective(objective, GRB.MAXIMIZE)
    
    # 求解参数
    m.setParam("OutputFlag", 1)  # 打开输出以查看详细信息
    m.setParam("NonConvex", 2)
    m.setParam("FeasibilityTol", 1e-6)
    m.setParam("OptimalityTol", 1e-6)
    
    # 尝试求解
    try:
        m.optimize()
    except Exception as e:
        print(f"求解异常: {e}")
        return None, None, None, None, None
    
    # 检查求解状态
    if m.status == GRB.OPTIMAL:
        delta_W_values = [delta_W[i].x for i in range(n_dim)]
        delta_PV_values = [delta_PV[i].x for i in range(n_dim)]
        delta_sum_values = [delta_sum[i].x for i in range(n_dim)]
        worst_cost = m.objVal
        
        nonneg_index = [1 if delta_sum_values[i] >= 0 else 0 for i in range(n_dim)]
        
        print(f"最坏情况分布求解成功，目标值: {worst_cost:.4f}")
        return delta_W_values, delta_PV_values, delta_sum_values, worst_cost, nonneg_index
    
    else:
        print(f"最坏情况分布求解失败，状态码: {m.status}")
        return None, None, None, None, None
    
