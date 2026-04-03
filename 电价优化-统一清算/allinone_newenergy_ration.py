import gurobipy as gp
from gurobipy import GRB, quicksum
from Wasserstein import *
import numpy as np
import matplotlib.pyplot as plt
from PV_96 import *
from W_96_平滑性惩罚项 import *
import random

# 1、CDP与VPPO变为火电与电网公司
# ——整体的模型描述
# 2、改进工作一
# （1）高比率渗透，光伏与风的比率约束，光伏与风的最大最小约束。
# （2）需求约束，外生需求。不同需求下系统分配
# （3）新能源的不确定性

# 主程序：main_iteration
# 子程序：CDP_op, ES_op, Wasserstein

# ---------确定全局基本参数--------------- 注释
# 设定时间分段
n = 96
# 根据时间段设置变量维度
time_slots = range(n)

# -------------------开始设置wasserstein球-----------------------------
# P_delta = np.random.choice(F1_samples, size=96, replace=True).reshape(4, 96)
# 设定基本误差分布
np.random.seed(42)
r = 1000
partial = 0.95

# 设置经验分布
Fr_samples = np.random.normal(loc=0, scale=0.5, size=1000)  # 均值为0，标准差为1
Fr_mean = np.mean(Fr_samples)
Fr_center = Fr_samples - Fr_mean
Fr_center_mean = np.mean(Fr_center ** 2)

optimal_rho, K = find_optimal_rho(Fr_center_mean)
epsilon = compute_eplsilon(K, partial, r)

# 检验是否构造了了一个Wasserstein球
in_ball = wasserstein_ball(Fr_samples, epsilon)
print(f"Test distribution is within Wasserstein ball (ε={epsilon}): {in_ball}")

# 生成分布 F1
F1_samples = generate_distribution_in_ball(Fr_samples, epsilon)

# 验证是否满足 W(Fr, F1) ≤ ε
w_dist_F1 = wasserstein_distance(Fr_samples.flatten(), F1_samples.flatten())
print(f"W(Fr, F1) = {w_dist_F1:.4f} ≤ ε? {w_dist_F1 <= epsilon}")

P_TP_path = './regression_data/TP.csv'
P_W_path = './regression_data/W.csv'
P_PV_path = './regression_data/PV.csv'
bids_W_path = './regression_data/lambda_W.csv'
bids_PV_path = './regression_data/lambda_PV.csv'
demand_path = './regression_data/demand.csv'

P_TP = pd.read_csv(P_TP_path)
P_W = pd.read_csv(P_W_path)
P_PV = pd.read_csv(P_PV_path)

lambda_clear_W = pd.read_csv(bids_W_path)
lambda_clear_PV = pd.read_csv(bids_PV_path)
demand = pd.read_csv(demand_path)

P_TP_base = 96 * [5]
P_demand = [13.1, 12.8, 12.4, 12, 11.7, 11.4, 11.1, 10.8, 10.5, 10.2, 9.7, 9.2, 8.7, 9.1, 9.3, 9.5, 9.7, 9.7, 9.7, 9.7,
            9.7, 10, 10.2, 10.5, 10.7, 11, 11.2, 11.5, 11.7, 12.2, 12.7, 13.2, 12.7, 13.7, 14.7, 15.7, 14.7, 15.7, 16.7,
            17.7, 17.2, 18.2, 19.2, 20.2, 19.2, 18.2, 17.7, 17.2, 16.7, 16.2, 15.7, 15.2, 14.7, 14.7, 14.7, 14.7, 14.7,
            15, 15.2, 15.5, 15.7, 16.2, 16.7, 17.2, 17.7, 18.2, 18.7, 19.2, 19.7, 21, 22.2, 23.5, 24.7, 24.7, 24.7,
            24.7, 24.7, 22.5, 22.2, 22, 21.7, 21.2, 20.7, 20.2, 19.7, 19.2, 18.7, 18.2, 17.7, 17.2, 16.7, 16.2, 15.7,
            15.2, 14.7, 14.2]
P_demand = np.subtract(P_demand, P_TP_base)
P_demand_all = (1500 - np.sum(P_TP_base))

df = pd.read_csv('price_income.csv')

ration = 0.6

# 计算风电和光伏预测出力
total_demand = np.sum(P_demand)
# print(total_demand)

P_W_portion = df['W']
total_W_portioin = np.sum(P_W_portion)
P_W_pre = ration * P_W_portion / total_W_portioin * total_demand
# print(P_W_pre)

P_PV_portion = df['PV']
total_PV_portioin = np.sum(P_PV_portion)
P_PV_pre = ration * P_PV_portion / total_PV_portioin * total_demand
# print(P_PV_pre)

t = 0
predictions_W = []
predictions_PV = []


def compute_p_deltas(delta_W_0, delta_PV_0):
    """
    计算 P_delta[1] 到 P_delta[4]，每个 P_delta[i] = delta_W_0 + delta_PV_0

    Args:
        delta_W_0 (list): 96 维分布
        delta_PV_0 (list): 96 维分布
        n (int): 生成 P_delta 的数量（默认为 1）

    Returns:
        dict: {1: P_delta_1, 2: P_delta_2, ..., n: P_delta_n}
    """
    # 计算 delta_W_0 + delta_PV_0
    sum_distribution = np.array(delta_W_0) + np.array(delta_PV_0)

    # 生成 n 个相同的 P_delta
    p_deltas = sum_distribution.tolist()

    return p_deltas


# 各个CDP所需要平衡
# P_delta = compute_p_deltas(delta_W_0, delta_PV_0)

# P_TP_base = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
P_TP_base = 96 * [5]
# P_TP_base = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


# 设定CDP的基本参数---------------
# 最大功率
P_CDP_max = 25
# 最小功率
P_CDP_min = 4
# 最大功率爬升
P_CDP_down = - 5
# 最小功率爬升
P_CDP_up = 5
# 运营成本系数
a = 0.008
b = 0.017
c = 300
# -----------------------------------

# 设定ES的基本参数--------------------
# ES的最大充电功率
P_ES_c_max = 2
# ES的最大放电功率
P_ES_d_max = 2
# ES最大容量
E_ES_max = 4 * 0.9
# ES最小容量
E_ES_min = 4 * 0.1
# ES每小时运行成本
lambda_ES = 150
# --------------------------

# ------------------------------------
# PRM的最低中标能力
P_PRM_min = 0
# PRM的最大削峰能力
P_PRM_rp_max = 75
# PRM的最大填谷能力
P_PRM_rv_max = 75
# 设定电力现货市场出清价格
lambda_EM_clear = [400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400,
                   400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 488, 575, 663, 750, 656, 563, 469, 375,
                   356, 338, 319, 300, 290, 280, 270, 260, 258, 255, 253, 250, 238, 225, 213, 200, 188, 175, 163, 150,
                   300, 450, 600, 750, 863, 975, 1088, 1200, 1125, 1050, 975, 900, 863, 825, 788, 750, 713, 675, 638,
                   600, 588, 575, 563, 550, 613, 675, 738, 800, 800, 800, 800, 800, 788, 775, 763, 750, 663, 575, 488,
                   400]
# W直接上网电价，变成火电价
lambda_G_sell_W = 300
# PV直接上网电价
lambda_G_sell_PV = 300
# 设置CDP价格约束
lambda_CDP_min = 0
lambda_G_buy_VPPO = [375, 375, 375, 375, 375, 375, 375, 375, 375, 375, 375, 375, 375, 375, 375, 375, 375, 375, 375, 375,
                     375, 375, 375, 375, 375, 375, 375, 375, 506, 638, 769, 900, 900, 900, 900, 900, 900, 900, 900, 900,
                     1013, 1125, 1238, 1350, 1350, 1350, 1350, 1350, 1238, 1125, 1013, 900, 900, 900, 900, 900, 900,
                     900, 900, 900, 1050, 1200, 1350, 1500, 1500, 1500, 1500, 1500, 1463, 1425, 1388, 1350, 1350, 1350,
                     1350, 1350, 1350, 1350, 1350, 1350, 1350, 1350, 1350, 1350, 1238, 1125, 1013, 900, 900, 900, 900,
                     900, 769, 638, 506, 375]

# 设置柔性符合价格约束--------------------------------
# PRM削峰价格限制
lambda_PRM_sell_rp_min = 100
lambda_PRM_sell_rp_max = 300
# PRM平谷价格限制
lambda_PRM_sell_rv_min = 100
lambda_PRM_sell_rv_max = 200

# 设置储能价格约束
# PRM削峰价格限制
lambda_PRM_sell_rp_min = 50
lambda_PRM_sell_rp_max = 300
# PRM平谷价格限制
lambda_PRM_sell_rv_min = 50
lambda_PRM_sell_rv_max = 400

# 给定一天内的居民用电基准情况------------------------------
P_demand = [13.1, 12.8, 12.4, 12, 11.7, 11.4, 11.1, 10.8, 10.5, 10.2, 9.7, 9.2, 8.7, 9.1, 9.3, 9.5, 9.7, 9.7, 9.7, 9.7,
            9.7, 10, 10.2, 10.5, 10.7, 11, 11.2, 11.5, 11.7, 12.2, 12.7, 13.2, 12.7, 13.7, 14.7, 15.7, 14.7, 15.7, 16.7,
            17.7, 17.2, 18.2, 19.2, 20.2, 19.2, 18.2, 17.7, 17.2, 16.7, 16.2, 15.7, 15.2, 14.7, 14.7, 14.7, 14.7, 14.7,
            15, 15.2, 15.5, 15.7, 16.2, 16.7, 17.2, 17.7, 18.2, 18.7, 19.2, 19.7, 21, 22.2, 23.5, 24.7, 24.7, 24.7,
            24.7, 24.7, 22.5, 22.2, 22, 21.7, 21.2, 20.7, 20.2, 19.7, 19.2, 18.7, 18.2, 17.7, 17.2, 16.7, 16.2, 15.7,
            15.2, 14.7, 14.2]
P_demand = np.subtract(P_demand, P_TP_base)
P_demand_all = 1500 - np.sum(P_TP_base)
# P_demand = [14, 13.5, 13, 12.5, 12, 12.5, 14, 13.5, 15, 17, 16, 19, 19, 20.5, 17, 17, 16, 17.5, 18, 18, 18.5, 17, 16.5, 15]
# P_demand = [7, 7.5, 7, 6, 6, 6.5, 7, 6.5, 7.5, 8.5, 8, 9.5, 9.5, 10.5, 9, 9, 8, 9, 9, 9, 9.5, 8.5, 8, 7.5]
# 给出新能源发电占比
# R = 0.5
# -----------------------------------

# 设定主博弈价格初始值（应由最大值或中位数确定）--------------
# VPPO向CDP提供的初始价格
lambda_VPPO_CDP = 96 * [350]
# VPPO向CDP提供的初始向上补偿价格
lambda_VPPO_CDP_up = 96 * [500]
# VPPO向CDP提供的初始向下补偿价格
lambda_VPPO_CDP_down = 96 * [150]
# VPPO向ES提供的初始放电价格
lambda_VPPO_ES_d = 96 * [300]
# VPPO向ES提供的初始充电价格
lambda_VPPO_ES_c = 96 * [210]
# --------------------------

# W运营成本系数----------------------
W_1 = 0.01
W_2 = 0.02
W_3 = 200

# PV运营成本系数----------------------
PV_1 = 0.01
PV_2 = 0.02
PV_3 = 150

p_max = 300


# 风电优化子程序
def WPV_sort(demand, R, p_max):
    n_periods = len(demand)

    results_W, delta_W_0 = ElectricityMarket_W.main_optimization_W(R, p_max, demand)

    results_PV, delta_PV_0 = ElectricityMarket_PV.main_optimization_PV(R, p_max, demand)

    # 直接访问您需要的变量
    if results_W['wind']['success']:
        P_W_act = results_W['wind']['P_W_act']
        bids_W_act = results_W['wind']['bids_W_act']

    if results_PV['solar']['success']:
        P_PV_act = results_PV['solar']['P_PV_act']
        bids_PV_act = results_PV['solar']['bids_PV_act']

    allocation_result = []
    demand_lo = []
    total_allocated = 0

    for t in range(n_periods):
        # 当前时段的参数
        D_t = demand[t]

        P_W_t = P_W_act[t]
        bid_W_t = bids_W_act[t]

        P_PV_t = P_PV_act[t]
        bid_PV_t = bids_PV_act[t]

        # 创建供应商列表 (报价, 电量, 类型, 时段)
        suppliers = [
            (bid_W_t, P_W_t, 'wind', t),
            (bid_PV_t, P_PV_t, 'solar', t)
        ]

        # 按报价排序，如果报价相同则随机决定顺序
        if bid_W_t == bid_PV_t:
            # 50%的概率交换顺序
            if random.random() < 0.5:
                suppliers = [suppliers[1], suppliers[0]]  # 交换位置

        # 按报价排序（报价低的优先）
        suppliers_sorted = sorted(suppliers, key=lambda x: x[0])
        # print(suppliers_sorted)

        # 开始分配，只分配需求范围内的部分
        remaining_demand = D_t
        Q_cum = 0
        period_allocations = []

        for bid, capacity, supplier_type, period in suppliers_sorted:
            alloc = min(capacity, remaining_demand)
            Q_cum += alloc
            p_bid = bid

        if Q_cum < D_t / 2:
            p_clear = p_max
        if D_t / 2 <= Q_cum <= 4 * D_t / 5:
            p_clear = 4 * p_max / 5
        if Q_cum > 4 * D_t / 5:
            p_clear = p_max / 2

        p_setting = (p_bid + p_clear) / 2

        Q_cum = 0
        for bid, capacity, supplier_type, period in suppliers_sorted:
            alloc = min(capacity, remaining_demand)
            Q_cum += alloc

            period_allocations.append({
                'supplier': supplier_type,
                'bid_price': bid,
                'allocated': alloc,
                'p_setting': p_setting
            })

            remaining_demand = remaining_demand - alloc
            total_allocated += alloc

        demand_lo.append(max(0, remaining_demand))

        # 记录分配信息
        allocation_info = {
            'period': t,
            'demand': D_t,
            'total_available': P_W_t + P_PV_t,
            'excess': max(0, (P_W_t + P_PV_t) - D_t),  # 超出需求的部分
            'allocations': period_allocations
        }

        allocation_result.append(allocation_info)

    wind_bids = []
    wind_allocs = []
    p_setting_W = []

    solar_bids = []
    solar_allocs = []
    p_setting_PV = []

    for period_info in allocation_result:
        # 使用列表推导式提取数据
        wind_alloc = next((alloc for alloc in period_info['allocations'] if alloc['supplier'] == 'wind'), None)
        solar_alloc = next((alloc for alloc in period_info['allocations'] if alloc['supplier'] == 'solar'), None)

        wind_bids.append(wind_alloc['bid_price'] if wind_alloc else 0)
        wind_allocs.append(wind_alloc['allocated'] if wind_alloc else 0)
        p_setting_W.append(wind_alloc['p_setting'] if wind_alloc else 0)

        solar_bids.append(solar_alloc['bid_price'] if solar_alloc else 0)
        solar_allocs.append(solar_alloc['allocated'] if solar_alloc else 0)
        p_setting_PV.append(solar_alloc['p_setting'] if solar_alloc else 0)

    return wind_allocs, solar_allocs, wind_bids, solar_bids, demand_lo, delta_PV_0, delta_W_0, allocation_result, p_setting_W, p_setting_PV


# 火和储能优化子程序-----------------------------------
CDP_ES_profit_all = []


def CDP_ES(n, time_slots, P_ES_c_max, P_ES_d_max, E_ES_max, E_ES_min, lambda_ES,
           lambda_VPPO_ES_c, lambda_VPPO_ES_d, lambda_VPPO_CDP,
           lambda_VPPO_CDP_up, lambda_VPPO_CDP_down, P_CDP_max, P_CDP_min,
           P_CDP_down, P_CDP_up, a, b, c, P_delta, P_W, P_PV, P_deltaS, P_demand):
    m = gp.Model("CDP_ES_slave")
    m.setParam("NonConvex", 2)
    # m.setParam('MIPGap', 0.011)  # 2.5%的间隙容忍度
    # m.setParam('TimeLimit', 300)  # 5分钟时间限制
    m.setParam('OutputFlag', 1)

    # 1. 正确定义变量
    # ES变量
    P_ES_c = m.addVars(4, n, lb=0, ub=P_ES_c_max, name="P_ES_c")
    P_ES_d = m.addVars(4, n, lb=0, ub=P_ES_d_max, name="P_ES_d")

    P_ES_c_all = m.addVars(n, name="P_ES_c_all")
    P_ES_d_all = m.addVars(n, name="P_ES_d_all")

    E_ES = m.addVars(4, n + 1, lb=E_ES_min, ub=E_ES_max, name="E_ES")

    # CDP变量
    P_CDP_act = m.addVars(n, lb=P_CDP_min, ub=P_CDP_max, name="P_CDP")

    TP_load_pay = m.addVars(n, lb=0, name="P_PAY")

    # 二进制变量用于条件逻辑
    is_nonnegative = m.addVars(n, vtype=GRB.BINARY, name="is_nonnegative")
    u_ES = m.addVars(4, n, vtype=GRB.BINARY, name="u_ES")  # 充放电互斥

    # 2. ES约束
    # 初始和最终能量状态
    for j in range(4):
        m.addConstr(E_ES[j, 0] == 1.2, name=f"E_ES_{j}_init")
        m.addConstr(E_ES[j, n] == 1.2, name=f"E_ES_{j}_final")

    # 能量动态
    for j in range(4):
        for i in range(n):
            m.addConstr(E_ES[j, i + 1] == E_ES[j, i] + 0.95 * P_ES_c[j, i] - P_ES_d[j, i] / 0.95,
                        name=f"E_ES_{j}_{i}_dynamics")

    # 3. 充放电互斥约束（使用大M法）
    M = 1000
    for j in range(4):
        for i in range(n):
            m.addConstr(P_ES_c[j, i] <= M * u_ES[j, i], name=f"charge_active_{j}_{i}")
            m.addConstr(P_ES_d[j, i] <= M * (1 - u_ES[j, i]), name=f"discharge_active_{j}_{i}")

    # 4. CDP约束
    # 爬坡约束
    for i in range(1, n):
        m.addConstr(P_CDP_act[i] - P_CDP_act[i - 1] <= P_CDP_up, name=f"P_CDP_up_{i}")
        m.addConstr(P_CDP_act[i] - P_CDP_act[i - 1] >= P_CDP_down, name=f"P_CDP_down_{i}")

    # 5. 功率平衡约束（修正版）
    for i in range(1, n):
        m.addConstr(P_ES_c_all[i] == quicksum(
            P_ES_c[j, i]
            for j in range(4)
        ), name=f"P_ES_c_all{i}")
        m.addConstr(P_ES_d_all[i] == quicksum(
            P_ES_d[j, i]
            for j in range(4)
        ), name=f"P_ES_d_all{i}")
        m.addConstr(P_ES_c_all[i] * P_ES_d_all[i] <= 1e-8, name=f"no_coexist")

    for i in range(n):
        total_ES_discharge = quicksum(P_ES_d[j, i] for j in range(4))
        total_ES_charge = quicksum(P_ES_c[j, i] for j in range(4))

        # 计算可再生能源出力
        renewable_power = P_W[i] + P_PV[i]

        # 计算净负荷（需要由CDP和ES满足的部分）
        net_load = max(P_demand[i] - renewable_power, 0)

        m.addConstr(
            P_CDP_act[i] + total_ES_discharge - total_ES_charge >= net_load + P_deltaS[i],
            name=f"balance_{i}"
        )

        expr = P_demand[i] + P_deltaS[i] - renewable_power - total_ES_discharge + total_ES_charge

        is_positive = m.addVar(vtype=gp.GRB.BINARY, name=f"is_positive_{i}")
        M = 10000
        m.addConstr(TP_load_pay[i] >= expr)
        m.addConstr(TP_load_pay[i] >= 0)
        m.addConstr(TP_load_pay[i] <= expr + M * (1 - is_positive))
        m.addConstr(TP_load_pay[i] <= M * is_positive)
        m.addConstr(expr <= M * is_positive - 1e-6)

    # 6. 目标函数组件
    # ES收益和成本
    ES_income = quicksum(
        - lambda_VPPO_ES_c[i] * quicksum(P_ES_c[j, i] for j in range(4)) +
        lambda_VPPO_ES_d[i] * quicksum(P_ES_d[j, i] for j in range(4))
        for i in range(n)
    )

    ES_cost = quicksum(
        lambda_ES * quicksum(P_ES_c[j, i] + P_ES_d[j, i] for j in range(4))
        for i in range(n)
    )

    # CDP收益和成本
    CDP_income = quicksum(lambda_VPPO_CDP[i] * TP_load_pay[i] for i in range(n))
    CDP_cost = quicksum(
        a * P_CDP_act[i] ** 2 + c * P_CDP_act[i] / 2 + b for i in range(n))  # 单位成本是不是太低了？这里应该展现的成本基本上只有300块，而不是300/MkW

    # 惩罚项（正确建模）
    penalty_terms = quicksum(
        is_nonnegative[i] * lambda_VPPO_CDP_down[i] * P_delta[i] -
        (1 - is_nonnegative[i]) * lambda_VPPO_CDP_up[i] * P_delta[i]
        for i in range(n)
    )

    # 关联二进制变量和条件
    M_penalty = 1e6
    for i in range(n):
        m.addConstr(P_delta[i] >= -M_penalty * (1 - is_nonnegative[i]), name=f"penalty_nonneg_{i}")
        m.addConstr(P_delta[i] <= M_penalty * is_nonnegative[i] - 1e-6, name=f"penalty_neg_{i}")

    # 7. 利润非负约束
    m.addConstr(CDP_income + penalty_terms - CDP_cost >= 1e-6, name="Profit_nneg")

    # 8. 设置目标函数
    total_profit = CDP_income + penalty_terms + ES_income - CDP_cost - ES_cost
    m.setObjective(total_profit, GRB.MAXIMIZE)

    # 9. 求解
    try:
        m.optimize()

        if m.status == GRB.OPTIMAL:
            CDP_ES_profit_all.append(m.ObjVal)
            # 提取结果
            P_ES_c_total = [sum(P_ES_c[j, i].x for j in range(4)) for i in range(n)]
            P_ES_d_total = [sum(P_ES_d[j, i].x for j in range(4)) for i in range(n)]
            P_CDP_values = [P_CDP_act[i].x for i in range(n)]

            return P_ES_c_total, P_ES_d_total, P_CDP_values, m.objVal

        elif m.status == GRB.INFEASIBLE:
            print("模型不可行，进行IIS分析")
            m.computeIIS()
            m.write("model_iis.ilp")
            return None, None, None, None

        else:
            print(f"优化失败，状态码: {m.status}")
            return None, None, None, None

    except Exception as e:
        print(f"求解异常: {e}")
        return None, None, None, None


# 给定一个空列表-----------------------

VPPO_profit = []


def Master(n, time_slots, lambda_VPPO_CDP_up, lambda_VPPO_CDP_down, P_W, P_PV, P_delta, P_CDP, P_ES_c, P_ES_d, bids_W,
           bids_PV, p_setting_W, p_setting_PV, record_profit=True):
    m = gp.Model("Master")
    m.setParam("NonConvex", 2)  # 允许非凸约束
    m.setParam("MIPGap", 0.015)  # 1.5% 间隙

    # 设定决策目标---------------------
    # CDP设施分别在EM和PRM的发电功率
    P_CDP_act = m.addVars(time_slots, lb=0, ub=GRB.INFINITY, name="P_CDP_act")

    # ES设施分别在EM和PRM的充放电功率
    P_ES_d_act = m.addVars(time_slots, lb=0, ub=GRB.INFINITY, name="P_ES_d")
    P_ES_c_act = m.addVars(time_slots, lb=0, ub=GRB.INFINITY, name="P_ES_c")

    # 设定主博弈价格变量--------------
    # VPPO向新能源提供的价格
    p_max_m = m.addVar(name="lambda_VPPO_W_PV")

    lb = 50  # 自定义计算下界的函数
    ub = 500  # 自定义计算上界的函数
    p_max_m.lb = lb
    p_max_m.ub = ub

    # VPPO向CDP提供的价格
    lambda_VPPO_CDP_m = m.addVars(time_slots, name="lambda_VPPO_CDP_m")
    for t in time_slots:
        lb = max(lambda_VPPO_CDP[t] - 20, lambda_CDP_min)  # 自定义计算下界的函数
        ub = min(lambda_VPPO_CDP[t] + 20, lambda_G_buy_VPPO[t])  # 自定义计算上界的函数
        lambda_VPPO_CDP_m[t].lb = lb
        lambda_VPPO_CDP_m[t].ub = ub

    # VPPO向CDP提供的向上补偿价格
    lambda_VPPO_CDP_up_m = m.addVars(time_slots, name="lambda_VPPO_CDP_up_m")
    for t in time_slots:
        # 使用与遗传变异相同的边界
        current_val = lambda_VPPO_CDP_up[t]
        lb = max(current_val - 20, 300)  # 与UP_MIN一致
        ub = min(current_val + 20, 500)  # 与UP_MAX一致
        lambda_VPPO_CDP_up_m[t].lb = lb
        lambda_VPPO_CDP_up_m[t].ub = ub

    # VPPO向CDP提供的向下补偿价格
    lambda_VPPO_CDP_down_m = m.addVars(time_slots, name="lambda_VPPO_CDP_down_m")
    for t in time_slots:
        current_val = lambda_VPPO_CDP_down[t]
        lb = max(current_val - 20, 100)  # 与DOWN_MIN一致
        ub = min(current_val + 20, 200)  # 与DOWN_MAX一致
        lambda_VPPO_CDP_down_m[t].lb = lb
        lambda_VPPO_CDP_down_m[t].ub = ub

    # VPPO向ES提供的放电价格
    lambda_VPPO_ES_d_m = m.addVars(time_slots, name="lambda_VPPO_ES_d_m")
    for t in time_slots:
        lb = max(lambda_VPPO_ES_d[t] - 20, lambda_PRM_sell_rp_min)  # 自定义计算下界的函数
        ub = min(lambda_VPPO_ES_d[t] + 20, lambda_PRM_sell_rp_max)  # 自定义计算上界的函数
        lambda_VPPO_ES_d_m[t].lb = lb
        lambda_VPPO_ES_d_m[t].ub = ub

    # VPPO向ES提供的充电价格
    lambda_VPPO_ES_c_m = m.addVars(time_slots, name="lambda_VPPO_ES_c_m")
    for t in time_slots:
        lb = max(lambda_VPPO_ES_c[t] - 20, lambda_PRM_sell_rv_min)  # 自定义计算下界的函数
        ub = min(lambda_VPPO_ES_c[t] + 20, lambda_PRM_sell_rv_max)  # 自定义计算上界的函数
        lambda_VPPO_ES_c_m[t].lb = lb
        lambda_VPPO_ES_c_m[t].ub = ub
    # --------------------------

    # 设定VPPO系统发电功率
    # 设定VPPO系统EM发电功率
    P_PGC = m.addVars(time_slots, lb=0, ub=GRB.INFINITY, name="P_PGC")

    m.addConstrs(
        (P_PGC[i] == P_W[i] + P_PV[i] + P_CDP_act[i] + P_ES_d_act[i] - P_ES_c_act[i]
         for i in range(n)),
        name="P_PGC_{i}"
    )

    m.addConstrs(
        (P_PGC[i] >= P_demand[i]
         for i in range(n)),
        name="P_demand"
    )

    # 设定EM运营收入
    C_PGC_income = 0
    C_PGC_income = quicksum(
        lambda_EM_clear[i] * P_demand[i]
        for i in range(n)
    )

    # 设定ES运营成本
    C_VPPO_ES_cost = 0
    C_VPPO_ES_cost = quicksum(
        - lambda_VPPO_ES_c_m[i] * P_ES_c_act[i] + lambda_VPPO_ES_d_m[i] * P_ES_d_act[i]
        for i in range(n)
    )

    m.addConstr(
        (C_VPPO_ES_cost >= 0),
        name="Profit_ES_nneg"
    )

    # 设定误差部分最差成本(丢给CDP)
    # delta_W, delta_PV, P_deltaS, worst_cost_1, nonneg_index = find_worst_case_distributions(Fr_samples, lambda_VPPO_CDP_up, lambda_VPPO_CDP_down, epsilon, p_max)

    C_VPPO_delta = 0
    for i in range(n):
        # 成本计算
        C_VPPO_delta += (-(1 - nonneg_index[i]) * lambda_VPPO_CDP_up_m[i] * P_deltaS[i]
                         + nonneg_index[i] * lambda_VPPO_CDP_down_m[i] * P_deltaS[i])

    # VPPO利用CDP平衡最差情况误差的成本
    worst_cost = gp.quicksum(p_max_m * delta_W[i] + p_max_m * delta_PV[i] for i in range(n)) + C_VPPO_delta

    # 设定W设施预期成本
    C_VPPO_W_cost = 0
    C_VPPO_W_cost = quicksum(
        p_setting_W[i] * P_W[i]
        for i in range(n)
    )

    # 设置风电收入为正
    m.addConstr(
        (C_VPPO_W_cost - 40 * gp.quicksum(P_W[i] for i in range(n)) >= 0),
        name="Profit_W_nneg"
    )

    # 设定PV设施预期成本
    C_VPPO_PV_cost = 0
    C_VPPO_PV_cost = quicksum(
        p_setting_PV[i] * P_PV[i]
        for i in range(n)
    )

    m.addConstr(
        (C_VPPO_PV_cost - 80 * gp.quicksum(P_PV[i] for i in range(n)) >= 0),
        name="Profit_PV_nneg"
    )

    # 设定CDP设施预期成本
    C_VPPO_CDP_cost = 0
    C_VPPO_CDP_cost = quicksum(
        lambda_VPPO_CDP_m[i] * P_CDP_act[i]
        for i in range(n)  # i=0,1,...,23
    )

    m.addConstr(
        (C_VPPO_CDP_cost >= quicksum(
            a * P_CDP_act[i] ** 2 + c * P_CDP_act[i] + b  # 替换b，c，使得火电成本提高
            for i in range(n)
        )),
        name="Profit_CDP_nneg"
    )

    # 设定优化目标--------------------------------
    m.setObjective(C_PGC_income - C_VPPO_W_cost - C_VPPO_PV_cost - C_VPPO_CDP_cost - C_VPPO_ES_cost - worst_cost,
                   GRB.MAXIMIZE)  #

    # 设定约束条件----------------------------------------
    # 设置发电恒等式
    # CDP
    m.addConstrs(
        (P_CDP_act[i] == P_CDP[i]
         for i in range(n)),
        name="P_CDP_STILL_{i}"
    )
    # ES放电
    m.addConstrs(
        (P_ES_d_act[i] == P_ES_d[i]
         for i in range(n)),
        name="P_ES_d_STILL_{i}"
    )
    # ES充电
    m.addConstrs(
        (P_ES_c_act[i] == P_ES_c[i]
         for i in range(n)),
        name="P_ES_c_STILL_{i}"
    )

    # VPPO从W购买电力的平均价格大于平均火电电价
    # m.addConstr(C_VPPO_W_cost >= I_G_sell_W_cost + 1e-6, "VPPO_W>SELL_W")

    # VPPO从PV购买电力的平均价格大于平均火电电价
    # m.addConstr(C_VPPO_PV_cost >= I_G_sell_PV_cost + 1e-6, "VPPO_PV>SELL_PV")

    # 处理限制，不限制+成本，反过来体现渗透率，最大功率的渗透率占比
    # m.addConstr(
    # ((quicksum(P_W[i] + P_PV[i] + P_delta[i] for i in range(n))) >= R * quicksum(P_W[i] + P_PV[i] + P_CDP[i] for i in range(n)) ),
    # name="P_WPV_ration"
    # )

    m.setParam("OutputFlag", 1)
    m.setParam("DualReductions", 0)
    # m.setParam("MIPGap", 0.015)  # 1.5% 间隙
    # m.setParam("TimeLimit", 300)

    # m.optimize()
    try:
        m.optimize()
        if m.status == GRB.OPTIMAL:
            if record_profit:
                VPPO_profit.append(m.ObjVal)
        else:
            print(f"模型未求得最优解，状态码为: {m.status}")
        if m.status == GRB.INFEASIBLE:
            print("模型不可行，导出IIS")
            m.computeIIS()
            m.write("model_infeasible.ilp")
            VPPO_profit.append(None)
    except Exception as e:
        print(f"Gurobi求解过程出现异常: {e}")
        VPPO_profit.append(None)

    P_PGC_values = []
    p_max_values = []
    lambda_VPPO_CDP_values = []
    lambda_VPPO_CDP_up_values = []
    lambda_VPPO_CDP_down_values = []
    lambda_VPPO_ES_d_values = []
    lambda_VPPO_ES_c_values = []

    PGC_profit = m.ObjVal

    # if m.status == GRB.INFEASIBLE:
    #    m.computeIIS()
    #    m.write("modelupdown.ilp")
    # m.computeIIS()
    # m.write("modelupdown.ilp")

    # VPPO_profit.append(m.ObjVal)

    for v in m.getVars():
        if v.varName.startswith("P_PGC"):
            P_PGC_values.append(v.x)
        if v.varName.startswith("lambda_VPPO_W_PV"):
            p_max_values.append(v.x)
        if v.varName.startswith("lambda_VPPO_CDP_m"):
            lambda_VPPO_CDP_values.append(v.x)
        if v.varName.startswith("lambda_VPPO_CDP_up_m"):
            lambda_VPPO_CDP_up_values.append(v.x)
        if v.varName.startswith("lambda_VPPO_CDP_down_m"):
            lambda_VPPO_CDP_down_values.append(v.x)
        if v.varName.startswith("lambda_VPPO_ES_d_m"):
            lambda_VPPO_ES_d_values.append(v.x)
        if v.varName.startswith("lambda_VPPO_ES_c_m"):
            lambda_VPPO_ES_c_values.append(v.x)

    if m.status == GRB.OPTIMAL:
        print("Optimization successful!")
        # 返回优化后的变量字典（包含 .X 属性）
        return PGC_profit, P_PGC_values, p_max, lambda_VPPO_CDP_values, lambda_VPPO_CDP_up_values, lambda_VPPO_CDP_down_values  # , lambda_VPPO_ES_d_values, lambda_VPPO_ES_c_values

    if m.status == GRB.Status.INFEASIBLE:
        print("模型不可行")
    elif m.status == GRB.Status.UNBOUNDED:
        print("模型无界")
    # elif m.status == GRB.Status.INF_OR_UNBD:
    #    print("模型不可行或无界")
    else:
        print(f"优化失败，状态码: {m.status}")


# 修改您的主迭代循环
P_PGC_all = []
W_Profit_all = []
PV_Profit_all = []
p_max_all = [100]
#

for R in range(80, 81):
    P_W, P_PV, bids_W, bids_PV, remain, delta_PV_0, delta_W_0, allocation_result, p_setting_W, p_setting_PV = WPV_sort(
        P_demand, R, p_max)
    P_delta = compute_p_deltas(delta_W_0, delta_PV_0)

    p_setting_W_current = np.array(p_setting_W)
    p_setting_PV_current = np.array(p_setting_PV)

    P_W_current = np.array(P_W)
    P_PV_current = np.array(P_PV)

    W_Profit = np.sum(p_setting_W_current * P_W_current)
    PV_Profit = np.sum(p_setting_PV_current * P_PV_current)

    W_Profit_all.append(W_Profit)
    PV_Profit_all.append(PV_Profit)

    delta_W, delta_PV, P_deltaS, worst_cost_1, nonneg_index = find_worst_case_distributions(
        Fr_samples, lambda_VPPO_CDP_up, lambda_VPPO_CDP_down, epsilon, p_max)

    P_deltaS = 96 * [3]

    P_ES_c, P_ES_d, P_CDP, CDP_ES_profit = CDP_ES(
        n, time_slots, P_ES_c_max, P_ES_d_max, E_ES_max, E_ES_min, lambda_ES,
        lambda_VPPO_ES_c, lambda_VPPO_ES_d, lambda_VPPO_CDP,
        lambda_VPPO_CDP_up, lambda_VPPO_CDP_down, P_CDP_max, P_CDP_min,
        P_CDP_down, P_CDP_up, a, b, c, P_delta, P_W, P_PV, P_deltaS, P_demand
    )

    iterations = 100

    for iters in range(iterations):
        print(f"\n=== 第 {iters + 1} 次迭代 ===")

        PGC_profit, P_PGC, p_max, lambda_VPPO_CDP, lambda_VPPO_CDP_up, lambda_VPPO_CDP_down = Master(
            n, time_slots, lambda_VPPO_CDP_up, lambda_VPPO_CDP_down,
            P_W, P_PV, P_delta, P_CDP, P_ES_c, P_ES_d, bids_W, bids_PV, p_setting_W, p_setting_PV, record_profit=True
        )
        # 更新其他子问题
        P_W, P_PV, bids_W, bids_PV, remain, delta_PV_0, delta_W_0, allocation_result, p_setting_W, p_setting_PV = WPV_sort(
            P_demand, R, p_max)

        p_setting_W_current = np.array(p_setting_W)
        p_setting_PV_current = np.array(p_setting_PV)

        P_W_current = np.array(P_W)
        P_PV_current = np.array(P_PV)

        W_Profit = np.sum(p_setting_W_current * P_W_current)
        PV_Profit = np.sum(p_setting_PV_current * P_PV_current)

        W_Profit_all.append(W_Profit)
        PV_Profit_all.append(PV_Profit)

        P_ES_c, P_ES_d, P_CDP, CDP_ES_profit = CDP_ES(
            n, time_slots, P_ES_c_max, P_ES_d_max, E_ES_max, E_ES_min, lambda_ES,
            lambda_VPPO_ES_c, lambda_VPPO_ES_d, lambda_VPPO_CDP,
            lambda_VPPO_CDP_up, lambda_VPPO_CDP_down, P_CDP_max, P_CDP_min,
            P_CDP_down, P_CDP_up, a, b, c, P_delta, P_W, P_PV, P_deltaS, P_demand
        )

        print(f"迭代 {iters} 完成")

        p_max_all.append(p_max)

    P_PGC_all.append(PGC_profit)

# print("P_PGC", P_PGC)
# print("P_ES_c", P_ES_c)
# print("P_ES_d", P_ES_d)
# print("lambda_VPPO_W", lambda_VPPO_W)
# print("lambda_VPPO_PV", lambda_VPPO_PV)
# print("CDP_profit", CDP_profit)
# print("ES_profit", ES_profit)

# plt.plot(range(iterations+1), W_profit_all, marker='o', linestyle='-', color='b')
# plt.xlabel("Iteration")
# plt.ylabel("Objective Value")
# plt.title("Convergence of W Profit")
# plt.grid(True)
# plt.show()

# plt.plot(range(iterations+1), PV_profit_all, marker='o', linestyle='-', color='b')
# plt.xlabel("Iteration")
# plt.ylabel("Objective Value")
# plt.title("Convergence of PV Profit")
# plt.grid(True)
# plt.show()

bids_W = np.array(bids_W)
p_setting_W = np.array(p_setting_W)

bids_PV = np.array(bids_PV)
p_setting_PV = np.array(p_setting_PV)

P_W = np.array(P_W)
P_PV = np.array(P_PV)

plt.plot(range(n), bids_W, marker='o', linestyle='-', color='b')
plt.xlabel("Quarter")
plt.ylabel("Objective Value")
plt.title("W Bid Price")
plt.grid(True)
plt.show()

plt.plot(range(n), bids_PV, marker='o', linestyle='-', color='b')
plt.xlabel("Quarter")
plt.ylabel("Objective Value")
plt.title("PV Bid Price")
plt.grid(True)
plt.show()

plt.plot(range(n), p_setting_W, marker='o', linestyle='-', color='b')
plt.xlabel("Quarter")
plt.ylabel("Objective Value")
plt.title("W Clear Price")
plt.grid(True)
plt.show()

plt.plot(range(n), p_setting_PV, marker='o', linestyle='-', color='b')
plt.xlabel("Quarter")
plt.ylabel("Objective Value")
plt.title("PV Clear Price")
plt.grid(True)
plt.show()

plt.plot(range(n), p_setting_W * P_W, marker='o', linestyle='-', color='b')
plt.xlabel("Quarter")
plt.ylabel("Objective Value")
plt.title("W Income per Quater")
plt.grid(True)
plt.show()

plt.plot(range(n), p_setting_PV * P_PV, marker='o', linestyle='-', color='b')
plt.xlabel("Quarter")
plt.ylabel("Objective Value")
plt.title("PV Income per Quater")
plt.grid(True)
plt.show()

plt.plot(range(iterations + 1), p_max_all, marker='o', linestyle='-', color='b')
plt.xlabel("Iteration")
plt.ylabel("Objective Value")
plt.title("Convergence of PGC Bid Price")
plt.grid(True)
plt.show()

plt.plot(range(iterations + 1), CDP_ES_profit_all, marker='o', linestyle='-', color='b')
plt.xlabel("Iteration")
plt.ylabel("Objective Value")
plt.title("Convergence of TP Profit")
plt.grid(True)
plt.show()

plt.plot(range(iterations), VPPO_profit, marker='o', linestyle='-', color='b')
plt.xlabel("Iteration")
plt.ylabel("Objective Value")
plt.title("Convergence of PGC Profit")
plt.grid(True)
plt.show()

plt.bar(range(n), P_W, color='b', alpha=0.7)
plt.xlabel("Quarter")
plt.ylabel("Objective Value")
plt.title("W Power")
plt.grid(True)
plt.show()

plt.bar(range(n), P_PV, color='b', alpha=0.7)
plt.xlabel("Quarter")
plt.ylabel("Objective Value")
plt.title("PV Power")
plt.grid(True)
plt.show()

plt.bar(range(n), P_ES_c, color='b', alpha=0.7)
plt.xlabel("Quarter")
plt.ylabel("Objective Value")
plt.title("ES_c Power")
plt.grid(True)
plt.show()

plt.bar(range(n), P_ES_d, color='b', alpha=0.7)
plt.xlabel("Quarter")
plt.ylabel("Objective Value")
plt.title("ES_d Power")
plt.grid(True)
plt.show()

plt.bar(range(n), P_CDP, color='b', alpha=0.7)
plt.xlabel("Quarter")
plt.ylabel("Objective Value")
plt.title("TP Power")
plt.grid(True)
plt.show()

P_W = np.array(P_W)
P_PV = np.array(P_PV)
P_ES_c = np.array(P_ES_c)
P_ES_d = np.array(P_ES_d)
P_CDP = np.array(P_CDP)

P_ES_c_negative = [-x for x in P_ES_c]

plt.figure(figsize=(15, 6))
dimensions = [f'Dim {i + 1}' for i in range(24)]  # x轴标签
x = np.arange(96)  # 24个维度的位置
width = 0.6  # 柱状图宽度（覆盖整个维度）

plt.bar(x, P_ES_c_negative, width=width, label='ES_c', color="#2ca02c", edgecolor='black')

# 绘制堆叠柱状图
plt.bar(x, P_TP_base, width=width, label='TP_BASE', color="#7effff", edgecolor='black')
plt.bar(x, P_W, width=width, label='W', color='#1f77b4', edgecolor='black', bottom=P_TP_base)
plt.bar(x, P_PV, width=width, label='PV', color='#ff7f0e', edgecolor='black', bottom=P_W + P_TP_base)
plt.bar(x, P_ES_d, width=width, label='ES_c', color='#2ca02c', edgecolor='black', bottom=P_TP_base + P_W + P_PV)
plt.bar(x, P_CDP, width=width, label='TP', color='#d62728', edgecolor='black', bottom=P_W + P_PV + P_ES_d + P_TP_base)
plt.plot(x, P_demand + P_TP_base, 'ro-', linewidth=2, markersize=8, label='Total Demand (Line)', color="#000000")
plt.plot(x, P_W + P_PV + P_ES_d + P_TP_base + P_CDP - P_ES_c, 'ro-', linewidth=2, markersize=8,
         label='Supply Demand (Line)', color="#F8FF38")

# 添加标题、标签和图例
plt.title('Stacked Bar Chart of Four 96 Quaters', fontsize=14)
plt.xlabel('Quater Hour', fontsize=12)
plt.ylabel('Power', fontsize=12)
plt.xticks(x, rotation=45)  # 设置x轴标签
plt.legend(loc='upper right')

# 显示网格线（可选）
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 自动调整布局
plt.tight_layout()
plt.show()

n_periods = 96

periods = list(range(n_periods))

# 为每个时段创建堆叠数据，保持分配顺序
wind_allocations = []
solar_allocations = []

for period_info in allocation_result:
    # 按照分配顺序提取数据
    wind_alloc = 0
    solar_alloc = 0

    for alloc in period_info['allocations']:
        if alloc['supplier'] == 'wind':
            wind_alloc = alloc['allocated']
        elif alloc['supplier'] == 'solar':
            solar_alloc = alloc['allocated']

    wind_allocations.append(wind_alloc)
    solar_allocations.append(solar_alloc)

# 绘制组合堆叠柱状图
plt.figure(figsize=(15, 8))
x = np.arange(n_periods)
width = 0.6

# 初始化底部数组
bottom_array = np.zeros(n_periods)

# 1. 首先堆叠 P_TP_base（最底层）
plt.bar(x, P_TP_base, width=width, label='TP_BASE', color="#7effff", edgecolor='black')
bottom_array += P_TP_base

# 2. 按照分配顺序堆叠 P_W 和 P_PV
for period_info in allocation_result:
    period = period_info['period']
    # 按照分配顺序绘制
    for alloc in period_info['allocations']:
        if alloc['supplier'] == 'wind':
            # 绘制风能分配部分
            wind_allocated = alloc['allocated']
            plt.bar(period, wind_allocated, width=width, bottom=bottom_array[period],
                    color='#1f77b4', label='W_allocated' if period == 0 else "",
                    edgecolor='black')
            bottom_array[period] += wind_allocated

        elif alloc['supplier'] == 'solar':
            # 绘制太阳能分配部分
            solar_allocated = alloc['allocated']
            plt.bar(period, solar_allocated, width=width, bottom=bottom_array[period],
                    color='#ff7f0e', label='PV_allocated' if period == 0 else "",
                    edgecolor='black')
            bottom_array[period] += solar_allocated

plt.bar(x, P_ES_c_negative, width=width, label='ES_c', color="#2ca02c", edgecolor='black')

plt.bar(x, P_ES_d, width=width, label='ES_c', color='#2ca02c', edgecolor='black', bottom=bottom_array)

# 3. 最后堆叠 P_CDP（最上层）
plt.bar(x, P_CDP, width=width, label='TP', color='#d62728', edgecolor='black',
        bottom=bottom_array + P_ES_d)

# 计算总供应量（用于线条绘制）
total_supply = P_TP_base + np.array(wind_allocations) + np.array(solar_allocations) + P_CDP - P_ES_c + P_ES_d

# 添加参考线条
plt.plot(x, P_demand + P_TP_base, 'k-', linewidth=2, markersize=4, label='Total Demand (Line)', color="#000000")
plt.plot(x, total_supply, 'y-', linewidth=2, markersize=4, label='Supply Demand (Line)', color="#F8FF38")

# 添加标题、标签和图例
plt.title('Power Allocation Stacked Bar Chart (96 Quarters)', fontsize=14)
plt.xlabel('Quarter Hour', fontsize=12)
plt.ylabel('Power', fontsize=12)
plt.xticks(x[::max(1, n_periods // 24)], rotation=45)  # 每24个时段显示一个标签
plt.legend(loc='upper right')

# 显示网格线
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 自动调整布局
plt.tight_layout()
plt.show()

# 准备共享的输入数据
df = pd.read_csv('price_income.csv')

# 计算风电和光伏预测出力
total_demand = np.sum(P_demand)
# print(total_demand)

P_W_portion = df['W']
total_W_portioin = np.sum(P_W_portion)
P_W_pre = 0.01 * R * P_W_portion / total_W_portioin * total_demand
# print(P_W_pre)

P_PV_portion = df['PV']
total_PV_portioin = np.sum(P_PV_portion)
P_PV_pre = 0.01 * R * P_PV_portion / total_PV_portioin * total_demand
# print(P_PV_pre)

plt.figure(figsize=(15, 6))
dimensions = [f'Dim {i + 1}' for i in range(24)]  # x轴标签
x = np.arange(96)  # 24个维度的位置
width = 0.6  # 柱状图宽度（覆盖整个维度）

# 绘制堆叠柱状图
plt.bar(x, P_W, width=width, label='W', color="#3B66DC", edgecolor='black')
plt.bar(x, P_W_pre + delta_PV_0 - P_W, width=width, label='W', color="#9B9B9B", edgecolor='black', bottom=P_W)

# 添加标题、标签和图例
plt.title('W Expile Power in 96 Quaters', fontsize=14)
plt.xlabel('Quater Hour', fontsize=12)
plt.ylabel('Power', fontsize=12)
plt.xticks(x, rotation=45)  # 设置x轴标签
plt.legend(loc='upper right')

# 显示网格线（可选）
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 自动调整布局
plt.tight_layout()
plt.show()

plt.figure(figsize=(15, 6))
dimensions = [f'Dim {i + 1}' for i in range(24)]  # x轴标签
x = np.arange(96)  # 24个维度的位置
width = 0.6  # 柱状图宽度（覆盖整个维度）

# 绘制堆叠柱状图
plt.bar(x, P_PV, width=width, label='W', color="#FF9900", edgecolor='black')
plt.bar(x, P_PV_pre + delta_PV_0 - P_PV, width=width, label='W', color="#9B9B9B", edgecolor='black', bottom=P_PV)

# 添加标题、标签和图例
plt.title('PV Expile Power in 96 Quaters', fontsize=14)
plt.xlabel('Quater Hour', fontsize=12)
plt.ylabel('Power', fontsize=12)
plt.xticks(x, rotation=45)  # 设置x轴标签
plt.legend(loc='upper right')

# 显示网格线（可选）
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 自动调整布局
plt.tight_layout()
plt.show()

# 涨跌幅度
PGC_Percent = VPPO_profit[iterations - 1] / VPPO_profit[0]
CDP_ES_Percent = CDP_ES_profit_all[iterations] / CDP_ES_profit_all[0]
W_Percent = W_Profit_all[iterations] / W_Profit_all[0]
PV_Percent = PV_Profit_all[iterations] / PV_Profit_all[0]

data = {
    '指标名称': ['PGC_Percent', 'CDP_ES_Percent', 'W_Percent', 'PV_Percent'],
    '百分比值': [PGC_Percent, CDP_ES_Percent, W_Percent, PV_Percent],
    '原始值': [VPPO_profit[iterations - 1], CDP_ES_profit_all[iterations],
               W_Profit_all[iterations], PV_Profit_all[iterations]],
    '初始值': [VPPO_profit[0], CDP_ES_profit_all[0],
               W_Profit_all[0], PV_Profit_all[0]]
}

df = pd.DataFrame(data)
df.to_csv('涨跌幅度数据.csv', index=False, encoding='utf-8-sig')
print("数据已导出到 '涨跌幅度数据.csv'")



