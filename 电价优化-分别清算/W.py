import numpy as np
from scipy.optimize import minimize
import pandas as pd
from competition import PV_price_predict
from Wasserstein import *

class ElectricityMarket:
    def __init__(self, demand, P_W_pre, P_PV_pre, bids_PV, delta_W_0):

        self.demand = np.array(demand)
        self.P_W_pre = np.array(P_W_pre)
        self.P_PV_pre = np.array(P_PV_pre)
        self.n_periods = len(demand)
        self.bid_PV = np.array(bids_PV)
        
        # 定义风电实际出力范围（预测值+Wasserstein随机向量）
        self.P_W_act = self.P_W_pre + delta_W_0
        
        # 假设光伏是确定性的（实际等于预测）
        self.P_PV_act = self.P_PV_pre.copy()
    
    def calculate_profit(self, P_W_bid, bids_W, bids_PV):
        """计算风电和光伏的总利润"""
        total_profit = 0
        
        for t in range(self.n_periods):
            # 当前时段参数
            D_t = self.demand[t]

            P_W_bid_t = P_W_bid[t]
            P_W_t = self.P_W_act[t]
            
            P_PV_t = self.P_PV_act[t]

            bid_W_t = bids_W[t]
            bid_PV_t = bids_PV[t]
            
            # 创建供应商列表 (报价, 电量, 类型)
            suppliers = [
                (bid_W_t, P_W_t, 'wind'),
                (bid_PV_t, P_PV_t, 'solar')
            ]
            
            # 按报价排序
            suppliers_sorted = sorted(suppliers, key=lambda x: x[0])
            
            # 市场出清
            Q_cum = 0
            period_profit = 0
            
            for bid, qty, stype in suppliers_sorted:
                if Q_cum >= D_t:
                    break
                    
                alloc = min(qty, D_t - Q_cum)

                # 计算电价
                p_max = 100

                # 计算成本（假设风电成本较低，光伏成本中等）
                if stype == 'wind':
                    # 直接计算三个区间的分配量
                    avail_phase1 = max(0, D_t/2 - Q_cum)
                    avail_phase2 = max(0, 4*D_t/5 - Q_cum - min(alloc, avail_phase1))
    
                    alloc1 = min(alloc, avail_phase1)
                    alloc2 = min(alloc - alloc1, avail_phase2)
                    alloc3 = alloc - alloc1 - alloc2
    
                    revenue = ((bid + p_max/2) * alloc1 + (bid + p_max) * alloc2 + (bid + p_max*1.5) * alloc3) / 2
    
                    cost = 8 * alloc + max(0, 300 * (P_W_bid_t - P_W_t))
                    period_profit += revenue - cost
                    Q_cum += alloc
            
            total_profit += period_profit
        
        return total_profit
    
    def optimize_strategy(self, bids_W_0):
        """优化风电实际出力和报价策略"""
        # 定义优化变量: [P_W_act0, P_W_act1, P_W_act2, bid_W0, bid_W1, bid_W2, bid_PV0, bid_PV1, bid_PV2]
        n_vars = 2 * self.n_periods  # 3个时段的P_W_act + 3个时段的bid_W + 3个时段的bid_PV
        
        def objective(x):
            P_W_bid = x[0 : n_vars/2]
            bids_W =  x[n_vars/2 : n_vars]
            
            # 计算总利润（负号因为我们要最小化负利润）
            profit = self.calculate_profit(P_W_bid, bids_W, bids_W)
            return -profit
        
        # 定义约束
        bounds = []
        
        # P_W_act 的边界（在预测值±3范围内）
        for i in range(self.n_periods):
            bounds.append((self.P_W_act[i] - 3, self.P_W_act[i] + 3))
        
        # bids_W 的边界（报价范围）
        for i in range(self.n_periods):
            bounds.append((0, 300))  # 风电报价范围
        
        # 初始猜测
        x0 = np.concatenate([
            self.P_W_pre,  # 初始P_W_act用预测值
            bids_W_0 # 初始风电报价
        ])
        
        # 优化
        res = minimize(objective, x0, bounds=bounds, method='L-BFGS-B')
        
        return res

# 使用示例
if __name__ == "__main__":
    n = 96
    # 根据时间段设置变量维度
    time_slots = range(n)

    # -------------------开始设置wasserstein球-----------------------------
    #P_delta = np.random.choice(F1_samples, size=96, replace=True).reshape(4, 96)
    # 设定基本误差分布
    np.random.seed(42)
    r=1000
    partial = 0.95

    # 设置经验分布
    Fr_samples = np.random.normal(loc=0, scale=0.5, size=1000) # 均值为0，标准差为1
    Fr_mean = np.mean(Fr_samples)
    Fr_center = Fr_samples - Fr_mean
    Fr_center_mean = np.mean(Fr_center**2)

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

    # 设定CDP所要平衡的误差电量(由wasserstein球确定)
    delta_W_0 = extract_96_elements(F1_samples)
    delta_PV_0 = extract_96_elements(F1_samples)
    delta_PV_0[0] = 0
    delta_PV_0[1] = 0
    delta_PV_0[2] = 0
    delta_PV_0[3] = 0
    delta_PV_0[4] = 0
    delta_PV_0[5] = 0
    delta_PV_0[6] = 0
    delta_PV_0[22] = 0
    delta_PV_0[23] = 0
    # 输入数据
    demand = [13.1, 12.8, 12.4, 12, 11.7, 11.4, 11.1, 10.8, 10.5, 10.2, 9.7, 9.2, 8.7, 9.1, 9.3, 9.5, 9.7, 9.7, 9.7, 9.7, 9.7, 10, 10.2, 10.5, 10.7, 11, 11.2, 11.5, 11.7, 12.2, 12.7, 13.2, 12.7, 13.7, 14.7, 15.7, 14.7, 15.7, 16.7, 17.7, 17.2, 18.2, 19.2, 20.2, 19.2, 18.2, 17.7, 17.2, 16.7, 16.2, 15.7, 15.2, 14.7, 14.7, 14.7, 14.7, 14.7, 15, 15.2, 15.5, 15.7, 16.2, 16.7, 17.2, 17.7, 18.2, 18.7, 19.2, 19.7, 21, 22.2, 23.5, 24.7, 24.7, 24.7, 24.7, 24.7, 22.5, 22.2, 22, 21.7, 21.2, 20.7, 20.2, 19.7, 19.2, 18.7, 18.2, 17.7, 17.2, 16.7, 16.2, 15.7, 15.2, 14.7, 14.2]
    total_demand = np.sum(demand)
    n_periods = len(demand)

    df = pd.read_csv('price_income.csv')

    P_W_portion = df['W']
    total_W_portioin = np.sum(P_W_portion)
    P_W_pre = P_W_portion / total_W_portioin * demand # 风电预测出力

    P_PV_portion = df['PV']
    total_PV_portioin = np.sum(P_PV_portion)
    P_PV_pre = P_PV_portion / total_PV_portioin * demand # 光伏预测出力

    bids_PV = PV_price_predict(df)
    
    # 创建市场实例
    market = ElectricityMarket(demand, P_W_pre, P_PV_pre, bids_PV, delta_W_0)
    
    print("=== 市场参数 ===")
    print(f"需求: {demand}")
    print(f"风电预测: {P_W_pre}")
    print(f"光伏预测: {P_PV_pre}")
    print(f"风电投标范围: [{market.P_W_act_min}, {market.P_W_act_max}]")
    
    # 优化策略
    print("\n=== 开始优化 ===")
    result = market.optimize_strategy()
    
    if result.success:
        print("优化成功!")
        
        # 解析结果
        P_W_act_opt = result.x[0:n_periods]
        bids_W_opt = result.x[n_periods:2 * n_periods]
        total_profit = -result.fun
        
        print(f"\n=== 最优策略 ===")
        print(f"风电实际出力: {[f'{p:.2f}' for p in P_W_act_opt]}")
        print(f"风电报价: {[f'{b:.2f}' for b in bids_W_opt]}")
        print(f"总利润: {total_profit:.2f}")
        
        # 验证约束
        print(f"\n=== 约束验证 ===")
        for t in range(3):
            within_bounds = (market.P_W_act_min[t] <= P_W_act_opt[t] <= market.P_W_act_max[t])
            print(f"时段{t}: P_W_act={P_W_act_opt[t]:.2f}, 约束范围[{market.P_W_act_min[t]}, {market.P_W_act_max[t]}], 满足约束: {within_bounds}")
    
    else:
        print("优化失败:", result.message)

# 扩展：多场景分析
def scenario_analysis():
    """分析不同需求场景下的最优策略"""
    print("\n" + "="*50)
    print("场景分析")
    print("="*50)
    
    scenarios = {
        "基准场景": [30, 40, 50],
        "高需求场景": [40, 50, 60],
        "低需求场景": [20, 30, 40]
    }
    
    for scenario_name, scenario_demand in scenarios.items():
        print(f"\n--- {scenario_name} ---")
        market_scenario = ElectricityMarket(scenario_demand, P_W_pre, P_PV_pre, bids_PV)
        result = market_scenario.optimize_strategy()
        
        if result.success:
            P_W_act_opt = result.x[0 : n_periods]
            bids_W_opt = result.x[n_periods : 2 * n_periods]
            total_profit = -result.fun
            
            print(f"风电出力: {[f'{p:.2f}' for p in P_W_act_opt]}")
            print(f"风电报价: {[f'{b:.2f}' for b in bids_W_opt]}")
            print(f"总利润: {total_profit:.2f}")

# 运行场景分析
scenario_analysis()