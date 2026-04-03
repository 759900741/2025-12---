import numpy as np
from scipy.optimize import minimize

class ElectricityMarket:
    def __init__(self, demand, P_W_pre, P_PV_pre, bids_W, bids_PV):
        """
        demand: 三个时段的需求 [30, 40, 50]
        P_W_pre: 风电预测出力 [15, 17, 13] 
        P_PV_pre: 光伏预测出力 [10, 20, 15]
        bids_PV: 光伏报价 [70, 80, 90]
        """
        self.demand = np.array(demand)
        self.P_W_pre = np.array(P_W_pre)
        self.P_PV_pre = np.array(P_PV_pre)
        self.n_periods = len(demand)
        self.bid_W = np.array(bids_W)
        self.bid_PV = np.array(bids_PV)
        
        # 定义风电实际出力范围（在预测值±3范围内）
        self.P_W_act = self.P_W_pre.copy()
        
        # 假设光伏是确定性的（实际等于预测）
        self.P_PV_act = self.P_PV_pre.copy()
        
        # 主题T的参数
        self.T_cost = 20  # 主题T的单位成本
    
    def pricing_curve(self, Q, D):
        """时段特定的定价曲线"""
        pmax = 100  # 最高电价
        if Q <= D / 2:
            return (Q / (D / 2)) * pmax
        else:
            return (Q / (D / 2)) * pmax * 2
    
    def calculate_profit(self, bids_W, bids_PV, bids_T):
        """计算风电和光伏的总利润"""
        total_profit = 0
        
        for t in range(self.n_periods):
            # 当前时段参数
            D_t = self.demand[t]
            P_W_t = self.P_W_act[t]
            P_PV_t = self.P_PV_act[t]
            P_T_t = self.P_T_act[t]
            bid_W_t = bids_W[t]
            bid_PV_t = bids_PV[t]
            bid_T_t = bids_T[t]
            
            # 创建供应商列表 (报价, 电量, 类型)
            suppliers = [
                (bid_W_t, P_W_t, 'wind'),
                (bid_PV_t, P_PV_t, 'solar'),
                (bid_T_t, P_T_t, 'thermal')
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
                Q_before = Q_cum
                Q_mid = Q_before + alloc / 2
                
                # 计算电价
                p = self.pricing_curve(Q_mid, D_t)
                
                # 计算该供应商收入
                if stype == 'thermal':
                    revenue = (bid + p) / 2 * alloc
                    cost = 20 * alloc
                else:
                    revenue = 0
                    cost = 0
                    
                profit_i = revenue - cost
                period_profit += profit_i
                Q_cum += alloc
            
            total_profit += period_profit
        
        return total_profit
    
    def calculate_T_cost(self, P_W_bid, bids_W):
        """计算主题T的总成本"""
        total_cost_T = 0
        
        for t in range(self.n_periods):
            # 当前时段参数
            D_t = self.demand[t]
            P_W_t = self.P_W_act[t]
            P_PV_t = self.P_PV_act[t]
            bid_W_t = bids_W[t]
            bid_PV_t = self.bid_PV[t]
            
            # 创建供应商列表 (报价, 电量, 类型)
            suppliers = [
                (bid_W_t, P_W_t, 'wind'),
                (bid_PV_t, P_PV_t, 'solar')
            ]
            
            # 按报价排序
            suppliers_sorted = sorted(suppliers, key=lambda x: x[0])
            
            # 市场出清
            Q_cum = 0
            period_cost_T = 0
            
            # 采购可再生能源
            for bid, qty, stype in suppliers_sorted:
                if Q_cum >= D_t:
                    break
                    
                alloc = min(qty, D_t - Q_cum)
                Q_before = Q_cum
                Q_mid = Q_before + alloc / 2
                
                # 计算电价
                p = self.pricing_curve(Q_mid, D_t)
                
                # 主题T支付给供应商的费用
                payment_to_supplier = (bid + p) / 2 * alloc
                period_cost_T += payment_to_supplier
                Q_cum += alloc
            
            # 如果可再生能源不能满足需求，主题T需要自己发电
            if Q_cum < D_t:
                T_generation = D_t - Q_cum
                T_generation_cost = T_generation * self.T_cost
                period_cost_T += T_generation_cost
            
            total_cost_T += period_cost_T
        
        return total_cost_T
    
    def optimize_W_strategy(self):
        """优化风电的策略（申报电量和报价）"""
        # 定义优化变量: [P_W_bid0, P_W_bid1, P_W_bid2, bid_W0, bid_W1, bid_W2]
        n_vars = 3 + 3
        
        def objective(x):
            P_W_bid = x[0:3]
            bids_W = x[3:6]
            
            # 计算总利润（负号因为我们要最小化负利润）
            profit = self.calculate_profit(P_W_bid, bids_W, self.bid_PV)
            return -profit
        
        # 定义约束
        bounds = []
        
        # P_W_bid 的边界（申报电量范围）
        for i in range(3):
            bounds.append((0, self.P_W_act_max[i]))
        
        # bids_W 的边界（报价范围）
        for i in range(3):
            bounds.append((0, 100))
        
        # 初始猜测
        x0 = np.concatenate([
            self.P_W_pre,
            np.array([30, 30, 30])
        ])
        
        # 优化
        res = minimize(objective, x0, bounds=bounds, method='L-BFGS-B')
        
        return res
    
    def optimize_T_strategy(self, P_W_bid_opt, bids_W_opt):
        """主题T优化策略：分析成本结构并给出建议"""
        print("\n" + "="*50)
        print("主题T成本分析")
        print("="*50)
        
        total_cost_T = self.calculate_T_cost(P_W_bid_opt, bids_W_opt)
        
        print(f"主题T单位成本: {self.T_cost}")
        print(f"主题T总成本: {total_cost_T:.2f}")
        
        # 详细分析每个时段
        print(f"\n=== 各时段成本分析 ===")
        for t in range(self.n_periods):
            D_t = self.demand[t]
            P_W_t = self.P_W_act[t]
            P_PV_t = self.P_PV_act[t]
            bid_W_t = bids_W_opt[t]
            bid_PV_t = self.bid_PV[t]
            
            # 模拟采购过程
            suppliers = [
                (bid_W_t, P_W_t, 'wind'),
                (bid_PV_t, P_PV_t, 'solar')
            ]
            suppliers_sorted = sorted(suppliers, key=lambda x: x[0])
            
            Q_cum = 0
            cost_breakdown = {'wind': 0, 'solar': 0, 'T': 0}
            
            # 采购可再生能源
            for bid, qty, stype in suppliers_sorted:
                if Q_cum >= D_t:
                    break
                alloc = min(qty, D_t - Q_cum)
                Q_before = Q_cum
                Q_mid = Q_before + alloc / 2
                p = self.pricing_curve(Q_mid, D_t)
                
                payment = (bid + p) / 2 * alloc
                cost_breakdown[stype] += payment
                Q_cum += alloc
            
            # 主题T自己发电
            if Q_cum < D_t:
                T_generation = D_t - Q_cum
                T_cost = T_generation * self.T_cost
                cost_breakdown['T'] = T_cost
            
            print(f"\n时段 {t}:")
            print(f"  需求: {D_t}, 可再生能源采购: {Q_cum:.2f}")
            print(f"  风电成本: {cost_breakdown['wind']:.2f} (报价: {bid_W_t:.2f})")
            print(f"  光伏成本: {cost_breakdown['solar']:.2f} (报价: {bid_PV_t:.2f})")
            print(f"  主题T发电成本: {cost_breakdown['T']:.2f}")
            print(f"  总成本: {sum(cost_breakdown.values()):.2f}")
            
            # 成本优化建议
            if cost_breakdown['T'] > 0:
                print(f"  ⚠️  建议: 可考虑提高可再生能源采购以减少主题T发电")
            elif bid_PV_t > self.T_cost:
                print(f"  💡 建议: 光伏报价高于主题T成本，可减少光伏采购")
        
        return total_cost_T

# 使用示例
if __name__ == "__main__":
    # 输入数据
    demand = [30, 40, 50]
    P_W_pre = [15, 17, 13]  # 风电预测出力
    P_PV_pre = [10, 20, 15] # 光伏预测出力
    bids_PV = [70, 80, 90]  # 光伏报价
    
    # 创建市场实例
    market = ElectricityMarket(demand, P_W_pre, P_PV_pre, bids_PV)
    
    print("=== 市场参数 ===")
    print(f"需求: {demand}")
    print(f"风电预测: {P_W_pre}")
    print(f"光伏预测: {P_PV_pre}")
    print(f"光伏报价: {bids_PV}")
    print(f"风电实际出力: {[f'{p:.2f}' for p in market.P_W_act]}")
    
    # 优化风电策略
    print("\n=== 优化风电策略 ===")
    result_W = market.optimize_W_strategy()
    
    if result_W.success:
        print("风电策略优化成功!")
        
        # 解析结果
        P_W_bid_opt = result_W.x[0:3]
        bids_W_opt = result_W.x[3:6]
        total_profit = -result_W.fun
        
        print(f"\n=== 风电最优策略 ===")
        print(f"风电申报电量: {[f'{p:.2f}' for p in P_W_bid_opt]}")
        print(f"风电报价: {[f'{b:.2f}' for b in bids_W_opt]}")
        print(f"风电总利润: {total_profit:.2f}")
        
        # 验证约束
        print(f"\n=== 约束验证 ===")
        for t in range(3):
            within_bounds = (market.P_W_act_min[t] <= P_W_bid_opt[t] <= market.P_W_act_max[t])
            print(f"时段{t}: P_W_bid={P_W_bid_opt[t]:.2f}, 约束范围[{market.P_W_act_min[t]}, {market.P_W_act_max[t]}], 满足约束: {within_bounds}")
        
        # 主题T成本分析
        market.optimize_T_strategy(P_W_bid_opt, bids_W_opt)
    
    else:
        print("优化失败:", result_W.message)

# 场景分析
def scenario_analysis():
    """分析不同场景下的策略"""
    print("\n" + "="*50)
    print("多场景分析")
    print("="*50)
    
    scenarios = {
        "基准场景": ([30, 40, 50], [70, 80, 90]),
        "高光伏报价": ([30, 40, 50], [90, 95, 100]),
        "低光伏报价": ([30, 40, 50], [50, 60, 70])
    }
    
    P_W_pre = [15, 17, 13]
    P_PV_pre = [10, 20, 15]
    
    for scenario_name, (scenario_demand, scenario_bids_PV) in scenarios.items():
        print(f"\n--- {scenario_name} ---")
        print(f"需求: {scenario_demand}, 光伏报价: {scenario_bids_PV}")
        
        market = ElectricityMarket(scenario_demand, P_W_pre, P_PV_pre, scenario_bids_PV)
        result = market.optimize_W_strategy()
        
        if result.success:
            P_W_bid_opt = result.x[0:3]
            bids_W_opt = result.x[3:6]
            total_profit = -result.fun
            
            print(f"风电申报: {[f'{p:.2f}' for p in P_W_bid_opt]}")
            print(f"风电报价: {[f'{b:.2f}' for b in bids_W_opt]}")
            print(f"风电利润: {total_profit:.2f}")
            
            # 主题T成本
            T_cost = market.calculate_T_cost(P_W_bid_opt, bids_W_opt)
            print(f"主题T成本: {T_cost:.2f}")

# 运行场景分析
scenario_analysis()