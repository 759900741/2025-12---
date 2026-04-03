import numpy as np
from scipy.optimize import minimize
import pandas as pd
from competition import W_price_predict
from Wasserstein import *
import statsmodels.api as sm


class ElectricityMarket_PV:
    def __init__(self, demand, P_W_pre, P_PV_pre, delta_PV_0, df, p_max):

        self.demand = np.array(demand)
        self.P_W_pre = np.array(P_W_pre)
        self.P_PV_pre = np.array(P_PV_pre)
        self.n_periods = len(demand)
        self.df = df
        self.p_max = np.array(p_max)

        # 修正：从W_price_predict返回的字典中提取预测值
        # w_pred_result = W_price_predict(df, P_W_pre, P_PV_pre)
        self.bids_W = np.array(pd.read_csv('wind_predictions.csv'))  # 提取预测值数组
        # print('W_bids_predict', self.bids_W)

        # 定义光电实际出力（预测值+Wasserstein随机向量）
        self.P_PV_act = P_PV_pre + delta_PV_0

        # 假设风是确定性的（实际等于预测）
        self.P_W_act = self.P_W_pre.copy()

    def calculate_profit(self, P_PV_bid, bids_PV):
        """计算风电和光伏的总利润"""
        total_profit = 0

        for t in range(self.n_periods):
            # 当前时段参数
            D_t = self.demand[t]
            P_PV_bid_t = P_PV_bid[t]

            P_PV_t = self.P_PV_act[t]
            P_W_t = self.P_W_act[t]

            # 确保使用正确的索引访问
            bid_W_t = self.bids_W[t]
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
                    Q_cum += alloc

                if stype == 'solar':
                    if Q_cum < D_t / 2:
                        if Q_cum + alloc < D_t / 2:
                            revenue = (bid + p_max) / 2 * alloc
                        if D_t / 2 <= Q_cum + alloc <= 4 * D_t / 5:
                            revenue = (bid + p_max) / 2 * (D_t / 2 - Q_cum) + (bid + 4 * p_max / 5) / 2 * (
                                        Q_cum + alloc - D_t / 2)
                        if 4 * D_t / 5 <= Q_cum + alloc <= D_t:
                            revenue = (bid + p_max) / 2 * (D_t / 2 - Q_cum) + (
                                        bid + 4 * p_max / 5) / 2 * 3 * D_t / 10 + (bid + p_max / 2) / 2 * (
                                                  Q_cum + alloc - 4 * D_t / 5)
                    if D_t / 2 <= Q_cum <= 4 * D_t / 5:
                        if Q_cum + alloc <= 4 * D_t / 5:
                            revenue = (bid + 4 * p_max / 5) / 2 * alloc
                        if 4 * D_t / 5 <= Q_cum + alloc <= D_t:
                            revenue = (bid + 4 * p_max / 5) / 2 * (4 * D_t / 5 - Q_cum) + (bid + p_max / 2) / 2 * (
                                        Q_cum + alloc - 4 * D_t / 5)
                    if Q_cum > 4 * D_t / 5:
                        revenue = (bid + p_max / 2) / 2 * alloc
                    cost = 150 * qty + max(0, 300 * (P_PV_bid_t - P_PV_t))
                    period_profit += revenue - cost
                    Q_cum += alloc

            total_profit += period_profit

        return total_profit

    def optimize_strategy(self, bids_PV_0):
        """优化光伏投标出力和报价策略"""

        def objective(x):
            P_PV_bid = x[0: self.n_periods]
            bids_PV = x[self.n_periods: 2 * self.n_periods]

            # 计算总利润（负号因为我们要最小化负利润）
            profit = self.calculate_profit(P_PV_bid, bids_PV)
            return -profit

        # 定义约束
        bounds = []

        # P_PV_bid 的边界（在预测值±3范围内）
        for i in range(self.n_periods):
            bounds.append((max(0, self.P_PV_pre[i] - 3), self.P_PV_pre[i] + 3))

        # bids_PV 的边界（报价范围）
        for i in range(self.n_periods):
            bounds.append((0, 400))  # 光伏报价范围

        # 初始猜测
        x0 = np.concatenate([
            self.P_PV_pre,
            bids_PV_0
        ])

        # 优化
        res = minimize(objective, x0, bounds=bounds, method='L-BFGS-B')

        return res

    def optimize_solar_strategy(demand, P_W_pre, P_PV_pre, delta_PV_0, df, p_max, bids_PV_0=None):

        # 创建市场实例
        market = ElectricityMarket_PV(demand, P_W_pre, P_PV_pre, delta_PV_0, df, p_max)

        # 设置初始猜测
        if bids_PV_0 is None:
            bids_PV_0 = df['PV_Ave'].values if 'PV_Ave' in df.columns else np.full(len(demand), 50)

        # 检查数据长度
        if len(bids_PV_0) < len(demand):
            bids_PV_0 = np.pad(bids_PV_0, (0, len(demand) - len(bids_PV_0)), 'edge')
        elif len(bids_PV_0) > len(demand):
            bids_PV_0 = bids_PV_0[:len(demand)]

        # 优化策略
        print("=== 开始优化 ===")
        result = market.optimize_strategy(bids_PV_0)

        if result.success:
            print("优化成功!")

            # 解析结果
            P_PV_bid_opt = result.x[0:len(demand)]
            bids_PV_opt = result.x[len(demand):2 * len(demand)]
            total_profit = -result.fun

            # 返回结果字典
            return {
                'success': True,
                'P_PV_act': P_PV_bid_opt,
                'bids_PV_act': bids_PV_opt,
                'total_profit': total_profit,
                'message': '优化成功'
            }
        else:
            return {
                'success': False,
                'P_PV_act': None,
                'bids_PV_act': None,
                'total_profit': 0,
                'message': f'优化失败: {result.message}'
            }

    def main_optimization_PV(ration, p_max, demand):
        PV_profit_all = []

        # 准备共享的输入数据
        df = pd.read_csv('price_income.csv')

        # 计算风电和光伏预测出力
        total_demand = np.sum(demand)
        # print(total_demand)

        P_W_portion = df['W']
        total_W_portioin = np.sum(P_W_portion)
        P_W_pre = 0.01 * ration * P_W_portion / total_W_portioin * total_demand
        # print(P_W_pre)

        P_PV_portion = df['PV']
        total_PV_portioin = np.sum(P_PV_portion)
        P_PV_pre = 0.01 * ration * P_PV_portion / total_PV_portioin * total_demand
        # print(P_PV_pre)

        # 设置Wasserstein误差
        np.random.seed(42)
        r = 1000
        partial = 0.95

        # 设置经验分布
        Fr_samples = np.random.normal(loc=0, scale=0.5, size=1000)
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

        delta_W_0 = extract_96_elements(F1_samples)
        delta_PV_0 = extract_96_elements(F1_samples)

        # 设置某些时段为0
        for i in range(28):
            delta_PV_0[i] = 0
        for i in range(89, 96):
            delta_PV_0[i] = 0

        print("=" * 60)
        print("开始综合优化程序")
        print("=" * 60)

        # ==================== 光伏电优化子程序 ====================
        print("\n>>> 开始光电优化 <<<")
        solar_result = ElectricityMarket_PV.optimize_solar_strategy(demand, P_W_pre, P_PV_pre, delta_PV_0, df, p_max)
        # 从结果中获取变量，无论优化是否成功
        P_PV_act = solar_result['P_PV_act']
        bids_PV_act = solar_result['bids_PV_act']
        solar_profit = solar_result['total_profit']

        if solar_result['success']:
            # 风电优化结果

            PV_profit_all.append(solar_profit)

            # 返回所有结果
        results = {
            'solar': {
                'P_PV_act': P_PV_act,
                'bids_PV_act': bids_PV_act,
                'profit': solar_profit,
                'success': solar_result['success']
            }
        }

        return results, delta_PV_0


# 使用示例
if __name__ == "__main__":
    ration = 1
    optimization_results = ElectricityMarket_PV.main_optimization_PV(ration, p_max)
    print(optimization_results)