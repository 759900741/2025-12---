import numpy as np
from scipy.optimize import minimize
import pandas as pd

from market_analysis_utils import generate_wasserstein_scenario, load_price_income, prepare_market_inputs


class ElectricityMarket_W:
    def __init__(self, demand, P_W_pre, P_PV_pre, delta_W_0, df, p_max):
        self.demand = np.array(demand, dtype=float)
        self.P_W_pre = np.array(P_W_pre, dtype=float)
        self.P_PV_pre = np.array(P_PV_pre, dtype=float)
        self.n_periods = len(self.demand)
        self.df = df
        self.p_max = float(np.asarray(p_max).reshape(-1)[0])

        self.bids_PV = np.array(pd.read_csv('pv_predictions.csv')).astype(float).reshape(-1)
        self.P_W_act = np.maximum(self.P_W_pre + np.array(delta_W_0, dtype=float), 0.0)
        self.P_PV_act = self.P_PV_pre.copy()

    def calculate_profit(self, P_W_bid, bids_W):
        total_profit = 0.0

        for t in range(self.n_periods):
            D_t = max(float(self.demand[t]), 1e-6)
            P_W_bid_t = float(P_W_bid[t])
            P_W_t = float(self.P_W_act[t])
            P_PV_t = float(self.P_PV_act[t])
            bid_W_t = float(bids_W[t])
            bid_PV_t = float(self.bids_PV[t])

            suppliers = [
                (bid_W_t, P_W_t, 'wind'),
                (bid_PV_t, P_PV_t, 'solar'),
            ]
            suppliers_sorted = sorted(suppliers, key=lambda item: item[0])

            Q_cum = 0.0
            alloc_W = 0.0
            p_bid = bid_W_t

            for bid, qty, supplier_type in suppliers_sorted:
                if Q_cum >= D_t:
                    break

                alloc = min(qty, D_t - Q_cum)
                Q_cum += alloc
                p_bid = bid

                if supplier_type == 'wind':
                    alloc_W = alloc

            p_max = 100.0
            if Q_cum < D_t / 2.0:
                p_setting = p_max
            elif Q_cum <= 4.0 * D_t / 5.0:
                p_setting = 4.0 * p_max / 5.0
            else:
                p_setting = p_max / 2.0

            revenue = (p_bid + p_setting) / 2.0 * alloc_W
            cost = 200.0 * alloc_W + max(0.0, 300.0 * (P_W_bid_t - P_W_t))
            total_profit += revenue - cost

        return total_profit

    def optimize_strategy_with_smoothing(self, bids_W_0, lambda_smooth=0.15):
        def objective(x):
            P_W_bid = x[0:self.n_periods]
            bids_W = x[self.n_periods:2 * self.n_periods]
            profit = self.calculate_profit(P_W_bid, bids_W)

            smoothing_penalty = 0.0
            for i in range(len(bids_W) - 1):
                price_change = abs(bids_W[i + 1] - bids_W[i])
                if price_change > 20.0:
                    smoothing_penalty += (price_change - 20.0) ** 2

            return -profit + lambda_smooth * smoothing_penalty

        bounds = []
        for i in range(self.n_periods):
            bounds.append((max(0.0, self.P_W_pre[i] - 3.0), self.P_W_pre[i] + 3.0))

        for i in range(self.n_periods):
            if i == 0:
                bounds.append((0.0, 300.0))
            else:
                prev_bid = bids_W_0[i - 1] if bids_W_0 is not None else 50.0
                bounds.append((max(0.0, prev_bid - 30.0), min(300.0, prev_bid + 30.0)))

        x0 = np.concatenate([
            self.P_W_pre,
            bids_W_0 if bids_W_0 is not None else np.full(self.n_periods, 50.0),
        ])

        return minimize(objective, x0, bounds=bounds, method='L-BFGS-B')

    def optimize_wind_strategy(demand, P_W_pre, P_PV_pre, delta_W_0, df, p_max, bids_W_0=None):
        market = ElectricityMarket_W(demand, P_W_pre, P_PV_pre, delta_W_0, df, p_max)

        if bids_W_0 is None:
            bids_W_0 = df['W_Ave'].values if 'W_Ave' in df.columns else np.full(len(demand), 50.0)

        if len(bids_W_0) < len(demand):
            bids_W_0 = np.pad(bids_W_0, (0, len(demand) - len(bids_W_0)), 'edge')
        elif len(bids_W_0) > len(demand):
            bids_W_0 = bids_W_0[:len(demand)]

        print("=== 开始风电优化 ===")
        result = market.optimize_strategy_with_smoothing(bids_W_0, lambda_smooth=0.15)

        if result.success:
            print("风电优化成功。")
            P_W_bid_opt = result.x[0:len(demand)]
            bids_W_opt = result.x[len(demand):2 * len(demand)]
            total_profit = -result.fun
            return {
                'success': True,
                'P_W_act': P_W_bid_opt,
                'bids_W_act': bids_W_opt,
                'total_profit': total_profit,
                'message': 'wind optimization succeeded',
            }

        print("风电优化失败。")
        return {
            'success': False,
            'P_W_act': None,
            'bids_W_act': None,
            'total_profit': 0.0,
            'message': f'wind optimization failed: {result.message}',
        }

    def main_optimization_W(ration, p_max, demand, scale_factor=1.0):
        wind_profit_all = []
        df = load_price_income('price_income.csv')
        inputs = prepare_market_inputs(demand, ration, df, scale_factor=scale_factor)
        P_W_pre = inputs['P_W_pre']
        P_PV_pre = inputs['P_PV_pre']

        scenario = generate_wasserstein_scenario(daylight_mask=P_PV_pre > 1e-6, seed=42)
        print(f"Test distribution is within Wasserstein ball (epsilon={scenario['epsilon']}): {scenario['in_ball']}")
        print(f"W(Fr, F1) = {scenario['w_dist_F1']:.4f}, within epsilon: {scenario['w_dist_F1'] <= scenario['epsilon']}")

        delta_W_0 = scenario['delta_W_0']

        print("=" * 60)
        print("开始综合风电优化")
        print("=" * 60)
        print("\n>>> 进入风电子问题 <<<")

        wind_result = ElectricityMarket_W.optimize_wind_strategy(demand, P_W_pre, P_PV_pre, delta_W_0, df, p_max)

        if wind_result['success']:
            P_W_act = wind_result['P_W_act']
            bids_W_act = wind_result['bids_W_act']
            wind_profit = wind_result['total_profit']
            wind_profit_all.append(wind_profit)
            print(f"  [OK] 优化成功，利润: {wind_profit:.2f}")
        else:
            P_W_act = None
            bids_W_act = None
            wind_profit = 0.0

        results = {
            'wind': {
                'P_W_act': P_W_act,
                'bids_W_act': bids_W_act,
                'profit': wind_profit,
                'success': wind_result['success'],
            }
        }

        return results, delta_W_0
