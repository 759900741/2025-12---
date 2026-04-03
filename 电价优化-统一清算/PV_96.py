import numpy as np
from scipy.optimize import minimize
import pandas as pd
from Wasserstein import *
from market_analysis_utils import generate_wasserstein_scenario, load_price_income, prepare_market_inputs


class ElectricityMarket_PV:
    def __init__(self, demand, P_W_pre, P_PV_pre, delta_PV_0, df, p_max):
        self.demand = np.array(demand, dtype=float)
        self.P_W_pre = np.array(P_W_pre, dtype=float)
        self.P_PV_pre = np.array(P_PV_pre, dtype=float)
        self.n_periods = len(demand)
        self.df = df
        self.p_max = float(np.asarray(p_max).reshape(-1)[0])
        self.daylight_mask = self.P_PV_pre > 1e-6

        self.bids_W = np.array(pd.read_csv('wind_predictions.csv')).astype(float).reshape(-1)
        self.P_PV_act = np.maximum(self.P_PV_pre + np.array(delta_PV_0, dtype=float), 0.0)
        self.P_PV_act[~self.daylight_mask] = 0.0
        self.P_W_act = self.P_W_pre.copy()
        self.active_mask = self.P_PV_act > 1e-6

        self.smooth_weight = 3.0
        self.curvature_weight = 1.5
        self.anchor_weight = 0.8
        self.edge_weight = 4.0
        self.noise_scale = 2.0
        self.noise_phi = 0.85

    def _continuous_clear_price(self, cleared_ratio):
        lower_bound = 0.45 * self.p_max
        upper_bound = self.p_max
        clear_price = self.p_max * (1.05 - 0.7 * cleared_ratio)
        return float(np.clip(clear_price, lower_bound, upper_bound))

    def _build_bid_anchor(self, bids_PV_0):
        if 'PV_Ave' in self.df.columns:
            base = self.df['PV_Ave'].to_numpy(dtype=float)
        else:
            base = np.asarray(bids_PV_0, dtype=float)

        base = np.clip(base, 20, self.p_max * 0.88)
        net_load = np.maximum(self.demand - self.P_W_pre - self.P_PV_pre, 0.0)

        if np.max(net_load) > 0:
            load_component = net_load / np.max(net_load)
        else:
            load_component = np.zeros_like(net_load)

        solar_component = self.P_PV_pre.copy()
        if np.max(solar_component) > 0:
            solar_component = solar_component / np.max(solar_component)

        peak_flag = ((np.arange(self.n_periods) >= 32) & (np.arange(self.n_periods) <= 80)).astype(float)
        anchor = base + 45.0 * load_component - 20.0 * solar_component + 10.0 * peak_flag
        anchor = np.clip(anchor, 10.0, self.p_max * 0.92)
        anchor[~self.active_mask] = 0.0
        return anchor

    def _add_correlated_noise(self, bids_PV):
        rng = np.random.default_rng(42)
        adjusted = np.array(bids_PV, dtype=float).copy()
        noise_level = 0.0

        for t in np.where(self.active_mask)[0]:
            noise_level = self.noise_phi * noise_level + rng.normal(0.0, self.noise_scale)
            adjusted[t] += noise_level

        adjusted[~self.active_mask] = 0.0
        return np.clip(adjusted, 0.0, self.p_max)

    def calculate_profit(self, P_PV_bid, bids_PV):
        total_profit = 0.0

        for t in range(self.n_periods):
            if not self.active_mask[t]:
                continue

            D_t = max(float(self.demand[t]), 1e-6)
            P_PV_bid_t = float(P_PV_bid[t])
            P_PV_t = float(self.P_PV_act[t])
            P_W_t = float(self.P_W_act[t])

            bid_W_t = float(self.bids_W[t])
            bid_PV_t = float(bids_PV[t])

            suppliers = [
                (bid_W_t, P_W_t, 'wind'),
                (bid_PV_t, P_PV_t, 'solar')
            ]
            suppliers_sorted = sorted(suppliers, key=lambda x: x[0])

            Q_cum = 0.0
            alloc_PV = 0.0
            p_bid = bid_PV_t

            for bid, qty, stype in suppliers_sorted:
                if Q_cum >= D_t:
                    break

                alloc = min(qty, D_t - Q_cum)
                Q_cum += alloc
                p_bid = bid

                if stype == 'solar':
                    alloc_PV = alloc

            cleared_ratio = Q_cum / D_t
            p_setting = self._continuous_clear_price(cleared_ratio)

            revenue = (p_bid + p_setting) / 2.0 * alloc_PV
            deviation_penalty = max(0.0, 220.0 * (P_PV_bid_t - P_PV_t))
            cost = 150.0 * alloc_PV + deviation_penalty
            total_profit += revenue - cost

        return total_profit

    def optimize_strategy(self, bids_PV_0):
        bid_anchor = self._build_bid_anchor(bids_PV_0)

        def objective(x):
            P_PV_bid = x[0:self.n_periods]
            bids_PV = x[self.n_periods:2 * self.n_periods]

            profit = self.calculate_profit(P_PV_bid, bids_PV)
            active_bids = bids_PV[self.active_mask]
            active_anchor = bid_anchor[self.active_mask]

            if active_bids.size >= 2:
                first_diff = np.diff(active_bids)
                smooth_penalty = np.sum(first_diff ** 2)
            else:
                smooth_penalty = 0.0

            if active_bids.size >= 3:
                second_diff = np.diff(active_bids, n=2)
                curvature_penalty = np.sum(second_diff ** 2)
            else:
                curvature_penalty = 0.0

            anchor_penalty = np.sum((active_bids - active_anchor) ** 2)
            normalized = (active_bids - 0.5 * self.p_max) / max(0.5 * self.p_max, 1.0)
            edge_penalty = np.sum(normalized ** 4)

            return (
                -profit
                + self.smooth_weight * smooth_penalty
                + self.curvature_weight * curvature_penalty
                + self.anchor_weight * anchor_penalty
                + self.edge_weight * edge_penalty
            )

        bounds = []
        for i in range(self.n_periods):
            bounds.append((max(0.0, self.P_PV_pre[i] - 3.0), self.P_PV_pre[i] + 3.0))

        for is_active in self.active_mask:
            if is_active:
                bounds.append((10.0, self.p_max * 0.95))
            else:
                bounds.append((0.0, 0.0))

        x0 = np.concatenate([self.P_PV_pre, bid_anchor])
        return minimize(objective, x0, bounds=bounds, method='L-BFGS-B')

    def optimize_solar_strategy(demand, P_W_pre, P_PV_pre, delta_PV_0, df, p_max, bids_PV_0=None):
        market = ElectricityMarket_PV(demand, P_W_pre, P_PV_pre, delta_PV_0, df, p_max)

        if bids_PV_0 is None:
            bids_PV_0 = df['PV_Ave'].values if 'PV_Ave' in df.columns else np.full(len(demand), 50.0)

        if len(bids_PV_0) < len(demand):
            bids_PV_0 = np.pad(bids_PV_0, (0, len(demand) - len(bids_PV_0)), 'edge')
        elif len(bids_PV_0) > len(demand):
            bids_PV_0 = bids_PV_0[:len(demand)]

        print("=== Start PV optimization ===")
        result = market.optimize_strategy(bids_PV_0)

        if result.success:
            print("PV optimization succeeded.")
            P_PV_bid_opt = result.x[0:len(demand)]
            bids_PV_opt = result.x[len(demand):2 * len(demand)]
            bids_PV_opt = market._add_correlated_noise(bids_PV_opt)
            total_profit = market.calculate_profit(P_PV_bid_opt, bids_PV_opt)

            return {
                'success': True,
                'P_PV_act': P_PV_bid_opt,
                'bids_PV_act': bids_PV_opt,
                'total_profit': total_profit,
                'message': 'PV optimization succeeded'
            }

        return {
            'success': False,
            'P_PV_act': None,
            'bids_PV_act': None,
            'total_profit': 0,
            'message': f'PV optimization failed: {result.message}'
        }

    def main_optimization_PV(ration, p_max, demand, scale_factor=1.0):
        PV_profit_all = []
        df = load_price_income('price_income.csv')
        inputs = prepare_market_inputs(demand, ration, df, scale_factor=scale_factor)
        P_W_pre = inputs['P_W_pre']
        P_PV_pre = inputs['P_PV_pre']

        scenario = generate_wasserstein_scenario(daylight_mask=P_PV_pre > 1e-6, seed=42)
        print(f"Test distribution is within Wasserstein ball (epsilon={scenario['epsilon']}): {scenario['in_ball']}")
        print(f"W(Fr, F1) = {scenario['w_dist_F1']:.4f}, within epsilon: {scenario['w_dist_F1'] <= scenario['epsilon']}")

        delta_W_0 = scenario['delta_W_0']
        delta_PV_0 = scenario['delta_PV_0']

        print("=" * 60)
        print("Start integrated PV optimization")
        print("=" * 60)
        print("\n>>> Start PV sub-optimization <<<")
        solar_result = ElectricityMarket_PV.optimize_solar_strategy(demand, P_W_pre, P_PV_pre, delta_PV_0, df, p_max)

        if solar_result['success']:
            P_PV_act = solar_result['P_PV_act']
            bids_PV_act = solar_result['bids_PV_act']
            solar_profit = solar_result['total_profit']
            PV_profit_all.append(solar_profit)
        else:
            P_PV_act = None
            bids_PV_act = None
            solar_profit = 0

        results = {
            'solar': {
                'P_PV_act': P_PV_act,
                'bids_PV_act': bids_PV_act,
                'profit': solar_profit,
                'success': solar_result['success']
            }
        }

        return results, delta_PV_0
