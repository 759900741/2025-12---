import numpy as np

demand = [13.1, 12.8, 12.4, 12, 11.7, 11.4, 11.1, 10.8, 10.5, 10.2, 9.7, 9.2, 8.7, 9.1, 9.3, 9.5, 9.7, 9.7, 9.7, 9.7, 9.7, 10, 10.2, 10.5, 10.7, 11, 11.2, 11.5, 11.7, 12.2, 12.7, 13.2, 12.7, 13.7, 14.7, 15.7, 14.7, 15.7, 16.7, 17.7, 17.2, 18.2, 19.2, 20.2, 19.2, 18.2, 17.7, 17.2, 16.7, 16.2, 15.7, 15.2, 14.7, 14.7, 14.7, 14.7, 14.7, 15, 15.2, 15.5, 15.7, 16.2, 16.7, 17.2, 17.7, 18.2, 18.7, 19.2, 19.7, 21, 22.2, 23.5, 24.7, 24.7, 24.7, 24.7, 24.7, 22.5, 22.2, 22, 21.7, 21.2, 20.7, 20.2, 19.7, 19.2, 18.7, 18.2, 17.7, 17.2, 16.7, 16.2, 15.7, 15.2, 14.7, 14.2]
b = np.sum(demand)

P_W_act = [2, 2.5, 2.9, 3.3, 3.5, 3.2, 2.6, 2.5, 2.9, 3.3, 3.2, 2.5, 1.8, 1.7, 2.1, 2.5, 2.7, 2.8, 3.2, 3.5, 3.4, 3.2, 2.9, 2.8, 2.7, 2.6, 2.5, 2.7, 2.5, 2.8, 2.5, 2.5, 2.6, 2.9, 3.4, 3.8, 4.3, 4.7, 5.1, 5.3, 5.5, 5.5, 5.3, 4.9, 4.6, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.6, 4.8, 5.1, 5.3, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5, 5.6, 5.8, 6.1, 6.3, 6.5, 6.4, 6.2, 5.9, 5.7, 5.5, 5.3, 5.1, 4.8, 4.6, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.4, 4.2, 3.9, 3.7, 3.5, 3.5]
c = np.sum(P_W_act) #400

P_PV_act = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.6, 0.8, 1.1, 1.4, 1.7, 2.0, 2.3, 2.6, 2.9, 3.2, 3.5, 3.8, 4.1, 4.4, 4.7, 5.0, 5.3, 5.6, 5.9, 6.2, 6.5, 6.9, 7.3, 7.8, 8.2, 8.6, 9.0, 9.3, 9.6, 9.7, 9.9, 9.9, 9.8, 9.5, 9.1, 8.8, 8.4, 8.0, 7.6, 7.2, 6.8, 6.4, 6.0, 5.6, 5.2, 4.8, 4.4, 4.0, 3.6, 3.2, 2.8, 2.4, 2.0, 1.6, 1.2, 0.8, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
d = np.sum(P_PV_act) #

P_PV_bid = [13, 15]
bids_W = [30, 50]
bids_PV = [50, 30]
n_periods = 2

#def calculate_profit(P_PV_bid, bids_W, bids_PV):
#"""计算风电和光伏的总利润"""
total_profit = 0
        
for t in range(1,2):#n_periods):
        # 当前时段参数
        D_t = demand[t]
        P_PV_bid_t = P_PV_bid[t]
        P_W_t = P_W_act[t]
        P_PV_t = P_PV_act[t]
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
            # p = self.pricing_curve(Q_cum, D_t)
            p_max = 100
                
            # 计算该供应商收入
                
            # 计算成本（假设风电成本较低，光伏成本中等）
            if stype == 'solar':
                if Q_cum == 0:
                    if alloc <= D_t/2:
                        revenue = (bid + p_max / 2) / 2 * alloc
                    if alloc > D_t/2:
                        revenue = (bid + p_max / 2) / 2 * D_t / 2 + (bid + p_max) / 2 * (alloc - D_t / 2)
                if 0 < Q_cum <= D_t / 2:
                    if Q_cum + alloc <= D_t / 2:
                        revenue = (bid + p_max / 2) / 2 * alloc
                    else:
                        revenue = (bid + p_max / 2) / 2 * (D_t / 2 - Q_cum) + (bid + p_max) / 2 * (alloc - D_t / 2)
                if Q_cum > D_t / 2:
                    revenue = (bid + p_max) / 2 * alloc
                cost = 8 * qty + max(0, 50 * (P_PV_bid_t - P_PV_t)) # 风电低成本    
            profit_i = revenue - cost

            if stype == 'wind':
                revenue = 0
                cost = 0
            profit_i += revenue - cost
            period_profit += profit_i
            Q_cum += alloc
            
       
        total_profit += period_profit
    #return total_profit

a = calculate_profit(P_PV_bid, bids_W, bids_PV)
print(a)