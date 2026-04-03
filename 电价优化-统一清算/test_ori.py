demand = [20, 25]
P_W_act = [13, 15]
P_PV_act = [12, 13]
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