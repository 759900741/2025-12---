

P_PGC_all = []

for R in range(60,61):
    P_W, P_PV, bids_W, bids_PV, remain, delta_PV_0, delta_W_0 = WPV_sort(P_demand, R, p_max)
    
    P_delta = compute_p_deltas(delta_W_0, delta_PV_0)

    delta_W, delta_PV, P_deltaS, worst_cost_1, nonneg_index = find_worst_case_distributions(Fr_samples, 
                                  lambda_VPPO_CDP_up, lambda_VPPO_CDP_down, epsilon, p_max)
    P_deltaS = 96 *[3]
    #P_ES_c, P_ES_d, ES_profit = ES(n, time_slots, P_ES_c_max, P_ES_d_max, E_ES_max, E_ES_min, lambda_ES, lambda_VPPO_ES_c, lambda_VPPO_ES_d)

    P_ES_c, P_ES_d, P_CDP, CDP_ES_profit = CDP_ES(n, time_slots, P_ES_c_max, P_ES_d_max, E_ES_max, E_ES_min, lambda_ES, 
               lambda_VPPO_ES_c, lambda_VPPO_ES_d, lambda_VPPO_CDP, 
               lambda_VPPO_CDP_up, lambda_VPPO_CDP_down, P_CDP_max, P_CDP_min, 
               P_CDP_down, P_CDP_up, a, b, c, P_delta, P_W, P_PV, P_deltaS, P_demand)
    
    iterations = 30

    # 在迭代循环中使用
    temperature = 1000  # 初始温度
    cooling_rate = 0.95  # 冷却速率

    for iters in range(iterations):
        old_profit = VPPO_profit[-1] if VPPO_profit else -float('inf')
        PGC_profit, P_PGC, p_max, lambda_VPPO_CDP, lambda_VPPO_CDP_up, lambda_VPPO_CDP_down= Master(n, time_slots, lambda_VPPO_CDP_up, lambda_VPPO_CDP_down, P_W, P_PV, P_delta, P_CDP, P_ES_c, P_ES_d)
        #, P_ES_c, P_ES_d), lambda_VPPO_ES_d, lambda_VPPO_ES_c 
        
        P_W, P_PV, bids_W, bids_PV, remain, delta_PV_0, delta_W_0 = WPV_sort(P_demand, R, p_max)

        #P_ES_c, P_ES_d, ES_profit = ES(n, time_slots, P_ES_c_max, P_ES_d_max, E_ES_max, E_ES_min, lambda_ES, lambda_VPPO_ES_c, lambda_VPPO_ES_d)

        P_ES_c, P_ES_d, P_CDP, CDP_ES_profit = CDP_ES(n, time_slots, P_ES_c_max, P_ES_d_max, E_ES_max, E_ES_min, lambda_ES, 
               lambda_VPPO_ES_c, lambda_VPPO_ES_d, lambda_VPPO_CDP, 
               lambda_VPPO_CDP_up, lambda_VPPO_CDP_down, P_CDP_max, P_CDP_min, 
               P_CDP_down, P_CDP_up, a, b, c, P_delta, P_W, P_PV, P_deltaS, P_demand)
        
        print(iters)
    P_PGC_all.append(PGC_profit)


        ###############################加入遗传算法部分
    # 初始化最佳解
    current_best_prices = (lambda_VPPO_CDP_up.copy(), lambda_VPPO_CDP_down.copy(), lambda_VPPO_CDP.copy())
    current_best_profit = None

    for iters in range(iterations):
        print(f"\n=== 第 {iters + 1} 次迭代 ===")
    
    # 每10次迭代执行一次遗传变异
        if iters > 0 and iters % 10 == 0:
            print("执行遗传变异算法...")
            new_prices = fast_genetic_mutation(
                current_best_prices, P_W, P_PV, P_delta, P_CDP, P_ES_c, P_ES_d
            )
        
            if new_prices:
                lambda_VPPO_CDP_up, lambda_VPPO_CDP_down, lambda_VPPO_CDP = new_prices
                current_best_prices = new_prices
    
        # 正常迭代，添加随机扰动
        if iters > 0:
            lambda_VPPO_CDP_up = add_random_perturbation(lambda_VPPO_CDP_up, 0.03, 300, 500)
            lambda_VPPO_CDP_down = add_random_perturbation(lambda_VPPO_CDP_down, 0.03, 100, 200)
            lambda_VPPO_CDP = add_random_perturbation(lambda_VPPO_CDP, 0.02, 50, 500)
    
        # 求解主问题
        PGC_profit, P_PGC, p_max, lambda_VPPO_CDP, lambda_VPPO_CDP_up, lambda_VPPO_CDP_down = Master(
            n, time_slots, lambda_VPPO_CDP_up, lambda_VPPO_CDP_down, 
            P_W, P_PV, P_delta, P_CDP, P_ES_c, P_ES_d, bids_W, bids_PV, p_setting_W, p_setting_PV, record_profit=True
        )
    
        # 更新当前最佳价格
        if PGC_profit is not None:
            if current_best_profit is None or PGC_profit > current_best_profit:
                current_best_prices = (lambda_VPPO_CDP_up.copy(), lambda_VPPO_CDP_down.copy(), lambda_VPPO_CDP.copy())
                current_best_profit = PGC_profit
                print(f"🎯 发现新的最佳解，利润: {current_best_profit:.2f}")