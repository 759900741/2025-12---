import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson

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
P_demand = [13.1, 12.8, 12.4, 12, 11.7, 11.4, 11.1, 10.8, 10.5, 10.2, 9.7, 9.2, 8.7, 9.1, 9.3, 9.5, 9.7, 9.7, 9.7, 9.7, 9.7, 10, 10.2, 10.5, 10.7, 11, 11.2, 11.5, 11.7, 12.2, 12.7, 13.2, 12.7, 13.7, 14.7, 15.7, 14.7, 15.7, 16.7, 17.7, 17.2, 18.2, 19.2, 20.2, 19.2, 18.2, 17.7, 17.2, 16.7, 16.2, 15.7, 15.2, 14.7, 14.7, 14.7, 14.7, 14.7, 15, 15.2, 15.5, 15.7, 16.2, 16.7, 17.2, 17.7, 18.2, 18.7, 19.2, 19.7, 21, 22.2, 23.5, 24.7, 24.7, 24.7, 24.7, 24.7, 22.5, 22.2, 22, 21.7, 21.2, 20.7, 20.2, 19.7, 19.2, 18.7, 18.2, 17.7, 17.2, 16.7, 16.2, 15.7, 15.2, 14.7, 14.2]
P_demand = np.subtract(P_demand,P_TP_base) 
P_demand_all = (1500 - np.sum(P_TP_base))

df = pd.read_csv('price_income.csv')

ration = 0.6

# 计算风电和光伏预测出力
total_demand = np.sum(P_demand)
#print(total_demand)
    
P_W_portion = df['W']
total_W_portioin = np.sum(P_W_portion)
P_W_pre = ration * P_W_portion / total_W_portioin * total_demand
#print(P_W_pre)

P_PV_portion = df['PV'] 
total_PV_portioin = np.sum(P_PV_portion)
P_PV_pre = ration * P_PV_portion / total_PV_portioin * total_demand
#print(P_PV_pre)

t = 0
predictions_W = []
predictions_PV = []

def W_price_predict(t, demand, lambda_clear_W, P_W, P_PV, P_demand, P_W_pre, P_PV_pre, df_clean=None):
    """
    风电价格预测函数
    
    参数:
        demand_lo: 历史市场剩余电量
        bids_W_history: 历史风电报价
        bids_PV_history: 历史光伏报价
        P_W_history: 历史风电发电量
        P_PV_history: 历史光伏发电量
        P_demand: 预测期总需求
        P_W_pre: 预测期风电发电量
        P_PV_pre: 预测期光伏发电量
        df_clean: 包含历史数据的DataFrame，如果为None则从参数创建
        
    返回:
        dict: 包含系数和预测结果的字典
    """

    demand_lo = []
    P_W_history = []
    P_PV_history = []

    bids_W_history = []
    bids_PV_history = []

    for i in range(91):
        demand_lo_ij = np.array(demand.iloc[i, t]-P_TP.iloc[i, t])
        demand_lo.append(demand_lo_ij)
        P_W_history.append(P_W.iloc[i, t])
        P_PV_history.append(P_PV.iloc[i, t])
        bids_W_history.append(lambda_clear_W.iloc[i])
        bids_PV_history.append(lambda_clear_PV.iloc[i])
    
    # 如果未提供 df_clean，则从输入参数创建
    if df_clean is None:
        # 确保输入是一维数组
        demand_lo = np.array(demand_lo).flatten()
        bids_W_history = np.array(bids_W_history).flatten()
        P_W_history = np.array(P_W_history).flatten()
        P_PV_history = np.array(P_PV_history).flatten()
        
        # 获取最小长度
        min_length = min(len(demand_lo), len(bids_W_history), len(P_W_history), len(P_PV_history))
        
        # 截取相同长度
        demand_lo = demand_lo[:min_length]
        bids_W_history = bids_W_history[:min_length]
        P_W_history = P_W_history[:min_length]
        P_PV_history = P_PV_history[:min_length]
        
        # 创建 DataFrame
        df_clean = pd.DataFrame({
            'demand_lo': demand_lo,      # 市场剩余电量
            'W_power': P_W_history,      # 风电发电量
            'PV_power': P_PV_history,     # 光伏发电量
            'W_bid': bids_W_history
        })
        
        # 创建滞后变量
        #df_clean['demand_lo_lag1'] = df_clean['demand_lo'].shift(1)
        #df_clean['W_bid_lag1'] = df_clean['W_bid'].shift(1)
        #df_clean['PV_bid_lag1'] = df_clean['PV_bid'].shift(1)
        #df_clean['W_power_lag1'] = df_clean['W_power'].shift(1)
        #df_clean['PV_power_lag1'] = df_clean['PV_power'].shift(1)
    
        
        # 删除包含 NaN 的行
        df_clean = df_clean.dropna().reset_index(drop=True)
    
    # 计算标准化因子
    max_demand_lo = df_clean['demand_lo'].max()
    max_W_power = df_clean['W_power'].max()
    max_PV_power = df_clean['PV_power'].max()
    max_PV_power = max_PV_power if max_PV_power != 0 else 1
    
    # 创建标准化变量
    df_clean['demand_lo_norm'] = df_clean['demand_lo'] / max_demand_lo
    df_clean['W_power_norm'] = df_clean['W_power'] / max_W_power
    df_clean['PV_power_norm'] = df_clean['PV_power'] / max_PV_power
    
    #df_clean['demand_lo_lag1_norm'] = df_clean['demand_lo_lag1'] / sum_demand_lo
    #df_clean['W_power_lag1_norm'] = df_clean['W_power_lag1'] / sum_W_power
    #df_clean['PV_power_lag1_norm'] = df_clean['PV_power_lag1'] / sum_PV_power
    
    # 构建特征矩阵
    X = df_clean[['demand_lo_norm', 'W_power_norm', 'PV_power_norm']]
    X = sm.add_constant(X)
    y = df_clean['W_bid']
    
    # 拟合模型
    model = sm.OLS(y, X)
    results = model.fit()
    
    # 获取系数
    coefficients = results.params
    
    # ========== 预测部分 ==========
    # 确保预测数据是一维数组
    P_demand = np.array(P_demand).flatten()
    P_W_pre = np.array(P_W_pre).flatten()
    P_PV_pre = np.array(P_PV_pre).flatten()
    
    # 检查预测数据长度是否一致
    min_forecast_length = min(len(P_demand), len(P_W_pre), len(P_PV_pre))
    P_demand = P_demand[:min_forecast_length]
    P_W_pre = P_W_pre[:min_forecast_length]
    P_PV_pre = P_PV_pre[:min_forecast_length]
    
    # 标准化预测数据
    max_P_demand = max(P_demand)
    max_P_W_pre = max(P_W_pre)
    max_P_PV_pre = max(P_PV_pre)

    P_demand_norm = P_demand / max_P_demand
    W_power_pre_norm = P_W_pre / max_P_W_pre
    PV_power_pre_norm = P_PV_pre / max_P_PV_pre
    
    # 获取最后一个样本的值作为初始滞后项
    #last_demand_lo = df_clean['demand_lo'].iloc[-1]
    #last_W_bid = df_clean['W_bid'].iloc[-1]
    #last_PV_bid = df_clean['PV_bid'].iloc[-1]
    #last_W_power = df_clean['W_power'].iloc[-1]
    #last_PV_power = df_clean['PV_power'].iloc[-1]
    
    # 初始化滞后项（标准化）
    #demand_lo_lag1_norm = last_demand_lo / max_demand_lo
    #W_power_lag1_norm = last_W_power / max_W_power
    #PV_power_lag1_norm = last_PV_power / max_PV_power
    
    # 假设未来的报价使用历史平均值（可以根据需要修改）
    #W_bid_current_norm = df_clean['W_bid'].mean()
    #PV_bid_current_norm = df_clean['PV_bid'].mean()
    
    
    
    #for i in range(min_forecast_length):
        # 计算当前需求剩余（假设：需求剩余 = 总需求 - 新能源发电量）
        
        # 使用模型进行预测（标准化变量）
    prediction = (
        coefficients['const'] +
        coefficients['demand_lo_norm'] * P_demand_norm[t] +
        coefficients['W_power_norm'] * W_power_pre_norm[t] +
        coefficients['PV_power_norm'] * PV_power_pre_norm[t] 
    )
        
    # 转换回原始尺度
    predictions_W.append(prediction)
        
    
    # 准备返回结果
    result = {
        'coefficients': coefficients,
        'predictions': np.array(predictions_W),
        'r_squared': results.rsquared,
        'adjusted_r_squared': results.rsquared_adj,
        'sample_size': len(df_clean),
    }
    
    return result

def PV_price_predict(t, demand, lambdas_clear_PV, P_W, P_PV, P_demand, P_W_pre, P_PV_pre, df_clean=None):
    """
    光电价格预测函数
    
    参数:
        demand_lo: 历史市场剩余电量
        bids_W_history: 历史风电报价
        bids_PV_history: 历史光伏报价
        P_W_history: 历史风电发电量
        P_PV_history: 历史光伏发电量
        P_demand: 预测期总需求
        P_W_pre: 预测期风电发电量
        P_PV_pre: 预测期光伏发电量
        df_clean: 包含历史数据的DataFrame，如果为None则从参数创建
        
    返回:
        dict: 包含系数和预测结果的字典
    """

    demand_lo = []
    P_W_history = []
    P_PV_history = []

    bids_W_history = []
    bids_PV_history = []

    for i in range(91):
        demand_lo_ij = np.array(demand.iloc[i, t]-P_TP.iloc[i, t])
        demand_lo.append(demand_lo_ij)
        P_W_history.append(P_W.iloc[i, t])
        P_PV_history.append(P_PV.iloc[i, t])
        bids_W_history.append(lambda_clear_W.iloc[i])
        bids_PV_history.append(lambda_clear_PV.iloc[i])
    
    # 如果未提供 df_clean，则从输入参数创建
    if df_clean is None:
        # 确保输入是一维数组
        demand_lo = np.array(demand_lo).flatten()
        bids_PV_history = np.array(bids_PV_history).flatten()
        P_W_history = np.array(P_W_history).flatten()
        P_PV_history = np.array(P_PV_history).flatten()
        
        # 获取最小长度
        min_length = min(len(demand_lo), len(bids_PV_history), len(P_W_history), len(P_PV_history))
        
        # 截取相同长度
        demand_lo = demand_lo[:min_length]
        bids_PV_history = bids_PV_history[:min_length]
        P_W_history = P_W_history[:min_length]
        P_PV_history = P_PV_history[:min_length]
        
        # 创建 DataFrame
        df_clean = pd.DataFrame({
            'demand_lo': demand_lo,      # 市场剩余电量
            'W_power': P_W_history,      # 风电发电量
            'PV_power': P_PV_history,     # 光伏发电量
            'PV_bid': bids_PV_history
        })
        
        # 创建滞后变量
        #df_clean['demand_lo_lag1'] = df_clean['demand_lo'].shift(1)
        #df_clean['W_bid_lag1'] = df_clean['W_bid'].shift(1)
        #df_clean['PV_bid_lag1'] = df_clean['PV_bid'].shift(1)
        #df_clean['W_power_lag1'] = df_clean['W_power'].shift(1)
        #df_clean['PV_power_lag1'] = df_clean['PV_power'].shift(1)
    
        
        # 删除包含 NaN 的行
        df_clean = df_clean.dropna().reset_index(drop=True)
    
    # 计算标准化因子
    max_demand_lo = df_clean['demand_lo'].max()
    max_W_power = df_clean['W_power'].max()
    max_PV_power = df_clean['PV_power'].max()
    max_PV_power = max_PV_power if max_PV_power != 0 else 1
    
    # 创建标准化变量
    df_clean['demand_lo_norm'] = df_clean['demand_lo'] / max_demand_lo
    df_clean['W_power_norm'] = df_clean['W_power'] / max_W_power
    df_clean['PV_power_norm'] = df_clean['PV_power'] / max_PV_power
    
    #df_clean['demand_lo_lag1_norm'] = df_clean['demand_lo_lag1'] / sum_demand_lo
    #df_clean['W_power_lag1_norm'] = df_clean['W_power_lag1'] / sum_W_power
    #df_clean['PV_power_lag1_norm'] = df_clean['PV_power_lag1'] / sum_PV_power
    
    # 构建特征矩阵
    X = df_clean[['demand_lo_norm', 'W_power_norm', 'PV_power_norm']]
    X = sm.add_constant(X)
    y = df_clean['PV_bid']
    
    # 拟合模型
    model = sm.OLS(y, X)
    results = model.fit()
    
    # 获取系数
    coefficients = results.params
    
    # ========== 预测部分 ==========
    # 确保预测数据是一维数组
    P_demand = np.array(P_demand).flatten()
    P_W_pre = np.array(P_W_pre).flatten()
    P_PV_pre = np.array(P_PV_pre).flatten()
    
    # 检查预测数据长度是否一致
    min_forecast_length = min(len(P_demand), len(P_W_pre), len(P_PV_pre))
    P_demand = P_demand[:min_forecast_length]
    P_W_pre = P_W_pre[:min_forecast_length]
    P_PV_pre = P_PV_pre[:min_forecast_length]
    
    # 标准化预测数据
    max_P_demand = max(P_demand)
    max_P_W_pre = max(P_W_pre)
    max_P_PV_pre = max(P_PV_pre)

    P_demand_norm = P_demand / max_P_demand
    W_power_pre_norm = P_W_pre / max_P_W_pre
    PV_power_pre_norm = P_PV_pre / max_P_PV_pre
    
    # 获取最后一个样本的值作为初始滞后项
    #last_demand_lo = df_clean['demand_lo'].iloc[-1]
    #last_W_bid = df_clean['W_bid'].iloc[-1]
    #last_PV_bid = df_clean['PV_bid'].iloc[-1]
    #last_W_power = df_clean['W_power'].iloc[-1]
    #last_PV_power = df_clean['PV_power'].iloc[-1]
    
    # 初始化滞后项（标准化）
    #demand_lo_lag1_norm = last_demand_lo / max_demand_lo
    #W_power_lag1_norm = last_W_power / max_W_power
    #PV_power_lag1_norm = last_PV_power / max_PV_power
    
    # 假设未来的报价使用历史平均值（可以根据需要修改）
    #W_bid_current_norm = df_clean['W_bid'].mean()
    #PV_bid_current_norm = df_clean['PV_bid'].mean()
    
        # 计算当前需求剩余（假设：需求剩余 = 总需求 - 新能源发电量）
        
        # 使用模型进行预测（标准化变量）
    prediction = (
        coefficients['const'] +
        coefficients['demand_lo_norm'] * P_demand_norm[t] +
        coefficients['W_power_norm'] * W_power_pre_norm[t] +
        coefficients['PV_power_norm'] * PV_power_pre_norm[t] 
    )
        
    # 转换回原始尺度
    predictions_PV.append(prediction)
        
    
    # 准备返回结果
    result = {
        'coefficients': coefficients,
        'predictions': np.array(predictions_PV),
        'r_squared': results.rsquared,
        'adjusted_r_squared': results.rsquared_adj,
        'sample_size': len(df_clean),
    }
    
    return result

for t in range(96):
    result_W = W_price_predict(t, demand, lambda_clear_W, P_W, P_PV, 
                    P_demand, P_W_pre, P_PV_pre, df_clean=None)
    result_PV = PV_price_predict(t, demand, lambda_clear_PV, P_W, P_PV, 
                    P_demand, P_W_pre, P_PV_pre, df_clean=None)
W = result_W['predictions'] / 2
PV = result_PV['predictions'] / 2

df_W = pd.DataFrame({'W_predictions': W})
df_W.to_csv('wind_predictions.csv', index=False, encoding='utf-8-sig')
print(f"风电预测已保存到 wind_predictions.csv，包含 {len(W)} 条记录")

# 存储光伏预测
df_PV = pd.DataFrame({'PV_predictions': PV})
df_PV.to_csv('pv_predictions.csv', index=False, encoding='utf-8-sig')
print(f"光伏预测已保存到 pv_predictions.csv，包含 {len(PV)} 条记录")

result_PV = PV_price_predict(bids_W_history, bids_PV_history, P_W_history, P_PV_history)

# 输出结果
print("模型系数:")
for feature, coef in result_W['coefficients'].items():
    print(f"{feature}: {coef:.6f}")

print(f"\nR-squared: {result_W['r_squared']:.4f}")
print(f"调整R-squared: {result_W['adjusted_r_squared']:.4f}")
print(f"Durbin-Watson: {result_W['durbin_watson']:.3f}")
print(f"样本数量: {result_W['sample_size']}")

# 输出结果
print("模型系数:")
for feature, coef in result_PV['coefficients'].items():
    print(f"{feature}: {coef:.6f}")

print(f"\nR-squared: {result_PV['r_squared']:.4f}")
print(f"调整R-squared: {result_PV['adjusted_r_squared']:.4f}")
print(f"Durbin-Watson: {result_PV['durbin_watson']:.3f}")
print(f"样本数量: {result_PV['sample_size']}")