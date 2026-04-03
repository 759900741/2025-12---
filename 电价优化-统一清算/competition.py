import pandas as pd
import statsmodels.api as sm
import numpy as np
from statsmodels.stats.stattools import durbin_watson

def W_price_predict(df, P_W_pre, P_PV_pre):
    # 创建滞后变量
    df['W_Ave_lag1'] = df['W_Ave'].shift(1)

    sum_W = df['W'].sum()
    sum_PV = df['PV'].sum()
    
    # 根据和进行标准化
    df['W_normalized'] = df['W'] / sum_W
    df['PV_normalized'] = df['PV'] / sum_PV

    # 删除包含NaN的行（由于滞后操作）
    df_clean = df.dropna().copy()
    # 使用标准化后的变量进行建模
    X = df_clean[['W_normalized', 'PV_normalized', 'W_Ave_lag1']]
    #X = df_clean[['W', 'PV', 'W_Ave_lag1']]
    X = sm.add_constant(X)
    y = df_clean['W_Ave']

    model_W = sm.OLS(y, X)
    results_W = model_W.fit()

    dw_W = durbin_watson(results_W.resid)
    if dw_W > 1.5 and dw_W < 2.5:
        print("✅ 无显著自相关问题")
    else:
        print("⚠️ 可能存在自相关")

    coefficients_W = results_W.params
    
    # 准备返回结果
    result = {
        'coefficients': coefficients_W
    }
    
    # 对预测数据进行相同的标准化处理
    W_pre_normalized = P_W_pre / sum_W
    PV_pre_normalized = P_PV_pre / sum_PV
    
    # 获取最后一个可用的 W_Ave 值作为第一个预测的滞后项
    last_W_Ave = df_clean['W_Ave'].iloc[-1]
    
    # 使用递归方式计算预测值（因为每个预测都依赖于前一个预测值）
    predictions = []
    current_lag = last_W_Ave
    
    for i in range(len(P_W_pre)):
        prediction = (coefficients_W['const'] + 
                     coefficients_W['W_normalized'] * W_pre_normalized[i] + 
                     coefficients_W['PV_normalized'] * PV_pre_normalized[i] + 
                     coefficients_W['W_Ave_lag1'] * current_lag)
        
        predictions.append(prediction)
        current_lag = prediction
    
    result = {
        'coefficients': coefficients_W,
        'prediction': np.array(predictions)
    }
    
    return result

def PV_price_predict(df, P_W_pre, P_PV_pre):
    # 创建滞后变量
    df['PV_Ave_lag1'] = df['PV_Ave'].shift(1)

    sum_W = df['W'].sum()
    sum_PV = df['PV'].sum()
    
    # 根据和进行标准化
    df['W_normalized'] = df['W'] / sum_W
    df['PV_normalized'] = df['PV'] / sum_PV

    # 删除包含NaN的行（由于滞后操作）
    df_clean = df.dropna().copy()
    # 使用标准化后的变量进行建模
    X = df_clean[['W_normalized', 'PV_normalized', 'PV_Ave_lag1']]
    #X = df_clean[['W', 'PV', 'W_Ave_lag1']]
    X = sm.add_constant(X)
    y = df_clean['PV_Ave']

    model_PV = sm.OLS(y, X)
    results_PV = model_PV.fit()

    dw_PV = durbin_watson(results_PV.resid)
    if dw_PV > 1.5 and dw_PV < 2.5:
        print("✅ 无显著自相关问题")
    else:
        print("⚠️ 可能存在自相关")

    coefficients_PV = results_PV.params

    # 准备返回结果
    result = {
        'coefficients': coefficients_PV
    }
    
    # 对预测数据进行相同的标准化处理
    W_pre_normalized = P_W_pre / sum_W
    PV_pre_normalized = P_PV_pre / sum_PV
        
    # 获取最后一个可用的 W_Ave 值作为滞后项
    last_PV_Ave = df_clean['PV_Ave'].iloc[-1]

    # 使用递归方式计算预测值（因为每个预测都依赖于前一个预测值）
    predictions = []
    current_lag = last_PV_Ave
        
    for i in range(len(P_PV_pre)):
        prediction = (coefficients_PV['const'] + 
                     coefficients_PV['W_normalized'] * W_pre_normalized[i] + 
                     coefficients_PV['PV_normalized'] * PV_pre_normalized[i] + 
                     coefficients_PV['PV_Ave_lag1'] * current_lag)
            
        predictions.append(prediction)
        current_lag = prediction
        
    result = {
        'coefficients': coefficients_PV,
        'prediction': np.array(predictions)
    }
    
    return result