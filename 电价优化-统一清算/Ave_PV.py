import pandas as pd
import statsmodels.api as sm
import numpy as np

# 读取CSV文件
df = pd.read_csv('price_income.csv')

# 创建滞后变量
df['PV_Ave_lag1'] = df['PV_Ave'].shift(1)

# 删除包含NaN的行（由于滞后操作）
df_clean = df.dropna().copy()

# 检查PV=0的情况
pv_zero_mask = df_clean['PV'] == 0
pv_zero_count = pv_zero_mask.sum()

print("=" * 60)
print("数据检查 - PV=0的情况")
print("=" * 60)
print(f"PV=0的观测值数量: {pv_zero_count}")
if pv_zero_count > 0:
    print(f"PV=0时PV_Ave的平均值: {df_clean[pv_zero_mask]['PV_Ave'].mean():.2f}")
    print(f"PV=0时PV_Ave的标准差: {df_clean[pv_zero_mask]['PV_Ave'].std():.2f}")
    print("\nPV=0的观测值详情:")
    print(df_clean[pv_zero_mask][['PV', 'PV_Ave', 'PV_Ave_lag1']].head())

# 方案1：仅对PV>0的数据建模（推荐）
print("\n" + "=" * 60)
print("PV_Ave动态回归模型 (仅PV>0数据)")
print("=" * 60)

# 只使用PV>0的数据
pv_positive_data = df_clean[df_clean['PV'] > 0].copy()

if len(pv_positive_data) > 0:
    X_positive = pv_positive_data[['W', 'PV', 'PV_Ave_lag1']]
    X_positive = sm.add_constant(X_positive)
    y_positive = pv_positive_data['PV_Ave']

    model_positive = sm.OLS(y_positive, X_positive)
    results_positive = model_positive.fit()
    print(results_positive.summary())

    # 模型评估
    print("\n" + "=" * 60)
    print("模型评估")
    print("=" * 60)
    print(f"R-squared: {results_positive.rsquared:.4f}")
    print(f"调整R-squared: {results_positive.rsquared_adj:.4f}")
    print(f"AIC: {results_positive.aic:.2f}")
    print(f"BIC: {results_positive.bic:.2f}")
    print(f"使用的观测值数量: {len(pv_positive_data)}")

    # 自相关诊断
    print("\n" + "=" * 60)
    print("自相关诊断")
    print("=" * 60)
    from statsmodels.stats.stattools import durbin_watson

    dw = durbin_watson(results_positive.resid)
    print(f"Durbin-Watson: {dw:.3f}")

    if dw > 1.5 and dw < 2.5:
        print("✅ 无显著自相关问题")
    else:
        print("⚠️ 可能存在自相关")

    # 输出最终模型方程
    print("\n" + "=" * 60)
    print("最终模型方程")
    print("=" * 60)
    coefficients = results_positive.params
    print(f"PV_Ave_t = {coefficients['const']:.4f} + {coefficients['W']:.4f}×W_t + {coefficients['PV']:.4f}×PV_t + {coefficients['PV_Ave_lag1']:.4f}×PV_Ave_{{t-1}}")
    print("\n适用条件: PV_t > 0")
    print("当 PV_t = 0 时，PV_Ave_t = 0")

    # 计算拟合值
    fitted_values = results_positive.predict(X_positive)
    print(f"\n前5个拟合值 (仅PV>0数据):")
    for i in range(min(5, len(pv_positive_data))):
        actual = y_positive.iloc[i]
        fitted = fitted_values.iloc[i]
        residual = actual - fitted
        residual_pct = (residual / actual) * 100 if actual != 0 else 0
        print(f"观测值 {i+1}: 实际值 = {actual:.2f}, 拟合值 = {fitted:.2f}, 残差 = {residual:.2f} ({residual_pct:+.1f}%)")

    # 预测函数（考虑PV=0约束）
    def predict_pv_ave(W_value, PV_value, last_PV_Ave):
        """
        改进的预测函数，考虑PV=0的约束
        """
        if PV_value == 0:
            return 0.0  # 理论约束：PV=0时，PV_Ave=0
        else:
            return (coefficients['const'] + 
                   coefficients['W'] * W_value + 
                   coefficients['PV'] * PV_value + 
                   coefficients['PV_Ave_lag1'] * last_PV_Ave)

    # 预测示例
    print("\n" + "=" * 60)
    print("预测示例")
    print("=" * 60)
    
    # 使用最后一行PV>0的数据作为基础
    if len(pv_positive_data) > 0:
        last_positive_row = pv_positive_data.iloc[-1]
        
        # 案例1：正常预测 (PV>0)
        pred1 = predict_pv_ave(
            W_value=last_positive_row['W'],
            PV_value=last_positive_row['PV'], 
            last_PV_Ave=last_positive_row['PV_Ave']
        )
        print(f"案例1 (PV>0): W={last_positive_row['W']:.2f}, PV={last_positive_row['PV']:.2f}, PV_Ave_lag1={last_positive_row['PV_Ave']:.2f}")
        print(f"预测结果: {pred1:.2f}")
        
        # 案例2：PV=0的情况
        pred2 = predict_pv_ave(W_value=2.0, PV_value=0, last_PV_Ave=50.0)
        print(f"\n案例2 (PV=0): W=2.0, PV=0, PV_Ave_lag1=50.0")
        print(f"预测结果: {pred2:.2f} (强制为0)")
        
        # 案例3：边界情况
        pred3 = predict_pv_ave(W_value=1.5, PV_value=0.1, last_PV_Ave=60.0)
        print(f"\n案例3 (PV>0): W=1.5, PV=0.1, PV_Ave_lag1=60.0")
        print(f"预测结果: {pred3:.2f}")

else:
    print("没有PV>0的数据可用于建模")

# 方案2：对比原始全数据模型（供参考）
print("\n" + "=" * 60)
print("对比：使用全部数据的模型（供参考）")
print("=" * 60)

X_all = df_clean[['W', 'PV', 'PV_Ave_lag1']]
X_all = sm.add_constant(X_all)
y_all = df_clean['PV_Ave']

model_all = sm.OLS(y_all, X_all)
results_all = model_all.fit()

print(f"全数据模型 R-squared: {results_all.rsquared:.4f}")
print(f"仅PV>0模型 R-squared: {results_positive.rsquared:.4f}")

# 检查PV=0时的预测问题
if pv_zero_count > 0:
    print(f"\nPV=0时的预测问题分析:")
    pv_zero_predictions = results_all.predict(X_all[pv_zero_mask])
    print(f"PV=0时模型的平均预测值: {pv_zero_predictions.mean():.2f}")
    print(f"PV=0时模型预测的最小值: {pv_zero_predictions.min():.2f}")
    print(f"PV=0时模型预测的最大值: {pv_zero_predictions.max():.2f}")
    print("→ 这违反了PV=0时PV_Ave应该为0的理论约束")