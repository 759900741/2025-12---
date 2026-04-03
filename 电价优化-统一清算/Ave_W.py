import pandas as pd
import statsmodels.api as sm
import numpy as np

# 读取CSV文件
df = pd.read_csv('price_income.csv')

# 创建滞后变量
df['W_Ave_lag1'] = df['W_Ave'].shift(1)

# 删除包含NaN的行（由于滞后操作）
df_clean = df.dropna().copy()

# W_Ave动态回归模型
print("=" * 60)
print("W_Ave动态回归模型")
print("=" * 60)

X = df_clean[['W', 'PV', 'W_Ave_lag1']]
X = sm.add_constant(X)
y = df_clean['W_Ave']

model = sm.OLS(y, X)
results = model.fit()
print(results.summary())

# 模型评估
print("\n" + "=" * 60)
print("模型评估")
print("=" * 60)
print(f"R-squared: {results.rsquared:.4f}")
print(f"调整R-squared: {results.rsquared_adj:.4f}")
print(f"AIC: {results.aic:.2f}")
print(f"BIC: {results.bic:.2f}")

# 自相关诊断
print("\n" + "=" * 60)
print("自相关诊断")
print("=" * 60)
from statsmodels.stats.stattools import durbin_watson

dw = durbin_watson(results.resid)
print(f"Durbin-Watson: {dw:.3f}")

if dw > 1.5 and dw < 2.5:
    print("✅ 无显著自相关问题")
else:
    print("⚠️ 可能存在自相关")

# 拟合值分析
print("\n" + "=" * 60)
print("前10个拟合值分析")
print("=" * 60)
fitted_values = results.predict(X)
residuals = results.resid

print("Index | 实际值 | 拟合值 | 残差 | 残差百分比")
print("-" * 55)
for i in range(10):
    actual = y.iloc[i]
    fitted = fitted_values.iloc[i]
    residual = actual - fitted
    residual_pct = (residual / actual) * 100 if actual != 0 else 0
    print(f"{i+1:5} | {actual:6.2f} | {fitted:6.2f} | {residual:6.2f} | {residual_pct:7.1f}%")

# 模型精度指标
print(f"\n模型精度指标:")
print(f"平均绝对误差: {np.mean(np.abs(residuals)):.2f}")
print(f"均方根误差: {np.sqrt(np.mean(residuals**2)):.2f}")
print(f"平均绝对百分比误差: {np.mean(np.abs(residuals/y)) * 100:.1f}%")

# 输出最终模型方程
print("\n" + "=" * 60)
print("最终模型方程")
print("=" * 60)
coefficients = results.params
print(f"W_Ave_t = {coefficients['const']:.4f} + {coefficients['W']:.4f}×W_t + {coefficients['PV']:.4f}×PV_t + {coefficients['W_Ave_lag1']:.4f}×W_Ave_{{t-1}}")