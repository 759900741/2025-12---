import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 参数设置
c_over_one_minus_gamma = 2.0  # c/(1-γ)
theta = 0.5

# 定义函数
def p_H_star_star(q_L, c_over_one_minus_gamma=2.0, theta=0.5):
    """计算 p_H^{**}"""
    term1 = c_over_one_minus_gamma
    term2 = ((1 + q_L) * q_L) / (1 + q_L - theta)
    return 0.5 * (term1 + term2)

def p_bar_L_pool(q_L, theta=0.5):
    """计算 \bar{p}_L^{pool} (上界)"""
    denominator = 2 * (1 + q_L - theta)
    numerator = q_L * (1 + q_L + np.sqrt((1 + q_L) * theta))
    return numerator / denominator

def p_L_pool(q_L, theta=0.5):
    """计算 p_L^{pool} (下界)"""
    denominator = 2 * (1 + q_L - theta)
    numerator = q_L * (1 + q_L - np.sqrt((1 + q_L) * theta))
    return numerator / denominator

# 创建q_L的范围
q_min = 0.01
q_max = 10
q_values = np.linspace(q_min, q_max, 1000)

# 计算函数值
p_H_star_star_values = p_H_star_star(q_values)
p_bar_L_pool_values = p_bar_L_pool(q_values)
p_L_pool_values = p_L_pool(q_values)

# 计算较小的那条线（p_H^{**} 和 p_bar_L^{pool} 的较小值）
min_boundary = np.minimum(p_H_star_star_values, p_bar_L_pool_values)

# 找出交点位置（p_H^{**} = p_bar_L^{pool}）
diff = p_H_star_star_values - p_bar_L_pool_values
# 寻找符号变化的点
sign_changes = np.where(diff[:-1] * diff[1:] <= 0)[0]
intersection_points = []
for idx in sign_changes:
    if idx + 1 < len(q_values):
        # 线性插值找到精确交点
        q1, q2 = q_values[idx], q_values[idx+1]
        p_H1, p_H2 = p_H_star_star_values[idx], p_H_star_star_values[idx+1]
        p_bar1, p_bar2 = p_bar_L_pool_values[idx], p_bar_L_pool_values[idx+1]
        
        # 解线性方程
        t = (p_bar1 - p_H1) / ((p_H2 - p_H1) - (p_bar2 - p_bar1))
        if 0 <= t <= 1:
            q_intersect = q1 + t * (q2 - q1)
            p_intersect = p_H1 + t * (p_H2 - p_H1)
            intersection_points.append((q_intersect, p_intersect))

# 创建图形
plt.figure(figsize=(12, 8))

# 绘制三条曲线
plt.plot(q_values, p_H_star_star_values, 'm-', linewidth=3, 
         label=r'$p_H^{**}$')
plt.plot(q_values, p_bar_L_pool_values, 'c-', linewidth=3, 
         label=r'$\bar{p}_L^{pool}$')
plt.plot(q_values, p_L_pool_values, 'b-', linewidth=3, 
         label=r'$p_L^{pool}$')

# 用黑色虚线描出p_H^{**}和p_bar_L^{pool}之间较小的那条线
# 方法1：使用简单文本标签，避免LaTeX混合模式问题
plt.plot(q_values, min_boundary, 'k--', linewidth=2.5, 
         label='有效上界')

# 填充pooling价格区间（下界到有效上界）
plt.fill_between(q_values, p_L_pool_values, min_boundary, 
                 alpha=0.15, color='gray', label='有效价格区间')

# 标记交点
if intersection_points:
    for q_int, p_int in intersection_points:
        plt.scatter([q_int], [p_int], color='red', s=100, 
                    zorder=10, marker='o', label=f'交点')
        # 添加垂直线
        plt.axvline(x=q_int, color='gray', linestyle=':', 
                    linewidth=1.5, alpha=0.5)

# 添加网格和坐标轴
plt.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
plt.axvline(x=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)

# 设置图形属性
plt.xlabel(r'$q_L$', fontsize=14)
plt.ylabel('价格', fontsize=14)
plt.title(r'价格函数随 $q_L$ 变化（黑虚线为有效上界）', fontsize=16)

# 限制图例数量，避免重叠
handles, labels = plt.gca().get_legend_handles_labels()
# 只显示部分图例
plt.legend(handles[:6], labels[:6], fontsize=11, loc='upper left')

plt.grid(True, alpha=0.3)
plt.xlim([q_min, q_max])
max_y = max(max(p_H_star_star_values), max(p_bar_L_pool_values))
plt.ylim([0, max_y * 1.1])

# 添加说明文本（使用纯文本，避免LaTeX混合问题）
plt.text(q_max*0.65, max_y*0.8, 
         '有效上界 = min(p_H**, p̄_L^pool)', 
         fontsize=12, bbox=dict(boxstyle="round,pad=0.3", 
                                facecolor="yellow", alpha=0.8))

plt.tight_layout()
plt.show()

# 创建第二个图：更清晰地显示有效边界
plt.figure(figsize=(12, 8))

# 绘制三条原始曲线（淡色）
plt.plot(q_values, p_H_star_star_values, 'm-', linewidth=2, alpha=0.5, label=r'$p_H^{**}$')
plt.plot(q_values, p_bar_L_pool_values, 'c-', linewidth=2, alpha=0.5, label=r'$\bar{p}_L^{pool}$')
plt.plot(q_values, p_L_pool_values, 'b-', linewidth=2, alpha=0.5, label=r'$p_L^{pool}$')

# 突出显示有效边界
plt.plot(q_values, min_boundary, 'k--', linewidth=3, 
         label='有效上界')

# 填充有效区间
plt.fill_between(q_values, p_L_pool_values, min_boundary, 
                 alpha=0.2, color='lightgray', label='有效价格区间')

# 标记交点
if intersection_points:
    for i, (q_int, p_int) in enumerate(intersection_points):
        plt.scatter([q_int], [p_int], color='red', s=120, 
                    zorder=10, marker='o')
        plt.axvline(x=q_int, color='red', linestyle=':', 
                    linewidth=2, alpha=0.7)
        
        # 添加交点标注
        plt.annotate(f'交点', 
                     xy=(q_int, p_int),
                     xytext=(q_int+0.3, p_int+0.3),
                     arrowprops=dict(arrowstyle='->', color='red'),
                     fontsize=10, color='red')

# 设置图形属性
plt.xlabel(r'$q_L$', fontsize=14)
plt.ylabel('价格', fontsize=14)
plt.title(r'有效价格区间边界', fontsize=16)
plt.legend(fontsize=12, loc='upper left')
plt.grid(True, alpha=0.3)
plt.xlim([q_min, q_max])
plt.ylim([0, max_y * 1.1])

plt.tight_layout()
plt.show()

# 打印数值分析
print("参数设置：")
print(f"θ = {theta}")
print(f"c/(1-γ) = {c_over_one_minus_gamma}")

print("\n交点分析：")
if intersection_points:
    for i, (q_int, p_int) in enumerate(intersection_points):
        print(f"交点 {i+1}: q_L = {q_int:.4f}, p = {p_int:.4f}")
        print(f"  此时: p_H^{{**}} = {p_H_star_star(q_int):.4f}")
        print(f"        p̄_L^{{pool}} = {p_bar_L_pool(q_int):.4f}")
        print(f"        p_L^{{pool}} = {p_L_pool(q_int):.4f}")
        
        # 分析交点前后的情况
        q_before = q_int * 0.95
        q_after = q_int * 1.05
        
        print(f"\n  交点前 (q={q_before:.3f}):")
        p_H_before = p_H_star_star(q_before)
        p_bar_before = p_bar_L_pool(q_before)
        min_before = min(p_H_before, p_bar_before)
        bound_type_before = "p_H^{**}" if p_H_before < p_bar_before else "p̄_L^{pool}"
        print(f"    有效上界: {bound_type_before} = {min_before:.4f}")
        
        print(f"  交点后 (q={q_after:.3f}):")
        p_H_after = p_H_star_star(q_after)
        p_bar_after = p_bar_L_pool(q_after)
        min_after = min(p_H_after, p_bar_after)
        bound_type_after = "p_H^{**}" if p_H_after < p_bar_after else "p̄_L^{pool}"
        print(f"    有效上界: {bound_type_after} = {min_after:.4f}")
else:
    print("未发现交点")

print("\n数值表格（显示有效边界）：")
print(f"{'q_L':<8} {'p_H^{**}':<12} {'p̄_L^{pool}':<12} {'p_L^{pool}':<12} {'有效上界':<12} {'边界类型':<10} {'区间宽度':<12}")
print("-" * 90)

for q in [0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0]:
    p_H = p_H_star_star(q)
    p_bar_L = p_bar_L_pool(q)
    p_L = p_L_pool(q)
    
    # 确定有效上界
    if p_H < p_bar_L:
        effective_bound = p_H
        bound_type = "p_H^{**}"
    else:
        effective_bound = p_bar_L
        bound_type = "p̄_L^{pool}"
    
    interval_width = effective_bound - p_L
    
    print(f"{q:<8.2f} {p_H:<12.4f} {p_bar_L:<12.4f} {p_L:<12.4f} "
          f"{effective_bound:<12.4f} {bound_type:<10} {interval_width:<12.4f}")

print(f"\n函数表达式：")
print(r"1. $p_H^{**} = \frac{1}{2}\left(\frac{c}{1-\gamma} + \frac{(1+q_L)q_L}{1+q_L-\theta}\right)$")
print(r"2. $\bar{p}_L^{pool} = \frac{q_L}{2(1+q_L-\theta)}\left[1+q_L + \sqrt{(1+q_L)\theta}\right]$")
print(r"3. $p_L^{pool} = \frac{q_L}{2(1+q_L-\theta)}\left[1+q_L - \sqrt{(1+q_L)\theta}\right]$")
print(r"4. 有效上界 = $\min(p_H^{**}, \bar{p}_L^{pool})$")