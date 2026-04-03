import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 定义函数
def p_bar_sep(q_L):
    """计算上界价格"""
    return (1 + q_L)/2 + np.sqrt(1 + q_L)/2

def p_underline_sep(q_L):
    """计算下界价格"""
    return (1 + q_L)/2 - np.sqrt(1 + q_L)/2

def p_line(q_L, c_over_one_minus_gamma=2):
    """计算直线 p = (1 + q_L)/2 + c/(2*(1-γ))"""
    # c/(1-γ) = 2，所以 c/(2*(1-γ)) = 2/2 = 1.0
    return (1 + q_L)/2 + c_over_one_minus_gamma/2

# 创建q_L的范围 (q_L > 0)
q_min = 0.01
q_max = 10
q_values = np.linspace(q_min, q_max, 1000)

# 计算对应的价格
p_bar_values = p_bar_sep(q_values)
p_underline_values = p_underline_sep(q_values)
p_line_values = p_line(q_values)

# 计算较高的那条线（p_bar_sep 和 p_line 的较大值）
max_boundary = np.maximum(p_bar_values, p_line_values)

# 创建图形
plt.figure(figsize=(10, 7))

# 绘制三条曲线
plt.plot(q_values, p_bar_values, 'b-', linewidth=3, label=r'$\bar{p}^{sep}$')
plt.plot(q_values, p_underline_values, 'r-', linewidth=3, label=r'$p^{sep}$')
plt.plot(q_values, p_line_values, 'g--', linewidth=3, 
         label=r'$p = p_H^*$')

# 用黑色虚线描出p_bar_sep和p_line之间较高的那条线
plt.plot(q_values, max_boundary, 'k--', linewidth=2.5, 
         label='有效上界')

# 填充区间
plt.fill_between(q_values, p_underline_values, p_bar_values, alpha=0.15, color='gray')

# 添加垂直黑色实线（左侧，距离交点1.0）
q_left = 2.0  # 3.0 - 1.0 = 2.0
plt.axvline(x=q_left, color='black', linestyle='-', linewidth=1.2, alpha=0.7)

# 添加垂直黑色虚线（右侧，距离交点1.0）
q_right = 4.0  # 3.0 + 1.0 = 4.0
plt.axvline(x=q_right, color='black', linestyle='--', linewidth=1.2, alpha=0.7)

# 标记交点位置（可选，不添加标签）
q_intersection = 3.0
p_intersection = p_bar_sep(q_intersection)
plt.scatter([q_intersection], [p_intersection], color='purple', s=80, zorder=10, marker='o', alpha=0.7)

# 设置图形属性
plt.xlabel(r'$q_L$', fontsize=14)
plt.ylabel('价格', fontsize=14)
plt.title('价格随 $q_L$ 变化', fontsize=16)

# 将三条线的图例（标识）移到左边
plt.legend(fontsize=12, loc='upper left')

plt.grid(True, alpha=0.3)
plt.xlim([q_min, q_max])
plt.ylim([0, max(max(p_bar_values), max(p_line_values))])

# 将数学解释移到右侧，并与对应曲线位置接近
right_x = q_max * 0.7  # 使用q_max的70%作为x坐标

# 计算在right_x处各曲线的y值
q_ref = right_x
p_bar_at_ref = p_bar_sep(q_ref)
p_under_at_ref = p_underline_sep(q_ref)
p_line_at_ref = p_line(q_ref)

# 上界公式注释 - 放在蓝色曲线附近
plt.text(right_x, p_bar_at_ref + 0.3, r'上界: $\frac{1+q_L}{2} + \frac{\sqrt{1+q_L}}{2}$', 
         fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8),
         ha='left', va='bottom')

# 下界公式注释 - 放在红色曲线附近
plt.text(right_x, p_under_at_ref - 0.3, r'下界: $\frac{1+q_L}{2} - \frac{\sqrt{1+q_L}}{2}$', 
         fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.8),
         ha='left', va='top')

# 直线公式注释 - 放在绿色曲线附近
plt.text(right_x, p_line_at_ref, r'$p_H^*=\frac{1+q_L}{2} + \frac{c}{2(1-\gamma)}$', 
         fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8),
         ha='left', va='center')

# 在注释位置添加小标记点，显示注释与曲线的对应关系
plt.scatter([right_x], [p_bar_at_ref], color='blue', s=30, zorder=5, alpha=0.7)
plt.scatter([right_x], [p_under_at_ref], color='red', s=30, zorder=5, alpha=0.7)
plt.scatter([right_x], [p_line_at_ref], color='green', s=30, zorder=5, alpha=0.7)

plt.tight_layout()
plt.show()

# 创建第二个图：更清晰地显示有效上界
plt.figure(figsize=(10, 7))

# 绘制原始曲线（淡色）
plt.plot(q_values, p_bar_values, 'b-', linewidth=2, alpha=0.5, label=r'$\bar{p}^{sep}$')
plt.plot(q_values, p_line_values, 'g--', linewidth=2, alpha=0.5, label=r'$p_H^*$')

# 突出显示有效上界（最大值）
plt.plot(q_values, max_boundary, 'k--', linewidth=3, label='有效上界 (max)')

# 填充从下界到有效上界的区域
plt.fill_between(q_values, p_underline_values, max_boundary, 
                 alpha=0.2, color='lightgray', label='有效价格区间')

# 设置图形属性
plt.xlabel(r'$q_L$', fontsize=14)
plt.ylabel('价格', fontsize=14)
plt.title('有效上界：max($\bar{p}^{sep}$, $p_H^*$)', fontsize=16)
plt.legend(fontsize=12, loc='upper left')
plt.grid(True, alpha=0.3)
plt.xlim([q_min, q_max])
plt.ylim([0, max(max(p_bar_values), max(p_line_values))])

plt.tight_layout()
plt.show()

# 打印注释位置信息
print("注释位置信息：")
print(f"在 q_L = {q_ref:.1f} 处：")
print(f"  上界价格: {p_bar_at_ref:.4f}")
print(f"  下界价格: {p_under_at_ref:.4f}")
print(f"  直线价格: {p_line_at_ref:.4f}")
print(f"  有效上界: {max(p_bar_at_ref, p_line_at_ref):.4f}")

# 分析有效上界的变化
print(f"\n有效上界分析：")
print(f"在 q_L = {q_intersection:.2f} 处：")
print(f"  p_bar_sep = {p_bar_sep(q_intersection):.4f}")
print(f"  p_H^* = {p_line(q_intersection):.4f}")
print(f"  有效上界 = {max(p_bar_sep(q_intersection), p_line(q_intersection)):.4f}")

# 找出交点（p_bar_sep = p_H^* 的点）
diff_func = lambda q: p_bar_sep(q) - p_line(q)
# 寻找符号变化
sign_changes = np.where((p_bar_values[:-1] - p_line_values[:-1]) * 
                        (p_bar_values[1:] - p_line_values[1:]) <= 0)[0]

intersection_points = []
for idx in sign_changes:
    if idx + 1 < len(q_values):
        q1, q2 = q_values[idx], q_values[idx+1]
        p_bar1, p_bar2 = p_bar_values[idx], p_bar_values[idx+1]
        p_line1, p_line2 = p_line_values[idx], p_line_values[idx+1]
        
        # 线性插值
        t = (p_line1 - p_bar1) / ((p_bar2 - p_bar1) - (p_line2 - p_line1))
        if 0 <= t <= 1:
            q_intersect = q1 + t * (q2 - q1)
            p_intersect = p_bar1 + t * (p_bar2 - p_bar1)
            intersection_points.append((q_intersect, p_intersect))

print(f"\n交点分析：")
if intersection_points:
    for q_int, p_int in intersection_points:
        print(f"交点: q_L = {q_int:.4f}, p = {p_int:.4f}")
        print(f"  当 q_L < {q_int:.4f} 时，有效上界 = p_H^*")
        print(f"  当 q_L > {q_int:.4f} 时，有效上界 = p_bar_sep")
else:
    print("在给定范围内未找到交点")

# 打印数值表格
print(f"\n数值表格：")
print(f"{'q_L':<8} {'p_bar_sep':<12} {'p_H^*':<12} {'有效上界':<12} {'上界类型':<10}")
print("-" * 60)

for q in [q_left, q_intersection, q_right, 1.0, 2.0, 5.0, 10.0]:
    p_bar = p_bar_sep(q)
    p_l = p_line(q)
    max_val = max(p_bar, p_l)
    bound_type = "p_H^*" if p_l > p_bar else "p_bar_sep"
    print(f"{q:<8.2f} {p_bar:<12.4f} {p_l:<12.4f} {max_val:<12.4f} {bound_type:<10}")