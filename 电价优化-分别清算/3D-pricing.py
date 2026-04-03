import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.tri import Triangulation
from scipy.interpolate import griddata
import matplotlib

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # 支持中文的字体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

q_L = 2  # np.sqrt(3) / 2
t = 0.3

def income(q_L, r, c):
    """
    计算收入的函数
    """
    if c < np.sqrt(1+q_L)*(1-r):
        income_val = t*r*((1+q_L)/2+np.sqrt(1+q_L)/2)*(1/2-1/(2*np.sqrt(1+q_L)))+(1-t)*r*q_L/4
    else:
        income_val = t*r*((1+q_L)**2/4-(c/(2*(1-r)))**2)/(1+q_L)+(1-t)*r*q_L/4
    return income_val

# 创建密集的r和c网格
r_points = 100
c_points = 100
r_vals = np.linspace(0.01, 0.99, r_points)

# 创建参数化网格
R_list = []
C_list = []

for i, r in enumerate(r_vals):
    # 计算当前r对应的c_max
    c_max_r = (2*np.sqrt(1+q_L)-1+np.sqrt(8*q_L-8*np.sqrt(1+q_L)+5))/2 * (1-r)
    # 使用参数化的c值，确保边界平滑
    c_vals_r = np.linspace(0.01, c_max_r, c_points)
    
    for c in c_vals_r:
        R_list.append(r)
        C_list.append(c)

# 转换为numpy数组
R_array = np.array(R_list)
C_array = np.array(C_list)

# 计算pi值
Pi_list = [income(q_L, r, c) for r, c in zip(R_array, C_array)]
Pi_array = np.array(Pi_list)

# 创建三角化网格
tri = Triangulation(R_array, C_array)

# 创建三维光滑曲面图
fig = plt.figure(figsize=(16, 12))
ax = fig.add_subplot(111, projection='3d')

# 绘制三角化曲面 - 使用更鲜明的颜色映射
surf = ax.plot_trisurf(R_array, C_array, Pi_array, 
                      triangles=tri.triangles, 
                      cmap='plasma', alpha=0.95,  # 使用plasma颜色映射，更鲜明
                      edgecolor='none', antialiased=True,
                      vmin=np.min(Pi_array), vmax=np.max(Pi_array))  # 设置完整范围

# 绘制pi=0的平面
# 创建规则网格用于零平面
r_zero = np.linspace(0.01, 0.99, 50)
c_zero = np.linspace(0, np.max(C_array), 50)
R_zero, C_zero = np.meshgrid(r_zero, c_zero)
zero_plane = np.zeros_like(R_zero)

# 只绘制在有效区域内的零平面
mask = C_zero <= (2*np.sqrt(1+q_L)-1+np.sqrt(8*q_L-8*np.sqrt(1+q_L)+5))/2 * (1 - R_zero)
R_zero_masked = np.where(mask, R_zero, np.nan)
C_zero_masked = np.where(mask, C_zero, np.nan)
zero_plane_masked = np.where(mask, zero_plane, np.nan)

zero_surf = ax.plot_surface(R_zero_masked, C_zero_masked, zero_plane_masked, 
                           color='red', alpha=0.4, label='π=0 plane')

# 添加pi=0的等高线
# 插值到规则网格用于等高线
r_grid = np.linspace(0.01, 0.99, 1000)
c_grid = np.linspace(0.01, np.max(C_array), 1000)
R_grid, C_grid = np.meshgrid(r_grid, c_grid)

# 创建有效区域的掩码
valid_mask = C_grid <= (2*np.sqrt(1+q_L)-1+np.sqrt(8*q_L-8*np.sqrt(1+q_L)+5))/2 * (1 - R_grid)

# 插值pi值
Pi_grid = griddata((R_array, C_array), Pi_array, (R_grid, C_grid), method='cubic')
Pi_grid_masked = np.where(valid_mask, Pi_grid, np.nan)

# 在三维曲面上绘制pi=0的等高线
contour = ax.contour(R_grid, C_grid, Pi_grid_masked, 
                    levels=[0], colors='white', linewidths=4, linestyles='--')  # 改为白色更明显

# 在z=0平面上投影pi=0的等高线
ax.contour(R_grid, C_grid, Pi_grid_masked, 
          levels=[0], colors='red', linewidths=3, offset=0)

# 添加标签和标题
ax.set_xlabel('r', fontsize=14, labelpad=10)
ax.set_ylabel('c', fontsize=14, labelpad=10)
ax.set_zlabel('π', fontsize=14, labelpad=10)
ax.set_title('π 关于 r 和 c 的三维光滑曲面（红色平面为 π=0）', fontsize=16, pad=20)

# 添加颜色条
cbar = fig.colorbar(surf, ax=ax, shrink=0.6, aspect=20, pad=0.1)
cbar.set_label('π值', fontsize=12)

# 设置视角
ax.view_init(elev=25, azim=45)

# 设置坐标轴范围
ax.set_xlim([0, 1])
ax.set_ylim([0, np.max(C_array)])
ax.set_zlim([np.min(Pi_array), np.max(Pi_array)])

plt.tight_layout()
plt.show()

# 创建二维等高线图
fig2, ax2 = plt.subplots(figsize=(14, 10))

# 绘制填充等高线 - 使用更鲜明的颜色映射和更多层次
levels = 100  # 增加层次数使颜色过渡更平滑
contourf = ax2.contourf(R_grid, C_grid, Pi_grid_masked, levels=levels, cmap='plasma', alpha=0.9)

# 绘制等高线 - 减少数量并淡化
contour_lines = ax2.contour(R_grid, C_grid, Pi_grid_masked, levels=8, colors='white', alpha=0.4, linewidths=0.8)

# 特别标出pi=0的等高线 - 加粗并改为白色
zero_contour = ax2.contour(R_grid, C_grid, Pi_grid_masked, levels=[0], colors='white', linewidths=4, linestyles='--')
ax2.clabel(zero_contour, inline=True, fontsize=12, fmt='π=0', colors='white')

# 添加有效区域边界
r_boundary = np.linspace(0.01, 0.99, 100)
c_boundary = (2*np.sqrt(1+q_L)-1+np.sqrt(8*q_L-8*np.sqrt(1+q_L)+5))/2 * (1 - r_boundary)
c_mid_boundary = np.sqrt(1+q_L)*(1-r_boundary)
c_bd_1 = 2*np.sqrt(1+q_L)/(2+q_L/t)
#c_bd_2 = 
ax2.plot(r_boundary, c_boundary, 'w-', linewidth=2, label='有效区域边界', alpha=0.8)
ax2.plot(r_boundary, c_mid_boundary, 'w-', linewidth=2, label='成本计算分界线', alpha=0.8)

ax2.set_xlabel('r', fontsize=14)
ax2.set_ylabel('c', fontsize=14)
ax2.set_title('π 关于 r 和 c 的等高线图（颜色表示利润大小）', fontsize=16)
ax2.legend(fontsize=12)

# 添加颜色条
cbar2 = plt.colorbar(contourf, ax=ax2, label='π值', shrink=0.8)
cbar2.set_label('π值', fontsize=14)

plt.tight_layout()
plt.show()

# 创建第三个图：只显示颜色填充，不显示等高线，更清晰显示利润分布
fig3, ax3 = plt.subplots(figsize=(14, 10))

# 只使用颜色填充，不显示等高线
im = ax3.contourf(R_grid, C_grid, Pi_grid_masked, levels=levels, cmap='plasma', alpha=1.0)

# 只显示π=0的线
zero_contour3 = ax3.contour(R_grid, C_grid, Pi_grid_masked, levels=[0], colors='white', linewidths=4)
ax3.clabel(zero_contour3, inline=True, fontsize=14, fmt='π=0', colors='white')

# 添加边界
ax3.plot(r_boundary, c_boundary, 'w-', linewidth=2, alpha=0.8)
ax3.plot(r_boundary, c_mid_boundary, 'w-', linewidth=2, alpha=0.8)

ax3.set_xlabel('r', fontsize=14)
ax3.set_ylabel('c', fontsize=14)
ax3.set_title('利润分布图（颜色越亮表示利润越高）', fontsize=16)

# 添加颜色条
cbar3 = plt.colorbar(im, ax=ax3, label='π值', shrink=0.8)
cbar3.set_label('π值', fontsize=14)

plt.tight_layout()
plt.show()

# 分析信息
print(f"参数设置:")
print(f"q_L = {q_L:.4f}")
print(f"t = {t:.4f}")
print(f"r的范围: 0.01 - 0.99")
print(f"c的范围: 0.01 - {np.max(C_array):.4f}")
print(f"π的最小值: {np.min(Pi_array):.6f}")
print(f"π的最大值: {np.max(Pi_array):.6f}")
print(f"π为负值的区域比例: {np.sum(Pi_array < 0) / len(Pi_array) * 100:.2f}%")

# 显示颜色映射范围
pi_range = np.max(Pi_array) - np.min(Pi_array)
print(f"利润变化范围: {pi_range:.6f}")
print(f"利润变化倍数: {np.max(Pi_array)/max(0.0001, np.min(Pi_array)):.2f}倍")