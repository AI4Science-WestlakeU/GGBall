import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 设置参数 a（实数，位于0到1之间）
a = 0.9

def mobius_transform(z, t):
    """
    渐进式莫比乌斯变换 f_t(z) = (z - t*a) / (1 - t*a*z)
    注意：当 a 为实数时，conjugate(a)=a
    """
    return (z - t * a) / (1 - t * a * z)

# 生成庞加莱圆盘内的网格（同心圆和径向线）
theta = np.linspace(0, 2*np.pi, 300)
radii = np.linspace(0.2, 0.9, 5)  # 同心圆半径
radial_angles = np.linspace(0, 2*np.pi, 12, endpoint=False)  # 径向线角度

# 创建原始网格的列表（每个网格线为一组复数点）
circles = [r * np.exp(1j*theta) for r in radii]
rays = [np.linspace(0, 1, 300) * np.exp(1j*angle) for angle in radial_angles]

# 准备图形
fig, ax = plt.subplots(figsize=(6,6))
ax.set_aspect('equal')
ax.set_xlim([-1.1, 1.1])
ax.set_ylim([-1.1, 1.1])
ax.axis('off')

# 绘制单位圆（边界，不变）
boundary, = ax.plot(np.cos(theta), np.sin(theta), color='black', lw=2)

# 初始绘制网格，存储绘图对象
circle_lines = [ax.plot(np.real(circle), np.imag(circle), color='blue', lw=1, alpha=0.6)[0] for circle in circles]
ray_lines = [ax.plot(np.real(ray), np.imag(ray), color='green', lw=1, alpha=0.6)[0] for ray in rays]

# 在原图上标出待映射的点 a （靠近边界）和其目标点 0
point_a, = ax.plot([a], [0], 'ro', label='Point a')
center_point, = ax.plot([0], [0], 'ko', label='Center')

# 添加红点局部坐标系的网格（颜色稍浅以示区分）
local_radii = np.linspace(0.1, 0.3, 3)  # 小半径同心圆
local_theta = np.linspace(0, 2*np.pi, 100)
local_circles = [a + r * np.exp(1j*local_theta) for r in local_radii]  # 以a为中心的初始位置
local_rays = [a + np.linspace(0, 0.3, 100) * np.exp(1j*ang) for ang in radial_angles]

local_circle_lines = [ax.plot([], [], color='cornflowerblue', lw=1, alpha=0.4)[0] for _ in local_circles]
local_ray_lines = [ax.plot([], [], color='lightgreen', lw=1, alpha=0.4)[0] for _ in local_rays]

# # 添加局部坐标系线对象（红色十字）
# local_x, = ax.plot([], [], 'r-', lw=1.5, alpha=0.8)
# local_y, = ax.plot([], [], 'r-', lw=1.5, alpha=0.8)

# 添加 5 个随机点（在单位圆内均匀分布）
np.random.seed(42)  # 固定随机种子保证可重复性
random_radii = np.sqrt(np.random.rand(5))  # 均匀分布在圆内
random_angles = np.random.rand(5) * 2 * np.pi
random_points = random_radii * np.exp(1j * random_angles)
random_dots = [ax.plot(np.real(pt), np.imag(pt), 'o', color='purple', alpha=0.6)[0] for pt in random_points]

ax.legend(loc='upper right')

def update(frame):
    # 多种 t 计算方式（取消注释其中一种）：
    
    # 1. 单向线性运动 (t: 0 → 1)
    # t = frame / 200  # 总时长保持 200 帧
    
    # 2. 双向循环运动 (t: 0 → 2 → 0)
    # t = (frame % 200) / 100  # 每 200 帧完成一个循环
    
    # 3. 正弦振荡运动 (当前生效模式)
    t = 0.5 + 0.5 * np.cos(frame * np.pi / 50)  # 平滑振荡
    
    # 4. 阶梯式增长 (每 50 帧前进 0.2)
    # t = (frame // 50) * 0.2
    
    # 5. 随机波动 (实验性)
    # t = 0.5 + 0.4 * np.random.randn()  # 带随机扰动的运动
    
    # 更新每条同心圆的坐标
    for line, circle in zip(circle_lines, circles):
        transformed = mobius_transform(circle, t)
        line.set_data(np.real(transformed), np.imag(transformed))
    # 更新径向线
    for line, ray in zip(ray_lines, rays):
        transformed = mobius_transform(ray, t)
        line.set_data(np.real(transformed), np.imag(transformed))
    # 更新待映射点的位置（映射 a，即 a -> f_t(a)）
    transformed_a = mobius_transform(a, t)
    point_a.set_data([np.real(transformed_a)], [np.imag(transformed_a)])
    
    # # 更新局部坐标系（十字坐标轴）
    # axis_length = 0.2  # 坐标系显示长度
    # x_center = np.real(transformed_a)
    # y_center = np.imag(transformed_a)
    # local_x.set_data([x_center - axis_length/2, x_center + axis_length/2], [y_center, y_center])
    # local_y.set_data([x_center, x_center], [y_center - axis_length/2, y_center + axis_length/2])

    # 更新随机点的位置（修复警告）
    for dot, pt in zip(random_dots, random_points):
        transformed_pt = mobius_transform(pt, t)
        # 将标量值转换为列表
        dot.set_data([np.real(transformed_pt)], [np.imag(transformed_pt)])
    
    # 更新红点局部坐标系网格
    for line, circle in zip(local_circle_lines, local_circles):
        transformed = mobius_transform(circle, t)
        line.set_data(np.real(transformed), np.imag(transformed))
    
    for line, ray in zip(local_ray_lines, local_rays):
        transformed = mobius_transform(ray, t)
        line.set_data(np.real(transformed), np.imag(transformed))

    return circle_lines + ray_lines + [point_a] + random_dots + local_circle_lines + local_ray_lines

# 将总帧数改为 200 以完成两个周期（0->2）
anim = FuncAnimation(fig, update, frames=200, interval=50, blit=True)

anim.save('poincare.gif', writer='pillow', fps=20)

# plt.title("庞加莱圆盘中莫比乌斯变换动画\n(将点 a 从接近边界移动到中心)")
# plt.show()