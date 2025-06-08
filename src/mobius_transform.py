import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import torch

def mobius_map(z0, w0):
    """生成将z0映射到w0的莫比乌斯变换"""
    def transform(z):
        # 分两步变换：先映射到原点，再映射到w0
        t1 = (z - z0) / (1 - np.conj(z0)*z)
        return (t1 + w0) / (1 + np.conj(w0)*t1)
    return transform

# 原功能保持兼容
def mobius_to_origin(z0):
    """生成将z0映射到原点的莫比乌斯变换（兼容旧版）"""
    return mobius_map(z0, 0+0j)

def mobius_map_poincare_origin(z0, manifold):
    def transform(z):
       return manifold.mobius_add(-z0, z, project=False)
    return transform

def mobius_map_poincare_origin_reverse(z0, manifold):
    origin = torch.zeros_like(z0)
    z0_reverse = manifold.mobius_add(-z0, origin, project=False)
    print(f"z0_reverse is {z0_reverse}")
    def transform(z):
       return manifold.mobius_add(-z0_reverse, z, project=False)
    return transform

# # 将 0.5+0.5j 映射到 0.2+0.3j
# mobius_trans = mobius_map(z0=0.5+0.5j, w0=0.2+0.3j)

# # 原功能依然可用（映射到原点）
# mobius_trans_origin = mobius_to_origin(0.5+0.5j)

def plot_hyperbolic_disk(ax):
    """绘制庞加莱圆盘和网格线"""
    boundary = Circle((0, 0), 1, edgecolor='black', facecolor='none', lw=1)
    ax.add_patch(boundary)
    
    # 绘制双曲网格线
    theta = np.linspace(0, 2*np.pi, 100)
    for r in np.linspace(0.2, 0.8, 4):
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        ax.plot(x, y, color='gray', alpha=0.3, lw=0.5)
        
    ax.set_aspect('equal')
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)

def visualize_transform(z0, target_point):
    """可视化莫比乌斯变换效果"""
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # # 原始圆盘
    # plot_hyperbolic_disk(ax1)
    # ax1.scatter([0, z0.real], [0, z0.imag], c=['red', 'blue'], s=50)
    # ax1.set_title(f'Original Disk (z0={z0:.2f})')
    
    # # 生成测试点云用于展示形变效果
    # angles = np.linspace(0, 2*np.pi, 20)
    # radii = np.linspace(0.1, 0.9, 8)
    # points = [r * np.exp(1j*a) for a in angles for r in radii]
    
    # # 应用变换
    # transform = mobius_to_origin(target_point)
    # transformed = [transform(z) for z in points]
    # transformed_z0 = transform(z0)
    
    # # 变换后的圆盘
    # plot_hyperbolic_disk(ax2)
    # ax2.scatter([z.real for z in transformed],
    #             [z.imag for z in transformed], c='green', s=10, alpha=0.6)
    # ax2.scatter(transformed_z0.real, transformed_z0.imag, c='blue', s=50)
    # ax2.set_title(f'Transformed Disk (target moved to origin)')
    
    # # 添加连接线展示对应关系
    # for orig, t in zip(points[:30], transformed[:30]):
    #     ax1.plot([orig.real, z0.real], [orig.imag, z0.imag], 'k--', lw=0.5, alpha=0.3)
    #     ax2.plot([t.real, 0], [t.imag, 0], 'k--', lw=0.5, alpha=0.3)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # 原始圆盘
    plot_hyperbolic_disk(ax1)
    ax1.scatter([0, z0.real], [0, z0.imag], c=['red', 'blue'], s=50)
    # 生成测试点云（变换前）
    angles = np.linspace(0, 2*np.pi, 20)
    radii = np.linspace(0.1, 0.9, 8)
    points = [r * np.exp(1j*a) for a in angles for r in radii]
    ax1.scatter([z.real for z in points],  # 添加原始点云可视化
                [z.imag for z in points], 
                c='green', s=10, alpha=0.6, label='Original points')
    ax1.set_title(f'Original Disk (z0={z0:.2f})')
    
    # 应用变换
    transform = mobius_to_origin(target_point)
    transformed = [transform(z) for z in points]
    transformed_z0 = transform(z0)
    
    # 变换后的圆盘
    plot_hyperbolic_disk(ax2)
    ax2.scatter([z.real for z in transformed],  # 改为红色显示变换后点云
                [z.imag for z in transformed], 
                c='green', s=10, alpha=0.6, label='Transformed points')
    # 添加变换后的原点（原坐标系的0点）
    origin_transformed = transform(0)  # 计算原点经过变换后的位置
    ax2.scatter(origin_transformed.real, origin_transformed.imag,  # 新增红色点
            c='red', s=50, label='Transformed origin')
    ax2.scatter(transformed_z0.real, transformed_z0.imag, c='blue', s=50)
    ax2.set_title(f'Transformed Disk (target moved to origin)')
    
    # 添加连接线展示对应关系
    for orig, t in zip(points[:30], transformed[:30]):
        ax1.plot([orig.real, z0.real], [orig.imag, z0.imag], 
                'k--', lw=0.5, alpha=0.3)
        ax2.plot([t.real, 0], [t.imag, 0], 
                'k--', lw=0.5, alpha=0.3)
    
    plt.savefig('mobius_transform.png')

# 使用示例
if __name__ == "__main__":
    # 原点和目标点（用复数表示坐标）
    origin = 0 + 0j
    target = 0.5 + 0.5j  # 需要映射到原点的目标点
    
    # 创建将目标点映射到原点的变换
    mobius_trans = mobius_to_origin(target)
    
    # 验证变换效果
    print(f"变换后目标点坐标: {mobius_trans(target):.2f}")  # 应接近 (0+0j)
    print(f"变换后原点坐标: {mobius_trans(origin):.2f}")    # 应接近 (-0.5-0.5j)
    
    # 可视化变换效果
    visualize_transform(target, target)
