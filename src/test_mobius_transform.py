import torch
import matplotlib.pyplot as plt
import numpy as np
# from src.manifolds.poincareball import PoincareBall
# from src.manifolds.poincare import PoincareBall
from src.models.hyperbolic_nn_plusplus.geoopt_plusplus.manifolds import PoincareBall
from src.mobius_transform import mobius_map_poincare_origin, mobius_map_poincare_origin_reverse
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation

def plot_poincare_disk(ax, points, color='blue', label=None):
    """在Poincare圆盘上绘制点"""
    # 创建单位圆
    circle = plt.Circle((0, 0), 1, fill=False, color='black')
    ax.add_artist(circle)
    
    # 绘制点
    points_np = points.detach().cpu().numpy()
    ax.scatter(points_np[:, 0], points_np[:, 1], color=color, label=label, alpha=0.6)
    
    # 设置坐标轴
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect('equal')
    ax.grid(True)
    if label:
        ax.legend()

def test_mobius_transform():
    # 创建Poincare流形
    # manifold = PoincareBall(dim=2, c=1.0)
    manifold = PoincareBall()
    # 生成一些随机点
    # num_points = 100
    # points = torch.randn(num_points, 2) * 0.3  # 生成靠近原点的点
    
    points = torch.tensor([[0.0, 0.0],[0.25, 0.25], [0,0.75], [0.25,0.75], [0.8,0.2]], dtype=torch.float32)

    # 确保点在单位圆内
    norm = torch.norm(points, dim=-1, keepdim=True)
    # points = points / (norm + 1e-6) * 0.8  # 缩放以确保点在圆内
    
    # 选择变换中心点
    z0 = torch.tensor([0.5, 0.5], dtype=torch.float32)
    
    # 创建Mobius变换
    transform = mobius_map_poincare_origin(z0, manifold)
    
    # 应用变换
    transformed_points = transform(points)
    
    # 可视化
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # 绘制原始点
    plot_poincare_disk(ax1, points, color='blue', label='原始点')
    ax1.scatter([z0[0]], [z0[1]], color='red', label='变换中心', s=100)
    ax1.set_title('变换前')
    
    # 绘制变换后的点
    plot_poincare_disk(ax2, transformed_points, color='green', label='变换后点')
    ax2.scatter([0], [0], color='red', label='新中心', s=100)
    ax2.set_title('变换后')
    
    plt.tight_layout()
    plt.savefig('mobius_transform_poincare.png')
    plt.show()
    
    # 打印一些统计信息
    print("原始点范数统计:")
    print(f"最小范数: {torch.norm(points, dim=1).min().item():.4f}")
    print(f"最大范数: {torch.norm(points, dim=1).max().item():.4f}")
    print(f"平均范数: {torch.norm(points, dim=1).mean().item():.4f}")
    
    print("\n变换后点范数统计:")
    print(f"最小范数: {torch.norm(transformed_points, dim=1).min().item():.4f}")
    print(f"最大范数: {torch.norm(transformed_points, dim=1).max().item():.4f}")
    print(f"平均范数: {torch.norm(transformed_points, dim=1).mean().item():.4f}")

def generate_evolution_animation():
    """生成Mobius变换的三个关键状态"""
    # 创建Poincare流形
    # manifold = PoincareBall(dim=2, c=1.0)
    manifold = PoincareBall()
    # 生成一些随机点
    points = torch.tensor([[0.0, 0.0],[0.25, 0.25], [0,0.75], [0.35,0.75], [0.81,0.2]], dtype=torch.float32)
    
    # 确保点在单位圆内
    norm = torch.norm(points, dim=-1, keepdim=True) 
    scale = torch.where(norm > 1, norm + 1e-1, torch.ones_like(norm))
    points = points / scale 
    
    # 选择变换中心点
    z0 = torch.tensor([0.5, 0.5], dtype=torch.float32)
    
    # 创建正向Mobius变换
    transform1 = mobius_map_poincare_origin(z0, manifold)
    
    # 创建反向Mobius变换
    transform2 = mobius_map_poincare_origin_reverse(z0, manifold)
    
    # 计算三个状态
    # 状态1: 原始状态
    state1_points = points
    state1_center = z0
    
    # 状态2: 正向变换后
    state2_points = transform1(points)
    state2_center = transform1(z0)
    
    # 状态3: 反向变换后
    state3_points = transform2(state2_points)
    state3_center = transform2(state2_center)
    
    # 计算点之间的距离
    def compute_distances(points):
        n = points.shape[0]
        distances = torch.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    distances[i, j] = torch.norm(points[i] - points[j])
        return distances
    

    dist_between_points = manifold.dist(state1_points[3], state1_points[4])
    dist_between_points_reverse = manifold.dist(state1_points[4], state1_points[3])
    import pdb; pdb.set_trace()
    dist1 = manifold.cdist(state1_points,state1_points)
    dist2 = manifold.cdist(state2_points,state2_points)
    dist3 = manifold.cdist(state3_points,state3_points)
    
    print(f"dist_between_points {dist_between_points}")
    print(f"dist_between_points_inverse {dist_between_points_reverse}")
    # 打印距离信息
    print("\n原始点之间的距离:")
    print(dist1)
    
    print("\n正向变换后点之间的距离:")
    print(dist2)
    
    print("\n反向变换后点之间的距离:")
    print(dist3)
    
    # 计算距离变化率
    dist_change_rate = torch.abs(dist2 - dist1) / (dist1 + 1e-6)
    print("\n正向变换后的距离变化率:")
    print(dist_change_rate)
    
    # 验证变换的可逆性
    error = torch.norm(state3_points - state1_points, dim=1).mean()
    print(f"\n恢复误差: {error.item():.6f}")
    
    # 创建图形
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # 绘制状态1
    circle1 = plt.Circle((0, 0), 1, fill=False, color='black')
    ax1.add_artist(circle1)
    ax1.scatter(state1_points[:, 0].cpu(), state1_points[:, 1].cpu(), 
                color='blue', alpha=0.6, label='变换点')
    ax1.scatter([state1_center[0]], [state1_center[1]], 
                color='red', s=100, label='基准点')
    ax1.set_title('状态1: 原始状态')
    ax1.set_xlim(-1.1, 1.1)
    ax1.set_ylim(-1.1, 1.1)
    ax1.set_aspect('equal')
    ax1.grid(True)
    ax1.legend()
    
    # 绘制状态2
    circle2 = plt.Circle((0, 0), 1, fill=False, color='black')
    ax2.add_artist(circle2)
    ax2.scatter(state2_points[:, 0].cpu(), state2_points[:, 1].cpu(), 
                color='blue', alpha=0.6, label='变换点')
    ax2.scatter([state2_center[0]], [state2_center[1]], 
                color='red', s=100, label='基准点')
    ax2.set_title('状态2: 正向变换后')
    ax2.set_xlim(-1.1, 1.1)
    ax2.set_ylim(-1.1, 1.1)
    ax2.set_aspect('equal')
    ax2.grid(True)
    ax2.legend()
    
    # 绘制状态3
    circle3 = plt.Circle((0, 0), 1, fill=False, color='black')
    ax3.add_artist(circle3)
    ax3.scatter(state3_points[:, 0].cpu(), state3_points[:, 1].cpu(), 
                color='blue', alpha=0.6, label='变换点')
    ax3.scatter([state3_center[0]], [state3_center[1]], 
                color='red', s=100, label='基准点')
    ax3.set_title('状态3: 反向变换后')
    ax3.set_xlim(-1.1, 1.1)
    ax3.set_ylim(-1.1, 1.1)
    ax3.set_aspect('equal')
    ax3.grid(True)
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig('mobius_transform_states.png')
    plt.show()
    
    # 打印每个状态的统计信息
    print("\n状态1 (原始状态) 统计:")
    print(f"点范数范围: [{torch.norm(state1_points, dim=1).min().item():.4f}, {torch.norm(state1_points, dim=1).max().item():.4f}]")
    print(f"基准点范数: {torch.norm(state1_center).item():.4f}")
    
    print("\n状态2 (正向变换后) 统计:")
    print(f"点范数范围: [{torch.norm(state2_points, dim=1).min().item():.4f}, {torch.norm(state2_points, dim=1).max().item():.4f}]")
    print(f"基准点范数: {torch.norm(state2_center).item():.4f}")
    
    print("\n状态3 (反向变换后) 统计:")
    print(f"点范数范围: [{torch.norm(state3_points, dim=1).min().item():.4f}, {torch.norm(state3_points, dim=1).max().item():.4f}]")
    print(f"基准点范数: {torch.norm(state3_center).item():.4f}")

if __name__ == "__main__":
    test_mobius_transform()
    generate_evolution_animation() 