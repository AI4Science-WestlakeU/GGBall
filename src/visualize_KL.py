# import numpy as np
# import matplotlib.pyplot as plt

# # 指数映射函数：从切空间 R^2 映射到洛伦兹模型的 H^2
# def exp_map(v):
#     r = np.linalg.norm(v)
#     if r < 1e-8:
#         return np.array([1.0, 0.0, 0.0])
#     return np.array([np.cosh(r), (np.sinh(r)/r)*v[0], (np.sinh(r)/r)*v[1]])

# # 对数映射函数：从洛伦兹模型 H^2 映射回切空间 R^2（用于验证）
# def log_map(x):
#     # x: [x0, x1, x2] 在双曲面上，满足 x0^2 - x1^2 - x2^2 = 1, x0>0
#     r = np.arccosh(x[0])
#     if np.abs(r) < 1e-8:
#         return np.array([0.0, 0.0])
#     return r * np.array([x[1], x[2]]) / np.sinh(r)

# # 1. 从欧式空间采样二维高斯分布（均值0，协方差单位矩阵）
# N = 1000
# euclidean_samples = np.random.randn(N, 2)

# # 2. 可视化欧式空间的二维高斯样本
# plt.figure(figsize=(6,6))
# plt.scatter(euclidean_samples[:,0], euclidean_samples[:,1], alpha=0.5, s=15)
# plt.title("欧式空间中的二维高斯采样")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.axis('equal')
# plt.grid(True)
# plt.savefig("euclidean_samples.png")

# # 3. 将欧式样本映射到双曲洛伦兹空间（H^2，洛伦兹模型在 R^(1,2) 中）
# lorentz_points = np.array([exp_map(v) for v in euclidean_samples])

# # 验证映射后的点是否满足双曲面条件：x0^2 - x1^2 - x2^2 = 1
# residual = np.abs(lorentz_points[:,0]**2 - lorentz_points[:,1]**2 - lorentz_points[:,2]**2 - 1)
# print("最大残差：", np.max(residual))

# # 4. 可视化洛伦兹空间中的分布
# # 一种常用的方法是将洛伦兹模型投影到 Poincaré 圆盘：投影公式为
# #   p = (x1, x2) / (1 + x0)
# poincare = lorentz_points[:, 1:3] / (1 + lorentz_points[:, 0]).reshape(-1,1)

# plt.figure(figsize=(6,6))
# plt.scatter(poincare[:,0], poincare[:,1], alpha=0.5, s=15)
# plt.title("Poincaré 圆盘下的高斯分布 (经 exp_map 映射)")
# plt.xlabel("p1")
# plt.ylabel("p2")
# plt.axis('equal')
# plt.grid(True)
# # 圆盘边界
# circle = plt.Circle((0,0), 1, color='black', fill=False, linestyle='--')
# plt.gca().add_artist(circle)
# plt.savefig("lorentz_samples.png")

# # 5. （可选）利用对数映射验证 exp_map 的逆映射
# recovered_samples = np.array([log_map(x) for x in lorentz_points])
# error = np.mean(np.abs(euclidean_samples - recovered_samples))
# print("原始欧式样本与对数映射后样本的平均绝对误差：", error)

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 用于 3D 绘图
from scipy.stats import shapiro, chi2


# 支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# -------------------------------
# 定义指数映射与对数映射函数
# -------------------------------
def exp_map(v):
    """
    指数映射函数：从切空间 R^2 映射到洛伦兹模型的 H^2
    输入: v 为二维切向量
    输出: 对应洛伦兹模型上的点 (x0, x1, x2)
    """
    r = np.linalg.norm(v)
    if r < 1e-8:
        return np.array([1.0, 0.0, 0.0])
    return np.array([np.cosh(r), (np.sinh(r)/r)*v[0], (np.sinh(r)/r)*v[1]])

def log_map(x):
    """
    对数映射函数：从洛伦兹模型 H^2 上的点映射回切空间 R^2
    输入: x 为洛伦兹模型上的点 (x0, x1, x2)，满足 x0^2 - x1^2 - x2^2 = 1, x0>0
    输出: 对应的切向量
    """
    r = np.arccosh(x[0])
    if np.abs(r) < 1e-8:
        return np.array([0.0, 0.0])
    return r * np.array([x[1], x[2]]) / np.sinh(r)

# -------------------------------
# 1. 欧式空间中绘制二维格子并采样高斯分布
# -------------------------------
N = 500  # 采样点数量
mean = 1
euclidean_samples = np.random.randn(N, 2) +  mean # 均值为 0，协方差为单位矩阵

# 绘制二维格子背景
x_range = np.linspace(-4, 4, 50)
y_range = np.linspace(-4, 4, 50)

plt.figure(figsize=(8,8))
# 绘制网格线
for x in x_range:
    plt.axvline(x=x, color='lightgray', linewidth=0.5)
for y in y_range:
    plt.axhline(y=y, color='lightgray', linewidth=0.5)
# 绘制高斯采样点
plt.scatter(euclidean_samples[:,0], euclidean_samples[:,1], color='blue', s=20, alpha=0.6)
plt.title("euclidean space samples:", {mean})
plt.xlabel("x")
plt.ylabel("y")
plt.axis('equal')
plt.grid(False)
plt.savefig("euclidean_samples.png")

# -------------------------------
# 2. 将欧式高斯样本通过 exp_map 映射到洛伦兹模型上
# -------------------------------
lorentz_points = np.array([exp_map(v) for v in euclidean_samples])

# 验证映射后的点是否满足双曲面条件: x0^2 - x1^2 - x2^2 = 1
residuals = lorentz_points[:,0]**2 - lorentz_points[:,1]**2 - lorentz_points[:,2]**2
print("映射后点的最大残差（应接近0）：", np.max(np.abs(residuals - 1)))


tangent_samples = np.array([log_map(x) for x in lorentz_points])

# 计算切空间样本的均值和协方差矩阵
mean_tangent = np.mean(tangent_samples, axis=0)
cov_tangent = np.cov(tangent_samples.T)
print("切空间样本均值：", mean_tangent)
print("切空间样本协方差矩阵：\n", cov_tangent)

# 绘制切空间每个维度的直方图
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.hist(tangent_samples[:,0], bins=30, density=True, alpha=0.6, color='g')
plt.title("Tanget Space first dimension histogram")
plt.xlabel("value")
plt.ylabel("density")

plt.subplot(1,2,2)
plt.hist(tangent_samples[:,1], bins=30, density=True, alpha=0.6, color='b')
plt.title("Tanget Space second dimension histogram")
plt.xlabel("value")
plt.ylabel("density")
plt.tight_layout()
plt.savefig("tangent_samples_hist.png")

# 使用 Shapiro-Wilk 检验判断各维是否符合正态分布
stat1, p1 = shapiro(tangent_samples[:,0])
stat2, p2 = shapiro(tangent_samples[:,1])
print("第一维 Shapiro-Wilk 检验 p-value:", p1)
print("第二维 Shapiro-Wilk 检验 p-value:", p2)

# -------------------------------
# 3. 计算马氏距离平方并与卡方分布比较
# -------------------------------
# 计算每个样本的马氏距离平方
inv_cov = np.linalg.inv(cov_tangent)
mahal_sq = np.array([ (x - mean_tangent).T @ inv_cov @ (x - mean_tangent) for x in tangent_samples ])

# 绘制马氏距离平方的直方图，并绘制自由度为2的卡方分布密度曲线
plt.figure(figsize=(8,5))
plt.hist(mahal_sq, bins=30, density=True, alpha=0.6, color='c', label="mahalanobis distance square")
x_vals = np.linspace(0, np.max(mahal_sq), 100)
plt.plot(x_vals, chi2.pdf(x_vals, df=2), 'r--', label="chi2 distribution (df=2)")
plt.title("Comparison of Mahalanobis distance square and Chi2 distribution")
plt.xlabel("mahalanobis distance square")
plt.ylabel("density")
plt.legend()
plt.savefig("mahalanobis_chi2.png")


# -------------------------------
# 3. 在 3D 中绘制洛伦兹模型超曲面及映射后的样本点
# -------------------------------
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')

# 生成洛伦兹模型超曲面（参数化：x0 = cosh(v), x1 = sinh(v)*cos(u), x2 = sinh(v)*sin(u)）
u = np.linspace(0, 2*np.pi, 50)
v_param = np.linspace(0, 2.5, 50)  # v 参数越大，超曲面延伸得越远
U, V = np.meshgrid(u, v_param)
X_hyp = np.cosh(V)
Y_hyp = np.sinh(V) * np.cos(U)
Z_hyp = np.sinh(V) * np.sin(U)
ax.plot_surface(X_hyp, Y_hyp, Z_hyp, color='lightgray', alpha=0.3, rstride=2, cstride=2, edgecolor='none')

# 在超曲面上绘制映射后的样本点
ax.scatter(lorentz_points[:,0], lorentz_points[:,1], lorentz_points[:,2], color='red', s=20)

ax.set_xlabel("x0")
ax.set_ylabel("x1")
ax.set_zlabel("x2")
ax.set_title("lorentz model with samples:", {mean})
plt.savefig("lorentz_samples.png")

# -------------------------------
# 4. 验证映射是否保持高斯分布
#    通过对数映射将洛伦兹模型上的点映射回欧式切空间，
#    比较还原后的点与原始采样的差异
# -------------------------------
recovered_samples = np.array([log_map(x) for x in lorentz_points])
error = np.mean(np.linalg.norm(euclidean_samples - recovered_samples, axis=1))
print("原始欧式采样与逆映射后样本的平均误差：", error)

# 如果 error 非常小，则说明 exp_map 与 log_map 互为近似逆映射，
# 进而可以认为映射保持了高斯分布的特性。
