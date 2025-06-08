import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from src.manifolds.lorentz import Lorentz
from src.manifolds.poincareball import PoincareBall
from src.distribution.wrapped_normal import WrappedNormal
from src.models.hyper_layers import LorentzHypLinear, HypActivation
import torch.nn.functional as F
import pdb
import math

# 数据生成：生成一批随机的超曲面点 xi
def generate_batch_data(manifold_l, manifold_p, dim, batch_size):
    """
    Generate a batch of random points on the hyperboloid.
    """
    x = torch.tensor(manifold_l.random_normal(batch_size, dim + 1, mean=0, std=1))
    xi_prime = manifold_l.lorentz_to_poincare(x)  # 转换到庞加莱球
    xi = manifold_l.poincare_to_lorentz(xi_prime)  # 转换回洛伦兹空间
    
    mu = nn.Parameter(torch.zeros([x.shape[0], x.shape[1] - 1]), requires_grad=False)
    std = nn.Parameter(torch.zeros(1, 1), requires_grad=False)
    poincare_dis = WrappedNormal(mu.mul(1), F.softplus(std).div(math.log(2)).mul(0.1), manifold_p)
    x_p = poincare_dis.rsample()
    # x_p = torch.cat([x_p, x_p.sum(-1)], dim=-1)  ## 最后一个维度补0
    x_p_l = manifold_l.poincare_to_lorentz(x_p)
    
    x_all = torch.stack([xi, x_p_l], dim=1)
    
    
    return x, x_p, x_all


# 输入点x是在洛伦兹上面
# xi_prime是在庞加莱上面
# xi是回到了洛伦兹上面
# 我们的模型要从xi预测x


class FullyConnectedResNet(nn.Module):
    def __init__(self, manifold, input_dim, output_dim, hidden_dim=128, num_blocks=3):
        """
        Fully Connected ResNet for mapping y -> x.
        Parameters:
        - input_dim: Input feature dimension (dim of y)
        - output_dim: Output feature dimension (dim of x)
        - hidden_dim: Number of neurons in hidden layers
        - num_blocks: Number of residual blocks
        """
        super(FullyConnectedResNet, self).__init__()
        self.input_layer = LorentzHypLinear(manifold, input_dim, hidden_dim)
        self.blocks = nn.Sequential(
            *[ResidualBlock(manifold, hidden_dim) for _ in range(num_blocks)]
        )
        self.output_layer = LorentzHypLinear(manifold, hidden_dim, output_dim)
        self.relu = HypActivation(manifold, F.relu)

    def forward(self, x):
        x = self.relu(self.input_layer(x))
        x = self.blocks(x)
        x = self.output_layer(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, manifold, hidden_dim):
        """
        Residual block with two fully connected layers and a skip connection.
        """
        super(ResidualBlock, self).__init__()
        self.fc1 = LorentzHypLinear(manifold, hidden_dim, hidden_dim)
        self.fc2 = LorentzHypLinear(manifold, hidden_dim, hidden_dim)
        self.relu = HypActivation(manifold, F.relu)

    def forward(self, x):
        identity = x  # Skip connection
        out = self.relu(self.fc1(x))
        out = self.fc2(out)
        return self.relu(out + identity)
    

def main():
    # 超参数设置
    k = torch.tensor(1.0)
    dim = 127  # 空间维度
    learning_rate = 1e-4
    epochs = 10000
    batch_size = 64

    # 创建数据和网络
    input_dim = dim + 1
    output_dim = dim + 1  # 输出维度是 xi 的维度
    manifold = Lorentz()
    manifold_p = PoincareBall(dim=dim)
    model = FullyConnectedResNet(manifold, input_dim, output_dim)

    # 优化器和损失函数
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    # 新增变量用于保存各个损失值
    loss_history = []
    loss_l_history = []
    loss_p_history = []


    batch_size = 512
    total_loss = 0.0
    # 训练循环
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    for epoch in range(epochs):
        model.train()
        # for _ in range(batch_size):
        # 生成数据对
        x, x_p, x_all = generate_batch_data(manifold, manifold_p, dim, batch_size)

        # pdb.set_trace()
        # 前向传播
        output_all = model(x_all)
        x_l_prediction = output_all[:, 0, ...]
        x_p_prediction = output_all[:, 1, ...]
        loss_distance_l = manifold.dist(x_l_prediction, x, keepdim=False).mean()

        x_p_prediction = manifold.lorentz_to_poincare(x_p_prediction)
        loss_distance_p = manifold_p.dist(x_p_prediction, x_p, keepdim=False).mean()
        
        loss = loss_distance_l + loss_distance_p
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        # 保存损失值
        loss_history.append(loss.item())
        loss_l_history.append(loss_distance_l.item())
        loss_p_history.append(loss_distance_p.item())

        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item():.4f}, '
                f'Loss_L: {loss_distance_l.item():.4f}, Loss_P: {loss_distance_p.item():.4f}')

        
    # 绘制并保存损失曲线
    plt.figure()
    plt.plot(range(epochs), loss_history, label='Total Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Total Loss over Time (Training Process)')
    plt.legend()
    plt.savefig('total_loss.png')

    plt.figure()
    plt.plot(range(epochs), loss_l_history, label='Loss Distance L')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Distance L over Time')
    plt.legend()
    plt.savefig('loss_distance_l.png')

    plt.figure()
    plt.plot(range(epochs), loss_p_history, label='Loss Distance P')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Distance P over Time')
    plt.legend()
    plt.savefig('loss_distance_p.png')

    # 保存模型
    torch.save(model.state_dict(), 'lorentz_to_poincare_model.pth')
    print("Model saved to 'lorentz_to_poincare_model.pth'")
