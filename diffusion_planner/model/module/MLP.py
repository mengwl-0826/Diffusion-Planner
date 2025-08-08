import torch
import torch.nn as nn

class TwoLayerMLP(nn.Module):
    """
    两层线性层的MLP结构
    结构：输入 → 线性层1 → 激活函数 → 线性层2 → 输出
    """
    def __init__(self, in_dim, hidden_dim, out_dim, act=nn.ReLU(), dropout=0.0):
        super().__init__()
        # 第一层线性变换：输入维度 → 隐藏层维度
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        # 激活函数（引入非线性）
        self.act = act
        # 可选的Dropout层（防止过拟合）
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        # 第二层线性变换：隐藏层维度 → 输出维度
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        # 前向传播过程
        x = self.fc1(x)    # 线性变换1
        x = self.act(x)    # 非线性激活
        x = self.dropout(x)# （可选）Dropout
        x = self.fc2(x)    # 线性变换2
        return x


# 示例用法
if __name__ == "__main__":
    # 创建一个输入维度为100，隐藏层维度为50，输出维度为10的两层MLP
    mlp = TwoLayerMLP(in_dim=100, hidden_dim=50, out_dim=10)
    
    # 随机生成输入数据 (batch_size=32, in_dim=100)
    x = torch.randn(32, 100)
    
    # 前向传播
    output = mlp(x)
    
    # 输出形状应为 (32, 10)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")


# 激活函数（如 ReLU、Sigmoid、GELU 等）是确定性的非线性变换函数，仅对输入张量进行逐元素或逐特征的变换，没有需要优化的参数
# 虽然激活层没有参数梯度，但它们会通过链式法则影响反向传播时的梯度计算：
# 前向传播：激活层将输入 x 变换为输出 y = f(x)（f 为激活函数）。
# 反向传播：梯度从上层流回时，需要计算 dy/dx（激活函数的导数），并与上层梯度相乘，得到流向下层的梯度。
