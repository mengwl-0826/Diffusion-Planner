import torch
import torch.nn as nn
import torch.nn.functional as F

class SimplifiedMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim  # 输入特征维度
        self.num_heads = num_heads  # 注意力头数量
        self.head_dim = embed_dim // num_heads  # 每个头的维度
        
        # 确保嵌入维度可被头数整除
        assert self.head_dim * num_heads == embed_dim, "嵌入维度必须是头数的整数倍"
        
        # 线性变换层：Q、K、V共享权重矩阵
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)  # 3倍用于Q、K、V
        self.out_proj = nn.Linear(embed_dim, embed_dim)      # 输出投影
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None):
        # query, key, value形状: (seq_len, batch_size, embed_dim)
        seq_len, batch_size, _ = query.shape
        
        # 1. 计算Q、K、V的线性变换
        # 输出形状: (seq_len, batch_size, 3*embed_dim)
        qkv = self.qkv_proj(query)
        
        # 拆分Q、K、V，并重塑为多头格式
        # 拆分后形状: (seq_len, batch_size, num_heads, 3*head_dim)
        qkv = qkv.view(seq_len, batch_size, self.num_heads, 3 * self.head_dim)
        q, k, v = torch.split(qkv, self.head_dim, dim=-1)  # 每个形状: (seq_len, batch_size, num_heads, head_dim)
        
        # 转置为 (batch_size, num_heads, seq_len, head_dim)，便于计算注意力
        q = q.transpose(0, 1).transpose(1, 2)  # (batch, heads, seq_len, head_dim)
        k = k.transpose(0, 1).transpose(1, 2)
        v = v.transpose(0, 1).transpose(1, 2)
        
        # 2. 计算缩放点积注意力
        # 计算Q·K^T / √(head_dim)
        attn_scores = torch.matmul(q, k.transpose(-2, -1))  # (batch, heads, seq_len, seq_len)
        attn_scores = attn_scores / (self.head_dim ** 0.5)
        
        # 应用掩码（可选）
        if attn_mask is not None:
            attn_scores = attn_scores + attn_mask  # 掩码值通常为-1e9，使softmax后权重接近0
        if key_padding_mask is not None:
            # key_padding_mask形状: (batch_size, seq_len)，扩展维度后应用
            attn_scores = attn_scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),  # 扩展为(batch, 1, 1, seq_len)
                -1e9
            )
        
        # 计算注意力权重
        attn_weights = F.softmax(attn_scores, dim=-1)  # (batch, heads, seq_len, seq_len)
        attn_weights = self.dropout(attn_weights)
        
        # 3. 注意力权重与V相乘
        output = torch.matmul(attn_weights, v)  # (batch, heads, seq_len, head_dim)
        
        # 4. 拼接所有头的结果
        output = output.transpose(1, 2).contiguous()  # (batch, seq_len, heads, head_dim)
        output = output.view(batch_size, seq_len, self.embed_dim)  # 拼接为(batch, seq_len, embed_dim)
        
        # 5. 输出投影
        output = self.out_proj(output)  # (batch, seq_len, embed_dim)
        
        # 转置回原始形状 (seq_len, batch_size, embed_dim)
        return output.transpose(0, 1), attn_weights
