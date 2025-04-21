import torch
from torch import nn
# from model.MultiHeadAttention import MultiHeadAttention
from torch import Tensor
import math
from einops import repeat
from einops.layers.torch import Rearrange
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class PositionalEncoding(nn.Module): # 位置编码
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1) # 生成一个max_len行1列的张量
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)) # 生成一个d_model/2行1列的张量
        pe = torch.zeros(max_len, 1, d_model) # 生成一个【max_len，1，d_model】的张量
        pe[:, 0, 0::2] = torch.sin(position * div_term)  # 偶数列
        pe[:, 0, 1::2] = torch.cos(position * div_term) # 奇数列
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:x.size(0)] # 位置编码
        return self.dropout(x) 


class EmbedPosEnc(nn.Module):
    def __init__(self, input_size, d_model):
        super(EmbedPosEnc, self).__init__()

        self.embedding = nn.Linear(input_size, d_model)
        #self.embedding = MultiScaleCNN(input_size, d_model)
        self.pos_enc = PositionalEncoding(d_model) # 位置编码

        self.arrange1 = Rearrange('b s e -> s b e')  # 重排列
        self.arrange2 = Rearrange('s b e -> b s e') # 重排列

    def forward(self, x, token):
        b = x.shape[0] # 获取批次大小
        y = self.embedding(x) # 嵌入
        token = repeat(token, '() s e -> b s e', b=b) # 重复token
        y = torch.cat([token, y], dim=1) # 拼接
        return self.arrange2(self.pos_enc(self.arrange1(y))) # 位置编码


class AttentionBlocks(nn.Module):
    def __init__(self, d_model, num_heads, rate=0.3, layer_norm_eps=1e-5):
        super(AttentionBlocks, self).__init__()

        self.att = nn.MultiheadAttention(d_model, num_heads=num_heads, batch_first=True) # 多头注意力
        self.drop = nn.Dropout(rate) 
        self.norm = nn.LayerNorm(d_model, eps=layer_norm_eps) # 归一化

    def forward(self, x, y=None):
        y = x if y is None else y # 如果y为空，则y=x
        att_out, att_w = self.att(x, y, y) # 多头注意力
        att_out = self.drop(att_out) # dropout
        y = self.norm(x + att_out) # 归一化
        return y




class Time_att(nn.Module): # 在时间维度上进行注意力
    def __init__(self, dims):
        super(Time_att, self).__init__()
        self.linear1 = nn.Linear(dims, dims, bias=False)
        self.linear2 = nn.Linear(dims, 1, bias=False)
        self.time = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        y = self.linear1(x.contiguous())
        y = self.linear2(torch.tanh(y))
        beta = F.softmax(y, dim=-1)
        c = beta * x
        return self.time(c.transpose(-1, -2)).transpose(-1, -2).contiguous().squeeze()


class TimeTransformer(nn.Module):
    """
    基于 QKV 的 Transformer 自注意力模块，
    用于对输入的时序特征进行融合，输出全局表示。
    """

    def __init__(self, d_model, num_heads, dropout):
        super(TimeTransformer, self).__init__()
        # 采用 PyTorch 内置的 MultiheadAttention
        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True)
        # 残差连接后 LayerNorm
        self.norm1 = nn.LayerNorm(d_model)
        # 前馈网络 (Feed Forward Network)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        # 实例化 AttentionPooling，d_model 应为一个整数
        self.attn_pool = AttentionPooling(d_model)
        # # 实例化 AttentionPooling，d_model 应为一个整数
        self.my_attn_pool = MyAttentionPooling(d_model)


    def forward(self, x):
        """
        输入：
          x: [B, T, d_model] 绿色特征时序表示
        输出：
          全局融合后的表示：[B, d_model]
        """
        # QKV 自注意力：输入的 Query, Key, Value 均为 x
        attn_output, _ = self.mha(x, x, x)  # 输出形状 [B, T, d_model]
        x = self.norm1(x + self.dropout(attn_output))

        # 前馈网络
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))

        # # 将时序信息融合成一个全局表示（例如用平均池化）
        # global_repr = torch.mean(x,dim=1)  # [B, d_model]

        # # 改为使用预先实例化的 self.attn_pool
        global_repr = self.attn_pool(x)  # [B, d_model]

        # 改为使用预先实例化的 self.my_attn_pool
        # global_repr = self.my_attn_pool(x)  # [B, d_model]

        return global_repr

class AttentionPooling(nn.Module):
    def __init__(self, d_model):
        super(AttentionPooling, self).__init__()
        # 利用一个全连接层将 token 特征映射到一个标量
        self.attn_fc = nn.Linear(d_model, 1, bias=False)

    def forward(self, x):
        """
        x: [B, T, d_model] 输入的时序特征
        输出:
            全局池化后的表示: [B, d_model]
        """
        # 计算每个 token 的注意力得分，形状 [B, T, 1]
        attn_scores = self.attn_fc(x)
        # 对时间步做 softmax 得到归一化的权重 [B, T, 1]
        attn_weights = F.softmax(attn_scores, dim=1)
        # 加权求和：将各 token 乘以权重后累加
        weighted_sum = torch.sum(attn_weights * x, dim=1)  # [B, d_model]
        return weighted_sum


class MyAttentionPooling(nn.Module):
    def __init__(self, dims):
        super(MyAttentionPooling, self).__init__()
        # 第一层将每个 token 的特征做线性变换，不带偏置
        self.linear1 = nn.Linear(dims, dims, bias=False)
        # 第二层将经过 tanh 激活后的结果映射为一个标量作为注意力得分
        self.linear2 = nn.Linear(dims, 1, bias=False)
        # 使用自适应平均池化将时间步降为 1
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [B, T, dims]，输入的时序特征
        Returns:
            Tensor of shape [B, dims]，全局融合后的表示
        """
        # 对输入做一次线性变换
        y = self.linear1(x.contiguous())
        # 应用 tanh 激活后再通过第二个线性层，将每个 token 映射为标量得分
        y = self.linear2(torch.tanh(y))  # 结果形状为 [B, T, 1]
        # 沿着时间维度（T 维度，即 dim=1）做 softmax 归一化，获得注意力权重
        beta = F.softmax(y, dim=1)  # [B, T, 1]
        # 将注意力权重与输入相乘，按加权方式聚合各时间步的特征
        weighted = beta * x  # [B, T, dims]
        # AdaptiveAvgPool1d 期望输入形状为 [B, C, L]，因此将维度交换为 [B, dims, T]
        weighted = weighted.transpose(1, 2)
        # 池化，将时间步降为1
        pooled = self.pool(weighted)  # [B, dims, 1]
        # 转换回 [B, 1, dims] 后 squeeze 得到 [B, dims]
        pooled = pooled.transpose(1, 2).contiguous().squeeze(1)
        return pooled