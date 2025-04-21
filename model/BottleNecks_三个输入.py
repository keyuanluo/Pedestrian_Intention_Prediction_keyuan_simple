import torch
from torch import nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Bottlenecks(nn.Module):  # 瓶颈结构
    def __init__(self, dims, args):
        super(Bottlenecks, self).__init__()
        self.dims = dims
        self.num_bnks = args.num_bnks  # 单元数目
        self.num_layers = args.bnks_layers  # 层数

        # 分别为 bbox, vel, acc 建立线性层列表
        self.bbox = nn.ModuleList()
        self.vel = nn.ModuleList()
        self.acc = nn.ModuleList()  # <-- 新增的

        # 第 1 层
        self.bbox.append(nn.Linear(dims, dims + self.num_bnks, bias=True))
        self.vel.append(nn.Linear(dims, dims + self.num_bnks, bias=True))
        self.acc.append(nn.Linear(dims, dims + self.num_bnks, bias=True))  # <-- 新增的

        # 后续 (num_layers - 1) 层
        for _ in range(self.num_layers - 1):
            self.bbox.append(nn.Linear(dims + self.num_bnks, dims + self.num_bnks, bias=True))
            self.vel.append(nn.Linear(dims + self.num_bnks, dims + self.num_bnks, bias=True))
            self.acc.append(nn.Linear(dims + self.num_bnks, dims + self.num_bnks, bias=True))  # <-- 新增的

        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def cut(self, x):
        """
        将 x 从最后一维切片：
        x[:, :, :self.dims]   --> 主体部分
        x[:, :, -self.num_bnks:] --> bottlenecks 特征
        """
        return x[:, :, :self.dims], x[:, :, -self.num_bnks:]

    def forward(self, bbox, vel, acc):
        """
        同时处理 bbox, vel, acc 三个输入
        bbox, vel, acc 的形状一般是 [batch_size, seq_len, dims]
        """
        # =========== 第 1 层 =============
        # 1) bbox
        out_bbox = self.bbox[0](bbox)        # [B, T, dims] -> [B, T, dims+num_bnks]
        out_bbox = self.dropout(self.relu(out_bbox))
        bbox_main, bnk_bbox = self.cut(out_bbox)

        # 2) vel
        out_vel = self.vel[0](vel)
        out_vel = self.dropout(self.relu(out_vel))
        vel_main, bnk_vel = self.cut(out_vel)

        # 3) acc
        out_acc = self.acc[0](acc)
        out_acc = self.dropout(self.relu(out_acc))
        acc_main, bnk_acc = self.cut(out_acc)

        # 将三者的 bottlenecks 部分进行合并
        # 这里示例为简单加和：bottlenecks = bnk_bbox + bnk_vel + bnk_acc
        # 你也可以改成 concat 或其他操作
        bottlenecks = bnk_bbox + bnk_vel + bnk_acc

        # =========== 后续层 (num_layers - 1) ===========
        for i in range(self.num_layers - 1):
            # bbox 的下一层输入：将主体部分 bbox_main 与 bottlenecks 拼接
            # 这里示例为在最后一维 cat
            cat_bbox = torch.cat((bbox_main, bottlenecks), dim=-1)
            out_bbox = self.bbox[i + 1](cat_bbox)
            out_bbox = self.dropout(self.relu(out_bbox))
            bbox_main, bnk_bbox = self.cut(out_bbox)

            # vel 的下一层输入
            cat_vel = torch.cat((vel_main, bottlenecks), dim=-1)
            out_vel = self.vel[i + 1](cat_vel)
            out_vel = self.dropout(self.relu(out_vel))
            vel_main, bnk_vel = self.cut(out_vel)

            # acc 的下一层输入
            cat_acc = torch.cat((acc_main, bottlenecks), dim=-1)
            out_acc = self.acc[i + 1](cat_acc)
            out_acc = self.dropout(self.relu(out_acc))
            acc_main, bnk_acc = self.cut(out_acc)

            # 再将本层得到的三个 bottleneck 特征合并
            bottlenecks = bnk_bbox + bnk_vel + bnk_acc
            # 如果你想加上 bnk_token 之类的，也可以在这里进行

        return bottlenecks
