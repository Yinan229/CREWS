import torch.nn as nn
import math
import torch
from torch.nn.utils import weight_norm
import torch.nn.functional as F

def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod

def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    nn.init.constant_(conv.bias, 0)

def conv_init(conv):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)

def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if hasattr(m, 'bias') and m.bias is not None and isinstance(m.bias, torch.Tensor):
            nn.init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.data.normal_(1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.data.fill_(0)

class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super(TemporalConv, self).__init__()
        pad = (kernel_size + (kernel_size-1) * (dilation-1) - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
            dilation=(dilation, 1))
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class MFE(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 dilations=[1,2,3,4],
                 residual=True,
                 residual_kernel_size=1):
        super().__init__()
        assert out_channels % (len(dilations) + 2) == 0, '# out channels should be multiples of # branches'
        self.num_branches = len(dilations) + 2
        branch_channels = out_channels // self.num_branches
        if type(kernel_size) == list:
            assert len(kernel_size) == len(dilations)
        else:
            kernel_size = [kernel_size]*len(dilations)
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    branch_channels,
                    kernel_size=1,
                    padding=0),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU(inplace=True),
                TemporalConv(
                    branch_channels,
                    branch_channels,
                    kernel_size=ks,
                    stride=stride,
                    dilation=dilation),
            )
            for ks, dilation in zip(kernel_size, dilations)
        ])
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3,1), stride=(stride,1), padding=(1,0)),
            nn.BatchNorm2d(branch_channels)
        ))
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0, stride=(stride,1)),
            nn.BatchNorm2d(branch_channels)
        ))
        self.apply(weights_init)

    def forward(self, x):
        branch_outs = []
        for tempconv in self.branches:
            out = tempconv(x)
            branch_outs.append(out)
        out = torch.cat(branch_outs, dim=1)
        return out

class GNN_TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GNN_TemporalConv, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(1, 1))

    def forward(self, x):
        x = self.conv(x)
        return x

class temp_1(nn.Module):
    def __init__(self, in_channels, out_channels, attn_norm="l1"):
        super(temp_1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.attn_norm = attn_norm
        self.conv1 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=[2,1])
        self.tcnA1 = GNN_TemporalConv(144, 16)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(out_channels)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)

    def forward(self, x, num_point, A=None, alpha=0.5):
        x1, x2, x3 = x.mean(-2), x.mean(-2), self.conv1(x)
        y1 = x1.unsqueeze(2)-x2.unsqueeze(3)*0
        y2 = x1.unsqueeze(3)-x2.unsqueeze(2)*0
        s = torch.cat([y1.unsqueeze(2),y2.unsqueeze(2)],dim=2).view(y1.size(0),self.in_channels,2,num_point*num_point)
        s = self.conv2(s).view(y1.size(0),self.out_channels,num_point,num_point)
        attention = self.tanh(s)
        if self.attn_norm == "l1":
            attention = attention / (attention.abs().sum(dim=-1, keepdim=True) + 1e-6)
        elif self.attn_norm == "none":
            pass
        else:
            raise ValueError(f"Unknown attn_norm={self.attn_norm}")
        x1 = torch.einsum('ncuv,nctv->nctu', attention, x3)
        x1 = self.bn(x1)
        x1 = self.relu(x1)
        x1 = x1.permute(0,2,1,3).contiguous()
        x1 = self.tcnA1(x1)
        return x1

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 =weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, dilation=dilation))
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv1(x)
        return out

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class CTE(nn.Module):
    def __init__(self, in_channels, out_channels, adaptive=True, residual=True):
        super(CTE, self).__init__()
        self.conv=TemporalConvNet(in_channels, num_channels=[in_channels, out_channels])
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)

    def forward(self, x):
        x = x.permute(0,3,1,2)
        x = x.squeeze(1)
        out = self.conv(x)
        out = out.unsqueeze(1)
        y = out.permute(0, 2, 3, 1)
        y = self.bn(y)
        y = self.relu(y)
        return y

class Bottom(nn.Module):
    def __init__(self, cte_in_channels, cte_out_channels, mfe_in,mfe_out,stride=1, residual=True, kernel_size=5, dilations=[1,2], t_in_channels=1, t_out_channels=1):
        super(Bottom, self).__init__()
        self.cte = CTE(cte_in_channels, cte_out_channels)
        self.mfe = MFE( mfe_in,mfe_out, kernel_size=kernel_size, stride=stride, dilations=dilations, residual=False)

    def forward(self, x):
        x = self.cte(x)
        x = x.permute(0,2,1,3).contiguous()
        x = self.mfe(x)
        x = x.permute(0,2,1,3).contiguous()
        return x

class GNN_FrequencyConv(nn.Module):
    def __init__(self, in_channels, out_channels, residual=True):
        super(GNN_FrequencyConv, self).__init__()
        self.convsA = temp_1(in_channels, out_channels)
        self.num_subset = 3
        self.convsA = nn.ModuleList()
        for i in range(self.num_subset):
            self.convsA.append(temp_1(in_channels, out_channels))
        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU(inplace=True)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)

    def forward(self, x, num_point):
        y = None
        for i in range(self.num_subset):
            z = self.convsA[i](x, num_point)
            y = z + y if y is not None else z
        y = self.bn(y)
        return y

class DTAL(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, residual=True, kernel_size=5, dilations=[1,2], t_in_channels=1, t_out_channels=1):
        super(DTAL, self).__init__()
        self.gcn1 = GNN_FrequencyConv(in_channels, out_channels)

    def forward(self, x, num_point):
        x = self.gcn1(x, num_point)
        x = x.permute(0,2,1,3).contiguous()
        return x

class edge_jetson_model(nn.Module):
    def __init__(self, num_blocks=1, in_channels=100, base_channel=64):
        super(edge_jetson_model, self).__init__()
        self.num_blocks = num_blocks
        self.data_bn1 = nn.InstanceNorm1d(in_channels, affine=True)
        in_ch = in_channels
        out_ch = base_channel
        all_layers_config = [
            ("bottom_1", Bottom(in_ch, out_ch, 797, 144))
        ]
        if num_blocks > 0:
            self.blocks = nn.ModuleDict()
            for name, layer in all_layers_config[:num_blocks]:
                self.blocks[name] = layer

    def forward(self, x):
        x = self.data_bn1(x)
        x = x.unsqueeze(-1)
        if self.num_blocks > 0:
            for name, layer in self.blocks.items():
                x = layer(x)
        return x

    def get_name(self):
        return "edge_jetson_model"

class edge_split_model(nn.Module):
    def __init__(self, num_blocks=2, in_channels=144, base_channel=64):
        super(edge_split_model, self).__init__()
        self.num_blocks = num_blocks
        self.identity_layer = nn.Identity()
        all_layers_config = [
            ("bottom_1", Bottom(in_channels, base_channel, 797, 144))
        ]
        if num_blocks < 1:
            self.blocks = nn.ModuleDict()
            for name, layer in all_layers_config[num_blocks:]:
                self.blocks[name] = layer

    def forward(self, x):
        x = self.identity_layer(x)
        if self.num_blocks < 1:
            for name, layer in self.blocks.items():
                x = layer(x)
        return x

    def get_name(self):
        return 'edge_split_model'

class server_bottom_model(nn.Module):
    def __init__(self,  drop_out=0.5):
        super(server_bottom_model, self).__init__()
        base_channel = 64
        out_channel = 16
        self.l2 = DTAL(base_channel, out_channel, residual=False, t_in_channels=144, t_out_channels=16)
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x

    def embed_nodes(self, xA, num_point, l2norm=True):
        """
        xA: [B, C, T, P]；num_point=P
        return: [B, P, d_phi]，其中 d_phi = C' * T'（DTAL 输出的 C'×T' 展平）
        """
        H = self.l2(xA, num_point=num_point)
        H = H.permute(0,2,1,3).contiguous()
        H = H.mean(dim=2)
        H = H.permute(0, 2, 1).contiguous()
        if l2norm:
            H = F.normalize(H, p=2, dim=-1)
        return H

    def forward(self, xA, num_point):
        N=xA.shape[0]
        xA = self.l2(xA,num_point=num_point)
        c_new = xA.size(2)
        xA = xA.permute(0,2,1,3).contiguous().view(N, c_new, -1)
        xA = xA.mean(2)
        return xA

    def get_name(self):
         return "server_bottom_model"

class server_top_model(nn.Module):
    def __init__(self, num_classA=6):
        super(server_top_model, self).__init__()
        self.num_classA = num_classA
        self.fcA = nn.Linear(16, num_classA)
        nn.init.normal_(self.fcA.weight, 0, math.sqrt(2. / num_classA))

    def forward(self, xA):
        xA = self.fcA(xA)
        return xA

    def get_name(self):
         return "server_top_model"
