import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt

# ==================== 小波变换工具函数 ====================

def create_1d_wavelet_filter(wave, in_size, out_size, type=torch.float):
    w = pywt.Wavelet(wave)
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=type)
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=type)
    dec_filters = torch.stack([dec_lo, dec_hi], dim=0)
    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1)

    rec_hi = torch.tensor(w.rec_hi, dtype=type)
    rec_lo = torch.tensor(w.rec_lo, dtype=type)
    rec_filters = torch.stack([rec_lo, rec_hi], dim=0)
    rec_filters = rec_filters[:, None].repeat(out_size, 1, 1)

    return dec_filters, rec_filters


def wavelet_1d_transform(x, filters):
    b, c, l = x.shape
    pad = (filters.shape[2] // 2 - 1)
    x = F.conv1d(x, filters, stride=2, groups=c, padding=pad)
    x = x.reshape(b, c, 2, l // 2)
    return x


def inverse_1d_wavelet_transform(x, filters):
    b, c, _, l_half = x.shape
    pad = (filters.shape[2] // 2 - 1)
    x = x.reshape(b, c * 2, l_half)
    x = F.conv_transpose1d(x, filters, stride=2, groups=c, padding=pad)
    return x


# ==================== 小波卷积模块（修复版） ====================

class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1.0, init_bias=0):
        super(_ScaleModule, self).__init__()
        self.dims = dims
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)
        self.bias = None
    
    def forward(self, x):
        return torch.mul(self.weight, x)


class WTConv1d(nn.Module):
    """
    一维小波卷积模块（保持时间维度不变）
    """
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, bias=True, 
                 wt_levels=1, wt_type='db1'):
        super(WTConv1d, self).__init__()

        assert in_channels == out_channels, "WTConv1d要求输入输出通道数相同"

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.wt_levels = wt_levels
        self.stride = stride

        # 创建小波滤波器
        self.wt_filter, self.iwt_filter = create_1d_wavelet_filter(
            wt_type, in_channels, in_channels, torch.float
        )
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)

        # 基础卷积（深度可分离）
        self.base_conv = nn.Conv1d(
            in_channels, in_channels, kernel_size, 
            padding='same', stride=1, dilation=1, 
            groups=in_channels, bias=bias
        )
        self.base_scale = _ScaleModule([1, in_channels, 1])

        # 小波域卷积（每层小波分解对应一个卷积）
        self.wavelet_convs = nn.ModuleList([
            nn.Conv1d(
                in_channels * 2, in_channels * 2, kernel_size, 
                padding='same', stride=1, dilation=1, 
                groups=in_channels * 2, bias=False
            ) for _ in range(self.wt_levels)
        ])
        self.wavelet_scale = nn.ModuleList([
            _ScaleModule([1, in_channels * 2, 1], init_scale=0.1) 
            for _ in range(self.wt_levels)
        ])

        # 下采样
        if self.stride > 1:
            self.do_stride = nn.AvgPool1d(kernel_size=stride, stride=stride)
        else:
            self.do_stride = None

    def forward(self, x):
        original_length = x.shape[2]  # 保存原始长度
        x_ll_in_levels = []
        x_h_in_levels = []
        shapes_in_levels = []

        curr_x_ll = x

        # 小波分解（前向）
        for i in range(self.wt_levels):
            curr_shape = curr_x_ll.shape
            shapes_in_levels.append(curr_shape)
            
            # 如果长度为奇数，进行填充
            if curr_shape[2] % 2 > 0:
                curr_pads = (0, curr_shape[2] % 2)
                curr_x_ll = F.pad(curr_x_ll, curr_pads)

            # 小波变换
            curr_x = wavelet_1d_transform(curr_x_ll, self.wt_filter)
            curr_x_ll = curr_x[:, :, 0, :]  # 低频分量
            
            # 在高频分量上应用卷积
            shape_x = curr_x.shape
            curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 2, shape_x[3])
            curr_x_tag = self.wavelet_scale[i](self.wavelet_convs[i](curr_x_tag))
            curr_x_tag = curr_x_tag.reshape(shape_x)

            x_ll_in_levels.append(curr_x_tag[:, :, 0, :])
            x_h_in_levels.append(curr_x_tag[:, :, 1:2, :])

        # 小波重构（反向）
        next_x_ll = 0
        for i in range(self.wt_levels - 1, -1, -1):
            curr_x_ll = x_ll_in_levels.pop()
            curr_x_h = x_h_in_levels.pop()
            curr_shape = shapes_in_levels.pop()

            curr_x_ll = curr_x_ll + next_x_ll
            curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)
            next_x_ll = inverse_1d_wavelet_transform(curr_x, self.iwt_filter)
            next_x_ll = next_x_ll[:, :, :curr_shape[2]]

        x_tag = next_x_ll
        
        # 确保小波分支输出长度与输入相同
        if x_tag.shape[2] != original_length:
            x_tag = F.interpolate(x_tag, size=original_length, mode='linear', align_corners=False)
        
        # 基础卷积分支 + 小波分支
        x = self.base_scale(self.base_conv(x))
        x = x + x_tag
        
        if self.do_stride is not None:
            x = self.do_stride(x)

        return x


# ==================== 基于WTConv1d的TCN模块 ====================

class Chomp1d(nn.Module):
    """裁剪因果卷积的额外填充"""
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class WTConvTemporalBlock(nn.Module):
    """
    使用WTConv1d的TCN时序块
    WTConv1d内部保持时间维度，但Chomp1d会裁剪padding
    """
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, 
                 dropout=0.2, wt_levels=1, wt_type='db1'):
        super(WTConvTemporalBlock, self).__init__()
        
        # 第一个WTConv1d层
        self.conv1x1_in = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        
        # 计算WTConv1d的有效padding（保持时间维度）
        # WTConv1d使用'same' padding，这个地方得改成因果padding
        self.wt_conv1 = WTConv1d(
            n_outputs, n_outputs, 
            kernel_size=kernel_size, 
            stride=1, 
            wt_levels=wt_levels, 
            wt_type=wt_type
        )
        # 手动裁剪最后的padding部分以实现因果性，感觉有点不合理，但是这样实现简单
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.wt_conv2 = WTConv1d(
            n_outputs, n_outputs, 
            kernel_size=kernel_size, 
            stride=1, 
            wt_levels=wt_levels, 
            wt_type=wt_type
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # 残差连接的下采样
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        
        self.init_weights()

    def init_weights(self):
        if self.conv1x1_in is not None:
            self.conv1x1_in.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        # 调整输入通道数
        if self.conv1x1_in is not None:
            out = self.conv1x1_in(x)
        else:
            out = x
            
        # 第一个WTConv1d分支
        out = self.wt_conv1(out)
        out = self.chomp1(out)  # 裁剪padding实现因果性
        out = self.relu1(out)
        out = self.dropout1(out)
        
        # 第二个WTConv1d分支
        out = self.wt_conv2(out)
        out = self.chomp2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        # 残差连接 - 确保维度匹配
        res = x if self.downsample is None else self.downsample(x)
        # 裁剪残差连接以匹配输出长度
        if res.shape[2] != out.shape[2]:
            res = res[:, :, :out.shape[2]]
        return self.relu(out + res)


# ==================== TCN分类器 ====================
# 4060训不动，后面有个简化版本的
class WTCNClassifier(nn.Module): 
    """
    基于WTConv1d的TCN分类器
    输入: (batch_size, 12, 10000)
    输出: (batch_size, num_classes)
    """
    def __init__(self, num_inputs=12, num_channels=[64, 128, 256], kernel_size=7, 
                 dropout=0.2, num_classes=5, wt_levels=2, wt_type='db3'):
        super(WTCNClassifier, self).__init__()
        
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            # 因果卷积的填充
            padding = (kernel_size - 1) * dilation_size
            
            layers += [WTConvTemporalBlock(
                in_channels, out_channels, kernel_size, 
                stride=1, dilation=dilation_size, 
                padding=padding, dropout=dropout,
                wt_levels=wt_levels, wt_type=wt_type
            )]
        
        self.network = nn.Sequential(*layers)
        
        # 全局池化和分类
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(num_channels[-1], num_classes)
        
    def forward(self, x):
        out = self.network(x)
        out = self.global_pool(out)
        out = out.squeeze(-1)
        out = self.fc(out)
        return out

# 这个简化版本勉强能训
class WTCNClassifierSimple(nn.Module):
    """
    输入: (batch_size, 12, 10000)
    输出: (batch_size, num_classes)
    """
    def __init__(self, num_inputs=12, num_channels=[64, 128, 256], kernel_size=7, 
                 dropout=0.2, num_classes=5, wt_levels=2, wt_type='db3'):
        super(WTCNClassifierSimple, self).__init__()
        
        # 只有一个WTConvTemporalBlock
        self.block1 = WTConvTemporalBlock(
            num_inputs, num_channels[0], kernel_size, 
            stride=1, dilation=1,  # dilation=1，不使用膨胀
            padding=kernel_size-1, dropout=dropout,
            wt_levels=wt_levels, wt_type=wt_type
        )
        self.block2 = WTConvTemporalBlock(
            num_channels[0], num_channels[1], kernel_size, 
            stride=1, dilation=1,  # dilation=1，不使用膨胀
            padding=kernel_size-1, dropout=dropout,
            wt_levels=wt_levels, wt_type=wt_type
        )
        self.block3 = WTConvTemporalBlock(
            num_channels[1], num_channels[2], kernel_size, 
            stride=1, dilation=1,  # dilation=1，不使用膨胀
            padding=kernel_size-1, dropout=dropout,
            wt_levels=wt_levels, wt_type=wt_type
        )
        
        
        # 全局池化和分类
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(num_channels[2], num_classes)
        
    def forward(self, x):
        out = self.block1(x)  # (batch, num_channels, seq_len)
        out = self.block2(out)  # (batch, num_channels, seq_len)
        out = self.block3(out)  # (batch, num_channels, seq_len)
        out = self.global_pool(out)  # (batch, num_channels, 1)
        out = out.squeeze(-1)  # (batch, num_channels)
        out = self.fc(out)  # (batch, num_classes)
        return out

# ==================== 测试 ====================

if __name__ == "__main__":
    batch_size = 2
    num_channels = 12
    seq_length = 10000
    num_classes = 5
    
    # 创建模型，注意，这个在4060上得训25-30个小时
    model = WTCNClassifierSimple(
        num_inputs=num_channels,
        num_channels=[64, 128, 256],
        kernel_size=7,
        dropout=0.3,
        num_classes=num_classes,
        wt_levels=2,      # 小波分解层数
        wt_type='haar'     # 小波类型：'db1', 'db2', 'db3', 'haar'等
    )
    
    # 测试输入
    x = torch.randn(batch_size, num_channels, seq_length)
    
    # 前向传播
    output = model(x)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"模型参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # 测试梯度回传
    loss = output.sum()
    loss.backward()
    print("梯度回传测试通过！")
    
    # 打印每层的参数量
    print("\n主要层参数量:")
    total_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad and 'wavelet' not in name:  # 跳过小波滤波器（不可训练）
            print(f"{name}: {param.numel():,} parameters")
            total_params += param.numel()
    print(f"\n总参数量（可训练）: {total_params:,}")