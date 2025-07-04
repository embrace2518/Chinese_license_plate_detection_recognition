import self as self
import torch.nn as nn
import torch
import torch.nn.functional as F


class myNet(nn.Module):
    def __init__(self, cfg=None, num_classes=78, export=False, se_scale_factor=0.8):
        super(myNet, self).__init__()  # 在PyTorch中，所有自定义神经网络都必须继承nn.Module并在构造函数中调用此父类初始化方法。
        if cfg is None:
            cfg = [16, 16, 32, 32, 'M', 64, 64, 'M', 96, 96, 'M', 128, 128]  # medium model
        self.export = export
        self.se_scale_factor = se_scale_factor
        self.feature = self.make_layers(cfg, True)
        self.newCnn = nn.Conv2d(cfg[-1], num_classes, 1, 1)
        self.loc = nn.MaxPool2d((5, 2), (1, 1), (0, 1), ceil_mode=False)
        # self.lstm = nn.LSTM(input_size=cfg[-1], hidden_size=cfg[-1], num_layers=2, bidirectional=True)
        self.attention = nn.Sequential(nn.Linear(256, 64), nn.Tanh(), nn.Linear(64, 1))
        self.gru = nn.GRU(cfg[-1], cfg[-1], num_layers=2, bidirectional=True, batch_first=False)
        self.fc = nn.Linear(cfg[-1] * 2, num_classes)

    # 新增SE Block模块
    class SEBlock(nn.Module):
        def __init__(self, channel, reduction=16, scale_factor=0.8):
            super().__init__()  # 更规范的父类调用方式
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            # 激励阶段(Excitation)
            self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),  # 降维（默认缩减16倍）
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),  # 恢复原始通道数
                nn.Sigmoid()  # 产生0-1的通道权重
            )
            self.scale_factor = scale_factor

        def forward(self, x):
            b, c, _, _ = x.size()
            y = self.avg_pool(x).view(b, c)
            y = self.fc(y).view(b, c, 1, 1)
            return x * (self.scale_factor * y + (1 - self.scale_factor))  # 保持至少20%的原始特征

    def make_layers(self, cfg, batch_norm=True):
        layers = []
        in_channels = 3
        for i, layer_cfg in enumerate(cfg):
            if cfg[i] == 'M':
                layers += [nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)]
            else:
                if i == 0:
                    conv2d = nn.Conv2d(in_channels, cfg[i], kernel_size=5, stride=1)
                else:
                    conv2d = nn.Conv2d(in_channels, cfg[i], kernel_size=3, padding=(1, 1), stride=1)
                layers.append(conv2d)
                if batch_norm:
                    layers.append(nn.BatchNorm2d(cfg[i]))
                layers += [self.SEBlock(cfg[i]), nn.ReLU(inplace=True)]  # 添加SE Block
                in_channels = cfg[i]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.feature(x)
        x = self.loc(x)
        b, c, h, w = x.size()
        assert h == 1, "the height of conv must be 1"
        conv = self.newCnn(x)
        x = x.squeeze(2).permute(2, 0, 1)  # [w b c]
        # GRU处理
        self.gru.flatten_parameters()
        gru_out, _ = self.gru(x)  # [w, b, hidden_size*2]
        # 注意力机制
        output = gru_out * F.softmax(self.attention(gru_out), dim=0)
        # 全连接层处理
        output = output.permute(1, 0, 2)  # [b, w, features]
        output = self.fc(output)  # [b, w, num_classes]
        output = output.permute(1, 0, 2)  # [w b c]
        if not self.export:
            output = F.log_softmax(output, dim=2)
            conv = F.log_softmax(conv.squeeze(2).permute(2, 0, 1), dim=2)
        return conv


if __name__ == '__main__':
    x = torch.randn(1, 3, 48, 168)
    cfg = [32, 'M', 64, 'M', 128, 'M', 256]
    model = myNet(num_classes=78, export=True, cfg=cfg)
    print(model)
    out = model(x)
    print(out.shape)
