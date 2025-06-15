import self as self
import torch.nn as nn
import torch
import torch.nn.functional as F


class myNet_ocr(nn.Module):
    def __init__(self, cfg=None, num_classes=78, chinese_classes=43, export=False):
        super(myNet_ocr, self).__init__()  # 在PyTorch中，所有自定义神经网络都必须继承nn.Module并在构造函数中调用此父类初始化方法。
        if cfg is None:
            cfg = [16, 16, 32, 32, 'M', 64, 64, 'M', 96, 96, 'M', 128, 128]  # medium model
        self.export = export
        self.feature = self.make_layers(cfg, True)
        # self.classifier = nn.Linear(cfg[-1], num_classes)
        # self.loc =  nn.MaxPool2d((2, 2), (5, 1), (0, 1),ceil_mode=True)
        # self.loc =  nn.AvgPool2d((2, 2), (5, 2), (0, 1),ceil_mode=False)
        self.newCnn = nn.Conv2d(cfg[-1], num_classes, 1, 1)
        self.loc = nn.MaxPool2d((5, 2), (1, 1), (0, 1), ceil_mode=False)
        self.lstm = nn.LSTM(input_size=cfg[-1], hidden_size=cfg[-1], num_layers=2, bidirectional=True)
        self.attention = nn.Sequential(nn.Linear(256, 64), nn.Tanh(), nn.Linear(64, 1))
        self.fc = nn.Linear(cfg[-1] * 2, num_classes)
        # # 增加汉字识别专用分支
        # self.chinese_head = nn.Sequential(
        #     nn.Conv2d(256, 128, 3, padding=1),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(),
        #     nn.AdaptiveAvgPool2d((1, 1))
        # )
        # self.chinese_fc = nn.Linear(128, chinese_classes)  # 单独汉字分类

    # 新增SE Block模块
    class SEBlock(nn.Module):
        def __init__(self, channel, reduction=16):
            super().__init__()  # 更规范的父类调用方式
            self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 将特征图压缩到1x1，得到通道级全局信息
            # 激励阶段(Excitation)
            self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),  # 降维（默认缩减16倍）
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5),  # 添加dropout防止过拟合
                nn.Linear(channel // reduction, channel),  # 恢复原始通道数
                nn.Sigmoid()  # 产生0-1的通道权重
            )

        def forward(self, x):
            b, c, _, _ = x.size()
            y = self.avg_pool(x).view(b, c)
            y = self.fc(y).view(b, c, 1, 1)
            # 添加缩放因子和残差连接
            scale_factor = 0.8  # 平衡原始特征与注意力权重
            return x * (scale_factor * y + 0.2)  # 保持至少20%的原始特征

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

        # # 汉字专用分支
        # chinese_feat = self.chinese_head(x).squeeze()
        # chinese_pred = F.log_softmax(self.chinese_fc(chinese_feat), dim=-1)

        x = self.feature(x)
        x = self.loc(x)
        # x = self.newCnn(x)
        b, c, h, w = x.size()
        assert h == 1, "the height of conv must be 1"
        conv = x.squeeze(2)  # b *512 * width
        conv = conv.permute(2, 0, 1)  # [seq_len, batch, channels]
        # if not self.export:
        #     conv = F.log_softmax(conv, dim=2)  # 输出时在特征维度做log_softmax
        # return conv
        # 增强序列处理
        self.lstm.flatten_parameters()  # 确保参数在内存中是连续的，从而加速计算
        lstm_out, _ = self.lstm(conv)  # [seq_len, batch, hidden_size*2]
        # 注意力机制
        attn_weights = F.softmax(self.attention(lstm_out) * 0.5, dim=0)  # 温度系数0.5
        output = lstm_out * attn_weights
        # 全连接层处理
        output = output.permute(1, 0, 2)  # 调整为 [batch, seq_len, features]
        output = self.fc(output)  # [batch, seq_len, num_classes]
        output = output.permute(1, 0, 2)  # 恢复为 [seq_len, batch, num_classes]
        if not self.export:
            output = F.log_softmax(output, dim=2)  # 训练时在特征维度做log_softmax
        return output


if __name__ == '__main__':
    x = torch.randn(1, 3, 48, 168)
    cfg = [32, 'M', 64, 'M', 128, 'M', 256]
    model = myNet_ocr(num_classes=78, export=True, cfg=cfg)
    print(model)
    out = model(x)
    print(out.shape)
