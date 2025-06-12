import self as self
import torch.nn as nn
import torch
import torch.nn.functional as F


class myNet_ocr(nn.Module):
    def __init__(self, cfg=None, num_classes=78, chinese_classes=43, export=False):
        super(myNet_ocr, self).__init__()  # 在PyTorch中，所有自定义神经网络都必须继承nn.Module并在构造函数中调用此父类初始化方法。
        if cfg is None:
            cfg = [32, 32, 64, 64, 'M', 128, 128, 'M', 196, 196, 'M', 256, 256]
            # cfg =[32,32,'M',64,64,'M',128,128,'M',256,256]
        self.feature = self.make_layers(cfg, True)
        self.export = export
        # self.classifier = nn.Linear(cfg[-1], num_classes)
        # self.loc =  nn.MaxPool2d((2, 2), (5, 1), (0, 1),ceil_mode=True)
        # self.loc =  nn.AvgPool2d((2, 2), (5, 2), (0, 1),ceil_mode=False)
        self.loc = nn.MaxPool2d((5, 2), (1, 1), (0, 1), ceil_mode=False)
        self.fc = nn.Linear(512, num_classes)
        self.lstm = nn.LSTM(input_size=256, hidden_size=256, num_layers=2, bidirectional=True)
        self.attention = nn.Sequential(nn.Linear(512, 128), nn.Tanh(), nn.Linear(128, 1))
        # 增加汉字识别专用分支
        self.chinese_head = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.chinese_fc = nn.Linear(128, chinese_classes)  # 单独汉字分类

    # 新增SE Block模块
    class SEBlock(nn.Module):
        def __init__(self, channel, reduction=16):
            super(myNet_ocr.SEBlock, self).__init__()
            self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 将特征图压缩到1x1，得到通道级全局信息
            # 激励阶段(Excitation)
            self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),  # 降维（默认缩减16倍）
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),  # 恢复原始通道数
                nn.Sigmoid()  # 产生0-1的通道权重
            )

        def forward(self, x):
            b, c, _, _ = x.size()
            y = self.avg_pool(x).view(b, c)
            y = self.fc(y).view(b, c, 1, 1)
            return x * y

    def make_layers(self, cfg, batch_norm=True):
        layers = []
        in_channels = 3
        for i, layer_cfg in enumerate(cfg):
            if i == 0:
                conv2d = nn.Conv2d(in_channels, cfg[i], kernel_size=5, stride=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(cfg[i]), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                layers.append(self.SEBlock(cfg[i]))  # 添加SE Block
                in_channels = cfg[i]
            else:
                if cfg[i] == 'M':
                    layers += [nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)]
                else:
                    conv2d = nn.Conv2d(in_channels, cfg[i], kernel_size=3, padding=(1, 1), stride=1)
                    if batch_norm:
                        layers += [conv2d, nn.BatchNorm2d(cfg[i]), nn.ReLU(inplace=True)]
                    else:
                        layers += [conv2d, nn.ReLU(inplace=True)]
                    layers.append(self.SEBlock(cfg[i]))  # 添加SE Block
                    in_channels = cfg[i]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.feature(x)
        # 汉字专用分支
        chinese_feat = self.chinese_head(x).squeeze()
        chinese_pred = F.log_softmax(self.chinese_fc(chinese_feat), dim=-1)

        x = self.loc(x)

        if self.export:
            conv = x.squeeze(2)  # b *512 * width
            conv = conv.transpose(2, 1)  # [w, b, c]
            return conv
        else:
            b, c, h, w = x.size()
            assert h == 1, "the height of conv must be 1"
            conv = x.squeeze(2)  # b *512 * width

            # 增强序列处理
            conv = conv.permute(2, 0, 1)  # [seq_len, batch, channels]是为了适应LSTM的输入格式
            self.lstm.flatten_parameters()  # 确保参数在内存中是连续的，从而加速计算
            lstm_out, _ = self.lstm(conv)  # [seq_len, batch, hidden_size*2]

            # 添加注意力机制
            attn_weights = F.softmax(self.attention(lstm_out), dim=0)
            output = lstm_out * attn_weights  # 保持三维结构 [seq_len, batch, features]
            output = output.permute(1, 0, 2)  # 调整为 [batch, seq_len, features]

            # 全连接层处理
            output = self.fc(output)  # [batch, seq_len, num_classes]
            output = F.log_softmax(output, dim=2)  # 在特征维度做softmax
            output = output.permute(1, 0, 2)  # 恢复为 [seq_len, batch, num_classes]

            return output, chinese_pred


if __name__ == '__main__':
    x = torch.randn(1, 3, 48, 168)
    cfg = [32, 'M', 64, 'M', 128, 'M', 256]
    model = myNet_ocr(num_classes=78, export=True, cfg=cfg)
    print(model)
    out = model(x)
    print(out.shape)
