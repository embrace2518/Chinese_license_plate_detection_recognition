import self as self
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.functional import conv2d


class myNet_ocr(nn.Module):
    def __init__(self, cfg=None, num_classes=78, export=False):
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
        self.newCnn = nn.Conv2d(cfg[-1], num_classes, 1, 1)
        self.newBn=nn.BatchNorm2d(num_classes)
        self.relu = nn.ReLU(inplace=True)

    # 新增SE Block模块
    class SEBlock(nn.Module):
        def __init__(self, channel, reduction=16):
            super(myNet_ocr.SEBlock, self).__init__()
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()
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
        x = self.loc(x)
        x = self.newCnn(x)
        x = self.relu(self.newBn(x))  # 增加激活和BN
        if self.export:
            conv = x.squeeze(2)  # b *512 * width
            conv = conv.transpose(2, 1)  # [w, b, c]
            return conv
        else:
            b, c, h, w = x.size()
            assert h == 1, "the height of conv must be 1"
            conv = x.squeeze(2)  # b *512 * width
            conv = conv.permute(2, 0, 1)  # [w, b, c]
            output = F.log_softmax(conv, dim=2)
            # output = torch.softmax(conv, dim=2)
            return output


if __name__ == '__main__':
    x = torch.randn(1, 3, 48, 168)
    cfg = [32, 'M', 64, 'M', 128, 'M', 256]
    model = myNet_ocr(num_classes=78, export=True, cfg=cfg)
    print(model)
    out = model(x)
    print(out.shape)
