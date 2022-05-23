import torch
import torch.nn as nn
import torch.nn.functional as F


def default_conv(in_channels, out_channels, kernel_size, stride=1, padding=None, bias=True, groups=1):
    if not padding and stride == 1:
        padding = kernel_size // 2
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias, groups=groups)


class ESA(nn.Module):
    def __init__(self, n_feats, conv=default_conv):
        super(ESA, self).__init__()
        f = n_feats // 4
        self.conv1 = conv(n_feats, f, kernel_size=1)
        self.conv_f = conv(f, f, kernel_size=1)
        self.conv_max = conv(f, f, kernel_size=3, padding=1)
        self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = conv(f, f, kernel_size=3, padding=1)
        self.conv3_ = conv(f, f, kernel_size=3, padding=1)
        self.conv4 = conv(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        c1_ = (self.conv1(x))
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        v_range = self.relu(self.conv_max(v_max))
        c3 = self.relu(self.conv3(v_range))
        c3 = self.conv3_(c3)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)), mode='bilinear')
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3 + cf)
        m = self.sigmoid(c4)

        return x * m


class RFABlock(nn.Module):
    def __init__(self, n_feat):
        super(RFABlock, self).__init__()
        self.rb1 = ESA(n_feat)
        self.rb2 = ESA(n_feat)
        self.rb3 = ESA(n_feat)
        self.rb4 = ESA(n_feat)
        self.conv= default_conv(n_feat, n_feat, 3)

    def forward(self, x0):
        x1 = self.rb1(x0)
        x2 = self.rb2(x0+x1)
        x3 = self.rb3(x0+x1+x2)
        x4 = self.rb4(x0+x1+x2+x3)
        x5 = self.conv(x1+x2+x3+x4)
        out= x0+x5
        return out


class RFANet(nn.Module):
    def __init__(self, num_channels, base_filter, upscale_factor):
        super(RFANet, self).__init__()

        self.scale = upscale_factor
        self.layers = torch.nn.Sequential(
            nn.Conv2d(in_channels=num_channels, out_channels=base_filter, kernel_size=9, stride=1, padding=4, bias=True),
            nn.ReLU(inplace=True),

            RFABlock(base_filter),
            RFABlock(base_filter),
            # RFABlock(base_filter),
            # RFABlock(base_filter),
            # RFABlock(base_filter),
            # RFABlock(base_filter),
            # RFABlock(base_filter),
            # RFABlock(base_filter),
            # RFABlock(base_filter),
            # RFABlock(base_filter),
            # RFABlock(base_filter),
            # RFABlock(base_filter),
            # RFABlock(base_filter),
            # RFABlock(base_filter),
            # RFABlock(base_filter),
            # RFABlock(base_filter),

            nn.Conv2d(in_channels=base_filter, out_channels=base_filter // 2, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=base_filter // 2, out_channels=num_channels * (upscale_factor ** 2), kernel_size=5, stride=1, padding=2, bias=True),
            nn.PixelShuffle(upscale_factor)
        )

    def forward(self, x):
        out = self.layers(x)
        return  out


if __name__ == '__main__':
    import numpy as np
    device = torch.device('cuda:0')
    model = RFANet(num_channels=2, base_filter=32, upscale_factor=2).float().to(device)
    data = torch.from_numpy(np.random.random((1,2,100,100))).float().to(device)
    output = model(data)
    print(output.shape)