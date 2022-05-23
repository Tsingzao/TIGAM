import torch.nn as nn
from Lib.MetaLib import default_conv, Upsampler


class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel//reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel//reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )
    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x*y

class RCAB(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat,n_feat,kernel_size,bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i==0: modules_body.append(act)
        modules_body.append(CALayer(n_feat,reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale
    def forward(self, x):
        res = self.body(x)
        res += x
        return res

class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = [
            RCAB(conv,n_feat,kernel_size,reduction,bias=True,bn=False,act=nn.ReLU(True),res_scale=1)
            for _ in range(n_resblocks)
        ]
        modules_body.append(conv(n_feat,n_feat,kernel_size))
        self.body = nn.Sequential(*modules_body)
    def forward(self, x):
        res = self.body(x)
        res += x
        return res

class RCAN(nn.Module):
    def __init__(self, inC, n_feat, upscale_factor, n_resblock, n_resgroup, reduction, pre, conv=default_conv):
        super(RCAN, self).__init__()
        self.n_resblock = n_resblock
        self.n_resgroup = n_resgroup
        self.inC = inC
        self.n_feat = n_feat
        self.scale = upscale_factor

        modules_head = [conv(self.inC, n_feat, 3)]
        modules_body = [
            ResidualGroup(conv,n_feat,3,reduction,n_resblock)
            for _ in range(n_resgroup)
        ]
        modules_body.append(conv(n_feat,n_feat,3))
        if pre:
            modules_tail = [conv(n_feat,self.inC,3)]
        else:
            modules_tail = [Upsampler(conv,self.scale,n_feat,act=False),
                            conv(n_feat,self.inC,3)]

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)
    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        res += x
        x = self.tail(res)
        return x