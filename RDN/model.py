import torch
import torch.nn as nn

class RDB_Conv(nn.Module):
    def __init__(self, inC, n_feat, kSize=3):
        super(RDB_Conv, self).__init__()
        self.conv = nn.Sequential(*[
            nn.Conv2d(inC, n_feat, kSize, padding=(kSize-1)//2, stride=1),
            nn.ReLU()
        ])
    def forward(self, x):
        out = self.conv(x)
        return torch.cat([x, out], 1)

class RDB(nn.Module):
    def __init__(self, inC, n_feat, n_block):
        super(RDB, self).__init__()
        convs = []
        for c in range(n_block):
            convs.append(RDB_Conv(inC+c*n_feat, n_feat))
        self.convs = nn.Sequential(*convs)
        self.LFF = nn.Conv2d(inC+n_block*n_feat, inC, 1, padding=0, stride=1)
    def forward(self, x):
        return self.LFF(self.convs(x))+x

class RDN(nn.Module):
    def __init__(self, inC, n_feat, upscale_factor, cfg):
        super(RDN, self).__init__()
        self.inC = inC
        self.n_feat = n_feat
        self.scale = upscale_factor
        self.D, C, G = {'A': (20, 6, 32), 'B': (16, 8, 64)}[cfg]

        self.SFENet1 = nn.Conv2d(inC, n_feat, 3, padding=1, stride=1)
        self.SFENet2 = nn.Conv2d(n_feat, n_feat, 3, padding=1, stride=1)
        self.RDBs = nn.ModuleList()
        for i in range(self.D):
            self.RDBs.append(RDB(n_feat, G, C))

        self.GFF = nn.Sequential(*[
            nn.Conv2d(self.D*n_feat, n_feat, 1, padding=0, stride=1),
            nn.Conv2d(n_feat, n_feat, 3, padding=1, stride=1)
        ])

        if self.scale == 2 or self.scale == 3 or self.scale == 5:
            self.UPNet = nn.Sequential(*[
                nn.Conv2d(n_feat, G * self.scale * self.scale, 3, padding=1, stride=1),
                nn.PixelShuffle(self.scale),
                nn.Conv2d(G, inC, 3, padding=1, stride=1)
            ])
        elif self.scale == 4:
            self.UPNet = nn.Sequential(*[
                nn.Conv2d(n_feat, G * 4, 3, padding=1, stride=1),
                nn.PixelShuffle(2),
                nn.Conv2d(G, G * 4, 3, padding=1, stride=1),
                nn.PixelShuffle(2),
                nn.Conv2d(G, inC, 3, padding=1, stride=1)
            ])
        else:
            raise ValueError("scale must be 2 or 3 or 4.")

    def forward(self, x):
        f1 = self.SFENet1(x)
        x = self.SFENet2(f1)

        RDBs_out = []
        for i in range(self.D):
            x = self.RDBs[i](x)
            RDBs_out.append(x)

        x = self.GFF(torch.cat(RDBs_out, 1))
        x += f1
        x = self.UPNet(x)

        return x