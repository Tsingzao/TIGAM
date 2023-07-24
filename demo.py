from __future__ import print_function

import numpy as np
import argparse
import os
import torch
from torch.utils.data import DataLoader, Dataset
from Lib.DataSet import *
from DBPN.solver import DBPNTrainer
from DRCN.solver import DRCNTrainer
from EDSR.solver import EDSRTrainer
from FSRCNN.solver import FSRCNNTrainer
from SRCNN.solver import SRCNNTrainer
from SRGAN.solver import SRGANTrainer
from SubPixelCNN.solver import SubPixelTrainer
from VDSR.solver import VDSRTrainer
from RDN.solver import RDNTrainer
from RCAN.solver import RCANTrainer
from SAN.solver import SANTrainer
from LapSRN.solver import LAPSRNTrainer
from SRDenseNet.solver import SRDenseNetTrainer
from USRNet.solver import USRNetTrainer
from NLSN.solver import NLSNTrainer
from RNAN.solver import RNANTrainer
from SRFBN.solver import SRFBNTrainer
from RFANet.solver import RFANetTrainer
from CSNLA.solver import CSNLATrainer
from SwinIR.solver import SwinIRTrainer
from Restormer.solver import RestormerTrainer
from WindTopo.solver import WindTopoTrainer
from Lib.DataSet import Ch3jLoader
from Lib.transforms import ToTensor, Normalize, Compose
from Lib.ProgressBar import progress_bar


class DemoDataLoader(Dataset):
    def __init__(self):
        super(DemoDataLoader, self).__init__()
        self.data = np.random.random((100,1,256,256))
        self.dem = np.random.random((1,256,256))

    def __getitem__(self, item):
        return self.data[item][:,::2,::2], self.data[item], self.dem

    def __len__(self):
        return len(self.data)


import torch.nn as nn
class SRCNN_Simple(torch.nn.Module):
    def __init__(self, num_channels=1, out_channels=1, base_filter=32, upscale_factor=2):
        super(SRCNN_Simple, self).__init__()

        self.scale = upscale_factor
        self.layers = torch.nn.Sequential(
            nn.Conv2d(in_channels=num_channels, out_channels=base_filter, kernel_size=9, stride=1, padding=4, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=base_filter, out_channels=base_filter // 2, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=base_filter // 2, out_channels=out_channels * (upscale_factor ** 2), kernel_size=5, stride=1, padding=2, bias=True),
            nn.PixelShuffle(upscale_factor)
        )

    def forward(self, x):
        out = self.layers(x)
        return out

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


class DeepSD(nn.Module):
    def __init__(self):
        super(DeepSD, self).__init__()
        self.s1 = SRCNN_Simple(2,1)
        self.s2 = SRCNN_Simple(2,1)
        self.s3 = SRCNN_Simple(2,1)

    def forward(self, x, d):
        d1 = d[:,:,::8,::8]
        x1 = torch.cat([x, d1], dim=1)
        y1 = self.s1(x1)

        d2 = d[:,:,::4,::4]
        x2 = torch.cat([y1, d2], dim=1)
        y2 = self.s2(x2)

        d3 = d[:,:,::2,::2]
        x3 = torch.cat([y2, d3], dim=1)
        y3 = self.s3(x3)

        return y3



from USRNet.model import *
class TIGAM(nn.Module):
    def __init__(self, n_iter=8, h_nc=64, in_nc=4, out_nc=3, nc=[64, 128, 256, 512], nb=2, act_mode='R',
                 downsample_mode='strideconv', upsample_mode='convtranspose', upscale_factor=2):
        super(TIGAM, self).__init__()

        self.d = DataNet()
        self.p = ResUNet(in_nc=in_nc, out_nc=out_nc, nc=nc, nb=nb, act_mode=act_mode, downsample_mode=downsample_mode,
                         upsample_mode=upsample_mode)
        self.h = HyPaNet(in_nc=2, out_nc=n_iter * 2, channel=h_nc)
        self.n = n_iter
        self.sf = upscale_factor

        # self.axial = torch.nn.Sequential(
        #     Axial_Layer(in_nc-1, num_heads=1, kernel_size=400, height_dim=True),
        #     Axial_Layer(in_nc-1, num_heads=1, kernel_size=400, height_dim=False))
            # Axial_Layer(in_nc-1, num_heads=1, kernel_size=200, height_dim=True),
            # Axial_Layer(in_nc-1, num_heads=1, kernel_size=200, height_dim=False))

        kernel_width_default_x1234 = [0.4, 0.7, 1.5, 2.0]
        kernel_width = kernel_width_default_x1234[self.sf - 1]

        k = fspecial_gaussian(25, kernel_width)
        k = shift_pixel(k, self.sf)  # shift the kernel
        k /= np.sum(k)

        self.kernel = single2tensor4(k[..., np.newaxis])
        self.sigma = torch.tensor(2/255.0).float().view([1, 1, 1, 1])

    def forward(self, x):
        '''
        x: tensor, NxCxWxH
        k: tensor, Nx(1,3)xwxh
        sigma: tensor, Nx1x1x1
        '''

        # initialization & pre-calculation
        w, h = x.shape[-2:]
        FB = p2o(self.kernel.to(self.p.m_head.weight.device), (w * self.sf, h * self.sf))
        FBC = torch.conj(FB)
        F2B = torch.pow(torch.abs(FB), 2)
        STy = upsample(x, sf=self.sf)
        FBFy = FBC * torch.fft.fftn(STy, dim=(-2, -1))
        x = nn.functional.interpolate(x, scale_factor=self.sf, mode='nearest')
        # x = self.axial(x)

        # hyper-parameter, alpha & beta
        self.sigma = self.sigma.to(self.p.m_head.weight.device)
        ab = self.h(torch.cat((self.sigma, torch.tensor(self.sf).type_as(self.sigma).expand_as(self.sigma)), dim=1))

        # unfolding
        for i in range(self.n):
            x = self.d(x, FB, FBC, F2B, FBFy, ab[:, i:i + 1, ...], self.sf)
            x = self.p(torch.cat((x, ab[:, i + self.n:i + self.n + 1, ...].repeat(x.size(0), 1, x.size(2), x.size(3))), dim=1))

        b, c, h, w = x.shape
        std = torch.std(torch.nn.functional.unfold(x, 8, stride=8).view(b, c, 64, h//8, w//8), dim=2)
        return x, std



def run(model, train_loader):
    OldLoss = 1e5
    device=torch.device('cuda:0')
    criterion = torch.nn.L1Loss().to(device)
    aux_loss = torch.nn.L1Loss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9,0.999), eps=1e-8)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20], gamma=0.5)
    model.to(device)
    for epoch in range(200):
        print("\n===> Epoch {} starts:".format(epoch))
        model.train()
        train_loss = 0
        for batch_num, (data, target, dem) in enumerate(train_loader):
            data = data.to(device).float()
            target = target.to(device).float()
            dem = dem.to(device).float()
            optimizer.zero_grad()

            # SRCNN
            # output = model(data)
            # loss = criterion(output, target)

            # DeepSD
            output = model(data[:,:,::4,::4], dem)
            loss = criterion(output, target)

            # TIGAM
            # output, std = model(data)
            # b, c, h, w = dem.shape
            # dem = torch.std(torch.nn.functional.unfold(dem, 8, stride=8).view(b, c, 64, h//8, w//8), dim=2)
            # aux = aux_loss(std, dem)
            # loss = criterion(output, target)+0.01*aux

            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            progress_bar(batch_num, len(train_loader), 'IterLoss: %.4f, Average Loss: %.4f' % (loss, train_loss / (batch_num + 1)))

        print("    Average Loss: {:.4f}".format(train_loss / len(train_loader)))
        CurrentLoss = train_loss / len(train_loader)
        scheduler.step(epoch)
        if CurrentLoss < OldLoss:
            print('Current Loss %.4f < Old Loss %.4f. Save New Model.'%(CurrentLoss, OldLoss))
            OldLoss = CurrentLoss
            model_out_path = "./demo_checkpoint/best_epoch_%05d.pth" % (epoch + 1)
            os.makedirs(os.path.dirname(model_out_path), exist_ok=True)
            torch.save(model, model_out_path)
            print("Checkpoint saved to {}".format(model_out_path))
        else:
            print('Current Loss %.4f > Old Loss %.4f. Do Not Save Any Model.' % (CurrentLoss, OldLoss))



def main():
    train_loader = DataLoader(DemoDataLoader(), batch_size=1, shuffle=True)

    # model = SRCNN_Simple()
    model = DeepSD()
    # model = TIGAM(in_nc=2, out_nc=1)
    run(model, train_loader)


if __name__ == '__main__':
    main()
