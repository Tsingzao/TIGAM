from __future__ import print_function
import math
import torch.nn as nn
import torch.nn.functional as F
import scipy.ndimage
from tensorboardX import SummaryWriter
import os
import torch
import numpy as np
from Lib.ProgressBar import progress_bar


def default_conv(inC, outC, kernel_size, bias=True):
    return nn.Conv2d(inC, outC, kernel_size,
                     padding=(kernel_size//2), bias=bias)


class Pos2Weight(nn.Module):
    def __init__(self, inC, kernel_size=3, outC=3):
        super(Pos2Weight, self).__init__()
        self.inC = inC
        self.kernel_size = kernel_size
        self.outC = outC
        self.meta_block = nn.Sequential(
            nn.Linear(3, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, self.inC*self.kernel_size*self.kernel_size*self.outC)
        )
    def forward(self, x):
        output = self.meta_block(x)
        return output

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn: m.append(nn.BatchNorm2d(n_feats))

                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3 or scale == 5:
            m.append(conv(n_feats, n_feats*(scale**2), 3, bias))
            m.append(nn.PixelShuffle(scale))
            if bn: m.append(nn.BatchNorm2d(n_feats))

            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)

class NormalTrainer(object):
    def __init__(self, config, train_loader, test_loader):
        super(NormalTrainer, self).__init__()
        self.GPU_IN_USE = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.GPU_IN_USE else 'cpu')
        self.model = None
        self.lr = config.lr
        self.nEpochs = config.nEpochs
        self.criterion = torch.nn.L1Loss()
        self.optimizer = None
        self.scheduler = None
        self.seed = config.seed
        self.upscale_factor = config.upscale_factor
        self.train_loader = train_loader
        self.valid_loader = test_loader
        self.test_loader = test_loader
        self.type = config.model
        self.pre = False
        self.meta = False
        self.curEpoch = 0
        self.state_dict = None
        self.resume = config.resume
        self.early = False
        self.aux_loss = torch.nn.L1Loss().to(self.device)
        if not config.testOnly and config.withLog:
            self.writer = SummaryWriter(comment=self.type+'_x%s'%self.upscale_factor)

    def save(self):
        model_out_path = "checkpoint/%s_x%d_best_epoch_%05d.pth"%(self.type, self.upscale_factor, self.curEpoch+1)
        os.makedirs(os.path.dirname(model_out_path), exist_ok=True)
        torch.save(self.model, model_out_path)
        print("Checkpoint saved to {}".format(model_out_path))

    def load_state_dict(self, state_dict):
        own_state = self.state_dict
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                if own_state[name].shape == param.shape:
                    own_state[name].copy_(param)
        print('Load State Done!')

    def _train_dem_(self):  # note: modify the test function, prediction, _ = self.model(data)
        self.model.train()
        train_loss = 0
        for batch_num, (data, dem, target) in enumerate(self.train_loader):
            data, target = data.to(self.device).float(), target.to(self.device).float()
            dem = dem.to(self.device).float()
            self.optimizer.zero_grad()

            output, std = self.model(data)

            # mode #1: predefined and shared ratio
            # b, c, h, w = output.shape
            # dem = torch.std(torch.nn.functional.unfold(dem, 2, stride=2), dim=-1)
            # std = torch.mean(torch.std(torch.nn.functional.unfold(output, 2, stride=2).view(b, c, 4, -1), dim=-1), dim=1)
            # aux = self.aux_loss(std*2, dem)

            # mode #2: learnable ratio
            # dem = torch.std(torch.nn.functional.unfold(dem, 2, stride=2), dim=-1)
            # aux = self.aux_loss(std, dem)

            # mode #3: learnable ratio with spatial kept.
            dem = dem[:,:,::self.upscale_factor,::self.upscale_factor]
            b, c, h, w = dem.shape
            dem = torch.std(torch.nn.functional.unfold(dem, 8, stride=8).view(b, c, 64, h//8, w//8), dim=2)
            aux = self.aux_loss(std, dem)

            loss = self.criterion(output, target)+0.05*aux
            if loss > 1e5:
                self.early = True
                break
            train_loss += loss.item()
            loss.backward()
            self.optimizer.step()
            progress_bar(batch_num, len(self.train_loader), 'IterLoss: %.4f, Average Loss: %.4f' % (loss, train_loss / (batch_num + 1)))
            self.writer.add_scalar('Train/TrainIterLoss_%s'%self.type, loss, self.curEpoch*len(self.train_loader)+batch_num)
            self.writer.add_scalar('Train/TrainAvgLoss_%s'%self.type, train_loss / (batch_num + 1), self.curEpoch*len(self.train_loader)+batch_num)

        print("    Average Loss: {:.4f}".format(train_loss / len(self.train_loader)))

    def train(self):
        self.model.train()
        train_loss = 0
        for batch_num, (data, dem, target) in enumerate(self.train_loader):
            data, target = data.to(self.device).float(), target.to(self.device).float()
            self.optimizer.zero_grad()

            output = self.model(data)

            loss = self.criterion(output, target)
            if loss > 1e5:
                self.early = True
                break
            train_loss += loss.item()
            loss.backward()
            self.optimizer.step()
            progress_bar(batch_num, len(self.train_loader), 'IterLoss: %.4f, Average Loss: %.4f' % (loss, train_loss / (batch_num + 1)))
            self.writer.add_scalar('Train/TrainIterLoss_%s'%self.type, loss, self.curEpoch*len(self.train_loader)+batch_num)
            self.writer.add_scalar('Train/TrainAvgLoss_%s'%self.type, train_loss / (batch_num + 1), self.curEpoch*len(self.train_loader)+batch_num)

        print("    Average Loss: {:.4f}".format(train_loss / len(self.train_loader)))

    def test(self, loader):
        self.model.eval()
        avg_psnr = 0

        preds, labels = [], []
        with torch.no_grad():
            for batch_num, (data, _, target) in enumerate(loader):
                data, target = data.to(self.device).float(), target.to(self.device).float()

                prediction, _ = self.model(data)
                # prediction = self.model(data)
                preds.append(prediction.cpu().numpy())
                labels.append(target.cpu().numpy())

                mse = self.criterion(prediction, target)
                avg_psnr += mse
                progress_bar(batch_num, len(loader), 'IterLoss: %.4f, Average Loss: %.4f' % (mse, avg_psnr / (batch_num + 1)))

        print("    Average Loss: {:.4f}".format(avg_psnr / len(loader)))

        preds, labels = np.concatenate(preds), np.concatenate(labels)
        from Lib.CheckData import denorm
        preds, labels = denorm(preds), denorm(labels)
        from Lib.CheckData import get_all_metrics
        get_all_metrics(preds, labels)

        return avg_psnr / len(loader)

    def run(self):
        OldLoss = 1e5
        beginEpoch = 1
        if self.resume:
            if os.path.exists(self.resume):
                self.load_state_dict(torch.load(self.resume).state_dict())
                print('train from resume')
                if self.upscale_factor == int(self.resume.split('_')[-4][-1]):
                    beginEpoch = int(self.resume.split('_')[-1].split('.pth')[0])+1
            else:
                print('%s not exist, train from scratch'%self.resume)
        for epoch in range(beginEpoch, self.nEpochs + 1):
            self.curEpoch = epoch-1
            print("\n===> Epoch {} starts:".format(epoch))
            # self.train()
            self._train_dem_()
            if self.early:
                break
            print("\n===> Epoch {} validation:".format(epoch))
            CurrentLoss = self.test(self.test_loader)
            self.scheduler.step(epoch)
            if CurrentLoss < OldLoss:
                print('Current Loss %.4f < Old Loss %.4f. Save New Model.'%(CurrentLoss, OldLoss))
                OldLoss = CurrentLoss
                self.save()
            else:
                print('Current Loss %.4f > Old Loss %.4f. Do Not Save Any Model.' % (CurrentLoss, OldLoss))

    def eval(self):
        state_dict = torch.load(self.resume, map_location='cpu')
        self.model.load_state_dict(state_dict.state_dict())
        self.model.eval()

        preds, labels = [], []
        loader = self.test_loader
        with torch.no_grad():
            for batch_num, (data, _, target) in enumerate(loader):
                data, target = data.to(self.device).float(), target.to(self.device).float()
                preds.append(self.model(data)[0].cpu().numpy())
                labels.append(target.cpu().numpy())
                progress_bar(batch_num, len(loader))

        preds, labels = np.concatenate(preds), np.concatenate(labels)
        from Lib.CheckData import denorm
        preds, labels = denorm(preds), denorm(labels)

        np.save('./results/preds_%s_x%s.npy'%(self.type, self.upscale_factor), preds)

