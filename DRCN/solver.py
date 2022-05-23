from __future__ import print_function
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from DRCN.model import Net
from Lib.ProgressBar import progress_bar
from Lib.MetaLib import NormalTrainer


def img_preprocess(data, upscale_factor, interpolation='bicubic'):
    import scipy.ndimage
    if interpolation == 'bicubic':
        order = 3
    elif interpolation == 'bilinear':
        order = 1
    elif interpolation == 'nearest':
        order = 0

    size = list(data.shape)

    if len(size) == 4:
        npdata = data.data.cpu().numpy()
        npdata = scipy.ndimage.zoom(npdata, (1,1,upscale_factor,upscale_factor), order=order)
        return torch.from_numpy(npdata)
    else:
        npdata = data.data.cpu().numpy()
        npdata = scipy.ndimage.zoom(npdata, (1,upscale_factor,upscale_factor), order=order)
        return torch.from_numpy(npdata)


class DRCNTrainer(NormalTrainer):
    def __init__(self, config, training_loader, testing_loader):
        super(DRCNTrainer, self).__init__(config, training_loader, testing_loader)

        # DRCN setup
        self.momentum = 0.9
        self.weight_decay = 0.0001
        self.loss_alpha = 1.0
        self.loss_alpha_zero_epoch = 25
        self.loss_alpha_decay = self.loss_alpha / self.loss_alpha_zero_epoch
        self.loss_beta = 0.001
        self.num_recursions = 16
        self.training_loader = training_loader
        self.testing_loader = testing_loader

    def build_model(self):
        self.model = Net(num_channels=2, base_channel=64, num_recursions=self.num_recursions, device=self.device, scale_factor=self.upscale_factor).to(self.device)
        self.state_dict = self.model.state_dict()
        self.model.weight_init()
        self.criterion = nn.MSELoss()
        torch.manual_seed(self.seed)

        if self.GPU_IN_USE:
            torch.cuda.manual_seed(self.seed)
            cudnn.benchmark = True
            self.criterion.cuda()

        # setup optimizer and scheduler
        param_groups = [{'params': list(self.model.parameters())}]
        param_groups += [{'params': [self.model.w]}]
        self.optimizer = optim.Adam(param_groups, lr=self.lr)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[50, 75, 100], gamma=0.5)  # lr decay

    def train(self):
        """
        data: [torch.cuda.FloatTensor], 4 batches: [64, 64, 64, 8]
        """
        self.model.train()
        train_loss = 0
        for batch_num, (data, _, target) in enumerate(self.training_loader):
            data = img_preprocess(data, self.upscale_factor)  # resize input image size
            data, target = data.to(self.device), target.to(self.device)
            target_d, output = self.model(data)

            # loss1
            loss_1 = 0
            for d in range(self.num_recursions):
                loss_1 += (self.criterion(target_d[d], target) / self.num_recursions)

            # loss2
            loss_2 = self.criterion(output, target)

            # regularization
            reg_term = 0
            for theta in self.model.parameters():
                reg_term += torch.mean(torch.sum(theta ** 2))

            # total loss
            loss = self.loss_alpha * loss_1 + (1 - self.loss_alpha) * loss_2 + self.loss_beta * reg_term
            loss.backward()

            train_loss += loss.item()
            self.optimizer.step()
            progress_bar(batch_num, len(self.training_loader), 'Loss: %.4f' % (train_loss / (batch_num + 1)))

        print("    Average Loss: {:.4f}".format(train_loss / len(self.training_loader)))

    def test(self):
        """
        data: [torch.cuda.FloatTensor], 10 batches: [10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
        """
        self.model.eval()
        avg_psnr = 0

        with torch.no_grad():
            for batch_num, (data, target) in enumerate(self.testing_loader):
                data = img_preprocess(data, self.upscale_factor)  # resize input image size
                data, target = data.to(self.device), target.to(self.device)
                _, prediction = self.model(data)
                mse = self.criterion(prediction, target)
                avg_psnr += mse
                progress_bar(batch_num, len(self.testing_loader), 'PSNR: %.4f' % (avg_psnr / (batch_num + 1)))

        print("    Average PSNR: {:.4f} dB".format(avg_psnr / len(self.testing_loader)))
        return avg_psnr / len(self.testing_loader)

    def run(self):
        self.build_model()
        OldLoss = 99999999
        if self.resume:
            if os.path.exists(self.resume):
                self.load_state_dict(torch.load(self.resume).state_dict())
                print('train from resume')
            else:
                print('%s not exist, train from scratch'%self.resume)
        for epoch in range(1, self.nEpochs + 1):
            print("\n===> Epoch {} starts:".format(epoch))
            self.loss_alpha = max(0.0, self.loss_alpha - self.loss_alpha_decay)
            self.train()
            print("\n===> Epoch {} validation:".format(epoch))
            CurrentLoss = self.test()
            self.scheduler.step(epoch)
            if CurrentLoss < OldLoss:
                print('Current Loss %.4f < Old Loss %.4f. Save New Model.'%(CurrentLoss, OldLoss))
                OldLoss = CurrentLoss
                self.save()
            else:
                print('Current Loss %.4f > Old Loss %.4f. Do Not Save Any Model.' % (CurrentLoss, OldLoss))
