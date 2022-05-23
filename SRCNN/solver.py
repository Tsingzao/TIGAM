from __future__ import print_function

import torch
import torch.backends.cudnn as cudnn

from SRCNN.model import Net
from Lib.MetaLib import NormalTrainer


class SRCNNTrainer(NormalTrainer):
    def __init__(self, config, training_loader, testing_loader):
        super(SRCNNTrainer, self).__init__(config, training_loader, testing_loader)

        self.model = Net(num_channels=2, base_filter=64, upscale_factor=self.upscale_factor).to(self.device)
        self.state_dict = self.model.state_dict()
        self.model.weight_init(mean=0.0, std=0.01)
        torch.manual_seed(self.seed)

        if self.GPU_IN_USE:
            torch.cuda.manual_seed(self.seed)
            cudnn.benchmark = True
            self.criterion.cuda()

        # mode #3: dem and flatten #3
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.9,0.999), eps=1e-8)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[10, 20], gamma=0.5)
        # org:
        # self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[50, 75, 100], gamma=0.5)
        # flatten #1: terrible
        # self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.lr)
        # self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[10, 20], gamma=0.5)
