from __future__ import print_function

import torch
import torch.backends.cudnn as cudnn

from VDSR.model import Net
from Lib.MetaLib import NormalTrainer


class VDSRTrainer(NormalTrainer):
    def __init__(self, config, training_loader, testing_loader):
        super(VDSRTrainer, self).__init__(config, training_loader, testing_loader)

        self.pre = True
        self.model = Net(num_channels=2, base_channels=64, num_residuals=4, scale_factor=self.upscale_factor).to(self.device)
        self.state_dict = self.model.state_dict()
        self.model.weight_init()
        self.criterion = torch.nn.L1Loss()
        torch.manual_seed(self.seed)

        if self.GPU_IN_USE:
            torch.cuda.manual_seed(self.seed)
            cudnn.benchmark = True
            self.criterion.cuda()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.9,0.999), eps=1e-8)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[50, 75, 100], gamma=0.5)
