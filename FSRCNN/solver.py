from __future__ import print_function

import torch
import torch.backends.cudnn as cudnn
from FSRCNN.model import Net
from Lib.MetaLib import NormalTrainer


class FSRCNNTrainer(NormalTrainer):
    def __init__(self, config, training_loader, testing_loader):
        super(FSRCNNTrainer, self).__init__(config, training_loader, testing_loader)

        self.model = Net(num_channels=2, upscale_factor=self.upscale_factor).to(self.device)
        self.state_dict = self.model.state_dict()
        self.model.weight_init(mean=0.0, std=0.2)
        self.criterion = torch.nn.MSELoss()
        torch.manual_seed(self.seed)

        if self.GPU_IN_USE:
            torch.cuda.manual_seed(self.seed)
            cudnn.benchmark = True
            self.criterion.cuda()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[50, 75, 100], gamma=0.5)  # lr decay
