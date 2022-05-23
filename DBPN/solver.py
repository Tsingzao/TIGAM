from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from DBPN.model import DBPN, DBPNS, DBPNLL
from Lib.MetaLib import NormalTrainer


class DBPNTrainer(NormalTrainer):
    def __init__(self, config, training_loader, testing_loader):
        super(DBPNTrainer, self).__init__(config, training_loader, testing_loader)

        self.model = DBPN(num_channels=2, base_channels=64, feat_channels=256, num_stages=7,
                          scale_factor=self.upscale_factor).to(self.device)
        self.state_dict = self.model.state_dict()
        self.model.weight_init()
        self.criterion = nn.L1Loss()
        torch.manual_seed(self.seed)

        if self.GPU_IN_USE:
            torch.cuda.manual_seed(self.seed)
            cudnn.benchmark = True
            self.criterion.cuda()

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[50, 75, 100], gamma=0.5)  # lr decay
