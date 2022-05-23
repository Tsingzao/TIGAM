from __future__ import print_function

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from RDN.model import RDN
from Lib.MetaLib import NormalTrainer

class RDNTrainer(NormalTrainer):
    def __init__(self, config, training_loader, testing_loader):
        super(RDNTrainer, self).__init__(config, training_loader, testing_loader)

        self.model = RDN(inC=2, n_feat=64, upscale_factor=self.upscale_factor, cfg='B').to(self.device)
        self.state_dict = self.model.state_dict()
        self.criterion = torch.nn.L1Loss()
        torch.manual_seed(self.seed)
        if self.GPU_IN_USE:
            torch.cuda.manual_seed(self.seed)
            cudnn.benchmark = True
            self.criterion.cuda()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.9,0.999), eps=1e-8)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[50,75,100], gamma=0.5)
