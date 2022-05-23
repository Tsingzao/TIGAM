from __future__ import print_function

import numpy as np
import argparse
import os
import torch
from torch.utils.data import DataLoader

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
from Lib.DataSet import Ch3jLoader
from Lib.transforms import ToTensor, Normalize, Compose

# ===========================================================
# Training settings
# ===========================================================
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
# hyper-parameters
parser.add_argument('--batchSize', type=int, default=32, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--nEpochs', type=int, default=20, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.01, help='Learning Rate. Default=0.01')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')

parser.add_argument('--gpuID', type=str, default='0', help='GPU ID to Use')
parser.add_argument('--resume', type=str, default='', help='Resume from saved model')
parser.add_argument('--testOnly', action='store_true', help='Test Mode')
parser.add_argument('--checkpoint', type=str, default='', help='Checkpoint path for testing')
parser.add_argument('--pre', action='store_true', help='Preprocess input shape')
parser.add_argument('--withLog', type=bool, default=False, help='Preprocess input shape')

# model configuration
parser.add_argument('--upscale_factor', '-uf',  type=int, default=2, help="super resolution upscale factor")
parser.add_argument('--model', '-m', type=str, default='srfbn', help='choose which model is going to use')

# python main.py --model srcnn --gpuID 0 --upscale_factor 2 --batchSize 32
# python main.py --model subpixel --gpuID 0 --upscale_factor 2 --batchSize 32
# python main.py --model vdsr --gpuID 0 --upscale_factor 2 --batchSize 32
# python main.py --model edsr --gpuID 0 --upscale_factor 2 --batchSize 32
# python main.py --model fsrcnn --gpuID 0 --upscale_factor 2 --batchSize 32
# python main.py --model dbpn --gpuID 1 --upscale_factor 2 --batchSize 16 --lr 1e-3
# python main.py --model rdn --gpuID 2 --upscale_factor 2 --batchSize 8 --lr 1e-3
# python main.py --model lapsrn --gpuID 3 --upscale_factor 2 --batchSize 16 --lr 1e-3
# python main.py --model srdensenet --gpuID 0 --upscale_factor 2 --batchSize 32 --lr 1e-3
# python main.py --model usrnet --gpuID 1 --upscale_factor 2 --batchSize 8 --lr 1e-4
# python main.py --model nlsn --gpuID 2 --upscale_factor 2 --batchSize 16 --lr 1e-3
# python main.py --model rcan --gpuID 3 --upscale_factor 2 --batchSize 16 --lr 1e-3
# python main.py --model san --gpuID 2 --upscale_factor 2 --batchSize 2 --lr 1e-4
# python main.py --model rnan --gpuID 3 --upscale_factor 2 --batchSize 8 --lr 1e-3 --withLog True
# python main.py --model srfbn --gpuID 2 --upscale_factor 2 --batchSize 8 --lr 1e-3 --withLog True
# python main.py --model rfanet --gpuID 2 --upscale_factor 2 --batchSize 8 --lr 1e-3 --withLog True
# python main.py --model csnla --gpuID 0 --upscale_factor 2 --batchSize 4 --lr 1e-4 --withLog True

# python main.py --testOnly --model usrnet --resume ./checkpoint/usrnet_x2_best_epoch_00018.pth

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuID
os.system('export PYTHONUNBUFFERED=1')

def main():
    if not args.testOnly:
        # ===========================================================
        # Set train dataset & valid dataset
        # ===========================================================
        transforms = Compose([Normalize(), ToTensor()])
        train_loader = DataLoader(Ch3jLoader('train', args.upscale_factor, transform=transforms), batch_size=args.batchSize, shuffle=True)
        # valid_loader = DataLoader(Ch3jLoader('valid', args.upscale_factor), batch_size=args.batchSize, shuffle=True)
        test_loader = DataLoader(Ch3jLoader('test', args.upscale_factor, transform=transforms), batch_size=args.testBatchSize, shuffle=False)
    else:
        transforms = Compose([Normalize(), ToTensor()])
        train_loader = None
        test_loader = DataLoader(Ch3jLoader('test', args.upscale_factor, transform=transforms), batch_size=args.testBatchSize, shuffle=False)

    if args.model.lower() == 'subpixel':
        model = SubPixelTrainer(args, train_loader, test_loader)
    elif args.model.lower() == 'srcnn':
        model = SRCNNTrainer(args, train_loader, test_loader)
    elif args.model.lower() == 'vdsr':
        model = VDSRTrainer(args, train_loader, test_loader)
    elif args.model.lower() == 'edsr':
        model = EDSRTrainer(args, train_loader, test_loader)
    elif args.model.lower() == 'fsrcnn':
        model = FSRCNNTrainer(args, train_loader, test_loader)
    elif args.model.lower() == 'dbpn':
        model = DBPNTrainer(args, train_loader, test_loader)
    elif args.model.lower() == 'rdn':   # Need lower learning rate
        model = RDNTrainer(args, train_loader, test_loader)
    elif args.model.lower() == 'lapsrn':
        model = LAPSRNTrainer(args, train_loader, test_loader)
    elif args.model.lower() == 'srdensenet':
        model = SRDenseNetTrainer(args, train_loader, test_loader)
    elif args.model.lower() == 'usrnet':
        model = USRNetTrainer(args, train_loader, test_loader)
    elif args.model.lower() == 'nlsn':
        model = NLSNTrainer(args, train_loader, test_loader)
    elif args.model.lower() == 'rcan':
        model = RCANTrainer(args, train_loader, test_loader)
    elif args.model.lower() == 'san':
        model = SANTrainer(args, train_loader, test_loader)
    elif args.model.lower() == 'rnan':
        model = RNANTrainer(args, train_loader, test_loader)
    elif args.model.lower() == 'srfbn':
        model = SRFBNTrainer(args, train_loader, test_loader)
    elif args.model.lower() == 'rfanet':
        model = RFANetTrainer(args, train_loader, test_loader)
    elif args.model.lower() == 'csnla':
        model = CSNLATrainer(args, train_loader, test_loader)
    elif args.model.lower() == 'srgan':
        model = SRGANTrainer(args, train_loader, test_loader)
    elif args.model.lower() == 'drcn':
        model = DRCNTrainer(args, train_loader, test_loader)
    else:
        raise Exception("the model does not exist")

    if not args.testOnly:
        model.run()
    else:
        model.eval()



if __name__ == '__main__':
    main()
