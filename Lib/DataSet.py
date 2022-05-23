from torch.utils.data import Dataset, DataLoader
from Lib.transforms import *
import h5py


class Ch3jLoader(Dataset):
    def __init__(self, dmode='train', factor=2, transform=None):
        super(Ch3jLoader, self).__init__()
        print('Init %s dataloader ...'%dmode)
        dpath = './Dataset/%s.h5'%dmode
        with h5py.File(dpath, 'r') as f:
            u = f['us'][:]
            v = f['vs'][:]
        with h5py.File('./Dataset/dem.h5', 'r') as f:
            d = f['dem'][:]
        self.data = np.array([u, v]).transpose((1,0,2,3))
        self.dem = np.expand_dims(d, 0)/1500.0
        self.factor = 4//factor
        self.transform = transform

    def __getitem__(self, item):
        data_item = self.transform(self.data[item]/100.0)
        d = ToTensor()(self.dem)
        x, y = data_item[:,::4,::4], data_item[:,::self.factor,::self.factor]
        return x, d, y

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    transforms = Compose([Normalize(), ToTensor()])
    loader = DataLoader(Ch3jLoader(transform=transforms, dmode='test'))
    for i, (low, _, high) in enumerate(loader):
        print('hello')