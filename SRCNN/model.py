import torch
import torch.nn as nn
import torch.nn.functional as F


# https://github.com/MartinGer/Stand-Alone-Axial-Attention/blob/master/Axial_Layer.py
class Axial_Layer(nn.Module):
    def __init__(self, in_channels, num_heads=8, kernel_size=100, stride=1, height_dim=True, inference=False):
        super(Axial_Layer, self).__init__()
        self.depth = in_channels
        self.num_heads = num_heads
        self.kernel_size = kernel_size
        self.stride = stride
        self.height_dim = height_dim
        self.dh = self.depth // self.num_heads

        assert self.depth % self.num_heads == 0, "depth should be divided by num_heads. (example: depth: 32, num_heads: 8)"

        self.kqv_conv = nn.Conv1d(in_channels, self.depth * 2, kernel_size=1, bias=False)
        self.kqv_bn = nn.BatchNorm1d(self.depth * 2)
        self.logits_bn = nn.BatchNorm2d(num_heads * 3)
        # Positional encodings
        self.rel_encoding = nn.Parameter(torch.randn(self.dh * 2, kernel_size * 2 - 1), requires_grad=True)
        key_index = torch.arange(kernel_size)
        query_index = torch.arange(kernel_size)
        # Shift the distance_matrix so that it is >= 0. Each entry of the
        # distance_matrix distance will index a relative positional embedding.
        distance_matrix = (key_index[None, :] - query_index[:, None]) + kernel_size - 1
        self.register_buffer('distance_matrix', distance_matrix.reshape(kernel_size * kernel_size))

        # later access attention weights
        self.inference = inference
        if self.inference:
            self.register_parameter('weights', None)

    def forward(self, x):
        if self.height_dim:
            x = x.permute(0, 3, 1, 2)  # batch_size, width, depth, height
        else:
            x = x.permute(0, 2, 1, 3)  # batch_size, height, depth, width

        batch_size, width, depth, height = x.size()
        x = x.reshape(batch_size * width, depth, height)

        # Compute q, k, v
        kqv = self.kqv_conv(x)
        kqv = self.kqv_bn(kqv)  # apply batch normalization on k, q, v
        k, q, v = torch.split(kqv.reshape(batch_size * width, self.num_heads, self.dh * 2, height),
                              [self.dh // 2, self.dh // 2, self.dh], dim=2)

        # Positional encodings
        rel_encodings = torch.index_select(self.rel_encoding, 1, self.distance_matrix).reshape(self.dh * 2,
                                                                                               self.kernel_size,
                                                                                               self.kernel_size)
        q_encoding, k_encoding, v_encoding = torch.split(rel_encodings, [self.dh // 2, self.dh // 2, self.dh], dim=0)

        # qk + qr + kr
        qk = torch.matmul(q.transpose(2, 3), k)
        qr = torch.einsum('bhdx,dxy->bhxy', q, q_encoding)
        kr = torch.einsum('bhdx,dxy->bhxy', k, k_encoding).transpose(2, 3)

        logits = torch.cat([qk, qr, kr], dim=1)
        logits = self.logits_bn(logits)  # apply batch normalization on qk, qr, kr
        logits = logits.reshape(batch_size * width, 3, self.num_heads, height, height).sum(dim=1)

        weights = F.softmax(logits, dim=3)

        if self.inference:
            self.weights = nn.Parameter(weights)

        attn = torch.matmul(weights, v.transpose(2, 3)).transpose(2, 3)
        attn_encoding = torch.einsum('bhxy,dxy->bhdx', weights, v_encoding)
        attn_out = torch.cat([attn, attn_encoding], dim=-1).reshape(batch_size * width, self.depth * 2, height)
        output = attn_out.reshape(batch_size, width, self.depth, 2, height).sum(dim=-2)

        if self.height_dim:
            output = output.permute(0, 2, 3, 1)
        else:
            output = output.permute(0, 2, 1, 3)

        return output


class FlattenGRU(torch.nn.Module):
    def __init__(self, num_channels, hidden):
        super(FlattenGRU, self).__init__()
        self.hidden = hidden
        self.par = nn.Parameter(torch.ones(1,1,100,100), requires_grad=True)
        self.conv_zr = nn.Conv2d(num_channels+1, 2*hidden, kernel_size=9, stride=1, padding=4, bias=True)
        self.conv_h = nn.Conv2d(hidden+num_channels, hidden, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        xx = torch.cat([self.par.repeat(x.shape[0],1,1,1), x], dim=1)
        zr = self.conv_zr(xx)
        z, r = torch.split(zr, self.hidden, dim=1)
        z = torch.sigmoid(z)
        r = torch.sigmoid(r)
        x = torch.cat([r*self.par.repeat(x.shape[0],1,1,1), x], dim=1)
        h = self.conv_h(x)
        h = torch.tanh(h)
        o = (1-z)*self.par+z*h
        return o


class Net(torch.nn.Module):
    def __init__(self, num_channels, base_filter, upscale_factor=2):
        super(Net, self).__init__()

        self.scale = upscale_factor
        self.layers = torch.nn.Sequential(
            # flatten #1: terrible
            # FlattenGRU(num_channels, base_filter),
            # nn.Conv2d(in_channels=base_filter, out_channels=base_filter, kernel_size=9, stride=1, padding=4, bias=True),
            nn.Conv2d(in_channels=num_channels, out_channels=base_filter, kernel_size=9, stride=1, padding=4, bias=True),
            nn.ReLU(inplace=True),
            # flatten #2:
            # FlattenGRU(base_filter, base_filter),
            # Axial #2: best
            # Axial_Layer(base_filter, num_heads=8, kernel_size=100, height_dim=True),
            # Axial_Layer(base_filter, num_heads=8, kernel_size=100, height_dim=False),
            nn.Conv2d(in_channels=base_filter, out_channels=base_filter // 2, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            # flatten #3: best
            # FlattenGRU(base_filter // 2, base_filter // 2),
            # Axial #1:
            # Axial_Layer(base_filter // 2, num_heads=8, kernel_size=100, height_dim=True),
            # Axial_Layer(base_filter // 2, num_heads=8, kernel_size=100, height_dim=False),
            nn.Conv2d(in_channels=base_filter // 2, out_channels=num_channels * (upscale_factor ** 2), kernel_size=5, stride=1, padding=2, bias=True),
            nn.PixelShuffle(upscale_factor)
        )

        # mode #3: best
        self.auxp = nn.Parameter(torch.ones(num_channels, 1), requires_grad=True)

    def forward(self, x):
        out = self.layers(x)
        # flattenGRU
        # return out
        # mode #3:
        b, c, h, w = out.shape
        std = torch.std(torch.nn.functional.unfold(out, 8, stride=8).view(b, c, 64, h//8, w//8), dim=2)
        return out, std
        # mode #2:
        # std = torch.einsum('bcn,cd->bdn', std, self.auxp).squeeze(1)

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


if __name__ == '__main__':
    import numpy as np
    # a = FlattenGRU(2, 32).float().to(torch.device('cuda:0'))
    # a = Net(2, 32).float().to(torch.device('cuda:0'))
    a = Axial_Layer(32).float().to(torch.device('cuda:0'))
    b = torch.from_numpy(np.random.random((2, 32, 100, 100))).float().to(torch.device('cuda:0'))
    c = a(b)
    print(c.shape)