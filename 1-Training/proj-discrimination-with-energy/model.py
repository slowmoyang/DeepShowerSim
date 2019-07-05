from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel

import warnings
from numbers import Real


class ResConvT2d(nn.Module):
    def __init__(self, in_channels, **kwargs):
        super(ResConvT2d, self).__init__()

        self.conv_transpose = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=in_channels,
            **kwargs)
        self.conv_transpose.weight.data.normal_(2.0, 0.02)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, input):
        hidden = self.conv_transpose(input)
        hidden = self.activation(hidden)

        size = hidden.shape[2:]
        residual = F.upsample_bilinear(input, size)

        output = torch.cat([hidden, residual], dim=1)
        return output

        

class Generator(nn.Module):
    def __init__(self, dim_latent):
        super(Generator, self).__init__()

        # torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1)

        self.model = nn.Sequential(
             ResConvT2d(in_channels=dim_latent, kernel_size=4, bias=False),
            # (B, 1, 4, 4)
            nn.ConvTranspose2d(
                in_channels=2*dim_latent,
                out_channels=int(dim_latent/4),
                kernel_size=4,
                bias=False),
            nn.ReLU(inplace=True),
            # (B, 1, 7, 7)
            nn.ConvTranspose2d(
                in_channels=int(dim_latent/4),
                out_channels=1,
                kernel_size=3,
                bias=False),
            nn.ReLU())
            # (B, 1, 9, 9)

        self.dim_latent = dim_latent

    def forward(self, noise, energy):
        # noise: (B, dim_z, 1, 1)
        batch_size = noise.size(0)

        if isinstance(energy, Real):
            scaled_noise = energy * noise
        else:
            scaled_noise = energy.view(batch_size, 1, 1, 1) * noise

        output = self.model(scaled_noise)
        return output

class MiniBatchDiscrimination(nn.Module):
    def __init__(self, A, B, C):
        """
        arXiv:1606.03498 [cs.LG]
        """
        super(MiniBatchDiscrimination, self).__init__()

        self.T = nn.Parameter(torch.rand(A, B * C, requires_grad=True))
        nn.init.normal(self.T, 0, 1)

        self.A = A
        self.B = B
        self.C = C

    def forward(self, f):
        """
        Args
          f: A torch.FloatTensor. size (n, a). the intermediate features.

        Symbols
          n: batch_size
          b: out_features
          c: dimension of kernel
        """
        M = f.mm(self.T) # (n, b * C)
        M = M.view(-1, self.B, self.C) # (n, b, c)

        # compute the L_1-distance between the rows of the resulting matrix M_i
        # across samples i \in {1, 2,\ldots,n}
        M = M.unsqueeze(0) # (1, n, B, C)
        M_transposed = M.permute(1, 0, 2, 3)
        L1_distance = torch.abs(M - M_transposed).sum(dim=3)

        # M - M_transposed: (n, n, B, C)
        # sum(dim=3): (n, n, out_features)

        # apply a negative exponential
        c = torch.exp(-L1_distance)
        # (n, n, out_features)

        # The output o(x_i) for this minbatch layer for a sample x_i is then
        # defined as the sum of the c_b(x_i, x_j)'s to all other samples:
        o = c.sum(dim=0)

        o = torch.cat([f, o], dim=1)

        return o

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv_block = nn.Sequential(
            # input: (B, 1, 9, 9)
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=4, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # (B, 32, 6, 6)
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # (B, 16, 3, 3)
        )

        # Minibatch discrimination branch
        self.classifier = nn.Sequential(
            MiniBatchDiscrimination(A=64*3*3 + 1, B=32, C=16),
            nn.Linear(in_features=(64*3*3+1)+32, out_features=32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(in_features=32, out_features=1),
            nn.Sigmoid()
        )

    def forward(self, readout, energy):
        batch_size = readout.size(0)

        hidden = self.conv_block(readout)
        hidden = hidden.view(batch_size, -1)

        reco_energy = readout.view(batch_size, -1).sum(dim=1)
        energy_diff = abs(reco_energy - energy).view(-1, 1)

        hidden = torch.cat((hidden, energy_diff), dim=1)

        output = self.classifier(hidden) 
        output = output.squeeze()

        return output


def _test():
    dim_latent = 64
    batch_size = 256
    print("dim_latent: {}".format(dim_latent))
    print("batc_size: {}".format(batch_size))

    generator = Generator(dim_latent=dim_latent)
    print(generator)

    z = torch.randn(batch_size, dim_latent, 1, 1)
    energy = 100 * torch.randn(batch_size)


    print("z: {}".format(z.shape))
    fake = generator(z, energy)
    print("fake: {}".format(fake.shape))

    discriminator = Discriminator()
    print(discriminator)

    output = discriminator(fake, energy)
    print("D output: {}".format(output.shape))

if __name__ == "__main__":
    _test()
