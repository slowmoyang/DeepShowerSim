from __future__ import division
from __future__ import print_function

from model import Generator
from model import Discriminator

import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import PbWO4Dtaset

import argparse

FAKE_LABEL = 0
REAL_LABEL = 1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-set', type=str, dest="train_set",
        default="/data/slowmoyang/DeepShowerSim/PbWO4_positron_uniform-1-100.root")
    parser.add_argument('--batch-size', type=int, default=64, dest="batch_size", help='batch size')
    parser.add_argument('--epoch', type=int, default=100, dest="num_epoch", help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate, default=0.0002')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--dim-latent', type=int, default=16, dest="dim_latent", help='size of the latent z vector')
    parser.add_argument('--no-gpu', action='store_true', dest="no_gpu", help='disables cuda')
    parser.add_argument('--num-gpu', type=int, default=1, dest="num_gpu", help='number of GPUs to use')
    args = parser.parse_args()

    device = torch.device("cuda:1")

    ######################################
    #
    ########################################
    train_set = PbWO4Dataset(path=args.train_set)
    train_loader = DataLoader(train_set, batch_size=args.batch_size)

    #####################
    #
    ######################
    generator = Generator(dim_latent=args.dim_latent)
    generator = generator.to(device)
    print(generator)

    discriminator = Discriminator()
    discriminator = discriminator.to(device)
    print(discriminator)

    ############################
    # NOTE 
    ##################################
    energy_criterion = nn.SmoothL1Loss()

    # setup optimizer
    optimizer_gen = optim.Adam(
        generator.parameters(),
        lr=args.lr,
        betas=(args.beta1, 0.999))

    optimizer_disc = optim.Adam(
        discriminator.parameters(),
        lr=args.lr,
        betas=(args.beta1, 0.999))

    ########################################
    # NOTE
    #######################################
    for epoch in xrange(args.num_epoch):
        for batch_idx, batch in enumerate(train_loader, 0):
            real = batch["energy_deposit"].to(device)
            energy = batch["total_energy"].to(device)
            batch_size = real.size(0)

            ################################
            # NOTE Update Discriminator
            ###############################
            # Train with real
            discriminator.zero_grad()
            output_real = discriminator(real, energy)

            # Train with Fake
            z = torch.randn(batch_size, args.dim_latent, 1, 1, device=device)
            fake = generator(z, energy)
            output_fake = discriminator(fake.detach(), energy)

            loss_disc = -torch.mean(output_real) + torch.mean(output_fake)
            loss_disc.backward()
            optimizer_disc.step()

            for param in discriminator.parameters():
                param.data.clamp_(-0.01, 0.01)


            ################################
            # NOTE Update Generator
            ###############################
            if batch_idx % 5 == 0:
                generator.zero_grad()

                z = torch.randn(batch_size, args.dim_latent, 1, 1, device=device)
                gen = generator(z, energy)
                output_gen = discriminator(gen, energy)

                loss_gen = -torch.mean(output_gen)
                loss_gen.backward()

                optimizer_gen.step()

            # FIXME
            if batch_idx % 100 == 0:
                print("[{:d}/{:d}][{:d}/{:d}]".format(
                    epoch, args.num_epoch, batch_idx, len(train_loader)))
                print("\tLoss D: {:.4f} Loss G: {:.4f}".format(
                    loss_disc.item(), loss_gen.item()))

        torch.save(
            generator.state_dict(),
            "/data/slowmoyang/DeepShowerSim/1-Training/Dev-DiscWithEnergy/generator_epoch-{:04d}_GLoss-{:.4f}_DLoss-{:.4f}.pth".format(
                epoch, loss_gen.item(), loss_disc.item()))


    print("END")



if __name__ == "__main__":
    main()
