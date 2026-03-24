"""
Basic GAN for generating synthetic face-like images from a CelebA subset.

Dataset expected:
    Folder containing face images, e.g.:
        celeba_subset/
            img1.jpg
            img2.jpg
            ...

Run:
    python 06_celeba_basic_gan.py --data_dir celeba_subset --epochs 20

Working:
1. Generator converts random noise into fake images.
2. Discriminator tries to separate real and fake images.
3. Both networks compete and improve over time.
"""

import argparse
import os
from torchvision.utils import save_image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class Generator(nn.Module):
    def __init__(self, latent_dim=100, channels=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        return self.net(z)


class Discriminator(nn.Module):
    def __init__(self, channels=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x).view(-1, 1)


def weights_init(m):
    classname = m.__class__.__name__
    if "Conv" in classname:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif "BatchNorm" in classname:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="celeba_subset")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--latent_dim", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--image_size", type=int, default=64)
    args = parser.parse_args()

    os.makedirs("gan_samples", exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ])

    dataset = datasets.ImageFolder(root=args.data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    G = Generator(latent_dim=args.latent_dim).to(device)
    D = Discriminator().to(device)
    G.apply(weights_init)
    D.apply(weights_init)

    criterion = nn.BCELoss()
    opt_G = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))

    fixed_noise = torch.randn(64, args.latent_dim, 1, 1, device=device)

    for epoch in range(1, args.epochs + 1):
        for real_imgs, _ in loader:
            real_imgs = real_imgs.to(device)
            batch_size = real_imgs.size(0)

            real_labels = torch.ones(batch_size, 1, device=device)
            fake_labels = torch.zeros(batch_size, 1, device=device)

            # Train discriminator
            z = torch.randn(batch_size, args.latent_dim, 1, 1, device=device)
            fake_imgs = G(z)

            D_real = D(real_imgs)
            D_fake = D(fake_imgs.detach())
            loss_D = criterion(D_real, real_labels) + criterion(D_fake, fake_labels)

            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()

            # Train generator
            D_fake_for_G = D(fake_imgs)
            loss_G = criterion(D_fake_for_G, real_labels)

            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()

        with torch.no_grad():
            samples = G(fixed_noise).detach().cpu()
            save_image(samples, f"gan_samples/epoch_{epoch:03d}.png", normalize=True)

        print(f"Epoch {epoch}/{args.epochs} | Loss_D: {loss_D.item():.4f} | Loss_G: {loss_G.item():.4f}")

    torch.save(G.state_dict(), "celeba_generator.pth")
    torch.save(D.state_dict(), "celeba_discriminator.pth")
    print("Saved GAN models.")


if __name__ == "__main__":
    main()
