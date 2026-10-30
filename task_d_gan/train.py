import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, utils

from config import (CHECKPOINT_ROOT, DATA_ROOT, DEFAULT_BATCH_SIZE,
                    DEFAULT_EPOCHS, DEFAULT_SEED, LATENT_DIM, RESULTS_ROOT,
                    SAMPLES_ROOT)
from models import Discriminator, Generator


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def make_loader(batch_size, limit=None):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    dataset = datasets.FashionMNIST(DATA_ROOT, train=True, download=True, transform=transform)
    if limit:
        dataset = Subset(dataset, range(min(limit, len(dataset))))
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)


def save_grid(images, path, title=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    utils.save_image(images, path, nrow=8, normalize=True, value_range=(-1, 1))
    if title:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(utils.make_grid(images, nrow=8, normalize=True).permute(1, 2, 0).squeeze(), cmap="gray")
        ax.set_title(title); ax.axis("off"); fig.tight_layout()
        fig.savefig(path.with_name(path.stem + "_plot.png"), dpi=140); plt.close(fig)


def train(epochs=DEFAULT_EPOCHS, batch_size=DEFAULT_BATCH_SIZE, limit=None, seed=DEFAULT_SEED, resume=False):
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = make_loader(batch_size, limit)
    generator, discriminator = Generator(LATENT_DIM).to(device), Discriminator().to(device)
    criterion = nn.BCEWithLogitsLoss()
    g_opt = optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    d_opt = optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    start_epoch = 1
    checkpoint_path = CHECKPOINT_ROOT / "training_state.pt"
    if resume and checkpoint_path.exists():
        state = torch.load(checkpoint_path, map_location=device)
        generator.load_state_dict(state["generator"])
        discriminator.load_state_dict(state["discriminator"])
        g_opt.load_state_dict(state["generator_optimizer"])
        d_opt.load_state_dict(state["discriminator_optimizer"])
        start_epoch = int(state["epoch"]) + 1
        print(f"Resuming from epoch {start_epoch}")
    fixed_noise = torch.randn(64, LATENT_DIM, 1, 1, device=device)
    history = []
    for epoch in range(start_epoch, epochs + 1):
        for real, _ in loader:
            real = real.to(device); n = real.size(0)
            d_opt.zero_grad()
            real_loss = criterion(discriminator(real), torch.ones(n, 1, device=device))
            noise = torch.randn(n, LATENT_DIM, 1, 1, device=device)
            fake = generator(noise)
            fake_loss = criterion(discriminator(fake.detach()), torch.zeros(n, 1, device=device))
            d_loss = real_loss + fake_loss; d_loss.backward(); d_opt.step()
            g_opt.zero_grad()
            g_loss = criterion(discriminator(fake), torch.ones(n, 1, device=device))
            g_loss.backward(); g_opt.step()
        samples = generator(fixed_noise).detach().cpu()
        sample_path = SAMPLES_ROOT / f"epoch_{epoch:03d}.png"
        save_grid(samples, sample_path, f"UrbanStyle GAN - epoch {epoch}")
        history.append({"epoch": epoch, "generator_loss": float(g_loss.detach()), "discriminator_loss": float(d_loss.detach())})
        torch.save({"epoch": epoch, "generator": generator.state_dict(),
                    "discriminator": discriminator.state_dict(),
                    "generator_optimizer": g_opt.state_dict(),
                    "discriminator_optimizer": d_opt.state_dict()}, checkpoint_path)
        print(f"epoch={epoch}/{epochs} d_loss={d_loss.item():.4f} g_loss={g_loss.item():.4f}")
    torch.save(generator.state_dict(), CHECKPOINT_ROOT / "generator.pt")
    torch.save(discriminator.state_dict(), CHECKPOINT_ROOT / "discriminator.pt")
    metadata = {"epochs": epochs, "batch_size": batch_size, "dataset_limit": limit, "seed": seed,
                "device": str(device), "latent_dim": LATENT_DIM, "history": history}
    (RESULTS_ROOT / "metrics.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return generator


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the UrbanStyle Fashion-MNIST GAN")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--limit", type=int, default=None, help="optional small subset for a quick demo")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--resume", action="store_true", help="resume from checkpoints/training_state.pt")
    args = parser.parse_args()
    train(args.epochs, args.batch_size, args.limit, args.seed, args.resume)
