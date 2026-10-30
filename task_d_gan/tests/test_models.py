import torch
try:
    from task_d_gan.models import Discriminator, Generator
except ModuleNotFoundError:
    from models import Discriminator, Generator


def test_gan_shapes():
    fake = Generator()(torch.randn(4, 64, 1, 1))
    assert fake.shape == (4, 1, 28, 28)
    assert Discriminator()(fake).shape == (4, 1)
