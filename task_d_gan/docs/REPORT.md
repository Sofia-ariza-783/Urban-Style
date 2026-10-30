# Task D: Promotion Images with a Small GAN

*UrbanStyle Project: Artificial Intelligence applied to fashion retail*

> Fashion-MNIST is used as a compact proxy for product images. The experiment demonstrates the GAN pipeline and interactive generation; it is not intended to produce high-resolution advertising artwork.

## 1. Objective

UrbanStyle needs a simple way to generate alternative product-image concepts for promotional material. The objective is to train a small generative adversarial network (GAN) that learns the visual distribution of Fashion-MNIST product images and produces new 28x28 grayscale samples from random noise.

The required components are a generator, a discriminator, a short training run, saved outputs and an interactive Pygame view.

## 2. Data and preprocessing

Fashion-MNIST is downloaded through `torchvision.datasets.FashionMNIST` into `data/fashion_mnist/`. Each training example is a 28x28 grayscale image. Pixel values are converted to tensors and normalized to [-1, 1], which matches the generator's final `tanh` activation.

The dataset is not copied into the repository. A fresh clone downloads it automatically on the first training run.

## 3. Model architecture

The generator receives a 64-dimensional latent vector reshaped to 64x1x1. Three transposed-convolution blocks increase the spatial size from 1x1 to 7x7, 14x14 and 28x28. Batch normalization and ReLU activations are used between the learned upsampling layers; the final `tanh` produces one grayscale channel.

The discriminator receives a 1x28x28 image. Two convolutional blocks reduce the spatial resolution while increasing the number of channels, followed by a linear layer that returns one real/fake logit. The discriminator output is trained with `BCEWithLogitsLoss`.

## 4. Training procedure

The two networks are trained alternately with Adam optimizers. For each batch, the discriminator scores real and generated images, and the generator is updated to make generated images score as real. A fixed noise batch is used after every epoch so that the saved figures show visual change over training.

The default learning rate is 2e-4 and the Adam parameters are (0.5, 0.999). The default seed is 42 and the default latent dimension is 64.

## 5. Results

The repository contains a 10-epoch CPU run with seed 42 and batch size 64. The final loss values are recorded in `results/metrics.json`:

| Epoch | Generator loss | Discriminator loss |
|---:|---:|---:|
| 1 | 1.3641 | 0.6651 |
| 5 | 1.3756 | 0.7381 |
| 10 | 2.2101 | 0.9963 |

The generated grid from epoch 10 is saved as `results/figures/epoch_010_plot.png`. The samples show recognizable clothing-like silhouettes, shoes and bags. They remain low-resolution and occasionally blurred, which is expected for a short GAN run on 28x28 images.

## 6. Interactive component

`pygame_ui/app.py` loads `models/generator.pt` and displays one generated sample. The Generate Image button samples the current seed, and the slider changes the seed from 0 to 999. The same controls are available from the keyboard for reproducible demonstrations. The `S` key saves the current screen.

## 7. Reproducibility and saved state

The seed controls Python, NumPy and PyTorch random generators. The generator and discriminator weights are saved in `models/`. `models/training_state.pt` also contains both optimizer states and the last completed epoch, allowing a run to continue:

```bash
python train.py --epochs 10
python train.py --epochs 20 --resume
```

The output directory contains one sample grid per epoch and the JSON loss history.

## 8. Limitations and future work

1. Fashion-MNIST is a low-resolution grayscale benchmark, not UrbanStyle's real product catalogue.
2. The experiment is unconditional, so it cannot request a particular product category.
3. Ten epochs are sufficient for the assignment demonstration but not for production-quality images.
4. A conditional GAN, higher-resolution colour data and a promotional poster compositor would be natural extensions, but are outside the required implementation.
