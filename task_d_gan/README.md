# UrbanStyle, Task D: Promotion Images with a Small GAN

- Academic report: [`docs/REPORT.md`](docs/REPORT.md)

This task trains a small DCGAN on Fashion-MNIST. The generator maps a 64-value noise vector to a 28x28 grayscale product image; the discriminator classifies real versus generated images. Samples, model weights and training metrics are saved under `results/` and `models/`.

## Architecture

```
Fashion-MNIST -> normalized 1x28x28 tensors
       |                    |
       +--> Generator       +--> Discriminator
            noise -> image       image -> real/fake logit
                         |
                         v
             train.py -> models/ + results/figures/
                         |
                         v
                    pygame_ui/app.py
```

## Quick start

Run all commands from this folder (`task_d_gan/`):

```bash
python -m pip install -r requirements.txt
python train.py --epochs 10
python -m pygame_ui.app
```

For a fast smoke test: `python train.py --epochs 1 --limit 1024`.

The final generator is saved as `models/generator.pt`, so Pygame can use it directly. The complete training state is saved as `models/training_state.pt`. To train to epoch 10 and later resume up to epoch 20, run:

```bash
python train.py --epochs 10
python train.py --epochs 20 --resume
```

## Layout

| Path | Purpose |
|---|---|
| `config.py` | Paths, image size, latent dimension and default training values |
| `models.py` | Generator and discriminator architectures |
| `train.py` | Dataset loading, GAN training, checkpointing and sample export |
| `pygame_ui/app.py` | Interactive image generation and seed control |
| `tests/` | Shape and model smoke tests |
| `models/` | Generator, discriminator and resumable training checkpoints |
| `results/` | Loss history and generated sample figures |

## Pygame controls

| Control | Action |
|---|---|
| Generate image | Sample a new product image using the current seed |
| Noise seed slider | Select a deterministic seed from 0 to 999 |
| `G` / Up / Down | Keyboard alternatives for generation and seed changes |
| `S` | Save the current Pygame screen |

The dataset is downloaded automatically to `../data/fashion_mnist/`.
