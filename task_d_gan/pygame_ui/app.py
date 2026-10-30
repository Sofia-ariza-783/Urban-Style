import argparse
from pathlib import Path
import pygame
import torch

from config import CHECKPOINT_ROOT, LATENT_DIM, RESULTS_ROOT
from models import Generator


class GANApp:
    def __init__(self, checkpoint: Path | None = None, seed: int = 42):
        pygame.init(); self.screen = pygame.display.set_mode((760, 560)); pygame.display.set_caption("UrbanStyle GAN")
        self.font = pygame.font.Font(None, 28); self.small = pygame.font.Font(None, 22)
        self.generator = Generator(LATENT_DIM).eval(); self.seed = seed; self.message = "Ready"
        self.button = pygame.Rect(480, 220, 220, 48)
        self.slider = pygame.Rect(480, 315, 220, 6)
        if checkpoint and checkpoint.exists(): self.generator.load_state_dict(torch.load(checkpoint, map_location="cpu")); self.message = "Model loaded"
        self.image = None; self.generate()

    def generate(self):
        torch.manual_seed(self.seed)
        with torch.no_grad(): self.image = self.generator(torch.randn(16, LATENT_DIM, 1, 1))[0, 0]
        self.message = f"Generated with noise seed {self.seed}"

    def draw(self):
        self.screen.fill((22, 27, 38)); title = self.font.render("UrbanStyle - GAN promotional images", True, (240, 240, 245)); self.screen.blit(title, (24, 20))
        if self.image is not None:
            arr = ((self.image.numpy() + 1) * 127.5).clip(0, 255).astype("uint8")
            surf = pygame.surfarray.make_surface(arr.T); surf = pygame.transform.scale(surf, (420, 420)); self.screen.blit(surf, (24, 70))
        pygame.draw.rect(self.screen, (55, 120, 210), self.button, border_radius=8)
        self.screen.blit(self.small.render("GENERATE IMAGE", True, (255, 255, 255)), (515, 235))
        pygame.draw.line(self.screen, (170, 180, 195), (self.slider.left, self.slider.centery), (self.slider.right, self.slider.centery), 4)
        knob_x = self.slider.left + int((self.seed / 999) * self.slider.width)
        pygame.draw.circle(self.screen, (245, 170, 55), (knob_x, self.slider.centery), 10)
        for text, y in [("Noise seed: 0 - 999", 275), ("Click slider or use UP/DOWN", 345), ("S / save current screen", 380), (self.message, 500)]:
            self.screen.blit(self.small.render(text, True, (220, 225, 235)), (480, y))
        pygame.display.flip()

    def run(self):
        clock = pygame.time.Clock(); running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: running = False
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if self.button.collidepoint(event.pos):
                        self.generate()
                    elif self.slider.inflate(0, 28).collidepoint(event.pos):
                        self.seed = max(0, min(999, round((event.pos[0] - self.slider.left) / self.slider.width * 999)))
                        self.generate()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_g: self.generate()
                    elif event.key == pygame.K_UP: self.seed = min(999, self.seed + 1); self.generate()
                    elif event.key == pygame.K_DOWN: self.seed = max(0, self.seed - 1); self.generate()
                    elif event.key == pygame.K_s:
                        RESULTS_ROOT.mkdir(exist_ok=True); pygame.image.save(self.screen, RESULTS_ROOT / f"pygame_seed_{self.seed}.png"); self.message = "Saved current screen"
            self.draw(); clock.tick(30)
        pygame.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_ROOT / "generator.pt"); parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(); GANApp(args.checkpoint, args.seed).run()
