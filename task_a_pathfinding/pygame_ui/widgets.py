"""Minimal Pygame widgets: Slider, Button, LineChart, plus a heat colour ramp."""
from __future__ import annotations

import pygame

BG = (246, 246, 244)
PANEL = (255, 255, 255)
INK = (20, 20, 20)
INK_2 = (90, 89, 85)
GRID = (225, 224, 220)
ACCENT = (42, 120, 214)

# Heat ramp: pale yellow -> orange -> deep orange (hotspot outlines are drawn in red)
HEAT_STOPS = [(255, 240, 170), (250, 170, 60), (214, 90, 40)]


def heat_color(frac: float) -> tuple[int, int, int]:
    """RGB colour for a heat value in [0, 1] (linear blend between ramp stops)."""
    frac = max(0.0, min(1.0, frac)) * (len(HEAT_STOPS) - 1)
    i = min(int(frac), len(HEAT_STOPS) - 2)
    t = frac - i
    return tuple(int(a + (b - a) * t) for a, b in zip(HEAT_STOPS[i], HEAT_STOPS[i + 1]))


class Slider:
    def __init__(self, rect, min_value, max_value, value, label, fmt="{:.2f}"):
        self.rect = pygame.Rect(rect)
        self.min, self.max, self.value = min_value, max_value, value
        self.label, self.fmt = label, fmt
        self.dragging = False

    def handle_event(self, event) -> bool:
        """Returns True when the value changed."""
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.inflate(0, 20).collidepoint(event.pos):
                self.dragging = True
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.dragging = False
        if self.dragging and event.type in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEMOTION):
            frac = (event.pos[0] - self.rect.x) / self.rect.w
            self.value = self.min + max(0.0, min(1.0, frac)) * (self.max - self.min)
            return True
        return False

    def draw(self, surf, font):
        surf.blit(font.render(f"{self.label}: {self.fmt.format(self.value)}", True, INK),
                  (self.rect.x, self.rect.y - 26))
        pygame.draw.rect(surf, GRID, self.rect, border_radius=4)
        frac = (self.value - self.min) / (self.max - self.min)
        filled = self.rect.copy()
        filled.w = int(self.rect.w * frac)
        pygame.draw.rect(surf, ACCENT, filled, border_radius=4)
        pygame.draw.circle(surf, ACCENT, (self.rect.x + filled.w, self.rect.centery), 10)
        pygame.draw.circle(surf, PANEL, (self.rect.x + filled.w, self.rect.centery), 5)


class Button:
    def __init__(self, rect, text, color=ACCENT):
        self.rect = pygame.Rect(rect)
        self.text, self.color = text, color

    def clicked(self, event) -> bool:
        return (event.type == pygame.MOUSEBUTTONDOWN and event.button == 1
                and self.rect.collidepoint(event.pos))

    def draw(self, surf, font):
        hover = self.rect.collidepoint(pygame.mouse.get_pos())
        color = tuple(min(255, c + 25) for c in self.color) if hover else self.color
        pygame.draw.rect(surf, color, self.rect, border_radius=8)
        label = font.render(self.text, True, (255, 255, 255))
        surf.blit(label, label.get_rect(center=self.rect.center))


class LineChart:
    """Line chart of one or more series on a shared y-axis."""

    def __init__(self, rect, title, x_max, zero_based=True):
        self.rect = pygame.Rect(rect)
        self.title, self.x_max = title, x_max
        self.zero_based = zero_based  # False: zoom the y-axis on the data range

    def draw(self, surf, font, small, series: list[tuple[str, tuple, list[float]]]):
        pygame.draw.rect(surf, PANEL, self.rect, border_radius=10)
        surf.blit(font.render(self.title, True, INK), (self.rect.x + 16, self.rect.y + 12))
        plot = pygame.Rect(self.rect.x + 70, self.rect.y + 50, self.rect.w - 100, self.rect.h - 90)
        values = [x for _, _, v in series for x in v]
        y_max = max(values + [1.0]) * 1.1
        y_min = 0.0 if self.zero_based or not values else min(values) * 0.85
        span = y_max - y_min

        for k in range(5):  # horizontal grid + y labels
            y = plot.bottom - k * plot.h / 4
            pygame.draw.line(surf, GRID, (plot.x, y), (plot.right, y))
            lbl = small.render(f"{y_min + span * k / 4:,.0f}", True, INK_2)
            surf.blit(lbl, (plot.x - lbl.get_width() - 8, y - 8))
        for d in range(0, self.x_max + 1, 5):
            x = plot.x + d / self.x_max * plot.w
            surf.blit(small.render(str(d), True, INK_2), (x - 6, plot.bottom + 6))

        legend_x = plot.right
        for name, color, values in reversed(series):
            lbl = small.render(name, True, INK)
            legend_x -= lbl.get_width() + 34
            pygame.draw.line(surf, color, (legend_x, self.rect.y + 22), (legend_x + 18, self.rect.y + 22), 3)
            surf.blit(lbl, (legend_x + 24, self.rect.y + 14))
            if len(values) >= 1:
                pts = [(plot.x + (i + 1) / self.x_max * plot.w, plot.bottom - (v - y_min) / span * plot.h)
                       for i, v in enumerate(values)]
                if len(pts) > 1:
                    pygame.draw.lines(surf, color, False, pts, 3)
                pygame.draw.circle(surf, color, pts[-1], 5)
