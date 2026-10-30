"""Interactive Pygame view of customers moving through the UrbanStyle store.

Run:  python -m pygame_ui.app                      (from task_a_pathfinding/)
      python -m pygame_ui.app --screenshot results/figures/app.png --ticks 120
                                                   (headless: simulates N ticks, saves an image)

Controls:
    Algorithm    click to cycle BFS -> A* -> Congestion-aware (restarts the run)
    Layout       click to cycle the store layouts (restarts the run)
    Customers    slider, applied when released (restarts the run)
    Speed        slider, simulation ticks per second
    Run / Pause  play or stop the simulation       Step    advance one tick
    Reset        start the same run again          Heatmap / Paths    toggle overlays

Customers are circles coloured by the section they are heading to. The heatmap
shows how often each floor cell was occupied; red outlines mark hotspots.
"""
from __future__ import annotations

import argparse
import os

import pygame

import config
from pygame_ui.widgets import BG, INK, INK_2, PANEL, Button, Slider, heat_color
from search.algorithms import ALGORITHMS
from simulation.store_sim import StoreSimulation

W, H = 1280, 760
CANVAS = pygame.Rect(380, 20, 880, 630)         # store drawing area
LEGEND = pygame.Rect(380, 662, 880, 78)

WALL = (70, 70, 76)
FLOOR = (240, 239, 235)
ENTRANCE_FILL, EXIT_FILL = (190, 235, 210), (245, 200, 195)
HOTSPOT_OUTLINE = (220, 20, 20)
EXIT_TARGET_COLOR = (90, 89, 85)                # customers walking to the exit
GREEN = (27, 140, 98)
SHORT_NAMES = {"W": "Women", "M": "Men", "B": "Beauty", "T": "Tech", "F": "Fitting", "C": "Checkout"}


def fit_text(font: pygame.font.Font, text: str, width: int) -> str:
    """Shorten text with '...' so that it fits in the given pixel width."""
    if font.size(text)[0] <= width:
        return text
    while text and font.size(text + "...")[0] > width:
        text = text[:-1]
    return text + "..."


class StoreApp:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("UrbanStyle - Customer Pathfinding")
        self.screen = pygame.display.set_mode((W, H))
        self.font = pygame.font.SysFont("segoeui,arial", 20)
        self.big = pygame.font.SysFont("segoeui,arial", 30, bold=True)
        self.small = pygame.font.SysFont("segoeui,arial", 15)
        self.clock = pygame.time.Clock()

        self.algo_names = list(ALGORITHMS)
        self.layout_names = list(config.LAYOUTS)
        self.algo_idx = self.algo_names.index(config.DEFAULT_ALGORITHM)
        self.layout_idx = self.layout_names.index(config.DEFAULT_LAYOUT)

        self.btn_algo = Button((40, 104, 300, 40), "", (98, 80, 214))
        self.btn_layout = Button((40, 152, 300, 40), "", (82, 81, 78))
        self.sl_customers = Slider((40, 236, 300, 10), 1, 60, config.DEFAULT_CUSTOMERS, "Customers", "{:.0f}")
        self.sl_speed = Slider((40, 290, 300, 10), 1, 30, 8, "Speed (ticks/s)", "{:.0f}")
        self.btn_run = Button((40, 318, 145, 40), "Run", GREEN)
        self.btn_step = Button((195, 318, 145, 40), "Step", (82, 81, 78))
        self.btn_reset = Button((40, 366, 145, 40), "Reset")
        self.btn_heat = Button((195, 366, 145, 40), "", (235, 104, 52))
        self.btn_paths = Button((40, 414, 300, 40), "", (235, 104, 52))

        self.show_heat = True
        self.show_paths = True
        self.running = False
        self.last_step = 0
        self.sim: StoreSimulation
        self.reset()

    # ------------------------------------------------------------------
    # Simulation control
    # ------------------------------------------------------------------
    @property
    def algorithm(self) -> str:
        return self.algo_names[self.algo_idx]

    @property
    def layout_name(self) -> str:
        return self.layout_names[self.layout_idx]

    def reset(self) -> None:
        """New run with the current layout, algorithm and customer count (same seed,
        so algorithms and layouts can be compared on the same customers)."""
        self.sim = StoreSimulation(self.layout_name, self.algorithm,
                                   n_customers=int(round(self.sl_customers.value)),
                                   seed=config.DEFAULT_SEED)
        self.running = False

    def step(self) -> None:
        if not self.sim.done:
            self.sim.step()
        if self.sim.done:
            self.running = False

    def handle(self, event) -> None:
        was_dragging = self.sl_customers.dragging
        if self.sl_customers.handle_event(event):
            self.sl_customers.value = round(self.sl_customers.value)    # whole customers only
        if was_dragging and not self.sl_customers.dragging:             # released
            if int(round(self.sl_customers.value)) != self.sim.n_customers:
                self.reset()
        self.sl_speed.handle_event(event)

        if self.btn_algo.clicked(event):
            self.algo_idx = (self.algo_idx + 1) % len(self.algo_names)
            self.reset()
        elif self.btn_layout.clicked(event):
            self.layout_idx = (self.layout_idx + 1) % len(self.layout_names)
            self.reset()
        elif self.btn_run.clicked(event):
            if self.sim.done:
                self.reset()
            self.running = not self.running
        elif self.btn_step.clicked(event):
            self.running = False
            self.step()
        elif self.btn_reset.clicked(event):
            self.reset()
        elif self.btn_heat.clicked(event):
            self.show_heat = not self.show_heat
        elif self.btn_paths.clicked(event):
            self.show_paths = not self.show_paths

    def update(self) -> None:
        now = pygame.time.get_ticks()
        if self.running and now - self.last_step >= 1000 / self.sl_speed.value:
            self.step()
            self.last_step = now

    # ------------------------------------------------------------------
    # Drawing: store
    # ------------------------------------------------------------------
    def grid_geometry(self) -> tuple[int, int, int]:
        """(x0, y0, cell size) of the grid, scaled to fit and centred in the canvas."""
        layout = self.sim.layout
        area = CANVAS.inflate(-40, -40)
        area.top += 36                                   # room for the header line
        area.h -= 36
        cs = min(area.w // layout.width, area.h // layout.height)
        x0 = area.x + (area.w - cs * layout.width) // 2
        y0 = area.y + (area.h - cs * layout.height) // 2
        return x0, y0, cs

    def cell_rect(self, cell, geometry) -> pygame.Rect:
        x0, y0, cs = geometry
        return pygame.Rect(x0 + cell[1] * cs, y0 + cell[0] * cs, cs, cs)

    def draw_store(self) -> None:
        s, sim, layout = self.screen, self.sim, self.sim.layout
        geo = self.grid_geometry()
        cs = geo[2]

        for r in range(layout.height):                   # walls, floor, shelves
            for c in range(layout.width):
                ch = layout.rows[r][c]
                rect = self.cell_rect((r, c), geo)
                if ch in config.SECTIONS:
                    color = config.SECTIONS[ch]["color"]
                elif ch == "#":
                    color = WALL
                else:
                    color = FLOOR
                pygame.draw.rect(s, color, rect)
                if layout.walkable((r, c)):
                    pygame.draw.rect(s, (225, 224, 220), rect, 1)

        for cell, fill in ((layout.entrance, ENTRANCE_FILL), (layout.exit, EXIT_FILL)):
            pygame.draw.rect(s, fill, self.cell_rect(cell, geo))

        if self.show_heat:
            self.draw_heat(geo)

        for cell, letter in ((layout.entrance, "In"), (layout.exit, "Out")):    # marked above the heat
            rect = self.cell_rect(cell, geo)
            label = self.small.render(letter, True, INK)
            s.blit(label, label.get_rect(midtop=(rect.centerx, rect.y + 3)))

        for code, shelves in layout.sections.items():    # section labels, centred on the shelves
            row = sum(r for r, _ in shelves) / len(shelves)
            col = sum(c for _, c in shelves) / len(shelves)
            label = self.small.render(SHORT_NAMES[code], True, (255, 255, 255))
            s.blit(label, label.get_rect(center=(geo[0] + (col + 0.5) * cs, geo[1] + (row + 0.5) * cs)))

        if self.show_paths:
            self.draw_paths(geo)
        self.draw_customers(geo)

    def draw_heat(self, geo) -> None:
        sim = self.sim
        cs = geo[2]
        peak = max(int(sim.heat.max()), 1)
        overlay = pygame.Surface((cs, cs), pygame.SRCALPHA)
        for cell in sim.layout.walkable_cells():
            if sim.heat[cell] > 0:
                overlay.fill((*heat_color(sim.heat[cell] / peak), 185))
                self.screen.blit(overlay, self.cell_rect(cell, geo))
        for r, c in zip(*sim.hotspot_mask().nonzero()):
            pygame.draw.rect(self.screen, HOTSPOT_OUTLINE, self.cell_rect((int(r), int(c)), geo), 3)

    @staticmethod
    def target_color(code: str) -> tuple[int, int, int]:
        return config.SECTIONS[code]["color"] if code in config.SECTIONS else EXIT_TARGET_COLOR

    def draw_paths(self, geo) -> None:
        for c in self.sim.active:
            if c.path:
                points = [self.cell_rect(cell, geo).center for cell in [c.pos, *c.path]]
                pygame.draw.lines(self.screen, self.target_color(c.current_target), False, points, 2)

    def draw_customers(self, geo) -> None:
        cs = geo[2]
        by_cell: dict = {}
        for c in self.sim.active:
            by_cell.setdefault(c.pos, []).append(c)
        radius = max(4, int(cs * 0.17))
        for cell, group in by_cell.items():
            cx, cy = self.cell_rect(cell, geo).center
            for i, c in enumerate(group):
                if len(group) > 1:                       # spread a crowd around the cell centre
                    angle = 6.2832 * i / len(group)
                    dx, dy = pygame.math.Vector2(0, -cs * 0.25).rotate_rad(angle)
                else:
                    dx = dy = 0
                pos = (int(cx + dx), int(cy + dy))
                pygame.draw.circle(self.screen, self.target_color(c.current_target), pos, radius)
                pygame.draw.circle(self.screen, INK, pos, radius, 2)

    # ------------------------------------------------------------------
    # Drawing: panels
    # ------------------------------------------------------------------
    def stat(self, label: str, value: str, y: int) -> None:
        self.screen.blit(self.small.render(label, True, INK_2), (40, y + 3))
        text = self.font.render(value, True, INK)
        self.screen.blit(text, (340 - text.get_width(), y))

    def draw_controls(self) -> None:
        s, sim = self.screen, self.sim
        pygame.draw.rect(s, PANEL, (20, 20, 340, 720), border_radius=10)
        s.blit(self.big.render("UrbanStyle Store", True, INK), (40, 34))
        s.blit(self.small.render("Customer pathfinding  |  BFS, A*, congestion-aware", True, INK_2), (40, 74))

        self.btn_algo.text = f"Algorithm: {self.algorithm}"
        self.btn_layout.text = f"Layout: {self.layout_name}"
        self.btn_run.text = "Pause" if self.running else "Run"
        self.btn_heat.text = f"Heatmap: {'on' if self.show_heat else 'off'}"
        self.btn_paths.text = f"Paths: {'on' if self.show_paths else 'off'}"
        for slider in (self.sl_customers, self.sl_speed):
            slider.draw(s, self.font)
        for b in (self.btn_algo, self.btn_layout, self.btn_run, self.btn_step,
                  self.btn_reset, self.btn_heat, self.btn_paths):
            b.draw(s, self.font)

        st = sim.stats()
        self.stat("Tick", f"{sim.tick}", 474)
        self.stat("In store / served / total", f"{len(sim.active)} / {st['served']} / {sim.n_customers}", 502)
        self.stat("Avg nodes expanded / search", f"{st['avg_nodes_expanded_per_search']:.1f}", 530)
        self.stat("Avg path length (steps)", f"{st['avg_path_length']:.1f}", 558)
        s.blit(self.font.render("Top hotspots", True, INK), (40, 596))
        spots = sim.hotspots(3)
        for i, (cell, count) in enumerate(spots):
            line = f"{i + 1}. ({cell[0]},{cell[1]})  {count} visits  {sim.layout.describe(cell)}"
            s.blit(self.small.render(fit_text(self.small, line, 300), True, INK_2), (40, 628 + 22 * i))
        if not spots:
            s.blit(self.small.render("none yet", True, INK_2), (40, 628))

    def draw_canvas(self) -> None:
        s, sim = self.screen, self.sim
        pygame.draw.rect(s, PANEL, CANVAS, border_radius=10)
        title = f"{self.layout_name}  |  {self.algorithm}"
        s.blit(self.font.render(title, True, INK), (CANVAS.x + 20, CANVAS.y + 12))
        if sim.done:
            msg = self.font.render("All customers served", True, GREEN)
            s.blit(msg, (CANVAS.right - 20 - msg.get_width(), CANVAS.y + 12))
        self.draw_store()

    def draw_legend(self) -> None:
        s = self.screen
        pygame.draw.rect(s, PANEL, LEGEND, border_radius=10)
        items = [(config.SECTIONS[code]["name"], config.SECTIONS[code]["color"]) for code in config.SECTIONS]
        items.append(("Heading to exit", EXIT_TARGET_COLOR))
        for i, (name, color) in enumerate(items):
            x = LEGEND.x + 20 + (i % 4) * 165
            y = LEGEND.y + 14 + (i // 4) * 26
            pygame.draw.circle(s, color, (x + 8, y + 9), 8)
            pygame.draw.circle(s, INK, (x + 8, y + 9), 8, 1)
            s.blit(self.small.render(name, True, INK), (x + 24, y))
        # heat gradient bar
        bar = pygame.Rect(LEGEND.right - 190, LEGEND.y + 40, 160, 14)
        for i in range(bar.w):
            pygame.draw.line(s, heat_color(i / (bar.w - 1)), (bar.x + i, bar.y), (bar.x + i, bar.bottom - 1))
        s.blit(self.small.render("Visits: low to high", True, INK_2), (bar.x, bar.y - 22))
        s.blit(self.small.render("Customer colour = section it is heading to", True, INK_2), (LEGEND.x + 20, LEGEND.y + 58))

    def draw(self) -> None:
        self.screen.fill(BG)
        self.draw_controls()
        self.draw_canvas()
        self.draw_legend()
        pygame.display.flip()

    def run(self) -> None:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                self.handle(event)
            self.update()
            self.draw()
            self.clock.tick(60)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--screenshot", help="headless mode: simulate --ticks ticks and save a PNG")
    parser.add_argument("--ticks", type=int, default=120, help="ticks to simulate in headless mode")
    args = parser.parse_args()
    if args.screenshot:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        app = StoreApp()
        for _ in range(args.ticks):
            app.step()
        app.draw()
        pygame.image.save(app.screen, args.screenshot)
        st = app.sim.stats()
        print(f"Saved {args.screenshot}  ({app.layout_name} / {app.algorithm}: tick {st['ticks']}, "
              f"served {st['served']}/{app.sim.n_customers})")
        pygame.quit()
    else:
        StoreApp().run()


if __name__ == "__main__":
    main()
