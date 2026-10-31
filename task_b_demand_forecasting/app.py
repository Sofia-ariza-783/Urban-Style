"""Interfaz Pygame de la Tarea B: pronóstico de demanda con encoder-decoder LSTM.

Uso:
    python app.py
    python app.py --data data/sample_retail_sales.csv

Controles:
    - Botones de serie (izquierda): eligen la categoría a pronosticar.
    - Sliders: máximo de épocas, lookback, horizonte (semanas a pronosticar) y semanas de historia.
    - "Entrenar": reentrena con los valores de los sliders.
    - "Media móvil": muestra/oculta el baseline.
    - "Exportar CSV + explicación": guarda las predicciones y un texto que las explica en outputs/.
"""
import argparse
import threading
from pathlib import Path

import pygame

import config
from data_prep import load_weekly_series
from forecasting import export_predictions, min_history_weeks, train_and_forecast
from narrator import write_narrative

WIDTH, HEIGHT = 1200, 740
PANEL_W = 300

BG = (245, 246, 250)
PANEL = (255, 255, 255)
TEXT = (30, 33, 40)
MUTED = (110, 116, 128)
GRID = (225, 228, 235)
ACCENT = (37, 99, 235)
ACTUAL = (55, 65, 81)
LSTM_C = (234, 88, 12)
BASE_C = (22, 163, 74)
VAL_C = (147, 51, 234)
VAL_BG = (245, 240, 255)
TEST_BG = (255, 247, 237)
FUTURE_BG = (236, 253, 245)


class Button:
    def __init__(self, rect, label, active=False):
        self.rect = pygame.Rect(rect)
        self.label = label
        self.active = active
        self.enabled = True

    def draw(self, surf, font):
        hover = self.rect.collidepoint(pygame.mouse.get_pos()) and self.enabled
        if self.active:
            fill, fg = ACCENT, (255, 255, 255)
        else:
            fill, fg = ((232, 237, 250) if hover else (240, 242, 247)), TEXT
        if not self.enabled:
            fg = MUTED
        pygame.draw.rect(surf, fill, self.rect, border_radius=6)
        txt = font.render(self.label, True, fg)
        surf.blit(txt, txt.get_rect(center=self.rect.center))

    def clicked(self, event):
        return (self.enabled and event.type == pygame.MOUSEBUTTONDOWN and event.button == 1
                and self.rect.collidepoint(event.pos))


class Slider:
    def __init__(self, x, y, w, label, vmin, vmax, value, step=1):
        self.track = pygame.Rect(x, y + 22, w, 6)
        self.label = label
        self.vmin, self.vmax, self.step = vmin, vmax, step
        self.value = value
        self.dragging = False

    def set_range(self, vmin, vmax):
        self.vmin, self.vmax = vmin, max(vmin, vmax)
        self.value = min(max(self.value, self.vmin), self.vmax)

    def _knob_x(self):
        span = self.vmax - self.vmin or 1
        return self.track.x + (self.value - self.vmin) / span * self.track.w

    def handle(self, event):
        """Devuelve True si el valor cambió."""
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.track.inflate(16, 24).collidepoint(event.pos):
                self.dragging = True
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.dragging = False
        if self.dragging and event.type in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEMOTION):
            ratio = min(max((event.pos[0] - self.track.x) / self.track.w, 0), 1)
            raw = self.vmin + ratio * (self.vmax - self.vmin)
            new = int(round((raw - self.vmin) / self.step) * self.step + self.vmin)
            new = min(max(new, self.vmin), self.vmax)
            if new != self.value:
                self.value = new
                return True
        return False

    def draw(self, surf, font):
        surf.blit(font.render(f"{self.label}: {self.value}", True, TEXT),
                  (self.track.x, self.track.y - 22))
        pygame.draw.rect(surf, GRID, self.track, border_radius=3)
        filled = self.track.copy()
        filled.w = self._knob_x() - self.track.x
        pygame.draw.rect(surf, ACCENT, filled, border_radius=3)
        pygame.draw.circle(surf, ACCENT, (int(self._knob_x()), self.track.centery), 9)


def fmt_money(v):
    return f"${v:,.0f}"


def safe_name(series_name):
    return series_name.replace(" - ", "_").replace(" ", "_")


class ForecastApp:
    def __init__(self, data_path):
        pygame.init()
        pygame.display.set_caption("UrbanStyle - Tarea B: Pronóstico de demanda (encoder-decoder LSTM)")
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("arial", 15)
        self.small = pygame.font.SysFont("arial", 12)
        self.title_font = pygame.font.SysFont("arial", 20, bold=True)
        self.h2 = pygame.font.SysFont("arial", 16, bold=True)

        self.weekly = load_weekly_series(data_path)
        self.n_weeks = len(self.weekly)
        self.series_names = list(self.weekly.columns)
        self.selected = 0
        self.result = None
        self.status = ""
        self.dirty = False
        self.show_baseline = True
        self.export_thread = None

        x, w = 20, PANEL_W - 40
        y = 70
        self.series_buttons = []
        for i, name in enumerate(self.series_names):
            self.series_buttons.append(Button((x, y, w, 30), name, active=(i == 0)))
            y += 36
        y += 8
        self.epochs = Slider(x, y, w, "Máx. épocas", 50, 1000, config.EPOCHS, step=50)
        y += 52
        self.lookback = Slider(x, y, w, "Lookback (semanas de entrada)", 2, 8, config.LOOKBACK)
        y += 52
        self.horizon = Slider(x, y, w, "Horizonte (semanas a pronosticar)", 1,
                              config.VAL_WEEKS, config.HORIZON)
        y += 52
        self.history = Slider(x, y, w, "Semanas de historia", 0, self.n_weeks, self.n_weeks)
        self._update_history_range()
        y += 58
        self.train_btn = Button((x, y, w, 34), "Entrenar")
        y += 40
        self.baseline_btn = Button((x, y, w, 28), "Media móvil", active=True)
        y += 34
        self.export_btn = Button((x, y, w, 28), "Exportar CSV + explicación")

        self.train()

    def _update_history_range(self):
        min_hist = min_history_weeks(self.lookback.value, self.horizon.value)
        self.history.set_range(min(min_hist, self.n_weeks), self.n_weeks)

    def train(self):
        name = self.series_names[self.selected]
        self.status = f"Entrenando '{name}'..."
        self.draw()
        pygame.display.flip()
        try:
            self.result = train_and_forecast(
                self.weekly[name],
                lookback=self.lookback.value,
                horizon=self.horizon.value,
                epochs=self.epochs.value,
                history_weeks=self.history.value,
            )
            r = self.result
            self.status = f"Listo · mejor época {r.best_epoch} de {r.epochs_run}"
        except ValueError as exc:
            self.result = None
            self.status = str(exc)
        self.dirty = False

    def export(self):
        """Exporta el CSV al instante y genera la explicación en segundo plano
        (la llamada al modelo de lenguaje puede tardar unos segundos)."""
        if not self.result or (self.export_thread and self.export_thread.is_alive()):
            return
        result = self.result
        name = safe_name(result.series_name)
        csv_path = export_predictions([result], config.OUTPUT_DIR / f"predictions_{name}.csv")
        self.status = "CSV listo. Generando explicación..."
        self.export_btn.enabled = False

        def worker():
            try:
                path, engine = write_narrative(result, config.OUTPUT_DIR / f"explicacion_{name}.md")
                motor = "Claude" if engine.startswith("Claude") else "reglas"
                self.status = f"Exportados {csv_path.name} y {path.name} ({motor})"
            except Exception as exc:
                self.status = f"Error en explicación: {exc}"
            finally:
                self.export_btn.enabled = True

        self.export_thread = threading.Thread(target=worker, daemon=True)
        self.export_thread.start()

    # ------------------------------------------------------------------ eventos
    def handle(self, event):
        for i, btn in enumerate(self.series_buttons):
            if btn.clicked(event) and i != self.selected:
                self.series_buttons[self.selected].active = False
                btn.active = True
                self.selected = i
                self.train()
                return
        for slider in (self.epochs, self.lookback, self.horizon, self.history):
            if slider.handle(event):
                self.dirty = True
                if slider in (self.lookback, self.horizon):
                    self._update_history_range()
        if self.train_btn.clicked(event):
            self.train()
        elif self.baseline_btn.clicked(event):
            self.show_baseline = not self.show_baseline
            self.baseline_btn.active = self.show_baseline
        elif self.export_btn.clicked(event):
            self.export()

    # ------------------------------------------------------------------ dibujo
    def draw(self):
        self.screen.fill(BG)
        pygame.draw.rect(self.screen, PANEL, (0, 0, PANEL_W, HEIGHT))
        self.screen.blit(self.title_font.render("Tarea B - Pronóstico", True, TEXT), (20, 18))
        self.screen.blit(self.small.render(f"Objetivo: {config.TARGET_COLUMN} semanal",
                                           True, MUTED), (20, 44))
        for btn in self.series_buttons:
            btn.draw(self.screen, self.font)
        for slider in (self.epochs, self.lookback, self.horizon, self.history):
            slider.draw(self.screen, self.font)
        self.train_btn.label = "Entrenar *" if self.dirty else "Entrenar"
        self.train_btn.active = self.dirty
        for btn in (self.train_btn, self.baseline_btn, self.export_btn):
            btn.draw(self.screen, self.font)
        self._draw_status()

        self.draw_forecast(pygame.Rect(PANEL_W + 70, 40, WIDTH - PANEL_W - 100, 400))
        self.draw_metrics(pygame.Rect(PANEL_W + 20, 490, 300, 230))
        self.draw_future(pygame.Rect(PANEL_W + 330, 490, 250, 230))
        self.draw_loss(pygame.Rect(PANEL_W + 590, 512, WIDTH - PANEL_W - 610, 208))

    def _draw_status(self):
        # Texto en hasta 2 líneas para que quepa en el panel
        words, lines, line = self.status.split(), [], ""
        for word in words:
            if self.small.size(line + " " + word)[0] > PANEL_W - 40:
                lines.append(line)
                line = word
            else:
                line = f"{line} {word}".strip()
        lines.append(line)
        for i, ln in enumerate(lines[-2:]):
            self.screen.blit(self.small.render(ln, True, MUTED), (20, HEIGHT - 38 + i * 15))

    def draw_forecast(self, area):
        r = self.result
        title = f"{r.series_name}: real vs pronóstico" if r else "Sin resultado"
        self.screen.blit(self.title_font.render(title, True, TEXT), (area.x - 40, 8))
        if not r:
            return

        n = len(r.actual)
        xs_total = n + r.horizon
        values = list(r.actual) + list(r.test_pred) + list(r.future_pred)
        if self.show_baseline:
            values += list(r.test_baseline) + list(r.future_baseline)
        vmax = max(values) * 1.1 or 1.0

        def px(i):
            return area.x + i / (xs_total - 1) * area.w

        def py(v):
            return area.bottom - v / vmax * area.h

        # Zonas: validación, prueba y pronóstico futuro
        zones = [(r.val_start, r.test_start, VAL_BG, "validación", VAL_C),
                 (r.test_start, n, TEST_BG, "prueba", LSTM_C),
                 (n - 0.5, xs_total - 1, FUTURE_BG, "pronóstico", BASE_C)]
        for start, end, color, label, lcolor in zones:
            x0, x1 = px(max(start - 0.5, 0)), px(end - 0.5) if end < xs_total - 1 else area.right
            pygame.draw.rect(self.screen, color, (x0, area.y, x1 - x0, area.h))
            self.screen.blit(self.small.render(label, True, lcolor), (x0 + 4, area.y + 4))

        # Rejilla y eje Y
        for k in range(5):
            v = vmax * k / 4
            y = py(v)
            pygame.draw.line(self.screen, GRID, (area.x, y), (area.right, y))
            lbl = self.small.render(fmt_money(v), True, MUTED)
            self.screen.blit(lbl, (area.x - lbl.get_width() - 6, y - 7))
        # Eje X: algunas fechas
        step = max(1, n // 6)
        for i in range(0, n, step):
            lbl = self.small.render(f"{r.weeks[i]:%d %b}", True, MUTED)
            self.screen.blit(lbl, (px(i) - lbl.get_width() / 2, area.bottom + 6))
        last_lbl = self.small.render(f"{r.future_weeks[-1]:%d %b}", True, BASE_C)
        self.screen.blit(last_lbl, (area.right - last_lbl.get_width(), area.bottom + 6))

        # Serie real
        pts = [(px(i), py(v)) for i, v in enumerate(r.actual)]
        pygame.draw.lines(self.screen, ACTUAL, False, pts, 2)

        split = r.test_start
        # Baseline (media móvil)
        if self.show_baseline:
            bpts = [(px(split + i), py(v)) for i, v in enumerate(r.test_baseline)]
            bpts += [(px(n + i), py(v)) for i, v in enumerate(r.future_baseline)]
            pygame.draw.lines(self.screen, BASE_C, False, bpts, 2)

        # LSTM: prueba (bloques de `horizon` semanas) + futuro
        lpts = [(px(split + i), py(v)) for i, v in enumerate(r.test_pred)]
        fpts = [(px(n + i), py(v)) for i, v in enumerate(r.future_pred)]
        pygame.draw.lines(self.screen, LSTM_C, False, lpts + fpts[:1], 3)
        if len(fpts) > 1:
            pygame.draw.lines(self.screen, LSTM_C, False, fpts, 3)
        for i, p in enumerate(lpts):
            radius = 5 if i % r.horizon == 0 else 3  # punto grande = inicio de cada bloque
            pygame.draw.circle(self.screen, LSTM_C, (int(p[0]), int(p[1])), radius)
        for p in fpts:
            pygame.draw.circle(self.screen, LSTM_C, (int(p[0]), int(p[1])), 7)
            pygame.draw.circle(self.screen, (255, 255, 255), (int(p[0]), int(p[1])), 3)

        # Leyenda
        lx, ly = area.x + 80, area.y + 6
        items = [(ACTUAL, "Real"), (LSTM_C, f"Encoder-decoder ({r.horizon} sem)")]
        if self.show_baseline:
            items.append((BASE_C, f"Media móvil ({config.MOVING_AVG_WINDOW} sem)"))
        for color, label in items:
            pygame.draw.line(self.screen, color, (lx, ly + 8), (lx + 22, ly + 8), 3)
            text = self.small.render(label, True, TEXT)
            self.screen.blit(text, (lx + 28, ly + 1))
            lx += text.get_width() + 50

    def draw_metrics(self, area):
        pygame.draw.rect(self.screen, PANEL, area, border_radius=8)
        r = self.result
        if not r:
            return
        x, y = area.x + 14, area.y + 10
        self.screen.blit(self.h2.render("Métricas (periodo de prueba)", True, TEXT), (x, y))
        y += 28
        cols = [x, x + 120, x + 200]
        for cx, h in zip(cols, ["Modelo", "MAE", "RMSE"]):
            self.screen.blit(self.small.render(h, True, MUTED), (cx, y))
        y += 18
        for name, m, s, color in [("LSTM", r.mae, r.rmse, LSTM_C),
                                  ("Media móvil", r.baseline_mae, r.baseline_rmse, BASE_C)]:
            self.screen.blit(self.font.render(name, True, color), (cols[0], y))
            self.screen.blit(self.font.render(fmt_money(m), True, TEXT), (cols[1], y))
            self.screen.blit(self.font.render(fmt_money(s), True, TEXT), (cols[2], y))
            y += 22
        y += 6
        self.screen.blit(self.small.render("MAE LSTM según semanas adelante:", True, MUTED), (x, y))
        y += 16
        hm = r.horizon_metrics
        for i in range(0, len(hm), 3):  # 3 por línea para que quepa en el recuadro
            line = "   ".join(f"{m['h']}s: {fmt_money(m['mae'])}" for m in hm[i:i + 3])
            self.screen.blit(self.small.render(line, True, TEXT), (x, y))
            y += 15
        y += 8 if len(hm) > 3 else 11
        stop = "sí" if r.early_stopped else "no (llegó al máximo)"
        self.screen.blit(self.small.render(f"Early stopping: {stop}", True, TEXT), (x, y))
        y += 16
        self.screen.blit(self.small.render(
            f"Mejor época: {r.best_epoch} de {r.epochs_run}", True, TEXT), (x, y))
        y += 16
        winner = "LSTM" if r.mae < r.baseline_mae else "Media móvil"
        self.screen.blit(self.small.render(f"Menor MAE: {winner}", True, MUTED), (x, y))

    def draw_future(self, area):
        pygame.draw.rect(self.screen, PANEL, area, border_radius=8)
        r = self.result
        if not r:
            return
        x, y = area.x + 14, area.y + 10
        self.screen.blit(self.h2.render(f"Próximas {r.horizon} semanas", True, TEXT), (x, y))
        y += 28
        cols = [x, x + 90, x + 165]
        for cx, h in zip(cols, ["Semana", "LSTM", "Media móvil"]):
            self.screen.blit(self.small.render(h, True, MUTED), (cx, y))
        y += 18
        for w, p, b in zip(r.future_weeks, r.future_pred, r.future_baseline):
            self.screen.blit(self.font.render(f"{w:%d-%b-%y}", True, TEXT), (cols[0], y))
            self.screen.blit(self.font.render(fmt_money(p), True, LSTM_C), (cols[1], y))
            self.screen.blit(self.font.render(fmt_money(b), True, BASE_C), (cols[2], y))
            y += 21
        pygame.draw.line(self.screen, GRID, (x, y + 2), (area.right - 14, y + 2))
        y += 8
        self.screen.blit(self.font.render("Total", True, TEXT), (cols[0], y))
        self.screen.blit(self.font.render(fmt_money(r.future_pred.sum()), True, LSTM_C), (cols[1], y))
        self.screen.blit(self.font.render(fmt_money(r.future_baseline.sum()), True, BASE_C),
                         (cols[2], y))

    def draw_loss(self, area):
        self.screen.blit(self.h2.render("Pérdida (MSE)", True, TEXT), (area.x, area.y - 22))
        pygame.draw.rect(self.screen, PANEL, area, border_radius=6)
        r = self.result
        if not r or len(r.train_loss) < 2:
            return
        lmax = max(max(r.train_loss), max(r.val_loss)) or 1.0
        n = len(r.train_loss)

        def pt(i, v):
            return (area.x + 8 + i / (n - 1) * (area.w - 16),
                    area.bottom - 22 - v / lmax * (area.h - 46))

        best_x = pt(r.best_epoch - 1, 0)[0]
        pygame.draw.line(self.screen, GRID, (best_x, area.y + 20), (best_x, area.bottom - 22), 2)
        pygame.draw.lines(self.screen, ACCENT, False,
                          [pt(i, v) for i, v in enumerate(r.train_loss)], 2)
        pygame.draw.lines(self.screen, VAL_C, False,
                          [pt(i, v) for i, v in enumerate(r.val_loss)], 2)
        self.screen.blit(self.small.render("entrenamiento", True, ACCENT), (area.x + 8, area.y + 4))
        self.screen.blit(self.small.render("validación", True, VAL_C), (area.x + 100, area.y + 4))
        self.screen.blit(self.small.render(f"mejor época: {r.best_epoch}", True, MUTED),
                         (area.x + 8, area.bottom - 18))

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (
                        event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    running = False
                else:
                    self.handle(event)
            self.draw()
            pygame.display.flip()
            self.clock.tick(30)
        pygame.quit()


def main():
    parser = argparse.ArgumentParser(description="Tarea B - App Pygame")
    parser.add_argument("--data", type=Path, default=config.DATA_PATH)
    args = parser.parse_args()
    ForecastApp(args.data).run()


if __name__ == "__main__":
    main()
