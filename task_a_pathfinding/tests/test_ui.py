"""Headless tests of the Pygame app's controls.  Run:  python -m pytest tests"""
import os

os.environ["SDL_VIDEODRIVER"] = "dummy"

import pygame
import pytest

import config
from pygame_ui.app import StoreApp


@pytest.fixture
def app():
    app = StoreApp()
    yield app
    pygame.quit()


def click(app, widget):
    """Send a left click at the centre of a button / slider."""
    pos = widget.rect.center
    app.handle(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=pos, button=1))
    app.handle(pygame.event.Event(pygame.MOUSEBUTTONUP, pos=pos, button=1))


def drag_slider(app, slider, frac):
    x = slider.rect.x + int(slider.rect.w * frac)
    y = slider.rect.centery
    app.handle(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=(x, y), button=1))
    app.handle(pygame.event.Event(pygame.MOUSEMOTION, pos=(x, y), rel=(0, 0), buttons=(1, 0, 0)))
    app.handle(pygame.event.Event(pygame.MOUSEBUTTONUP, pos=(x, y), button=1))


def test_algorithm_and_layout_buttons_cycle_and_reset(app):
    app.step()
    algorithms = [app.algorithm]
    for _ in range(3):
        click(app, app.btn_algo)
        algorithms.append(app.algorithm)
        assert app.sim.tick == 0 and app.sim.algorithm_name == app.algorithm
    assert algorithms[0] == algorithms[-1] and len(set(algorithms)) == 3   # full cycle

    layouts = [app.layout_name]
    for _ in range(3):
        click(app, app.btn_layout)
        layouts.append(app.layout_name)
        assert app.sim.layout.name == app.layout_name
        app.draw()                                    # every layout renders
    assert layouts[0] == layouts[-1] and len(set(layouts)) == 3


def test_run_pause_step_reset(app):
    click(app, app.btn_run)
    assert app.running
    app.last_step = -10_000
    app.update()
    assert app.sim.tick == 1
    click(app, app.btn_run)
    assert not app.running
    click(app, app.btn_step)
    assert app.sim.tick == 2
    click(app, app.btn_reset)
    assert app.sim.tick == 0 and not app.running


def test_toggles(app):
    heat, paths = app.show_heat, app.show_paths
    click(app, app.btn_heat)
    click(app, app.btn_paths)
    assert app.show_heat is not heat and app.show_paths is not paths
    app.draw()


def test_customers_slider_applies_on_release(app):
    drag_slider(app, app.sl_customers, 0.0)
    assert app.sl_customers.value == 1 and app.sim.n_customers == 1
    drag_slider(app, app.sl_customers, 0.5)
    assert app.sim.n_customers == app.sl_customers.value == round(1 + 0.5 * 59)
    # Grabbing the slider without moving it must not restart a running sim
    app.step()
    tick = app.sim.tick
    app.handle(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=app.sl_customers.rect.center, button=1))
    app.handle(pygame.event.Event(pygame.MOUSEBUTTONUP, pos=app.sl_customers.rect.center, button=1))
    assert app.sim.tick == tick


def test_speed_slider_changes_speed_without_reset(app):
    app.step()
    drag_slider(app, app.sl_speed, 0.99)
    assert app.sl_speed.value == pytest.approx(30, abs=0.5)
    assert app.sim.tick == 1


def test_run_to_completion_shows_everyone_served(app):
    drag_slider(app, app.sl_customers, 0.0)           # 1 customer: finishes quickly
    click(app, app.btn_run)
    for _ in range(config.MAX_TICKS):
        if not app.running:
            break
        app.step()
    assert app.sim.done and not app.running
    assert app.sim.heat.sum() > 0
    app.draw()                                        # heatmap stays on screen
    click(app, app.btn_run)                           # Run after the end restarts
    assert app.sim.tick == 0 and app.running
