"""Interactive Pygame dashboard for the UrbanStyle dynamic pricing agent.

Run:  python -m pygame_ui.dashboard                (from task_c_dynamic_pricing/)
      python -m pygame_ui.dashboard --screenshot results/figures/dashboard.png
                                                   (headless: runs one episode, saves an image)

Controls:
    Slider       initial price (applied on reset)
    Reset Env    start a new episode (new random market)
    Step         advance one day
    Run Episode  play the rest of the episode automatically
    Policy       switch between PPO / DQN / static pricing

A "shadow" environment with the same seed is always priced statically, so the
chart compares the selected policy against static pricing on the *same*
customers and demand shocks.
"""
from __future__ import annotations

import argparse
import os

import pygame

import config
from env.pricing_env import UrbanStylePricingEnv
from evaluation.baselines import StaticPricePolicy
from pygame_ui.widgets import BG, INK, INK_2, PANEL, Button, LineChart, Slider

W, H = 1180, 740
STEP_MS = 150
COLOR_AGENT = (42, 120, 214)
COLOR_STATIC = (237, 161, 0)
SEGMENT_RGB = {"Young": (42, 120, 214), "Adult": (235, 104, 52), "Senior": (27, 175, 122)}


def load_policies() -> dict:
    policies = {}
    for algo in ("ppo", "dqn"):
        path = config.MODELS_DIR / f"{algo}_urbanstyle_seed0.zip"
        if path.exists():
            from training.train import load_model
            policies[algo.upper()] = load_model(algo, 0)
    policies["Static"] = StaticPricePolicy(1.0)
    return policies


class Dashboard:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("UrbanStyle - Dynamic Pricing with RL")
        self.screen = pygame.display.set_mode((W, H))
        self.font = pygame.font.SysFont("segoeui,arial", 20)
        self.big = pygame.font.SysFont("segoeui,arial", 30, bold=True)
        self.small = pygame.font.SysFont("segoeui,arial", 15)
        self.clock = pygame.time.Clock()

        self.env = UrbanStylePricingEnv()
        self.shadow = UrbanStylePricingEnv()
        self.policies = load_policies()
        self.policy_names = list(self.policies)
        self.policy_idx = 0
        self.message = "" if len(self.policies) > 1 else "No trained model found - run training first (static only)."

        ref = self.env.ref_price
        self.slider = Slider((40, 130, 300, 10), self.env.min_price, self.env.max_price, ref, "Initial price")
        self.btn_reset = Button((40, 175, 140, 44), "Reset Env")
        self.btn_step = Button((200, 175, 140, 44), "Step", (82, 81, 78))
        self.btn_run = Button((40, 232, 300, 44), "Run Episode", (27, 140, 98))
        self.btn_policy = Button((40, 289, 300, 44), "", (98, 80, 214))
        self.chart_rev = LineChart((380, 20, 780, 380), "Cumulative revenue", config.EPISODE_LENGTH)
        self.chart_price = LineChart((380, 415, 780, 305), "Price history", config.EPISODE_LENGTH,
                                     zero_based=False)

        self.episode_seed = 0
        self.running = False
        self.last_step = 0
        self.reset()

    # ------------------------------------------------------------------
    @property
    def policy_name(self) -> str:
        return self.policy_names[self.policy_idx]

    def reset(self, new_market: bool = True):
        if new_market:
            self.episode_seed += 1
        opts = {"initial_price": self.slider.value}
        self.obs, self.info = self.env.reset(seed=self.episode_seed, options=opts)
        self.shadow.reset(seed=self.episode_seed, options=opts)
        self.done, self.running = False, False
        self.reward = 0.0
        self.hist_rev, self.hist_static, self.hist_price, self.hist_static_price = [], [], [], []
        self.upcoming_segment = self.info["segment"]

    def step(self):
        if self.done:
            return
        policy = self.policies[self.policy_name]
        action, _ = policy.predict(self.obs, deterministic=True)
        self.obs, self.reward, self.done, _, self.info = self.env.step(int(action))
        _, _, _, _, s_info = self.shadow.step(2)  # static: keep the initial price
        self.hist_rev.append(self.info["total_revenue"])
        self.hist_static.append(s_info["total_revenue"])
        self.hist_price.append(self.info["price"])
        self.hist_static_price.append(s_info["price"])
        self.upcoming_segment = self.env.segment_names[self.env.segment]
        if self.done:
            self.running = False

    # ------------------------------------------------------------------
    def handle(self, event):
        if self.slider.handle_event(event) and not self.hist_rev:
            self.reset(new_market=False)  # same market, new starting price
        if self.btn_reset.clicked(event):
            self.reset()
        elif self.btn_step.clicked(event):
            self.step()
        elif self.btn_run.clicked(event):
            if self.done:
                self.reset()
            self.running = True
        elif self.btn_policy.clicked(event):
            self.policy_idx = (self.policy_idx + 1) % len(self.policy_names)
            self.reset()

    def update(self):
        now = pygame.time.get_ticks()
        if self.running and now - self.last_step >= STEP_MS:
            self.step()
            self.last_step = now

    # ------------------------------------------------------------------
    def stat(self, label, value, y, color=INK):
        self.screen.blit(self.small.render(label, True, INK_2), (40, y))
        self.screen.blit(self.big.render(value, True, color), (40, y + 18))

    def draw(self):
        s = self.screen
        s.fill(BG)
        pygame.draw.rect(s, PANEL, (20, 20, 340, 700), border_radius=10)
        s.blit(self.big.render("UrbanStyle Pricing", True, INK), (40, 34))
        s.blit(self.small.render(f"Category: {self.env.params['category']}  |  RL + Gymnasium", True, INK_2), (40, 74))

        self.slider.draw(s, self.font)
        for b in (self.btn_reset, self.btn_step, self.btn_run):
            b.draw(s, self.font)
        self.btn_policy.text = f"Policy: {self.policy_name}  (click to switch)"
        self.btn_policy.draw(s, self.font)

        i = self.info
        seg_today = i["segment"] if self.hist_rev else "-"
        self.stat("Day", f"{len(self.hist_rev)} / {config.EPISODE_LENGTH}", 350)
        self.stat("Current price", f"{self.env.price:,.2f}", 410, COLOR_AGENT)
        self.stat("Customer segment (today -> next)", f"{seg_today} -> {self.upcoming_segment}", 470,
                  SEGMENT_RGB.get(self.upcoming_segment, INK))
        self.stat("Units sold today", f"{i['quantity']}", 530)
        self.stat("Reward (today's revenue)", f"{self.reward:,.0f}", 590)
        static_total = self.hist_static[-1] if self.hist_static else 0.0
        gap = (100 * (i["total_revenue"] / static_total - 1)) if static_total else 0.0
        self.stat("Total revenue (vs static)", f"{i['total_revenue']:,.0f}  ({gap:+.1f}%)", 650)

        self.chart_rev.draw(s, self.font, self.small, [
            (self.policy_name, COLOR_AGENT, self.hist_rev),
            ("Static price", COLOR_STATIC, self.hist_static),
        ])
        self.chart_price.draw(s, self.font, self.small, [
            (f"{self.policy_name} price", COLOR_AGENT, self.hist_price),
            ("Static price", COLOR_STATIC, self.hist_static_price),
        ])
        if self.message:
            s.blit(self.small.render(self.message, True, (200, 60, 60)), (390, 722))
        pygame.display.flip()

    def run(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                self.handle(event)
            self.update()
            self.draw()
            self.clock.tick(60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--screenshot", help="headless mode: run one episode and save a PNG")
    parser.add_argument("--seed", type=int, default=3, help="market seed for the headless episode")
    args = parser.parse_args()
    if args.screenshot:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        app = Dashboard()
        app.episode_seed = args.seed - 1
        app.reset()
        while not app.done:
            app.step()
        app.draw()
        pygame.image.save(app.screen, args.screenshot)
        print(f"Saved {args.screenshot}  ({app.policy_name}: {app.info['total_revenue']:,.0f} "
              f"vs static {app.hist_static[-1]:,.0f})")
        pygame.quit()
    else:
        Dashboard().run()


if __name__ == "__main__":
    main()
