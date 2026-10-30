"""Discrete-time customer simulation (no graphics, so it is easy to test).

Customers enter at the entrance, visit the sections on their shopping list,
pay at the checkout and leave through the exit. Each leg of the trip is
planned with the chosen search algorithm, and the number of customers that
stand on every cell is accumulated into a heat grid.

Customers do not block each other: congestion only shows up in two grids.
`heat` is the cumulative visit count (the heatmap and hotspots); `traffic` is
an exponentially decaying copy of the recent crowd, read by the
congestion-aware planner.

Run:  python -m simulation.store_sim         (from task_a_pathfinding/) quick demo
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

import config
from data.prepare_data import load_customer_profiles, load_demand_profile
from search.algorithms import ALGORITHMS
from store.layout import Cell, load_layout


# --------------------------------------------------------------------------
# Customers
# --------------------------------------------------------------------------
@dataclass
class Customer:
    id: int
    gender: str
    age: int
    category: str                   # product category of the sampled transaction
    targets: list[str]              # section codes to visit, ending with "C" and "X"
    pos: Cell                       # current cell
    path: list[Cell] = field(default_factory=list)   # cells still to walk (excludes pos)
    state: str = "walking"          # "walking" | "dwelling" | "done"
    dwell_left: int = 0
    target_idx: int = 0
    ticks_in_store: int = 0
    steps_walked: int = 0
    nodes_expanded: int = 0         # search effort spent on this customer's trips

    @property
    def current_target(self) -> str:
        """Section code the customer is heading to / standing at (its colour is
        config.SECTIONS[code]['color'])."""
        return self.targets[self.target_idx]


def section_for(category: str, gender: str) -> str:
    """Section that sells a product category (clothing depends on gender)."""
    if category == "Clothing":
        return config.CLOTHING_SECTION_BY_GENDER.get(gender, "W")
    return config.CATEGORY_TO_SECTION[category]


def build_shopping_list(category: str, gender: str, quantity: int,
                        category_share: dict[str, float], rng: np.random.Generator) -> list[str]:
    """Section codes for one customer, ending with checkout and exit.

    1. the section of the purchased category,
    2. (quantity - 1) extra browsing stops (at most MAX_EXTRA_STOPS), drawn from
       the store-wide category mix and never the same section twice in a row,
    3. a fitting-room visit after each clothing section with FITTING_PROB.
    """
    sections = [section_for(category, gender)]
    weight_by_section = {section_for(cat, gender): share for cat, share in category_share.items()}
    for _ in range(min(quantity - 1, config.MAX_EXTRA_STOPS)):
        options = {s: w for s, w in weight_by_section.items() if s != sections[-1]}
        probs = np.array(list(options.values()))
        sections.append(str(rng.choice(list(options), p=probs / probs.sum())))

    targets: list[str] = []
    for s in sections:
        targets.append(s)
        if s in config.CLOTHING_SECTIONS and rng.random() < config.FITTING_PROB:
            targets.append("F")
    return targets + ["C", "X"]


# --------------------------------------------------------------------------
# Simulation
# --------------------------------------------------------------------------
class StoreSimulation:
    """One run of the store with a given layout and path-search algorithm.

    Each call to step() is one tick:
      1. every customer in the store dwells or walks one cell along its path;
      2. if a customer is due (every SPAWN_INTERVAL ticks), a new one appears
         on the entrance cell;
      3. the heat and traffic grids are updated.

    Heat semantics: at the end of every tick, heat[row, col] += 1 for each
    customer in the store, at the cell the customer occupies. This includes
    the tick of entering (entrance cell) and the tick of arriving at the exit
    (exit cell), and counts dwelling customers on every tick. Therefore
    heat.sum() == sum(c.ticks_in_store for all customers).

    Traffic semantics: with the same per-cell customer count, every tick
        traffic = traffic * CONGESTION_DECAY + (customers on each cell this tick)
    so it is heat with a memory of roughly 1 / (1 - decay) ticks, and it
    shrinks towards zero when the store is empty.
    """

    def __init__(self, layout_name: str = config.DEFAULT_LAYOUT,
                 algorithm_name: str = config.DEFAULT_ALGORITHM,
                 n_customers: int = config.DEFAULT_CUSTOMERS, seed: int = config.DEFAULT_SEED):
        if algorithm_name not in ALGORITHMS:
            raise ValueError(f"Unknown algorithm '{algorithm_name}'. Available: {list(ALGORITHMS)}")
        self.layout = load_layout(layout_name)
        self.algorithm_name = algorithm_name
        self.search = ALGORITHMS[algorithm_name]
        self.n_customers = n_customers
        self.seed = seed

        self.rng = np.random.default_rng(seed)
        self.profiles = load_customer_profiles()
        self.category_share = load_demand_profile()["category_share"]

        self.tick = 0
        self.heat = np.zeros((self.layout.height, self.layout.width), dtype=int)
        self.traffic = np.zeros((self.layout.height, self.layout.width))
        self.customers: list[Customer] = []     # every customer spawned so far
        self.active: list[Customer] = []        # customers currently in the store

        # Search statistics, one entry per planned trip leg
        self.nodes_per_search: list[int] = []
        self.path_lengths: list[int] = []

    # ----------------------------------------------------------------------
    # Running
    # ----------------------------------------------------------------------
    @property
    def done(self) -> bool:
        return len(self.customers) == self.n_customers and not self.active

    def run(self, max_ticks: int = config.MAX_TICKS) -> bool:
        """Step until every customer has left (or max_ticks is reached).
        Returns True if the run completed."""
        while not self.done and self.tick < max_ticks:
            self.step()
        return self.done

    def step(self) -> None:
        for customer in self.active:
            self._advance(customer)
        if len(self.customers) < self.n_customers and self.tick % config.SPAWN_INTERVAL == 0:
            self._spawn()
        occupancy = np.zeros_like(self.heat)
        for customer in self.active:
            occupancy[customer.pos] += 1
            customer.ticks_in_store += 1
        self.heat += occupancy
        self.traffic = self.traffic * config.CONGESTION_DECAY + occupancy
        self.active = [c for c in self.active if c.state != "done"]
        self.tick += 1

    # ----------------------------------------------------------------------
    # Customer behaviour
    # ----------------------------------------------------------------------
    def _spawn(self) -> None:
        row = self.profiles.iloc[int(self.rng.integers(len(self.profiles)))]
        customer = Customer(
            id=len(self.customers), gender=row["Gender"], age=int(row["Age"]),
            category=row["Product Category"], pos=self.layout.entrance,
            targets=build_shopping_list(row["Product Category"], row["Gender"], int(row["Quantity"]),
                                        self.category_share, self.rng),
        )
        self._plan(customer)
        self.customers.append(customer)
        self.active.append(customer)

    def _plan(self, customer: Customer) -> None:
        """Search a path from the customer's position to the current target."""
        goals = self.layout.goal_cells(customer.current_target)
        result = self.search(self.layout, customer.pos, goals, self.traffic)   # nearest of all goals
        if result.path is None:
            raise RuntimeError(f"No path from {customer.pos} to section {customer.current_target}")
        customer.path = result.path[1:]
        customer.state = "walking"
        customer.nodes_expanded += result.nodes_expanded
        self.nodes_per_search.append(result.nodes_expanded)
        self.path_lengths.append(len(customer.path))

    def _advance(self, customer: Customer) -> None:
        if customer.state == "dwelling":
            customer.dwell_left -= 1
            if customer.dwell_left <= 0:
                customer.target_idx += 1
                self._plan(customer)
            return
        if customer.path:
            customer.pos = customer.path.pop(0)
            customer.steps_walked += 1
        if not customer.path:                       # arrived at the target
            if customer.current_target == "X":
                customer.state = "done"
            else:
                customer.state = "dwelling"
                customer.dwell_left = int(self.rng.integers(config.DWELL_TICKS[0], config.DWELL_TICKS[1] + 1))

    # ----------------------------------------------------------------------
    # Results
    # ----------------------------------------------------------------------
    def stats(self) -> dict:
        finished = [c for c in self.customers if c.state == "done"]
        heat_on_floor = self.heat[self.layout.walkable_mask()]
        return {
            "ticks": self.tick,
            "served": len(finished),
            "n_searches": len(self.nodes_per_search),
            "total_nodes_expanded": int(sum(self.nodes_per_search)),
            "avg_nodes_expanded_per_search": float(np.mean(self.nodes_per_search)) if self.nodes_per_search else 0.0,
            "avg_path_length": float(np.mean(self.path_lengths)) if self.path_lengths else 0.0,
            "avg_trip_ticks": float(np.mean([c.ticks_in_store for c in finished])) if finished else 0.0,
            "peak_heat": int(self.heat.max()),
            "mean_heat": float(heat_on_floor.mean()),
        }

    def hotspots(self, k: int = config.TOP_K_HOTSPOTS) -> list[tuple[Cell, int]]:
        """The k most visited walkable cells as (cell, count), busiest first."""
        cells = [(cell, int(self.heat[cell])) for cell in self.layout.walkable_cells()]
        cells = [item for item in cells if item[1] > 0]
        cells.sort(key=lambda item: (-item[1], item[0]))
        return cells[:k]

    def hotspot_mask(self) -> np.ndarray:
        """Boolean grid of walkable cells whose heat is at or above the
        HOTSPOT_PERCENTILE of the visited (non-zero) cells."""
        visited = self.layout.walkable_mask() & (self.heat > 0)
        if not visited.any():
            return visited
        threshold = np.percentile(self.heat[visited], config.HOTSPOT_PERCENTILE)
        return visited & (self.heat >= threshold)


if __name__ == "__main__":
    for algo in ALGORITHMS:
        sim = StoreSimulation(config.DEFAULT_LAYOUT, algo, n_customers=30, seed=0)
        sim.run()
        print(f"{algo:<17} {sim.stats()}")
        print(f"{'':<17} hotspots: {sim.hotspots()}")
