"""Tests for the store simulation.  Run:  python -m pytest tests"""
import numpy as np
import pytest

import config
from data.prepare_data import load_customer_profiles, load_demand_profile
from search.algorithms import ALGORITHMS
from simulation.store_sim import StoreSimulation, build_shopping_list

N_CUSTOMERS = 25
MAX_TICKS = 5000


def run_sim(layout="Compact 8x6", algorithm="A*", seed=0, n=N_CUSTOMERS):
    sim = StoreSimulation(layout, algorithm, n_customers=n, seed=seed)
    sim.run(MAX_TICKS)
    return sim


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def test_customer_profiles_schema():
    profiles = load_customer_profiles()
    assert list(profiles.columns) == ["Gender", "Age", "Product Category", "Quantity"]
    assert len(profiles) > 0 and profiles["Quantity"].min() >= 1


def test_demand_profile_shares_sum_to_one():
    profile = load_demand_profile()
    assert sum(profile["category_share"].values()) == pytest.approx(1.0, abs=1e-2)
    assert sum(profile["quantity_distribution"].values()) == pytest.approx(1.0, abs=1e-2)
    assert profile["source"] in ("kaggle", "synthetic")


# --------------------------------------------------------------------------
# Shopping lists
# --------------------------------------------------------------------------
def test_shopping_list_structure():
    share = load_demand_profile()["category_share"]
    rng = np.random.default_rng(0)
    for category, gender in [("Clothing", "Female"), ("Clothing", "Male"), ("Beauty", "Male"), ("Electronics", "Female")]:
        for quantity in (1, 2, 4):
            targets = build_shopping_list(category, gender, quantity, share, rng)
            assert targets[-2:] == ["C", "X"]
            assert set(targets) <= set(config.SECTIONS) | {"X"}
            shopping = [t for t in targets[:-2] if t != "F"]
            assert 1 <= len(shopping) <= 1 + config.MAX_EXTRA_STOPS
            assert all(a != b for a, b in zip(shopping, shopping[1:]))   # no immediate repeats
    # primary section follows the category (and gender for clothing)
    assert build_shopping_list("Clothing", "Female", 1, share, rng)[0] == "W"
    assert build_shopping_list("Clothing", "Male", 1, share, rng)[0] == "M"
    assert build_shopping_list("Beauty", "Male", 1, share, rng)[0] == "B"
    assert build_shopping_list("Electronics", "Female", 1, share, rng)[0] == "T"


# --------------------------------------------------------------------------
# Simulation
# --------------------------------------------------------------------------
@pytest.mark.parametrize("layout", list(config.LAYOUTS))
@pytest.mark.parametrize("algorithm", list(ALGORITHMS))
def test_all_customers_are_served(layout, algorithm):
    sim = run_sim(layout, algorithm)
    assert sim.done and sim.tick < MAX_TICKS
    assert sim.stats()["served"] == N_CUSTOMERS
    assert len(sim.customers) == N_CUSTOMERS
    for c in sim.customers:
        assert c.state == "done"
        assert c.targets[-1] == "X" and c.pos == sim.layout.exit


def test_customers_visit_their_targets_in_order():
    sim = StoreSimulation("Grid aisles", "A*", n_customers=10, seed=3)
    visited: dict[int, list[str]] = {}
    while not sim.done and sim.tick < MAX_TICKS:
        sim.step()
        for c in sim.active:
            if c.state == "dwelling":
                stops = visited.setdefault(c.id, [])
                if c.target_idx == len(stops):
                    stops.append(c.current_target)
    for c in sim.customers:
        assert visited[c.id] == c.targets[:-1]      # every stop except the exit, in order


def test_same_seed_is_reproducible_and_seeds_differ():
    a, b, other = run_sim(seed=5), run_sim(seed=5), run_sim(seed=6)
    assert a.stats() == b.stats()
    assert np.array_equal(a.heat, b.heat)
    assert [c.targets for c in a.customers] == [c.targets for c in b.customers]
    assert not np.array_equal(a.heat, other.heat)


def test_heat_counts_customer_ticks():
    sim = run_sim("Racetrack loop", "BFS")
    assert sim.heat.sum() == sum(c.ticks_in_store for c in sim.customers)
    assert sim.heat.sum() > 0
    assert sim.heat.min() >= 0
    assert sim.heat[~sim.layout.walkable_mask()].sum() == 0     # nobody stands on shelves
    # Every customer entered at the entrance, so it has at least one visit
    assert sim.heat[sim.layout.entrance] >= N_CUSTOMERS


def test_hotspots_and_mask():
    sim = run_sim("Grid aisles", "A*")
    top = sim.hotspots(3)
    assert len(top) == 3
    assert [count for _, count in top] == sorted((count for _, count in top), reverse=True)
    assert top[0][1] == sim.heat.max()
    mask = sim.hotspot_mask()
    assert mask.any()
    assert (mask & ~sim.layout.walkable_mask()).sum() == 0
    assert all(mask[cell] for cell, _ in top[:1])


def test_traffic_tracks_recent_crowd_and_decays():
    sim = StoreSimulation("Compact 8x6", "Congestion-aware", n_customers=5, seed=0)
    for _ in range(40):
        sim.step()
    assert sim.traffic.max() > 0 and sim.heat.sum() > 0
    # Traffic follows the recurrence traffic = traffic * decay + occupancy
    before, heat_before = sim.traffic.copy(), sim.heat.copy()
    sim.step()
    occupancy = sim.heat - heat_before
    assert np.allclose(sim.traffic, before * config.CONGESTION_DECAY + occupancy)
    # Once the store is empty it decays towards zero while heat stays fixed
    sim.run(MAX_TICKS)
    assert sim.done
    peak, heat_done = sim.traffic.max(), sim.heat.copy()
    for _ in range(20):
        sim.step()
    assert sim.traffic.max() == pytest.approx(peak * config.CONGESTION_DECAY ** 20)
    assert np.array_equal(sim.heat, heat_done)


def test_stats_are_consistent():
    sim = run_sim()
    s = sim.stats()
    assert s["n_searches"] == sum(len(c.targets) for c in sim.customers)   # one search per leg
    assert s["total_nodes_expanded"] == sum(c.nodes_expanded for c in sim.customers)
    assert s["avg_nodes_expanded_per_search"] == pytest.approx(s["total_nodes_expanded"] / s["n_searches"])
    assert s["peak_heat"] == sim.heat.max()
    assert s["avg_trip_ticks"] >= s["avg_path_length"]


def test_astar_expands_fewer_nodes_than_bfs_in_simulation():
    bfs_sim, astar_sim = run_sim("Grid aisles", "BFS"), run_sim("Grid aisles", "A*")
    assert astar_sim.stats()["total_nodes_expanded"] < bfs_sim.stats()["total_nodes_expanded"]
    # same seed -> same customers -> same total walking distance with equally short paths
    assert astar_sim.stats()["avg_path_length"] == bfs_sim.stats()["avg_path_length"]
