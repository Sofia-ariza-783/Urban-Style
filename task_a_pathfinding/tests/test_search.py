"""Tests for the store layouts and search algorithms.  Run:  python -m pytest tests"""
import numpy as np
import pytest

import config
from search.algorithms import ALGORITHMS, astar, bfs, dijkstra_congestion
from store.layout import StoreLayout, load_layout

LAYOUT_NAMES = list(config.LAYOUTS)


@pytest.fixture(params=LAYOUT_NAMES)
def layout(request):
    return load_layout(request.param)


def random_pairs(layout, n, seed=0):
    rng = np.random.default_rng(seed)
    cells = layout.walkable_cells()
    idx = rng.integers(len(cells), size=(n, 2))
    return [(cells[i], cells[j]) for i, j in idx]


# --------------------------------------------------------------------------
# Layouts
# --------------------------------------------------------------------------
def test_exactly_three_layouts():
    assert len(config.LAYOUTS) == 3


def test_compact_layout_is_8_by_6():
    layout = load_layout("Compact 8x6")
    assert (layout.width, layout.height) == (8, 6)


def test_layout_has_all_sections_and_walkable_entrance_exit(layout):
    assert set(layout.sections) == set(config.SECTIONS)
    assert layout.walkable(layout.entrance) and layout.walkable(layout.exit)


def test_everything_reachable_from_entrance(layout):
    reachable = layout.reachable_from_entrance()
    assert layout.exit in reachable
    for code in config.SECTIONS:
        assert reachable.intersection(layout.access_cells(code))
    # No walkable cell is cut off from the rest of the store
    assert reachable == set(layout.walkable_cells())


def test_neighbors_are_walkable_and_adjacent(layout):
    for cell in layout.walkable_cells():
        for n in layout.neighbors(cell):
            assert layout.walkable(n)
            assert abs(n[0] - cell[0]) + abs(n[1] - cell[1]) == 1


def test_access_cells_touch_the_section(layout):
    for code, shelves in layout.sections.items():
        for cell in layout.access_cells(code):
            assert layout.walkable(cell)
            assert any(abs(cell[0] - s[0]) + abs(cell[1] - s[1]) == 1 for s in shelves)


def test_invalid_layouts_are_rejected():
    with pytest.raises(ValueError, match="exactly one"):
        StoreLayout("no exit", ["E.."])
    with pytest.raises(ValueError, match="exit is not reachable"):
        StoreLayout("unreachable exit", ["E#X"])
    with pytest.raises(ValueError, match="section 'W' is not reachable"):
        StoreLayout("unreachable section", ["E.#W#", "..X##"])   # no floor next to the shelf
    with pytest.raises(ValueError, match="Unknown layout"):
        load_layout("does not exist")


# --------------------------------------------------------------------------
# Search algorithms
# --------------------------------------------------------------------------
@pytest.mark.parametrize("name", list(ALGORITHMS))
def test_paths_are_valid(layout, name):
    for start, goal in random_pairs(layout, 30):
        path = ALGORITHMS[name](layout, start, goal).path
        assert path is not None
        assert path[0] == start and path[-1] == goal
        assert all(layout.walkable(c) for c in path)
        assert all(abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1 for a, b in zip(path, path[1:]))


@pytest.mark.parametrize("name", list(ALGORITHMS))
def test_start_equals_goal(layout, name):
    result = ALGORITHMS[name](layout, layout.entrance, layout.entrance)
    assert result.path == [layout.entrance]
    assert result.cost == 0


def test_bfs_and_astar_find_equally_short_paths(layout):
    total_bfs = total_astar = 0
    for start, goal in random_pairs(layout, 60, seed=1):
        b, a = bfs(layout, start, goal), astar(layout, start, goal)
        assert len(a.path) == len(b.path)
        assert a.cost == b.cost == len(a.path) - 1
        assert a.nodes_expanded <= b.nodes_expanded
        total_bfs += b.nodes_expanded
        total_astar += a.nodes_expanded
    assert total_astar < total_bfs


def test_plain_dijkstra_matches_bfs(layout):
    for start, goal in random_pairs(layout, 20, seed=2):
        assert dijkstra_congestion(layout, start, goal).cost == bfs(layout, start, goal).cost
        zeros = np.zeros((layout.height, layout.width))
        assert dijkstra_congestion(layout, start, goal, zeros).cost == bfs(layout, start, goal).cost


@pytest.mark.parametrize("name", list(ALGORITHMS))
def test_unreachable_goal_returns_none(name):
    layout = StoreLayout("walled-off cell", [
        "E..X",
        "..#.",
        ".#.#",    # the floor cell at (2, 2) is sealed in by walls
    ])
    assert (2, 2) not in layout.reachable_from_entrance()
    result = ALGORITHMS[name](layout, layout.entrance, (2, 2))
    assert result.path is None
    assert ALGORITHMS[name](layout, layout.entrance, [(2, 2)]).path is None   # goal given as a collection


def test_multi_goal_search_finds_the_nearest_goal(layout):
    """With several goals, BFS and A* reach the goal that is nearest by walking
    distance, i.e. the minimum over the single-goal BFS lengths."""
    total_bfs = total_astar = 0
    for code in config.SECTIONS:
        goals = layout.access_cells(code)
        for start in [layout.entrance, layout.exit] + [g for g in goals[:2]]:
            shortest = min(len(bfs(layout, start, g).path) for g in goals)
            b, a = bfs(layout, start, goals), astar(layout, start, goals)
            d = dijkstra_congestion(layout, start, set(goals))
            assert len(b.path) == len(a.path) == len(d.path) == shortest
            assert b.path[-1] in goals and a.path[-1] in goals and d.path[-1] in goals
            assert a.nodes_expanded <= b.nodes_expanded
            total_bfs += b.nodes_expanded
            total_astar += a.nodes_expanded
    assert total_astar < total_bfs


def test_multi_goal_with_start_inside_goals():
    layout = load_layout("Compact 8x6")
    goals = layout.access_cells("T")
    for algorithm in ALGORITHMS.values():
        assert algorithm(layout, goals[0], goals).path == [goals[0]]


def test_congestion_aware_avoids_busy_corridor():
    #  E . . . X     short route along the top (4 steps),
    #  . # # # .     long detour along the bottom (8 steps)
    #  . . . . .
    layout = StoreLayout("two routes", [
        "E...X",
        ".###.",
        ".....",
    ])
    start, goal = layout.entrance, layout.exit
    heat = np.zeros((layout.height, layout.width))
    heat[0, 1:4] = 100                              # the short route is crowded

    short = astar(layout, start, goal, heat)
    detour = dijkstra_congestion(layout, start, goal, heat)
    assert len(short.path) - 1 == 4
    assert len(detour.path) - 1 == 8
    assert all(cell[0] != 0 or cell in (start, goal) for cell in detour.path)
    # Without heat the congestion-aware planner also takes the short route
    assert len(dijkstra_congestion(layout, start, goal).path) - 1 == 4
